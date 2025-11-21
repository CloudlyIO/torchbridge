"""
Validation and Profiling Tools for GPU Optimizations

Advanced validation framework for verifying correctness and profiling performance
of GPU kernel optimizations, compiler transformations, and memory optimizations.
"""

import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.profiler
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import warnings
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of validation"""
    NUMERICAL_ACCURACY = "numerical_accuracy"
    MEMORY_CORRECTNESS = "memory_correctness"
    PERFORMANCE_REGRESSION = "performance_regression"
    GRADIENT_CORRECTNESS = "gradient_correctness"
    COMPILER_CORRECTNESS = "compiler_correctness"


class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"           # Basic correctness checks
    THOROUGH = "thorough"     # Comprehensive validation
    EXHAUSTIVE = "exhaustive" # Exhaustive testing with edge cases


@dataclass
class ValidationConfig:
    """Configuration for validation testing"""
    validation_level: ValidationLevel = ValidationLevel.THOROUGH
    numerical_tolerance: Dict[str, float] = field(default_factory=lambda: {
        'rtol': 1e-5,
        'atol': 1e-8,
        'float32_rtol': 1e-4,
        'float32_atol': 1e-7,
        'float16_rtol': 1e-2,
        'float16_atol': 1e-3
    })
    gradient_tolerance: Dict[str, float] = field(default_factory=lambda: {
        'rtol': 1e-4,
        'atol': 1e-6
    })
    performance_regression_threshold: float = 0.05  # 5% regression threshold
    memory_overhead_threshold: float = 0.1         # 10% memory overhead threshold
    enable_profiling: bool = True
    profile_detailed_memory: bool = True
    profile_kernel_timing: bool = True


@dataclass
class ValidationResult:
    """Results from validation testing"""
    validation_type: ValidationType
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    tolerance_used: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)


class NumericalValidator:
    """
    Numerical accuracy validator for GPU optimizations

    Validates that optimized implementations produce numerically
    equivalent results to baseline implementations.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_accuracy(
        self,
        baseline_fn: Callable,
        optimized_fn: Callable,
        inputs: List[torch.Tensor],
        test_name: str,
        custom_tolerance: Optional[Dict[str, float]] = None
    ) -> ValidationResult:
        """
        Validate numerical accuracy between baseline and optimized implementations

        Args:
            baseline_fn: Baseline implementation
            optimized_fn: Optimized implementation
            inputs: Input tensors
            test_name: Name of the test
            custom_tolerance: Custom tolerance values

        Returns:
            ValidationResult with accuracy assessment
        """
        start_time = time.time()

        result = ValidationResult(
            validation_type=ValidationType.NUMERICAL_ACCURACY,
            test_name=test_name,
            passed=False
        )

        try:
            # Get tolerance values
            tolerance = custom_tolerance or self.config.numerical_tolerance
            device = inputs[0].device if inputs else torch.device('cpu')
            dtype = inputs[0].dtype if inputs else torch.float32

            # Adjust tolerance based on dtype
            if dtype == torch.float16:
                rtol = tolerance.get('float16_rtol', 1e-2)
                atol = tolerance.get('float16_atol', 1e-3)
            elif dtype == torch.float32:
                rtol = tolerance.get('float32_rtol', 1e-4)
                atol = tolerance.get('float32_atol', 1e-7)
            else:
                rtol = tolerance.get('rtol', 1e-5)
                atol = tolerance.get('atol', 1e-8)

            result.tolerance_used = {'rtol': rtol, 'atol': atol}

            # Execute baseline
            with torch.no_grad():
                baseline_output = baseline_fn(*inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Execute optimized
            with torch.no_grad():
                optimized_output = optimized_fn(*inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Compare outputs
            if isinstance(baseline_output, torch.Tensor):
                baseline_outputs = [baseline_output]
                optimized_outputs = [optimized_output]
            else:
                baseline_outputs = baseline_output
                optimized_outputs = optimized_output

            accuracy_metrics = {}
            all_close = True

            for i, (baseline_out, optimized_out) in enumerate(zip(baseline_outputs, optimized_outputs)):
                # Basic shape check
                if baseline_out.shape != optimized_out.shape:
                    result.error_message = f"Output {i} shape mismatch: {baseline_out.shape} vs {optimized_out.shape}"
                    all_close = False
                    break

                # Numerical comparison
                try:
                    is_close = torch.allclose(baseline_out, optimized_out, rtol=rtol, atol=atol)

                    # Calculate detailed metrics
                    abs_diff = torch.abs(baseline_out - optimized_out)
                    rel_diff = abs_diff / (torch.abs(baseline_out) + atol)

                    accuracy_metrics[f'output_{i}'] = {
                        'allclose': is_close,
                        'max_abs_diff': abs_diff.max().item(),
                        'mean_abs_diff': abs_diff.mean().item(),
                        'max_rel_diff': rel_diff.max().item(),
                        'mean_rel_diff': rel_diff.mean().item(),
                        'shape': list(baseline_out.shape),
                        'dtype': str(baseline_out.dtype)
                    }

                    if not is_close:
                        all_close = False

                        # Additional diagnostic information
                        if self.config.validation_level == ValidationLevel.EXHAUSTIVE:
                            accuracy_metrics[f'output_{i}']['mismatch_locations'] = self._find_mismatch_locations(
                                baseline_out, optimized_out, rtol, atol
                            )

                except Exception as e:
                    result.error_message = f"Comparison failed for output {i}: {str(e)}"
                    all_close = False
                    break

            result.passed = all_close
            result.metrics = accuracy_metrics

            if not all_close and not result.error_message:
                result.error_message = "Numerical accuracy validation failed"

        except Exception as e:
            result.error_message = f"Validation execution failed: {str(e)}"
            result.passed = False

        result.execution_time = time.time() - start_time
        return result

    def _find_mismatch_locations(
        self,
        baseline: torch.Tensor,
        optimized: torch.Tensor,
        rtol: float,
        atol: float
    ) -> Dict[str, Any]:
        """Find locations where tensors don't match"""
        abs_diff = torch.abs(baseline - optimized)
        rel_diff = abs_diff / (torch.abs(baseline) + atol)

        # Find elements that don't match
        mismatch_mask = ~torch.isclose(baseline, optimized, rtol=rtol, atol=atol)
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)

        if len(mismatch_indices) > 0:
            # Sample up to 10 mismatch locations
            sample_size = min(10, len(mismatch_indices))
            sample_indices = mismatch_indices[:sample_size]

            mismatch_info = []
            for idx in sample_indices:
                idx_tuple = tuple(idx.tolist())
                mismatch_info.append({
                    'index': idx_tuple,
                    'baseline_value': baseline[idx_tuple].item(),
                    'optimized_value': optimized[idx_tuple].item(),
                    'abs_diff': abs_diff[idx_tuple].item(),
                    'rel_diff': rel_diff[idx_tuple].item()
                })

            return {
                'total_mismatches': len(mismatch_indices),
                'mismatch_rate': len(mismatch_indices) / baseline.numel(),
                'sample_mismatches': mismatch_info
            }

        return {'total_mismatches': 0, 'mismatch_rate': 0.0}

    def validate_gradients(
        self,
        baseline_fn: Callable,
        optimized_fn: Callable,
        inputs: List[torch.Tensor],
        test_name: str
    ) -> ValidationResult:
        """Validate gradient correctness"""
        start_time = time.time()

        result = ValidationResult(
            validation_type=ValidationType.GRADIENT_CORRECTNESS,
            test_name=test_name,
            passed=False
        )

        try:
            # Ensure inputs require gradients
            grad_inputs = []
            for inp in inputs:
                if inp.dtype.is_floating_point:
                    grad_inp = inp.clone().detach().requires_grad_(True)
                    grad_inputs.append(grad_inp)
                else:
                    grad_inputs.append(inp)

            # Baseline gradient computation
            baseline_inputs = [inp.clone() if inp.requires_grad else inp for inp in grad_inputs]
            baseline_output = baseline_fn(*baseline_inputs)

            if isinstance(baseline_output, tuple):
                baseline_loss = baseline_output[0].sum()
            else:
                baseline_loss = baseline_output.sum()

            baseline_loss.backward()
            baseline_grads = [inp.grad.clone() if inp.grad is not None else None
                            for inp in baseline_inputs if inp.requires_grad]

            # Clear gradients
            for inp in baseline_inputs:
                if inp.grad is not None:
                    inp.grad.zero_()

            # Optimized gradient computation
            optimized_inputs = [inp.clone() if inp.requires_grad else inp for inp in grad_inputs]
            optimized_output = optimized_fn(*optimized_inputs)

            if isinstance(optimized_output, tuple):
                optimized_loss = optimized_output[0].sum()
            else:
                optimized_loss = optimized_output.sum()

            optimized_loss.backward()
            optimized_grads = [inp.grad.clone() if inp.grad is not None else None
                             for inp in optimized_inputs if inp.requires_grad]

            # Compare gradients
            tolerance = self.config.gradient_tolerance
            rtol = tolerance['rtol']
            atol = tolerance['atol']

            gradient_metrics = {}
            all_close = True

            for i, (baseline_grad, optimized_grad) in enumerate(zip(baseline_grads, optimized_grads)):
                if baseline_grad is None and optimized_grad is None:
                    continue

                if baseline_grad is None or optimized_grad is None:
                    result.error_message = f"Gradient {i} presence mismatch"
                    all_close = False
                    break

                is_close = torch.allclose(baseline_grad, optimized_grad, rtol=rtol, atol=atol)

                abs_diff = torch.abs(baseline_grad - optimized_grad)
                rel_diff = abs_diff / (torch.abs(baseline_grad) + atol)

                gradient_metrics[f'gradient_{i}'] = {
                    'allclose': is_close,
                    'max_abs_diff': abs_diff.max().item(),
                    'mean_abs_diff': abs_diff.mean().item(),
                    'max_rel_diff': rel_diff.max().item(),
                    'mean_rel_diff': rel_diff.mean().item()
                }

                if not is_close:
                    all_close = False

            result.passed = all_close
            result.metrics = gradient_metrics
            result.tolerance_used = {'rtol': rtol, 'atol': atol}

            if not all_close:
                result.error_message = "Gradient validation failed"

        except Exception as e:
            result.error_message = f"Gradient validation execution failed: {str(e)}"
            result.passed = False

        result.execution_time = time.time() - start_time
        return result


class MemoryValidator:
    """
    Memory usage and correctness validator

    Validates memory access patterns, detects memory leaks,
    and ensures memory safety of optimized implementations.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_memory_usage(
        self,
        baseline_fn: Callable,
        optimized_fn: Callable,
        inputs: List[torch.Tensor],
        test_name: str
    ) -> ValidationResult:
        """
        Validate memory usage patterns

        Args:
            baseline_fn: Baseline implementation
            optimized_fn: Optimized implementation
            inputs: Input tensors
            test_name: Name of the test

        Returns:
            ValidationResult with memory usage assessment
        """
        start_time = time.time()

        result = ValidationResult(
            validation_type=ValidationType.MEMORY_CORRECTNESS,
            test_name=test_name,
            passed=False
        )

        try:
            device = inputs[0].device if inputs else torch.device('cpu')

            # Measure baseline memory usage
            baseline_memory = self._measure_memory_usage(baseline_fn, inputs, device)

            # Measure optimized memory usage
            optimized_memory = self._measure_memory_usage(optimized_fn, inputs, device)

            # Calculate memory metrics
            memory_metrics = {
                'baseline_peak_mb': baseline_memory['peak_mb'],
                'optimized_peak_mb': optimized_memory['peak_mb'],
                'baseline_allocated_mb': baseline_memory['allocated_mb'],
                'optimized_allocated_mb': optimized_memory['allocated_mb'],
                'memory_overhead_ratio': (
                    optimized_memory['peak_mb'] / max(baseline_memory['peak_mb'], 1) - 1
                ),
                'allocation_overhead_ratio': (
                    optimized_memory['allocated_mb'] / max(baseline_memory['allocated_mb'], 1) - 1
                )
            }

            # Check for excessive memory overhead
            memory_overhead = memory_metrics['memory_overhead_ratio']
            passed = memory_overhead <= self.config.memory_overhead_threshold

            result.passed = passed
            result.metrics = memory_metrics
            result.memory_usage = {
                'baseline_peak_mb': baseline_memory['peak_mb'],
                'optimized_peak_mb': optimized_memory['peak_mb']
            }

            if not passed:
                result.error_message = (
                    f"Memory overhead too high: {memory_overhead:.2%} "
                    f"(threshold: {self.config.memory_overhead_threshold:.2%})"
                )

        except Exception as e:
            result.error_message = f"Memory validation failed: {str(e)}"
            result.passed = False

        result.execution_time = time.time() - start_time
        return result

    def _measure_memory_usage(
        self,
        fn: Callable,
        inputs: List[torch.Tensor],
        device: torch.device
    ) -> Dict[str, float]:
        """Measure memory usage during function execution"""
        if device.type == 'cuda':
            # Clear cache and reset peak memory stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            memory_before = torch.cuda.memory_allocated(device)

            # Execute function
            with torch.no_grad():
                _ = fn(*inputs)
                torch.cuda.synchronize(device)

            memory_after = torch.cuda.memory_allocated(device)
            peak_memory = torch.cuda.max_memory_allocated(device)

            return {
                'allocated_mb': (memory_after - memory_before) / (1024 ** 2),
                'peak_mb': peak_memory / (1024 ** 2)
            }
        else:
            # CPU memory measurement is more challenging
            # Use a simple approach for now
            return {
                'allocated_mb': 0.0,
                'peak_mb': 0.0
            }

    def detect_memory_leaks(
        self,
        fn: Callable,
        inputs: List[torch.Tensor],
        iterations: int = 10
    ) -> ValidationResult:
        """Detect memory leaks by running function multiple times"""
        start_time = time.time()

        result = ValidationResult(
            validation_type=ValidationType.MEMORY_CORRECTNESS,
            test_name="memory_leak_detection",
            passed=False
        )

        try:
            device = inputs[0].device if inputs else torch.device('cpu')

            if device.type != 'cuda':
                result.error_message = "Memory leak detection only supported on CUDA"
                return result

            memory_usage_history = []

            for i in range(iterations):
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(device)

                with torch.no_grad():
                    _ = fn(*inputs)
                    torch.cuda.synchronize(device)

                memory_after = torch.cuda.memory_allocated(device)
                memory_usage_history.append(memory_after - memory_before)

            # Analyze memory usage trend
            memory_trend = np.polyfit(range(iterations), memory_usage_history, 1)[0]
            memory_variance = np.var(memory_usage_history)

            # Check for memory leak (increasing trend)
            leak_threshold = 1024 * 1024  # 1MB per iteration
            has_leak = memory_trend > leak_threshold

            result.passed = not has_leak
            result.metrics = {
                'memory_trend_bytes_per_iter': memory_trend,
                'memory_variance': memory_variance,
                'iterations_tested': iterations,
                'memory_usage_history': memory_usage_history
            }

            if has_leak:
                result.error_message = f"Memory leak detected: {memory_trend / 1024 / 1024:.2f} MB/iter"

        except Exception as e:
            result.error_message = f"Memory leak detection failed: {str(e)}"
            result.passed = False

        result.execution_time = time.time() - start_time
        return result


class PerformanceProfiler:
    """
    Advanced performance profiler for GPU operations

    Provides detailed performance profiling including kernel timing,
    memory bandwidth utilization, and compute efficiency analysis.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def profile_execution(
        self,
        fn: Callable,
        inputs: List[torch.Tensor],
        test_name: str,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Profile function execution with detailed metrics

        Args:
            fn: Function to profile
            inputs: Input tensors
            test_name: Name of the profiling test
            iterations: Number of iterations to profile

        Returns:
            Detailed profiling results
        """
        device = inputs[0].device if inputs else torch.device('cpu')

        profiling_results = {
            'test_name': test_name,
            'device': str(device),
            'iterations': iterations,
            'timing': {},
            'memory': {},
            'kernel_info': {}
        }

        # Basic timing profiling
        profiling_results['timing'] = self._profile_timing(fn, inputs, iterations)

        # Memory profiling
        if device.type == 'cuda' and self.config.profile_detailed_memory:
            profiling_results['memory'] = self._profile_memory(fn, inputs)

        # PyTorch profiler (if available and enabled)
        if self.config.profile_kernel_timing and device.type == 'cuda':
            profiling_results['kernel_info'] = self._profile_kernels(fn, inputs)

        return profiling_results

    def _profile_timing(
        self,
        fn: Callable,
        inputs: List[torch.Tensor],
        iterations: int
    ) -> Dict[str, float]:
        """Profile execution timing"""
        device = inputs[0].device if inputs else torch.device('cpu')

        # Warmup
        for _ in range(min(10, iterations // 10)):
            with torch.no_grad():
                _ = fn(*inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)

        # Timing measurements
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()

            with torch.no_grad():
                _ = fn(*inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99)
        }

    def _profile_memory(
        self,
        fn: Callable,
        inputs: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Profile memory usage patterns"""
        device = inputs[0].device

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        memory_before = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            output = fn(*inputs)
            torch.cuda.synchronize(device)

        memory_after = torch.cuda.memory_allocated(device)
        peak_memory = torch.cuda.max_memory_allocated(device)

        # Estimate memory bandwidth
        input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in inputs)
        output_bytes = 0
        if isinstance(output, torch.Tensor):
            output_bytes = output.numel() * output.element_size()
        elif isinstance(output, (list, tuple)):
            for out_tensor in output:
                if isinstance(out_tensor, torch.Tensor):
                    output_bytes += out_tensor.numel() * out_tensor.element_size()

        total_bytes = input_bytes + output_bytes

        return {
            'allocated_mb': (memory_after - memory_before) / (1024 ** 2),
            'peak_mb': peak_memory / (1024 ** 2),
            'input_mb': input_bytes / (1024 ** 2),
            'output_mb': output_bytes / (1024 ** 2),
            'total_data_mb': total_bytes / (1024 ** 2)
        }

    def _profile_kernels(
        self,
        fn: Callable,
        inputs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Profile CUDA kernels using PyTorch profiler"""
        try:
            kernel_info = {}

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False
            ) as prof:
                with torch.no_grad():
                    _ = fn(*inputs)
                    torch.cuda.synchronize()

            # Extract kernel information
            events = prof.events()
            cuda_events = [event for event in events if event.device_type == torch.profiler.DeviceType.CUDA]

            if cuda_events:
                total_time = sum(event.cuda_time_total for event in cuda_events)
                kernel_count = len(cuda_events)

                kernel_info = {
                    'total_kernel_time_us': total_time,
                    'kernel_count': kernel_count,
                    'avg_kernel_time_us': total_time / max(kernel_count, 1),
                    'top_kernels': []
                }

                # Get top 5 most time-consuming kernels
                sorted_events = sorted(cuda_events, key=lambda e: e.cuda_time_total, reverse=True)
                for event in sorted_events[:5]:
                    kernel_info['top_kernels'].append({
                        'name': event.name,
                        'time_us': event.cuda_time_total,
                        'time_percentage': (event.cuda_time_total / total_time * 100) if total_time > 0 else 0
                    })

            return kernel_info

        except Exception as e:
            logger.warning(f"Kernel profiling failed: {e}")
            return {'error': str(e)}


class OptimizationValidator:
    """
    Comprehensive optimization validator

    Combines numerical, memory, and performance validation to provide
    complete validation of GPU optimizations.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()

        self.numerical_validator = NumericalValidator(self.config)
        self.memory_validator = MemoryValidator(self.config)
        self.profiler = PerformanceProfiler(self.config)

    def validate_optimization(
        self,
        baseline_fn: Callable,
        optimized_fn: Callable,
        inputs: List[torch.Tensor],
        test_name: str,
        enable_gradient_check: bool = False
    ) -> Dict[str, ValidationResult]:
        """
        Comprehensive validation of an optimization

        Args:
            baseline_fn: Baseline implementation
            optimized_fn: Optimized implementation
            inputs: Input tensors
            test_name: Name of the validation test
            enable_gradient_check: Whether to check gradient correctness

        Returns:
            Dictionary of validation results by validation type
        """
        results = {}

        # Numerical accuracy validation
        logger.info(f"Validating numerical accuracy: {test_name}")
        results['numerical'] = self.numerical_validator.validate_accuracy(
            baseline_fn, optimized_fn, inputs, test_name
        )

        # Gradient correctness validation
        if enable_gradient_check:
            logger.info(f"Validating gradient correctness: {test_name}")
            results['gradients'] = self.numerical_validator.validate_gradients(
                baseline_fn, optimized_fn, inputs, test_name
            )

        # Memory usage validation
        logger.info(f"Validating memory usage: {test_name}")
        results['memory'] = self.memory_validator.validate_memory_usage(
            baseline_fn, optimized_fn, inputs, test_name
        )

        # Memory leak detection
        if self.config.validation_level == ValidationLevel.EXHAUSTIVE:
            logger.info(f"Detecting memory leaks: {test_name}")
            results['memory_leaks'] = self.memory_validator.detect_memory_leaks(
                optimized_fn, inputs
            )

        return results

    def generate_validation_report(
        self,
        validation_results: Dict[str, Dict[str, ValidationResult]]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'summary': {
                'total_tests': len(validation_results),
                'passed_tests': 0,
                'failed_tests': 0,
                'validation_types': set()
            },
            'test_results': [],
            'failure_analysis': {},
            'recommendations': []
        }

        # Process results
        for test_name, test_results in validation_results.items():
            test_passed = all(result.passed for result in test_results.values())

            if test_passed:
                report['summary']['passed_tests'] += 1
            else:
                report['summary']['failed_tests'] += 1

            test_entry = {
                'test_name': test_name,
                'overall_passed': test_passed,
                'validations': {}
            }

            for validation_type, result in test_results.items():
                report['summary']['validation_types'].add(result.validation_type.value)

                test_entry['validations'][validation_type] = {
                    'passed': result.passed,
                    'error_message': result.error_message,
                    'execution_time': result.execution_time,
                    'key_metrics': self._extract_key_metrics(result)
                }

                # Collect failure patterns
                if not result.passed:
                    failure_key = f"{result.validation_type.value}_{result.error_message or 'unknown'}"
                    if failure_key not in report['failure_analysis']:
                        report['failure_analysis'][failure_key] = []
                    report['failure_analysis'][failure_key].append(test_name)

            report['test_results'].append(test_entry)

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['failure_analysis'])
        report['summary']['validation_types'] = list(report['summary']['validation_types'])

        return report

    def _extract_key_metrics(self, result: ValidationResult) -> Dict[str, Any]:
        """Extract key metrics from validation result"""
        key_metrics = {}

        if result.validation_type == ValidationType.NUMERICAL_ACCURACY:
            if 'output_0' in result.metrics:
                output_metrics = result.metrics['output_0']
                key_metrics = {
                    'max_abs_diff': output_metrics.get('max_abs_diff'),
                    'max_rel_diff': output_metrics.get('max_rel_diff'),
                    'tolerance_used': result.tolerance_used
                }

        elif result.validation_type == ValidationType.MEMORY_CORRECTNESS:
            key_metrics = {
                'memory_overhead_ratio': result.metrics.get('memory_overhead_ratio'),
                'peak_memory_mb': result.metrics.get('optimized_peak_mb')
            }

        elif result.validation_type == ValidationType.GRADIENT_CORRECTNESS:
            if 'gradient_0' in result.metrics:
                grad_metrics = result.metrics['gradient_0']
                key_metrics = {
                    'max_abs_diff': grad_metrics.get('max_abs_diff'),
                    'max_rel_diff': grad_metrics.get('max_rel_diff')
                }

        return key_metrics

    def _generate_recommendations(self, failure_analysis: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations based on failure patterns"""
        recommendations = []

        for failure_pattern, affected_tests in failure_analysis.items():
            if 'numerical_accuracy' in failure_pattern:
                recommendations.append(
                    f"Numerical accuracy issues detected in {len(affected_tests)} tests. "
                    "Consider adjusting tolerance levels or investigating precision loss."
                )

            elif 'memory_correctness' in failure_pattern:
                recommendations.append(
                    f"Memory usage issues detected in {len(affected_tests)} tests. "
                    "Review memory allocation patterns and consider optimization."
                )

            elif 'gradient_correctness' in failure_pattern:
                recommendations.append(
                    f"Gradient correctness issues detected in {len(affected_tests)} tests. "
                    "Verify that automatic differentiation is properly implemented."
                )

        if not recommendations:
            recommendations.append("All validations passed successfully!")

        return recommendations


def create_validation_suite(
    validation_level: str = "thorough",
    numerical_tolerance: Optional[Dict[str, float]] = None,
    enable_profiling: bool = True
) -> OptimizationValidator:
    """
    Factory function to create validation suite

    Args:
        validation_level: Level of validation ("basic", "thorough", "exhaustive")
        numerical_tolerance: Custom numerical tolerance values
        enable_profiling: Enable performance profiling

    Returns:
        Configured OptimizationValidator
    """
    config = ValidationConfig(
        validation_level=ValidationLevel(validation_level),
        enable_profiling=enable_profiling,
        profile_detailed_memory=enable_profiling,
        profile_kernel_timing=enable_profiling
    )

    if numerical_tolerance:
        config.numerical_tolerance.update(numerical_tolerance)

    return OptimizationValidator(config)