"""
Unified Validation Framework for GPU Optimizations

Consolidates all validation functionality into a single, comprehensive system
that provides numerical correctness, performance regression testing, and
memory validation for GPU optimization components.
"""

import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
import numpy as np

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of validation tests"""
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
    """Unified configuration for all validation testing"""
    # General settings
    validation_level: ValidationLevel = ValidationLevel.THOROUGH
    device: Optional[torch.device] = None
    enable_profiling: bool = True

    # Numerical tolerances
    numerical_tolerance: Dict[str, float] = field(default_factory=lambda: {
        'rtol': 1e-5,
        'atol': 1e-8,
        'float32_rtol': 1e-4,
        'float32_atol': 1e-7,
        'float16_rtol': 1e-2,
        'float16_atol': 1e-3
    })

    # Gradient tolerances
    gradient_tolerance: Dict[str, float] = field(default_factory=lambda: {
        'rtol': 1e-4,
        'atol': 1e-6
    })

    # Performance thresholds
    performance_regression_threshold: float = 0.05  # 5% regression threshold
    memory_overhead_threshold: float = 0.1         # 10% memory overhead threshold

    # Profiling settings
    profile_detailed_memory: bool = True
    profile_kernel_timing: bool = True


@dataclass
class ValidationResult:
    """Unified result structure for all validation tests"""
    validation_type: ValidationType
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    tolerance_used: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'validation_type': self.validation_type.value,
            'test_name': self.test_name,
            'passed': self.passed,
            'error_message': self.error_message,
            'metrics': self.metrics,
            'tolerance_used': self.tolerance_used,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage
        }


class UnifiedValidator:
    """
    Unified validation system that consolidates all validation functionality.

    Provides comprehensive validation for:
    - Numerical accuracy and correctness
    - Performance regression testing
    - Memory usage validation
    - Gradient correctness
    - Compiler optimization verification
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.device = self.config.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._validation_results: List[ValidationResult] = []

    def validate_numerical_accuracy(self,
                                   reference_func: Callable,
                                   optimized_func: Callable,
                                   test_inputs: List[torch.Tensor],
                                   test_name: str = "numerical_accuracy") -> ValidationResult:
        """
        Validate numerical accuracy between reference and optimized implementations.
        """
        start_time = time.perf_counter()

        try:
            # Move inputs to device
            device_inputs = [inp.to(self.device) for inp in test_inputs]

            # Get reference output
            with torch.no_grad():
                ref_output = reference_func(*device_inputs)

            # Get optimized output
            with torch.no_grad():
                opt_output = optimized_func(*device_inputs)

            # Determine tolerance based on data type
            if ref_output.dtype == torch.float16:
                rtol = self.config.numerical_tolerance['float16_rtol']
                atol = self.config.numerical_tolerance['float16_atol']
            elif ref_output.dtype == torch.float32:
                rtol = self.config.numerical_tolerance['float32_rtol']
                atol = self.config.numerical_tolerance['float32_atol']
            else:
                rtol = self.config.numerical_tolerance['rtol']
                atol = self.config.numerical_tolerance['atol']

            # Compare outputs
            if isinstance(ref_output, (list, tuple)):
                all_close = all(
                    torch.allclose(ref, opt, rtol=rtol, atol=atol)
                    for ref, opt in zip(ref_output, opt_output)
                )
                max_diff = max(
                    torch.max(torch.abs(ref - opt)).item()
                    for ref, opt in zip(ref_output, opt_output)
                )
            else:
                all_close = torch.allclose(ref_output, opt_output, rtol=rtol, atol=atol)
                max_diff = torch.max(torch.abs(ref_output - opt_output)).item()

            execution_time = time.perf_counter() - start_time

            result = ValidationResult(
                validation_type=ValidationType.NUMERICAL_ACCURACY,
                test_name=test_name,
                passed=all_close,
                metrics={
                    'max_absolute_difference': max_diff,
                    'reference_mean': torch.mean(ref_output).item() if not isinstance(ref_output, (list, tuple)) else 0.0,
                    'optimized_mean': torch.mean(opt_output).item() if not isinstance(opt_output, (list, tuple)) else 0.0
                },
                tolerance_used={'rtol': rtol, 'atol': atol},
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            result = ValidationResult(
                validation_type=ValidationType.NUMERICAL_ACCURACY,
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

        self._validation_results.append(result)
        return result

    def validate_performance(self,
                           baseline_func: Callable,
                           optimized_func: Callable,
                           test_inputs: List[torch.Tensor],
                           test_name: str = "performance",
                           num_warmup: int = 3,
                           num_iterations: int = 10) -> ValidationResult:
        """
        Validate that optimized function performs better than baseline.
        """
        start_time = time.perf_counter()

        try:
            device_inputs = [inp.to(self.device) for inp in test_inputs]

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = baseline_func(*device_inputs)
                    _ = optimized_func(*device_inputs)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # Benchmark baseline
            baseline_times = []
            for _ in range(num_iterations):
                iter_start = time.perf_counter()
                with torch.no_grad():
                    _ = baseline_func(*device_inputs)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                baseline_times.append(time.perf_counter() - iter_start)

            # Benchmark optimized
            optimized_times = []
            for _ in range(num_iterations):
                iter_start = time.perf_counter()
                with torch.no_grad():
                    _ = optimized_func(*device_inputs)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                optimized_times.append(time.perf_counter() - iter_start)

            baseline_avg = np.mean(baseline_times)
            optimized_avg = np.mean(optimized_times)
            speedup = baseline_avg / optimized_avg

            # Check for regression (optimized should be faster)
            is_regression = speedup < (1.0 - self.config.performance_regression_threshold)

            execution_time = time.perf_counter() - start_time

            result = ValidationResult(
                validation_type=ValidationType.PERFORMANCE_REGRESSION,
                test_name=test_name,
                passed=not is_regression,
                metrics={
                    'baseline_time_ms': baseline_avg * 1000,
                    'optimized_time_ms': optimized_avg * 1000,
                    'speedup': speedup,
                    'baseline_std': np.std(baseline_times) * 1000,
                    'optimized_std': np.std(optimized_times) * 1000
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            result = ValidationResult(
                validation_type=ValidationType.PERFORMANCE_REGRESSION,
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

        self._validation_results.append(result)
        return result

    def validate_memory_usage(self,
                             func: Callable,
                             test_inputs: List[torch.Tensor],
                             test_name: str = "memory_usage") -> ValidationResult:
        """
        Validate memory usage of a function.
        """
        start_time = time.perf_counter()

        try:
            device_inputs = [inp.to(self.device) for inp in test_inputs]

            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()

                with torch.no_grad():
                    _ = func(*device_inputs)

                peak_memory = torch.cuda.max_memory_allocated()
                final_memory = torch.cuda.memory_allocated()

                memory_metrics = {
                    'initial_memory_mb': initial_memory / 1024**2,
                    'peak_memory_mb': peak_memory / 1024**2,
                    'final_memory_mb': final_memory / 1024**2,
                    'memory_increase_mb': (peak_memory - initial_memory) / 1024**2
                }

                # Basic validation: no excessive memory usage
                passed = (peak_memory - initial_memory) / initial_memory < self.config.memory_overhead_threshold

            else:
                # CPU memory tracking is more complex, just run the function
                with torch.no_grad():
                    _ = func(*device_inputs)
                # TODO: Implement CPU memory tracking using psutil or tracemalloc
                # This would track process memory usage, peak allocations, and memory leaks
                memory_metrics = {'note': 'CPU memory tracking not implemented'}
                passed = True

            execution_time = time.perf_counter() - start_time

            result = ValidationResult(
                validation_type=ValidationType.MEMORY_CORRECTNESS,
                test_name=test_name,
                passed=passed,
                metrics=memory_metrics,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            result = ValidationResult(
                validation_type=ValidationType.MEMORY_CORRECTNESS,
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

        self._validation_results.append(result)
        return result

    def validate_gradients(self,
                          reference_func: Callable,
                          optimized_func: Callable,
                          test_inputs: List[torch.Tensor],
                          test_name: str = "gradient_correctness") -> ValidationResult:
        """
        Validate gradient correctness between reference and optimized implementations.
        """
        start_time = time.perf_counter()

        try:
            # Create inputs that require gradients
            device_inputs_ref = [inp.clone().detach().requires_grad_(True).to(self.device) for inp in test_inputs]
            device_inputs_opt = [inp.clone().detach().requires_grad_(True).to(self.device) for inp in test_inputs]

            # Forward pass and backward pass for reference
            ref_output = reference_func(*device_inputs_ref)
            ref_loss = torch.sum(ref_output) if not isinstance(ref_output, torch.Tensor) else torch.sum(ref_output)
            ref_loss.backward()

            # Forward pass and backward pass for optimized
            opt_output = optimized_func(*device_inputs_opt)
            opt_loss = torch.sum(opt_output) if not isinstance(opt_output, torch.Tensor) else torch.sum(opt_output)
            opt_loss.backward()

            # Compare gradients
            rtol = self.config.gradient_tolerance['rtol']
            atol = self.config.gradient_tolerance['atol']

            gradients_match = True
            max_grad_diff = 0.0

            for ref_inp, opt_inp in zip(device_inputs_ref, device_inputs_opt):
                if ref_inp.grad is not None and opt_inp.grad is not None:
                    if not torch.allclose(ref_inp.grad, opt_inp.grad, rtol=rtol, atol=atol):
                        gradients_match = False
                    max_grad_diff = max(max_grad_diff, torch.max(torch.abs(ref_inp.grad - opt_inp.grad)).item())

            execution_time = time.perf_counter() - start_time

            result = ValidationResult(
                validation_type=ValidationType.GRADIENT_CORRECTNESS,
                test_name=test_name,
                passed=gradients_match,
                metrics={
                    'max_gradient_difference': max_grad_diff,
                    'reference_loss': ref_loss.item(),
                    'optimized_loss': opt_loss.item()
                },
                tolerance_used={'rtol': rtol, 'atol': atol},
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            result = ValidationResult(
                validation_type=ValidationType.GRADIENT_CORRECTNESS,
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

        self._validation_results.append(result)
        return result

    def validate_component(self,
                          component: nn.Module,
                          reference_component: Optional[nn.Module] = None,
                          test_inputs: Optional[List[torch.Tensor]] = None,
                          validation_types: Optional[List[ValidationType]] = None) -> List[ValidationResult]:
        """
        Comprehensive validation of a component against reference implementation.
        """
        if validation_types is None:
            validation_types = [ValidationType.NUMERICAL_ACCURACY, ValidationType.PERFORMANCE_REGRESSION]

        if test_inputs is None:
            # Generate default test inputs based on component
            test_inputs = self._generate_test_inputs(component)

        results = []

        if reference_component is not None:
            if ValidationType.NUMERICAL_ACCURACY in validation_types:
                results.append(self.validate_numerical_accuracy(
                    reference_component,
                    component,
                    test_inputs,
                    f"{component.__class__.__name__}_numerical"
                ))

            if ValidationType.PERFORMANCE_REGRESSION in validation_types:
                results.append(self.validate_performance(
                    reference_component,
                    component,
                    test_inputs,
                    f"{component.__class__.__name__}_performance"
                ))

            if ValidationType.GRADIENT_CORRECTNESS in validation_types:
                results.append(self.validate_gradients(
                    reference_component,
                    component,
                    test_inputs,
                    f"{component.__class__.__name__}_gradients"
                ))

        if ValidationType.MEMORY_CORRECTNESS in validation_types:
            results.append(self.validate_memory_usage(
                component,
                test_inputs,
                f"{component.__class__.__name__}_memory"
            ))

        return results

    def _generate_test_inputs(self, component: nn.Module) -> List[torch.Tensor]:
        """Generate appropriate test inputs for a component"""
        # Default: assume it's an attention-like component
        batch_size, seq_len, embed_dim = 2, 128, 256
        return [torch.randn(batch_size, seq_len, embed_dim)]

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        total_tests = len(self._validation_results)
        passed_tests = sum(1 for result in self._validation_results if result.passed)

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / max(total_tests, 1),
            'results': [result.to_dict() for result in self._validation_results]
        }

    def clear_results(self):
        """Clear all validation results"""
        self._validation_results.clear()


# Convenience functions for backward compatibility and ease of use
def create_validator(config: Optional[ValidationConfig] = None) -> UnifiedValidator:
    """Create a unified validator instance"""
    return UnifiedValidator(config)


def validate_component_quick(component: nn.Module,
                           reference: Optional[nn.Module] = None,
                           device: Optional[torch.device] = None) -> bool:
    """Quick validation check for a component"""
    config = ValidationConfig(validation_level=ValidationLevel.BASIC, device=device)
    validator = UnifiedValidator(config)

    if reference is not None:
        results = validator.validate_component(component, reference)
        return all(result.passed for result in results)
    else:
        # Just test that component runs without error
        test_inputs = validator._generate_test_inputs(component)
        try:
            with torch.no_grad():
                _ = component(*test_inputs)
            return True
        except Exception:
            return False