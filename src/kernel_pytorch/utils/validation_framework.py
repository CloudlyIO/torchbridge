"""
Enhanced Validation Framework
============================

Comprehensive validation and testing framework for GPU optimization components,
ensuring correctness, performance, and reliability across all optimization levels.

This module provides:
1. Automated correctness validation
2. Performance regression testing
3. Numerical stability verification
4. Cross-platform compatibility testing
5. Educational validation examples
6. Continuous integration utilities

Author: Advanced GPU Optimization Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type
import numpy as np
import time
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import os
from pathlib import Path
import pytest
from collections import defaultdict
import matplotlib.pyplot as plt


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    error_message: Optional[str]
    metrics: Dict[str, float]
    execution_time: float
    memory_usage: float


@dataclass
class PerformanceRegression:
    """Performance regression test result."""
    component_name: str
    baseline_time: float
    current_time: float
    regression_percent: float
    threshold_percent: float
    is_regression: bool


class NumericalValidator:
    """
    Validates numerical correctness and stability of GPU optimization components.

    Ensures that optimized implementations produce mathematically equivalent
    results to reference implementations within acceptable tolerances.
    """

    def __init__(self,
                 rtol: float = 1e-5,
                 atol: float = 1e-8,
                 device: Optional[torch.device] = None):
        self.rtol = rtol
        self.atol = atol
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def validate_function_equivalence(self,
                                    reference_func: Callable,
                                    optimized_func: Callable,
                                    test_inputs: List[torch.Tensor],
                                    test_name: str = "function_equivalence") -> ValidationResult:
        """
        Validate that optimized function produces equivalent results to reference.

        Args:
            reference_func: Reference implementation
            optimized_func: Optimized implementation to validate
            test_inputs: List of test input tensors
            test_name: Name of the validation test

        Returns:
            ValidationResult with validation outcome
        """
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() if self.device.type == 'cuda' else 0

        try:
            # Move inputs to device
            test_inputs = [inp.to(self.device) for inp in test_inputs]

            # Run reference implementation
            with torch.no_grad():
                if len(test_inputs) == 1:
                    ref_output = reference_func(test_inputs[0])
                else:
                    ref_output = reference_func(*test_inputs)

            # Run optimized implementation
            with torch.no_grad():
                if len(test_inputs) == 1:
                    opt_output = optimized_func(test_inputs[0])
                else:
                    opt_output = optimized_func(*test_inputs)

            # Compare outputs
            if isinstance(ref_output, torch.Tensor):
                ref_outputs = [ref_output]
                opt_outputs = [opt_output]
            else:
                ref_outputs = ref_output if isinstance(ref_output, (list, tuple)) else [ref_output]
                opt_outputs = opt_output if isinstance(opt_output, (list, tuple)) else [opt_output]

            max_error = 0.0
            mean_error = 0.0
            all_close = True

            for ref_out, opt_out in zip(ref_outputs, opt_outputs):
                if isinstance(ref_out, torch.Tensor) and isinstance(opt_out, torch.Tensor):
                    # Check shapes match
                    if ref_out.shape != opt_out.shape:
                        raise ValueError(f"Shape mismatch: reference {ref_out.shape}, optimized {opt_out.shape}")

                    # Check values are close
                    is_close = torch.allclose(ref_out, opt_out, rtol=self.rtol, atol=self.atol)
                    if not is_close:
                        all_close = False

                    # Calculate error metrics
                    error = torch.abs(ref_out - opt_out)
                    max_error = max(max_error, error.max().item())
                    mean_error = max(mean_error, error.mean().item())

            execution_time = time.time() - start_time
            memory_usage = (torch.cuda.memory_allocated() - initial_memory) / 1024**2 if self.device.type == 'cuda' else 0

            return ValidationResult(
                test_name=test_name,
                passed=all_close,
                error_message=None if all_close else f"Numerical mismatch: max_error={max_error:.2e}, mean_error={mean_error:.2e}",
                metrics={
                    'max_absolute_error': max_error,
                    'mean_absolute_error': mean_error,
                    'relative_tolerance': self.rtol,
                    'absolute_tolerance': self.atol
                },
                execution_time=execution_time,
                memory_usage=memory_usage
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                passed=False,
                error_message=f"Exception during validation: {str(e)}",
                metrics={},
                execution_time=execution_time,
                memory_usage=0.0
            )

    def validate_gradient_equivalence(self,
                                    reference_func: Callable,
                                    optimized_func: Callable,
                                    test_inputs: List[torch.Tensor],
                                    test_name: str = "gradient_equivalence") -> ValidationResult:
        """
        Validate that gradients computed by optimized function match reference.

        Args:
            reference_func: Reference implementation
            optimized_func: Optimized implementation
            test_inputs: List of test input tensors (with requires_grad=True)
            test_name: Name of the validation test

        Returns:
            ValidationResult with gradient validation outcome
        """
        start_time = time.time()

        try:
            # Ensure inputs require gradients
            test_inputs_ref = [inp.clone().detach().requires_grad_(True).to(self.device) for inp in test_inputs]
            test_inputs_opt = [inp.clone().detach().requires_grad_(True).to(self.device) for inp in test_inputs]

            # Forward pass and backward pass for reference
            if len(test_inputs_ref) == 1:
                ref_output = reference_func(test_inputs_ref[0])
            else:
                ref_output = reference_func(*test_inputs_ref)

            if isinstance(ref_output, torch.Tensor):
                ref_loss = ref_output.sum()
            else:
                ref_loss = sum(out.sum() for out in ref_output if isinstance(out, torch.Tensor))

            ref_loss.backward()

            # Forward pass and backward pass for optimized
            if len(test_inputs_opt) == 1:
                opt_output = optimized_func(test_inputs_opt[0])
            else:
                opt_output = optimized_func(*test_inputs_opt)

            if isinstance(opt_output, torch.Tensor):
                opt_loss = opt_output.sum()
            else:
                opt_loss = sum(out.sum() for out in opt_output if isinstance(out, torch.Tensor))

            opt_loss.backward()

            # Compare gradients
            max_grad_error = 0.0
            mean_grad_error = 0.0
            gradients_close = True

            for ref_inp, opt_inp in zip(test_inputs_ref, test_inputs_opt):
                if ref_inp.grad is not None and opt_inp.grad is not None:
                    grad_close = torch.allclose(ref_inp.grad, opt_inp.grad, rtol=self.rtol, atol=self.atol)
                    if not grad_close:
                        gradients_close = False

                    grad_error = torch.abs(ref_inp.grad - opt_inp.grad)
                    max_grad_error = max(max_grad_error, grad_error.max().item())
                    mean_grad_error = max(mean_grad_error, grad_error.mean().item())

            execution_time = time.time() - start_time

            return ValidationResult(
                test_name=test_name,
                passed=gradients_close,
                error_message=None if gradients_close else f"Gradient mismatch: max_error={max_grad_error:.2e}",
                metrics={
                    'max_gradient_error': max_grad_error,
                    'mean_gradient_error': mean_grad_error,
                    'gradient_rtol': self.rtol,
                    'gradient_atol': self.atol
                },
                execution_time=execution_time,
                memory_usage=0.0
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                passed=False,
                error_message=f"Exception during gradient validation: {str(e)}",
                metrics={},
                execution_time=execution_time,
                memory_usage=0.0
            )


class PerformanceValidator:
    """
    Validates performance characteristics and detects performance regressions.

    Ensures that optimization components maintain or improve performance
    compared to baseline implementations.
    """

    def __init__(self,
                 warmup_iterations: int = 5,
                 benchmark_iterations: int = 50,
                 regression_threshold: float = 0.05):  # 5% threshold
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.regression_threshold = regression_threshold

    def benchmark_function(self,
                          func: Callable,
                          args: Tuple,
                          kwargs: Dict[str, Any] = None) -> float:
        """
        Benchmark a function and return average execution time.

        Args:
            func: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Average execution time in milliseconds
        """
        kwargs = kwargs or {}

        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.benchmark_iterations):
            start_time = time.perf_counter()

            with torch.no_grad():
                result = func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        return np.mean(times)

    def validate_performance_improvement(self,
                                       baseline_func: Callable,
                                       optimized_func: Callable,
                                       test_args: Tuple,
                                       test_name: str = "performance_improvement",
                                       expected_speedup: float = 1.0) -> ValidationResult:
        """
        Validate that optimized function performs better than baseline.

        Args:
            baseline_func: Baseline implementation
            optimized_func: Optimized implementation
            test_args: Test arguments
            test_name: Name of the test
            expected_speedup: Expected minimum speedup factor

        Returns:
            ValidationResult with performance comparison
        """
        start_time = time.time()

        try:
            # Benchmark baseline
            baseline_time = self.benchmark_function(baseline_func, test_args)

            # Benchmark optimized
            optimized_time = self.benchmark_function(optimized_func, test_args)

            # Calculate speedup
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0.0
            improvement_percent = (baseline_time - optimized_time) / baseline_time * 100

            # Check if performance meets expectations
            performance_met = speedup >= expected_speedup

            execution_time = time.time() - start_time

            return ValidationResult(
                test_name=test_name,
                passed=performance_met,
                error_message=None if performance_met else f"Performance below expectation: {speedup:.2f}x vs {expected_speedup:.2f}x expected",
                metrics={
                    'baseline_time_ms': baseline_time,
                    'optimized_time_ms': optimized_time,
                    'speedup': speedup,
                    'improvement_percent': improvement_percent,
                    'expected_speedup': expected_speedup
                },
                execution_time=execution_time,
                memory_usage=0.0
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                passed=False,
                error_message=f"Exception during performance validation: {str(e)}",
                metrics={},
                execution_time=execution_time,
                memory_usage=0.0
            )

    def detect_performance_regression(self,
                                    current_func: Callable,
                                    test_args: Tuple,
                                    baseline_time: float,
                                    component_name: str) -> PerformanceRegression:
        """
        Detect if there's a performance regression compared to baseline.

        Args:
            current_func: Current implementation
            test_args: Test arguments
            baseline_time: Baseline execution time
            component_name: Name of the component

        Returns:
            PerformanceRegression result
        """
        current_time = self.benchmark_function(current_func, test_args)
        regression_percent = (current_time - baseline_time) / baseline_time * 100
        is_regression = regression_percent > (self.regression_threshold * 100)

        return PerformanceRegression(
            component_name=component_name,
            baseline_time=baseline_time,
            current_time=current_time,
            regression_percent=regression_percent,
            threshold_percent=self.regression_threshold * 100,
            is_regression=is_regression
        )


class ComponentValidator:
    """
    Comprehensive validator for GPU optimization components.

    Combines numerical, performance, and integration validation for complete
    component testing.
    """

    def __init__(self,
                 device: Optional[torch.device] = None,
                 rtol: float = 1e-5,
                 atol: float = 1e-8):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.numerical_validator = NumericalValidator(rtol=rtol, atol=atol, device=self.device)
        self.performance_validator = PerformanceValidator()
        self.validation_results = []

    def validate_attention_component(self,
                                   attention_module: nn.Module,
                                   reference_attention: Optional[nn.Module] = None) -> List[ValidationResult]:
        """
        Comprehensive validation of attention components.

        Args:
            attention_module: Attention module to validate
            reference_attention: Reference implementation (if None, uses PyTorch's MHA)

        Returns:
            List of validation results
        """
        results = []

        # Test configurations
        test_configs = [
            (8, 64, 512, 8),    # batch, seq_len, embed_dim, num_heads
            (4, 128, 512, 8),
            (2, 256, 512, 8),
            (1, 1024, 512, 8),
        ]

        for batch_size, seq_len, embed_dim, num_heads in test_configs:
            # Create test inputs
            test_input = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

            # Create reference if not provided
            if reference_attention is None:
                ref_attention = nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True
                ).to(self.device)
            else:
                ref_attention = reference_attention

            # Test function equivalence
            def ref_func(x):
                return ref_attention(x, x, x)[0]  # MHA returns (output, weights)

            def opt_func(x):
                return attention_module(x)

            result = self.numerical_validator.validate_function_equivalence(
                ref_func, opt_func, [test_input],
                f"attention_equivalence_{batch_size}x{seq_len}x{embed_dim}"
            )
            results.append(result)

            # Test performance improvement
            if result.passed:
                perf_result = self.performance_validator.validate_performance_improvement(
                    ref_func, opt_func, (test_input,),
                    f"attention_performance_{batch_size}x{seq_len}x{embed_dim}",
                    expected_speedup=0.8  # Allow slight performance decrease for correctness
                )
                results.append(perf_result)

        return results

    def validate_linear_component(self,
                                linear_module: nn.Module,
                                input_features: int,
                                output_features: int) -> List[ValidationResult]:
        """
        Comprehensive validation of linear components.

        Args:
            linear_module: Linear module to validate
            input_features: Number of input features
            output_features: Number of output features

        Returns:
            List of validation results
        """
        results = []

        # Test configurations
        batch_sizes = [1, 4, 16, 64]

        # Reference linear layer
        ref_linear = nn.Linear(input_features, output_features).to(self.device)

        # Copy weights for fair comparison
        if hasattr(linear_module, 'weight'):
            linear_module.weight.data.copy_(ref_linear.weight.data)
        if hasattr(linear_module, 'bias') and hasattr(ref_linear, 'bias'):
            if linear_module.bias is not None and ref_linear.bias is not None:
                linear_module.bias.data.copy_(ref_linear.bias.data)

        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, input_features, device=self.device)

            # Test equivalence
            result = self.numerical_validator.validate_function_equivalence(
                ref_linear, linear_module, [test_input],
                f"linear_equivalence_{batch_size}x{input_features}x{output_features}"
            )
            results.append(result)

            # Test gradient equivalence
            grad_result = self.numerical_validator.validate_gradient_equivalence(
                ref_linear, linear_module, [test_input],
                f"linear_gradient_{batch_size}x{input_features}x{output_features}"
            )
            results.append(grad_result)

        return results

    def validate_normalization_component(self,
                                       norm_module: nn.Module,
                                       normalized_shape: Union[int, List[int]]) -> List[ValidationResult]:
        """
        Comprehensive validation of normalization components.

        Args:
            norm_module: Normalization module to validate
            normalized_shape: Shape to normalize over

        Returns:
            List of validation results
        """
        results = []

        # Reference normalization
        if isinstance(normalized_shape, int):
            ref_norm = nn.LayerNorm(normalized_shape).to(self.device)
            test_shapes = [
                (4, normalized_shape),
                (8, 32, normalized_shape),
                (2, 64, normalized_shape),
            ]
        else:
            ref_norm = nn.LayerNorm(normalized_shape).to(self.device)
            test_shapes = [
                (4, *normalized_shape),
                (8, *normalized_shape),
            ]

        # Copy parameters for fair comparison
        if hasattr(norm_module, 'weight') and hasattr(ref_norm, 'weight'):
            norm_module.weight.data.copy_(ref_norm.weight.data)
        if hasattr(norm_module, 'bias') and hasattr(ref_norm, 'bias'):
            norm_module.bias.data.copy_(ref_norm.bias.data)

        for test_shape in test_shapes:
            test_input = torch.randn(*test_shape, device=self.device)

            # Test equivalence
            result = self.numerical_validator.validate_function_equivalence(
                ref_norm, norm_module, [test_input],
                f"norm_equivalence_{'x'.join(map(str, test_shape))}"
            )
            results.append(result)

            # Test performance
            if result.passed:
                perf_result = self.performance_validator.validate_performance_improvement(
                    ref_norm, norm_module, (test_input,),
                    f"norm_performance_{'x'.join(map(str, test_shape))}",
                    expected_speedup=0.9
                )
                results.append(perf_result)

        return results

    def validate_complete_model(self,
                              model: nn.Module,
                              sample_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                              reference_model: Optional[nn.Module] = None) -> List[ValidationResult]:
        """
        Complete model validation including forward/backward passes.

        Args:
            model: Model to validate
            sample_inputs: Sample input tensors
            reference_model: Reference model (if provided)

        Returns:
            List of validation results
        """
        results = []
        model = model.to(self.device)

        if isinstance(sample_inputs, torch.Tensor):
            sample_inputs = sample_inputs.to(self.device)
        else:
            sample_inputs = tuple(inp.to(self.device) for inp in sample_inputs)

        # Test model execution without errors
        try:
            start_time = time.time()

            # Forward pass
            with torch.no_grad():
                if isinstance(sample_inputs, tuple):
                    outputs = model(*sample_inputs)
                else:
                    outputs = model(sample_inputs)

            execution_time = time.time() - start_time

            results.append(ValidationResult(
                test_name="model_forward_execution",
                passed=True,
                error_message=None,
                metrics={
                    'output_shape': str(outputs.shape) if isinstance(outputs, torch.Tensor) else str([out.shape for out in outputs]),
                    'output_dtype': str(outputs.dtype) if isinstance(outputs, torch.Tensor) else str([out.dtype for out in outputs])
                },
                execution_time=execution_time,
                memory_usage=0.0
            ))

        except Exception as e:
            results.append(ValidationResult(
                test_name="model_forward_execution",
                passed=False,
                error_message=f"Model execution failed: {str(e)}",
                metrics={},
                execution_time=0.0,
                memory_usage=0.0
            ))

        # Test backward pass if reference model provided
        if reference_model is not None:
            reference_model = reference_model.to(self.device)

            def model_forward(inputs):
                if isinstance(inputs, tuple):
                    return model(*inputs)
                else:
                    return model(inputs)

            def ref_forward(inputs):
                if isinstance(inputs, tuple):
                    return reference_model(*inputs)
                else:
                    return reference_model(inputs)

            # Test equivalence
            if isinstance(sample_inputs, tuple):
                equiv_result = self.numerical_validator.validate_function_equivalence(
                    ref_forward, model_forward, list(sample_inputs),
                    "model_equivalence"
                )
            else:
                equiv_result = self.numerical_validator.validate_function_equivalence(
                    ref_forward, model_forward, [sample_inputs],
                    "model_equivalence"
                )
            results.append(equiv_result)

        return results

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Returns:
            Dictionary with validation summary and recommendations
        """
        if not self.validation_results:
            return {"error": "No validation results available"}

        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result.passed)
        failed_tests = total_tests - passed_tests

        # Categorize results
        categories = defaultdict(list)
        for result in self.validation_results:
            category = result.test_name.split('_')[0]
            categories[category].append(result)

        category_summaries = {}
        for category, results in categories.items():
            category_passed = sum(1 for r in results if r.passed)
            category_total = len(results)
            avg_time = np.mean([r.execution_time for r in results])

            category_summaries[category] = {
                'passed': category_passed,
                'total': category_total,
                'pass_rate': category_passed / category_total,
                'avg_execution_time': avg_time
            }

        # Generate recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append(f"{failed_tests} tests failed - review failed test details")

        if any(result.test_name.startswith('performance') and not result.passed for result in self.validation_results):
            recommendations.append("Performance issues detected - consider optimization")

        if any('gradient' in result.test_name and not result.passed for result in self.validation_results):
            recommendations.append("Gradient computation issues - verify backward pass implementation")

        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0
            },
            'category_summaries': category_summaries,
            'recommendations': recommendations,
            'failed_tests': [
                {
                    'test_name': result.test_name,
                    'error_message': result.error_message,
                    'metrics': result.metrics
                }
                for result in self.validation_results if not result.passed
            ]
        }


def demonstrate_validation_framework():
    """
    Comprehensive demonstration of the validation framework.
    """
    print("✅ Enhanced Validation Framework Demonstration")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize validators
    validator = ComponentValidator(device=device)

    # Create sample optimized components to validate
    print(f"\n🔧 Creating Test Components:")

    # Sample optimized linear layer
    class OptimizedLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x):
            return self.linear(x)

    opt_linear = OptimizedLinear(512, 256).to(device)
    print(f"  ✓ Optimized Linear Layer: 512 → 256")

    # Sample optimized attention
    class OptimizedAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        def forward(self, x):
            return self.attention(x, x, x)[0]

    opt_attention = OptimizedAttention(512, 8).to(device)
    print(f"  ✓ Optimized Attention: 512 dim, 8 heads")

    # Sample optimized normalization
    class OptimizedLayerNorm(nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.norm = nn.LayerNorm(normalized_shape)

        def forward(self, x):
            return self.norm(x)

    opt_norm = OptimizedLayerNorm(512).to(device)
    print(f"  ✓ Optimized LayerNorm: 512 features")

    # Validate Linear Component
    print(f"\n📊 Validating Linear Component:")
    linear_results = validator.validate_linear_component(opt_linear, 512, 256)
    validator.validation_results.extend(linear_results)

    linear_passed = sum(1 for r in linear_results if r.passed)
    linear_total = len(linear_results)
    print(f"  Tests passed: {linear_passed}/{linear_total}")

    for result in linear_results[:3]:  # Show first 3 results
        status = "✅" if result.passed else "❌"
        print(f"    {status} {result.test_name}: {result.execution_time*1000:.2f} ms")

    # Validate Attention Component
    print(f"\n🎯 Validating Attention Component:")
    attention_results = validator.validate_attention_component(opt_attention)
    validator.validation_results.extend(attention_results)

    attention_passed = sum(1 for r in attention_results if r.passed)
    attention_total = len(attention_results)
    print(f"  Tests passed: {attention_passed}/{attention_total}")

    for result in attention_results[:3]:  # Show first 3 results
        status = "✅" if result.passed else "❌"
        print(f"    {status} {result.test_name}: {result.execution_time*1000:.2f} ms")
        if 'speedup' in result.metrics:
            print(f"        Speedup: {result.metrics['speedup']:.2f}x")

    # Validate Normalization Component
    print(f"\n📐 Validating Normalization Component:")
    norm_results = validator.validate_normalization_component(opt_norm, 512)
    validator.validation_results.extend(norm_results)

    norm_passed = sum(1 for r in norm_results if r.passed)
    norm_total = len(norm_results)
    print(f"  Tests passed: {norm_passed}/{norm_total}")

    # Complete Model Validation
    print(f"\n🏗️ Complete Model Validation:")

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = opt_attention
            self.norm1 = OptimizedLayerNorm(512)
            self.linear1 = OptimizedLinear(512, 1024)
            self.linear2 = OptimizedLinear(1024, 512)
            self.norm2 = OptimizedLayerNorm(512)

        def forward(self, x):
            # Attention block
            attn_out = self.attention(x)
            x = self.norm1(x + attn_out)

            # FFN block
            ffn_out = self.linear2(torch.relu(self.linear1(x)))
            x = self.norm2(x + ffn_out)

            return x

    test_model = TestModel().to(device)
    sample_input = torch.randn(4, 64, 512, device=device)

    model_results = validator.validate_complete_model(test_model, sample_input)
    validator.validation_results.extend(model_results)

    model_passed = sum(1 for r in model_results if r.passed)
    model_total = len(model_results)
    print(f"  Model tests passed: {model_passed}/{model_total}")

    # Generate Validation Report
    print(f"\n📋 Validation Report:")
    report = validator.generate_validation_report()

    print(f"  Overall Summary:")
    summary = report['summary']
    print(f"    Total tests: {summary['total_tests']}")
    print(f"    Passed: {summary['passed_tests']} ({summary['pass_rate']:.1%})")
    print(f"    Failed: {summary['failed_tests']}")

    print(f"\n  Category Breakdown:")
    for category, stats in report['category_summaries'].items():
        print(f"    {category.title()}: {stats['passed']}/{stats['total']} "
              f"({stats['pass_rate']:.1%}) - {stats['avg_execution_time']*1000:.2f} ms avg")

    if report['recommendations']:
        print(f"\n  💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"    • {rec}")

    # Performance Regression Testing
    print(f"\n⚠️ Performance Regression Testing:")

    # Simulate baseline performance data
    baseline_times = {
        'linear_forward': 0.5,  # ms
        'attention_forward': 2.0,  # ms
        'norm_forward': 0.2,  # ms
    }

    perf_validator = PerformanceValidator()

    # Test for regressions
    sample_linear_input = torch.randn(32, 512, device=device)
    sample_attn_input = torch.randn(8, 64, 512, device=device)
    sample_norm_input = torch.randn(32, 512, device=device)

    components = [
        ('linear_forward', lambda: opt_linear(sample_linear_input), (sample_linear_input,), 0.5),
        ('attention_forward', lambda: opt_attention(sample_attn_input), (sample_attn_input,), 2.0),
        ('norm_forward', lambda: opt_norm(sample_norm_input), (sample_norm_input,), 0.2),
    ]

    regressions_detected = 0
    for comp_name, comp_func, comp_args, baseline_time in components:
        regression = perf_validator.detect_performance_regression(
            comp_func, comp_args, baseline_time, comp_name
        )

        status = "⚠️" if regression.is_regression else "✅"
        print(f"    {status} {comp_name}: {regression.current_time:.2f} ms "
              f"({regression.regression_percent:+.1f}%)")

        if regression.is_regression:
            regressions_detected += 1

    print(f"\n  Performance regressions detected: {regressions_detected}")

    # Numerical Stability Testing
    print(f"\n🔬 Numerical Stability Testing:")

    numerical_validator = NumericalValidator(device=device)

    # Test with different input scales
    input_scales = [1e-3, 1e-1, 1.0, 10.0, 1000.0]
    stability_results = []

    for scale in input_scales:
        scaled_input = torch.randn(4, 512, device=device) * scale

        ref_result = torch.layer_norm(scaled_input, (512,))
        opt_result = opt_norm(scaled_input)

        # Check if outputs are finite
        is_stable = (torch.isfinite(ref_result).all() and
                    torch.isfinite(opt_result).all() and
                    torch.allclose(ref_result, opt_result, rtol=1e-4, atol=1e-6))

        stability_results.append((scale, is_stable))
        status = "✅" if is_stable else "❌"
        print(f"    {status} Input scale {scale}: {'Stable' if is_stable else 'Unstable'}")

    stable_count = sum(1 for _, stable in stability_results if stable)
    print(f"  Numerical stability: {stable_count}/{len(input_scales)} scales stable")

    print(f"\n✅ Validation framework demonstration complete!")
    print(f"Key validation capabilities demonstrated:")
    print(f"  • Numerical correctness verification")
    print(f"  • Performance regression detection")
    print(f"  • Gradient equivalence testing")
    print(f"  • Complete model validation")
    print(f"  • Numerical stability analysis")
    print(f"  • Comprehensive reporting")


if __name__ == "__main__":
    demonstrate_validation_framework()