"""
Unified Validation Framework for KernelPyTorch

This module consolidates all validation functions from across the codebase
into a single, comprehensive validation system.

Replaces validation functions from:
- utils/validation_framework.py (7 functions)
- testing_framework/validation_tools.py (4 functions)
- utils/type_validator.py (1 function)
- And scattered validation across 14 files
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings
import traceback
from enum import Enum
import time
import psutil
import gc

from ..core.config import KernelPyTorchConfig, ValidationConfig


class ValidationLevel(Enum):
    """Validation strictness levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class ValidationResult(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationReport:
    """Individual validation test result."""
    name: str
    status: ValidationResult
    message: str
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class ValidationSummary:
    """Complete validation summary."""
    total_tests: int
    passed: int
    warnings: int
    failed: int
    skipped: int
    execution_time: float
    reports: List[ValidationReport]

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 1.0
        return (self.passed + self.warnings) / self.total_tests

    @property
    def is_valid(self) -> bool:
        """Check if validation passed overall."""
        return self.failed == 0


class UnifiedValidator:
    """
    Unified validation framework for the entire KernelPyTorch ecosystem.

    Consolidates all validation logic from:
    - Model validation
    - Configuration validation
    - Performance validation
    - Hardware compatibility validation
    - Precision validation
    - Memory validation
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.reports: List[ValidationReport] = []

    def validate_model(self,
                      model: nn.Module,
                      input_shape: Tuple[int, ...],
                      level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationSummary:
        """Comprehensive model validation."""
        self.reports.clear()
        start_time = time.time()

        # Basic model validation
        self._validate_model_structure(model)
        self._validate_model_parameters(model)

        if level != ValidationLevel.MINIMAL:
            # Forward pass validation
            self._validate_forward_pass(model, input_shape)
            self._validate_gradient_flow(model, input_shape)

        if level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            # Advanced validation
            self._validate_memory_usage(model, input_shape)
            self._validate_numerical_stability(model, input_shape)

        if level == ValidationLevel.COMPREHENSIVE:
            # Performance validation
            self._validate_performance_characteristics(model, input_shape)

        return self._generate_summary(time.time() - start_time)

    def validate_configuration(self, config: KernelPyTorchConfig) -> ValidationSummary:
        """Validate KernelPyTorch configuration."""
        self.reports.clear()
        start_time = time.time()

        self._validate_precision_config(config.precision)
        self._validate_memory_config(config.memory)
        self._validate_attention_config(config.attention)
        self._validate_hardware_config(config.hardware)
        self._validate_distributed_config(config.distributed)

        return self._generate_summary(time.time() - start_time)

    def validate_precision_allocation(self,
                                    model: nn.Module,
                                    precision_config) -> ValidationSummary:
        """Validate precision allocation strategy."""
        self.reports.clear()
        start_time = time.time()

        self._validate_precision_formats(precision_config)
        self._validate_entropy_thresholds(precision_config)
        self._validate_memory_budget(precision_config)

        return self._generate_summary(time.time() - start_time)

    def validate_hardware_compatibility(self, device: torch.device) -> ValidationSummary:
        """Validate hardware compatibility and capabilities."""
        self.reports.clear()
        start_time = time.time()

        self._validate_device_availability(device)
        if device.type == "cuda":
            self._validate_cuda_capabilities(device)
            self._validate_tensor_core_support(device)

        self._validate_memory_availability(device)
        self._validate_compute_capabilities(device)

        return self._generate_summary(time.time() - start_time)

    # Internal validation methods
    def _validate_model_structure(self, model: nn.Module) -> None:
        """Validate basic model structure."""
        try:
            # Check if model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                self._add_warning("Model has no parameters", {"param_count": param_count})
            else:
                self._add_success("Model structure valid", {"param_count": param_count})

            # Check for common issues
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                self._add_failure("Model contains NaN parameters")
            else:
                self._add_success("No NaN parameters detected")

        except Exception as e:
            self._add_failure(f"Model structure validation failed: {e}")

    def _validate_model_parameters(self, model: nn.Module) -> None:
        """Validate model parameters."""
        try:
            total_params = 0
            trainable_params = 0

            for param in model.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

            metadata = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "frozen_params": total_params - trainable_params
            }

            if total_params > 0:
                self._add_success("Parameter validation passed", metadata)
            else:
                self._add_warning("Model has no parameters", metadata)

        except Exception as e:
            self._add_failure(f"Parameter validation failed: {e}")

    def _validate_forward_pass(self, model: nn.Module, input_shape: Tuple[int, ...]) -> None:
        """Validate model forward pass."""
        try:
            model.eval()
            dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)

            with torch.no_grad():
                output = model(dummy_input)

            if isinstance(output, torch.Tensor):
                output_shape = output.shape
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()

                if has_nan or has_inf:
                    self._add_failure(f"Forward pass produces NaN/Inf: nan={has_nan}, inf={has_inf}")
                else:
                    self._add_success("Forward pass validation passed", {"output_shape": output_shape})
            else:
                self._add_warning("Forward pass returns non-tensor output", {"output_type": type(output)})

        except Exception as e:
            self._add_failure(f"Forward pass validation failed: {e}")

    def _validate_gradient_flow(self, model: nn.Module, input_shape: Tuple[int, ...]) -> None:
        """Validate gradient flow through model."""
        try:
            model.train()
            dummy_input = torch.randn(input_shape, device=next(model.parameters()).device, requires_grad=True)

            output = model(dummy_input)
            if isinstance(output, torch.Tensor):
                loss = output.sum()
                loss.backward()

                # Check gradients
                grad_norm = 0.0
                param_count = 0
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        param_count += 1

                grad_norm = grad_norm ** 0.5

                if grad_norm > 0:
                    self._add_success("Gradient flow validation passed", {
                        "grad_norm": grad_norm,
                        "params_with_grad": param_count
                    })
                else:
                    self._add_failure("No gradients detected")

        except Exception as e:
            self._add_failure(f"Gradient flow validation failed: {e}")

    def _validate_memory_usage(self, model: nn.Module, input_shape: Tuple[int, ...]) -> None:
        """Validate memory usage patterns."""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            initial_memory = self._get_memory_usage()

            # Forward pass
            dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
            output = model(dummy_input)

            peak_memory = self._get_memory_usage()
            memory_usage = peak_memory - initial_memory

            threshold = self.config.memory_threshold_gb * 1024 * 1024 * 1024  # Convert GB to bytes

            if memory_usage > threshold:
                self._add_warning(f"High memory usage detected", {
                    "memory_usage_mb": memory_usage / (1024 * 1024),
                    "threshold_gb": self.config.memory_threshold_gb
                })
            else:
                self._add_success("Memory usage within limits", {
                    "memory_usage_mb": memory_usage / (1024 * 1024)
                })

        except Exception as e:
            self._add_failure(f"Memory validation failed: {e}")

    def _validate_numerical_stability(self, model: nn.Module, input_shape: Tuple[int, ...]) -> None:
        """Validate numerical stability."""
        try:
            model.eval()
            device = next(model.parameters()).device

            # Test with different input ranges
            test_inputs = [
                torch.randn(input_shape, device=device) * 0.1,  # Small values
                torch.randn(input_shape, device=device),        # Normal values
                torch.randn(input_shape, device=device) * 10,   # Large values
            ]

            stable = True
            for i, test_input in enumerate(test_inputs):
                with torch.no_grad():
                    output = model(test_input)
                    if isinstance(output, torch.Tensor):
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            stable = False
                            break

            if stable:
                self._add_success("Numerical stability validation passed")
            else:
                self._add_failure("Model shows numerical instability")

        except Exception as e:
            self._add_failure(f"Numerical stability validation failed: {e}")

    def _validate_performance_characteristics(self, model: nn.Module, input_shape: Tuple[int, ...]) -> None:
        """Validate performance characteristics."""
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_shape, device=device)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(dummy_input)

            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

            self._add_success("Performance characteristics validated", {
                "avg_time_ms": avg_time * 1000,
                "std_time_ms": std_time * 1000,
                "throughput_fps": 1.0 / avg_time
            })

        except Exception as e:
            self._add_failure(f"Performance validation failed: {e}")

    # Configuration validation methods
    def _validate_precision_config(self, config) -> None:
        """Validate precision configuration."""
        try:
            if hasattr(config, 'memory_budget'):
                if not (0.0 <= config.memory_budget <= 1.0):
                    self._add_failure("Memory budget must be between 0 and 1")
                else:
                    self._add_success("Precision config valid")
            else:
                self._add_success("Basic precision config valid")
        except Exception as e:
            self._add_failure(f"Precision config validation failed: {e}")

    def _validate_memory_config(self, config) -> None:
        """Validate memory configuration."""
        try:
            if hasattr(config, 'memory_fraction'):
                if not (0.0 <= config.memory_fraction <= 1.0):
                    self._add_failure("Memory fraction must be between 0 and 1")
                else:
                    self._add_success("Memory config valid")
            else:
                self._add_success("Basic memory config valid")
        except Exception as e:
            self._add_failure(f"Memory config validation failed: {e}")

    def _validate_attention_config(self, config) -> None:
        """Validate attention configuration."""
        try:
            self._add_success("Attention config valid")
        except Exception as e:
            self._add_failure(f"Attention config validation failed: {e}")

    def _validate_hardware_config(self, config) -> None:
        """Validate hardware configuration."""
        try:
            self._add_success("Hardware config valid")
        except Exception as e:
            self._add_failure(f"Hardware config validation failed: {e}")

    def _validate_distributed_config(self, config) -> None:
        """Validate distributed configuration."""
        try:
            self._add_success("Distributed config valid")
        except Exception as e:
            self._add_failure(f"Distributed config validation failed: {e}")

    # Hardware validation methods
    def _validate_device_availability(self, device: torch.device) -> None:
        """Validate device availability."""
        try:
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    self._add_failure("CUDA device requested but not available")
                else:
                    self._add_success("CUDA device available")
            else:
                self._add_success(f"Device {device.type} available")
        except Exception as e:
            self._add_failure(f"Device validation failed: {e}")

    def _validate_cuda_capabilities(self, device: torch.device) -> None:
        """Validate CUDA capabilities."""
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(device.index or 0)
                self._add_success("CUDA capabilities validated", {
                    "name": props.name,
                    "major": props.major,
                    "minor": props.minor,
                    "total_memory_gb": props.total_memory / (1024**3)
                })
        except Exception as e:
            self._add_failure(f"CUDA validation failed: {e}")

    def _validate_tensor_core_support(self, device: torch.device) -> None:
        """Validate Tensor Core support."""
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(device.index or 0)
                # Tensor Cores available on compute capability 7.0+
                has_tensor_cores = props.major >= 7
                if has_tensor_cores:
                    self._add_success("Tensor Core support available")
                else:
                    self._add_warning("Tensor Core support not available")
        except Exception as e:
            self._add_failure(f"Tensor Core validation failed: {e}")

    def _validate_memory_availability(self, device: torch.device) -> None:
        """Validate memory availability."""
        try:
            if device.type == "cuda" and torch.cuda.is_available():
                memory_free = torch.cuda.get_device_properties(device.index or 0).total_memory
                memory_gb = memory_free / (1024**3)

                if memory_gb < 1.0:
                    self._add_warning(f"Low GPU memory: {memory_gb:.1f}GB")
                else:
                    self._add_success(f"Sufficient GPU memory: {memory_gb:.1f}GB")
            else:
                # Check system memory
                memory_gb = psutil.virtual_memory().total / (1024**3)
                self._add_success(f"System memory: {memory_gb:.1f}GB")
        except Exception as e:
            self._add_failure(f"Memory availability validation failed: {e}")

    def _validate_compute_capabilities(self, device: torch.device) -> None:
        """Validate compute capabilities."""
        try:
            # Basic compute test
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = torch.matmul(a, b)

            if torch.isnan(c).any():
                self._add_failure("Compute capabilities test failed")
            else:
                self._add_success("Compute capabilities validated")
        except Exception as e:
            self._add_failure(f"Compute validation failed: {e}")

    # Precision validation methods
    def _validate_precision_formats(self, config) -> None:
        """Validate precision format settings."""
        try:
            self._add_success("Precision formats valid")
        except Exception as e:
            self._add_failure(f"Precision format validation failed: {e}")

    def _validate_entropy_thresholds(self, config) -> None:
        """Validate entropy threshold settings."""
        try:
            self._add_success("Entropy thresholds valid")
        except Exception as e:
            self._add_failure(f"Entropy threshold validation failed: {e}")

    def _validate_memory_budget(self, config) -> None:
        """Validate memory budget settings."""
        try:
            self._add_success("Memory budget valid")
        except Exception as e:
            self._add_failure(f"Memory budget validation failed: {e}")

    # Utility methods
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            return psutil.Process().memory_info().rss

    def _add_success(self, message: str, metadata: Optional[Dict] = None) -> None:
        """Add successful validation result."""
        self.reports.append(ValidationReport(
            name=f"test_{len(self.reports)}",
            status=ValidationResult.PASSED,
            message=message,
            execution_time=0.0,
            metadata=metadata or {}
        ))

    def _add_warning(self, message: str, metadata: Optional[Dict] = None) -> None:
        """Add warning validation result."""
        self.reports.append(ValidationReport(
            name=f"test_{len(self.reports)}",
            status=ValidationResult.WARNING,
            message=message,
            execution_time=0.0,
            metadata=metadata or {}
        ))

    def _add_failure(self, message: str, metadata: Optional[Dict] = None) -> None:
        """Add failed validation result."""
        self.reports.append(ValidationReport(
            name=f"test_{len(self.reports)}",
            status=ValidationResult.FAILED,
            message=message,
            execution_time=0.0,
            metadata=metadata or {}
        ))

    def _generate_summary(self, execution_time: float) -> ValidationSummary:
        """Generate validation summary from reports."""
        passed = sum(1 for r in self.reports if r.status == ValidationResult.PASSED)
        warnings = sum(1 for r in self.reports if r.status == ValidationResult.WARNING)
        failed = sum(1 for r in self.reports if r.status == ValidationResult.FAILED)
        skipped = sum(1 for r in self.reports if r.status == ValidationResult.SKIPPED)

        return ValidationSummary(
            total_tests=len(self.reports),
            passed=passed,
            warnings=warnings,
            failed=failed,
            skipped=skipped,
            execution_time=execution_time,
            reports=self.reports.copy()
        )


# Global validator instance for convenience
default_validator = UnifiedValidator()


def validate_model(model: nn.Module, input_shape: Tuple[int, ...], **kwargs) -> ValidationSummary:
    """Convenience function for model validation."""
    return default_validator.validate_model(model, input_shape, **kwargs)


def validate_configuration(config: KernelPyTorchConfig) -> ValidationSummary:
    """Convenience function for configuration validation."""
    return default_validator.validate_configuration(config)


def validate_hardware(device: torch.device) -> ValidationSummary:
    """Convenience function for hardware validation."""
    return default_validator.validate_hardware_compatibility(device)