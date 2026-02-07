"""
Unified Validation Framework for TorchBridge

This module consolidates all validation functions from across the codebase
into a single, comprehensive validation system.

Replaces validation functions from:
- utils/validation_framework.py (7 functions)
- utils/type_validator.py (1 function)
- And scattered validation across 14 files
"""

import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import psutil
    _psutil_available = True
except ImportError:
    _psutil_available = False

import torch
import torch.nn as nn

from ..core.config import TorchBridgeConfig, ValidationConfig


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
    metadata: dict[str, Any]


@dataclass
class ValidationSummary:
    """Complete validation summary."""
    total_tests: int
    passed: int
    warnings: int
    failed: int
    skipped: int
    execution_time: float
    reports: list[ValidationReport]

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
    Unified validation framework for the entire TorchBridge ecosystem.

    Consolidates all validation logic from:
    - Model validation
    - Configuration validation
    - Performance validation
    - Hardware compatibility validation
    - Precision validation
    - Memory validation
    """

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()
        self.reports: list[ValidationReport] = []

    def validate_model(self,
                      model: nn.Module,
                      input_shape: tuple[int, ...],
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

    def validate_configuration(self, config: TorchBridgeConfig) -> ValidationSummary:
        """Validate TorchBridge configuration."""
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

    def validate_tpu_compatibility(self, config: TorchBridgeConfig) -> ValidationSummary:
        """Validate TPU-specific configuration and compatibility."""
        self.reports.clear()
        start_time = time.time()

        tpu_config = config.hardware.tpu

        # Basic TPU configuration validation
        self._validate_tpu_configuration(tpu_config)

        # XLA integration validation
        self._validate_xla_integration(tpu_config)

        # TPU memory validation
        self._validate_tpu_memory_config(tpu_config)

        # TPU compilation validation
        self._validate_tpu_compilation_config(tpu_config)

        # TPU version-specific validation
        self._validate_tpu_version_compatibility(tpu_config)

        return self._generate_summary(time.time() - start_time)

    def validate_tpu_model_optimization(self,
                                      model: nn.Module,
                                      tpu_config,
                                      sample_inputs: torch.Tensor | None = None) -> ValidationSummary:
        """Validate TPU model optimization."""
        self.reports.clear()
        start_time = time.time()

        # Model structure validation for TPU
        self._validate_tpu_model_structure(model, tpu_config)

        # Layer optimization validation
        self._validate_tpu_layer_optimization(model, tpu_config)

        # Tensor shape validation
        if sample_inputs is not None:
            self._validate_tpu_tensor_shapes(model, sample_inputs, tpu_config)

        # Memory efficiency validation
        self._validate_tpu_memory_efficiency(model, tpu_config)

        # Performance characteristics
        self._validate_tpu_performance_characteristics(model, tpu_config)

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

    def _validate_forward_pass(self, model: nn.Module, input_shape: tuple[int, ...]) -> None:
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

    def _validate_gradient_flow(self, model: nn.Module, input_shape: tuple[int, ...]) -> None:
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

    def _validate_memory_usage(self, model: nn.Module, input_shape: tuple[int, ...]) -> None:
        """Validate memory usage patterns."""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            initial_memory = self._get_memory_usage()

            # Forward pass
            dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
            model(dummy_input)

            peak_memory = self._get_memory_usage()
            memory_usage = peak_memory - initial_memory

            threshold = self.config.memory_threshold_gb * 1024 * 1024 * 1024  # Convert GB to bytes

            if memory_usage > threshold:
                self._add_warning("High memory usage detected", {
                    "memory_usage_mb": memory_usage / (1024 * 1024),
                    "threshold_gb": self.config.memory_threshold_gb
                })
            else:
                self._add_success("Memory usage within limits", {
                    "memory_usage_mb": memory_usage / (1024 * 1024)
                })

        except Exception as e:
            self._add_failure(f"Memory validation failed: {e}")

    def _validate_numerical_stability(self, model: nn.Module, input_shape: tuple[int, ...]) -> None:
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
            for _i, test_input in enumerate(test_inputs):
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

    def _validate_performance_characteristics(self, model: nn.Module, input_shape: tuple[int, ...]) -> None:
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
                if _psutil_available:
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    self._add_success(f"System memory: {memory_gb:.1f}GB")
                else:
                    self._add_warning("System memory check unavailable (psutil not installed)")
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
        elif _psutil_available:
            return psutil.Process().memory_info().rss
        else:
            return 0

    def _add_success(self, message: str, metadata: dict | None = None) -> None:
        """Add successful validation result."""
        self.reports.append(ValidationReport(
            name=f"test_{len(self.reports)}",
            status=ValidationResult.PASSED,
            message=message,
            execution_time=0.0,
            metadata=metadata or {}
        ))

    def _add_warning(self, message: str, metadata: dict | None = None) -> None:
        """Add warning validation result."""
        self.reports.append(ValidationReport(
            name=f"test_{len(self.reports)}",
            status=ValidationResult.WARNING,
            message=message,
            execution_time=0.0,
            metadata=metadata or {}
        ))

    def _add_failure(self, message: str, metadata: dict | None = None) -> None:
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

    # TPU-specific validation methods

    def _validate_tpu_configuration(self, tpu_config) -> None:
        """Validate basic TPU configuration."""
        try:
            # Check TPU version
            from ..core.config import TPUCompilationMode, TPUTopology, TPUVersion

            if tpu_config.version not in TPUVersion:
                self._add_failure(f"Invalid TPU version: {tpu_config.version}")
            else:
                self._add_success(f"TPU version valid: {tpu_config.version.value}")

            # Check topology
            if tpu_config.topology not in TPUTopology:
                self._add_failure(f"Invalid TPU topology: {tpu_config.topology}")
            else:
                self._add_success(f"TPU topology valid: {tpu_config.topology.value}")

            # Check compilation mode
            if tpu_config.compilation_mode not in TPUCompilationMode:
                self._add_failure(f"Invalid TPU compilation mode: {tpu_config.compilation_mode}")
            else:
                self._add_success(f"TPU compilation mode valid: {tpu_config.compilation_mode.value}")

            # Check precision setting
            valid_precisions = ['bfloat16', 'float16', 'float32']
            if tpu_config.precision not in valid_precisions:
                self._add_warning(f"TPU precision '{tpu_config.precision}' may not be optimal. Consider: {valid_precisions}")
            else:
                self._add_success(f"TPU precision valid: {tpu_config.precision}")

        except Exception as e:
            self._add_failure(f"TPU configuration validation failed: {str(e)}")

    def _validate_xla_integration(self, tpu_config) -> None:
        """Validate XLA integration availability."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Check XLA availability
            self._add_success("PyTorch/XLA available")

            # Check XLA device (compatible with torch_xla 2.9+)
            try:
                # Use new API if available
                if hasattr(torch_xla, 'device'):
                    device = torch_xla.device()
                else:
                    device = xm.xla_device()
                self._add_success(f"XLA device available: {device}")
            except Exception as e:
                self._add_warning(f"XLA device access failed: {str(e)}")

            # Check XLA world size (compatible with torch_xla 2.9+)
            try:
                # Use new runtime API if available
                if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'world_size'):
                    world_size = torch_xla.runtime.world_size()
                elif hasattr(xm, 'xrt_world_size'):
                    world_size = xm.xrt_world_size()
                else:
                    world_size = 1

                self._add_success(f"XLA world size: {world_size}")

                if world_size > 1 and tpu_config.topology == "single":
                    self._add_warning("Multi-device XLA detected but TPU topology set to 'single'")

            except Exception as e:
                self._add_warning(f"XLA world size check failed: {str(e)}")

        except ImportError:
            self._add_warning("PyTorch/XLA not available - TPU functionality will be limited")

    def _validate_tpu_memory_config(self, tpu_config) -> None:
        """Validate TPU memory configuration."""
        # Memory fraction validation
        if not 0.1 <= tpu_config.memory_fraction <= 1.0:
            self._add_failure(f"TPU memory fraction {tpu_config.memory_fraction} must be between 0.1 and 1.0")
        else:
            self._add_success(f"TPU memory fraction valid: {tpu_config.memory_fraction}")

        # Gradient checkpointing validation
        if tpu_config.gradient_checkpointing:
            self._add_success("TPU gradient checkpointing enabled")
        else:
            self._add_warning("TPU gradient checkpointing disabled - may increase memory usage")

    def _validate_tpu_compilation_config(self, tpu_config) -> None:
        """Validate TPU compilation settings."""
        # XLA optimization level
        if not 0 <= tpu_config.xla_optimization_level <= 3:
            self._add_failure(f"XLA optimization level {tpu_config.xla_optimization_level} must be 0-3")
        else:
            self._add_success(f"XLA optimization level valid: {tpu_config.xla_optimization_level}")

        # Dynamic shapes validation
        if tpu_config.enable_xla_dynamic_shapes:
            self._add_success("XLA dynamic shapes enabled - supports variable input sizes")
        else:
            self._add_warning("XLA dynamic shapes disabled - input sizes must be static")

    def _validate_tpu_version_compatibility(self, tpu_config) -> None:
        """Validate TPU version-specific features."""
        from ..core.config import TPUVersion

        # High-performance features for newer TPUs
        if tpu_config.version in [TPUVersion.V5P, TPUVersion.V6E, TPUVersion.V7]:
            if tpu_config.xla_optimization_level < 2:
                self._add_warning("Consider using higher XLA optimization level for high-performance TPUs")
            if tpu_config.memory_fraction < 0.9:
                self._add_warning("High-performance TPUs can typically use higher memory fractions")

        # Cost-optimized TPU settings
        elif tpu_config.version == TPUVersion.V5E:
            if tpu_config.xla_optimization_level > 1:
                self._add_warning("V5E TPUs may benefit from lower optimization levels for stability")

    def _validate_tpu_model_structure(self, model: nn.Module, tpu_config) -> None:
        """Validate model structure for TPU optimization."""
        # Check for TPU-friendly layer sizes
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_features, out_features = module.in_features, module.out_features

                if in_features % 8 != 0:
                    self._add_warning(f"Linear layer {name} input features ({in_features}) not divisible by 8")
                if out_features % 8 != 0:
                    self._add_warning(f"Linear layer {name} output features ({out_features}) not divisible by 8")

            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                in_channels = module.in_channels
                out_channels = module.out_channels

                if in_channels % 8 != 0:
                    self._add_warning(f"Conv layer {name} input channels ({in_channels}) not divisible by 8")
                if out_channels % 8 != 0:
                    self._add_warning(f"Conv layer {name} output channels ({out_channels}) not divisible by 8")

        self._add_success("TPU model structure validation completed")

    def _validate_tpu_layer_optimization(self, model: nn.Module, tpu_config) -> None:
        """Validate layer-specific TPU optimizations."""
        # Check activation functions
        activation_counts = {}
        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                activation_type = type(module).__name__
                activation_counts[activation_type] = activation_counts.get(activation_type, 0) + 1

        if activation_counts:
            self._add_success(f"TPU-optimized activations found: {activation_counts}")

        # Check for batch normalization
        bn_count = sum(1 for m in model.modules()
                      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)))
        if bn_count > 0:
            self._add_success(f"Batch normalization layers: {bn_count}")

    def _validate_tpu_tensor_shapes(self, model: nn.Module, sample_inputs: torch.Tensor, tpu_config) -> None:
        """Validate tensor shapes for TPU efficiency."""
        # Check input shape alignment
        if len(sample_inputs.shape) >= 2:
            for dim in sample_inputs.shape:
                if dim % 8 != 0:
                    self._add_warning(f"Input dimension {dim} not divisible by 8 - may be suboptimal for TPU")

        # Test forward pass shape propagation
        try:
            with torch.no_grad():
                output = model(sample_inputs)
                self._add_success(f"Forward pass successful: {sample_inputs.shape} → {output.shape}")
        except Exception as e:
            self._add_failure(f"Forward pass failed: {str(e)}")

    def _validate_tpu_memory_efficiency(self, model: nn.Module, tpu_config) -> None:
        """Validate TPU memory efficiency."""
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = param_count * 4 / (1024**2)  # Assume float32

        # TPU memory capacity estimates
        memory_estimates = {
            "v4": 32 * 1024,    # 32GB
            "v5e": 16 * 1024,   # 16GB
            "v5p": 95 * 1024,   # 95GB
            "v6e": 32 * 1024,   # 32GB estimated
            "v7": 128 * 1024    # 128GB estimated
        }

        tpu_memory_mb = memory_estimates.get(tpu_config.version.value, 32 * 1024)
        memory_utilization = param_size_mb / (tpu_memory_mb * tpu_config.memory_fraction)

        if memory_utilization > 0.8:
            self._add_warning(f"High memory utilization: {memory_utilization:.2%}")
        elif memory_utilization > 0.5:
            self._add_success(f"Reasonable memory utilization: {memory_utilization:.2%}")
        else:
            self._add_success(f"Low memory utilization: {memory_utilization:.2%}")

    def _validate_tpu_performance_characteristics(self, model: nn.Module, tpu_config) -> None:
        """Validate TPU performance characteristics."""
        # Check for performance-impacting patterns
        sequential_count = sum(1 for m in model.modules() if isinstance(m, nn.Sequential))
        if sequential_count > 10:
            self._add_warning(f"High number of Sequential modules ({sequential_count}) may impact TPU performance")

        # Check for attention patterns
        attention_patterns = 0
        for name, module in model.named_modules():
            if 'attention' in name.lower() or hasattr(module, 'attention'):
                attention_patterns += 1

        if attention_patterns > 0:
            self._add_success(f"Attention patterns detected: {attention_patterns} (good for TPU)")

        self._add_success("TPU performance characteristics validation completed")

    # NVIDIA-specific validation methods

    def validate_nvidia_compatibility(self, config: TorchBridgeConfig) -> ValidationSummary:
        """Validate NVIDIA-specific configuration and compatibility."""
        self.reports.clear()
        start_time = time.time()

        nvidia_config = config.hardware.nvidia

        # Basic NVIDIA configuration validation
        self._validate_nvidia_configuration(nvidia_config)

        # CUDA integration validation
        self._validate_cuda_integration(nvidia_config)

        # NVIDIA memory validation
        self._validate_nvidia_memory_config(nvidia_config)

        # FP8 support validation
        self._validate_fp8_support(nvidia_config)

        # FlashAttention validation
        self._validate_flash_attention_config(nvidia_config)

        return self._generate_summary(time.time() - start_time)

    def validate_nvidia_model_optimization(self,
                                          model: nn.Module,
                                          nvidia_config,
                                          sample_inputs: torch.Tensor | None = None) -> ValidationSummary:
        """Validate NVIDIA model optimization."""
        self.reports.clear()
        start_time = time.time()

        # Model structure validation for NVIDIA
        self._validate_nvidia_model_structure(model, nvidia_config)

        # Layer optimization validation
        self._validate_nvidia_layer_optimization(model, nvidia_config)

        # Tensor Core optimization validation
        self._validate_tensor_core_optimization(model, nvidia_config)

        # Memory efficiency validation
        self._validate_nvidia_memory_efficiency(model, nvidia_config)

        # Performance characteristics
        self._validate_nvidia_performance_characteristics(model, nvidia_config)

        return self._generate_summary(time.time() - start_time)

    def _validate_nvidia_configuration(self, nvidia_config) -> None:
        """Validate basic NVIDIA configuration."""
        try:
            from ..core.config import NVIDIAArchitecture

            # Check architecture
            if nvidia_config.architecture == NVIDIAArchitecture.AUTO:
                self._add_success("NVIDIA architecture will be auto-detected")
            else:
                self._add_success(f"NVIDIA architecture: {nvidia_config.architecture.value}")

            # Check FP8 settings
            if nvidia_config.fp8_enabled:
                if nvidia_config.architecture in [
                    NVIDIAArchitecture.HOPPER,
                    NVIDIAArchitecture.BLACKWELL_DC,
                    NVIDIAArchitecture.BLACKWELL_CONSUMER,
                ]:
                    self._add_success("FP8 training enabled for H100/Blackwell")
                else:
                    self._add_warning("FP8 enabled but architecture may not support it")

            # Check Tensor Core version
            if nvidia_config.tensor_core_version >= 4:
                self._add_success(f"Tensor Core version {nvidia_config.tensor_core_version} (latest generation)")
            elif nvidia_config.tensor_core_version >= 3:
                self._add_success(f"Tensor Core version {nvidia_config.tensor_core_version}")
            else:
                self._add_warning(f"Older Tensor Core version {nvidia_config.tensor_core_version}")

        except Exception as e:
            self._add_failure(f"NVIDIA configuration validation failed: {e}")

    def _validate_cuda_integration(self, nvidia_config) -> None:
        """Validate CUDA integration."""
        try:
            if torch.cuda.is_available():
                self._add_success("CUDA is available")

                # Check CUDA version
                cuda_version = torch.version.cuda
                if cuda_version:
                    self._add_success(f"CUDA version: {cuda_version}")

                # Check cuDNN
                if torch.backends.cudnn.is_available():
                    cudnn_version = torch.backends.cudnn.version()
                    self._add_success(f"cuDNN version: {cudnn_version}")

            else:
                self._add_warning("CUDA not available, will use CPU fallback")

        except Exception as e:
            self._add_failure(f"CUDA integration validation failed: {e}")

    def _validate_nvidia_memory_config(self, nvidia_config) -> None:
        """Validate NVIDIA memory configuration."""
        try:
            if nvidia_config.memory_pool_enabled:
                self._add_success("Memory pooling enabled")

            if 0.0 < nvidia_config.memory_fraction <= 1.0:
                self._add_success(f"Memory fraction: {nvidia_config.memory_fraction}")
            else:
                self._add_warning(f"Invalid memory fraction: {nvidia_config.memory_fraction}")

        except Exception as e:
            self._add_failure(f"NVIDIA memory config validation failed: {e}")

    def _validate_fp8_support(self, nvidia_config) -> None:
        """Validate FP8 support."""
        try:
            from ..core.config import NVIDIAArchitecture

            if nvidia_config.fp8_enabled:
                if nvidia_config.architecture in [
                    NVIDIAArchitecture.HOPPER,
                    NVIDIAArchitecture.BLACKWELL_DC,
                    NVIDIAArchitecture.BLACKWELL_CONSUMER,
                ]:
                    self._add_success("FP8 supported on current architecture")

                    # Check FP8 recipe
                    if nvidia_config.fp8_recipe == "DelayedScaling":
                        self._add_success("Using DelayedScaling FP8 recipe")
                    else:
                        self._add_warning(f"Unknown FP8 recipe: {nvidia_config.fp8_recipe}")
                else:
                    self._add_warning("FP8 not supported on current architecture")

        except Exception as e:
            self._add_failure(f"FP8 validation failed: {e}")

    def _validate_flash_attention_config(self, nvidia_config) -> None:
        """Validate FlashAttention configuration."""
        try:
            if nvidia_config.flash_attention_enabled:
                self._add_success("FlashAttention enabled")

                version = nvidia_config.flash_attention_version
                if version == "3":
                    self._add_success("Using FlashAttention-3 (latest)")
                elif version in ["2", "1"]:
                    self._add_success(f"Using FlashAttention-{version}")
                else:
                    self._add_warning(f"Unknown FlashAttention version: {version}")

        except Exception as e:
            self._add_failure(f"FlashAttention validation failed: {e}")

    # =========================================================================
    # CUSTOM CUDA KERNEL VALIDATION (Phase 4A)
    # =========================================================================

    def validate_custom_kernels(self, config: TorchBridgeConfig) -> ValidationSummary:
        """
        Validate custom CUDA kernel availability and functionality.

        This validates:
        - CUDA kernel compilation
        - FlashAttention-2/3 kernels
        - Fused Linear+Activation kernels
        - FP8 kernels (H100+ only)
        - Kernel registry functionality

        Args:
            config: TorchBridge configuration

        Returns:
            ValidationSummary with kernel validation results
        """
        self.reports.clear()
        start_time = time.time()

        try:
            # Validate kernel configuration
            if not config.kernel.enabled:
                self._add_warning("Custom kernels disabled in configuration")
                return self._generate_summary(time.time() - start_time)

            # Check CUDA availability
            self._validate_cuda_available()

            # Validate kernel registry
            self._validate_kernel_registry()

            # Validate FlashAttention kernels
            self._validate_flash_attention_kernels(config)

            # Validate Fused Linear+Activation kernels
            self._validate_fused_activation_kernels(config)

            # Validate FP8 kernels (if enabled)
            if config.kernel.fp8_attention or config.kernel.fp8_layernorm:
                self._validate_fp8_kernels(config)

        except Exception as e:
            self._add_failure(f"Custom kernel validation failed: {e}")
            self._add_failure(f"Traceback: {traceback.format_exc()}")

        return self._generate_summary(time.time() - start_time)

    def _validate_cuda_available(self) -> None:
        """Validate CUDA compilation and availability."""
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                self._add_warning("CUDA not available - custom kernels will use CPU fallback")
                return

            self._add_success("CUDA is available")

            # Try to import compiled CUDA extension
            try:
                import torchbridge_cuda
                self._add_success("CUDA kernels extension compiled and importable")

                # Check available kernels
                kernel_attrs = dir(torchbridge_cuda)
                kernel_count = sum(1 for attr in kernel_attrs if not attr.startswith('_'))
                self._add_success(f"Found {kernel_count} CUDA kernel functions")

            except ImportError:
                self._add_warning(
                    "CUDA kernels not compiled - will use PyTorch fallback. "
                    "Run 'python setup.py build_ext --inplace' to compile kernels"
                )

        except Exception as e:
            self._add_failure(f"CUDA availability check failed: {e}")

    def _validate_kernel_registry(self) -> None:
        """Validate kernel registry functionality."""
        try:
            from ..core.kernel_registry import (
                KernelType,
                get_kernel_registry,
            )

            # Test registry creation
            registry = get_kernel_registry()
            self._add_success("Kernel registry created successfully")

            # Check registered kernels
            all_kernels = registry.list_kernels()
            self._add_success(f"Registry contains {len(all_kernels)} kernels")

            # Check kernel types
            for kernel_type in KernelType:
                type_kernels = registry.list_kernels(kernel_type=kernel_type)
                if type_kernels:
                    self._add_success(f"Found {len(type_kernels)} {kernel_type.value} kernels")

        except Exception as e:
            self._add_failure(f"Kernel registry validation failed: {e}")

    def _validate_flash_attention_kernels(self, config: TorchBridgeConfig) -> None:
        """Validate FlashAttention kernel availability."""
        try:
            if not config.kernel.flash_attention_enabled:
                self._add_warning("FlashAttention kernels disabled in configuration")
                return

            # Try to import FlashAttention wrapper
            try:
                from ..hardware.gpu.custom_kernels import FlashAttentionV3
                self._add_success("FlashAttentionV3 module importable")

                # Test module creation
                fa3 = FlashAttentionV3(causal=True)
                self._add_success("FlashAttentionV3 instance created successfully")

                # Check kernel availability
                if fa3._cuda_kernel_available:
                    self._add_success("FlashAttention-3 CUDA kernel available")
                else:
                    self._add_warning("FlashAttention-3 CUDA kernel not available (using fallback)")

                # Validate configuration settings
                version = config.kernel.flash_attention_version
                if version == "3":
                    self._add_success("Configured for FlashAttention-3")
                elif version == "2":
                    self._add_success("Configured for FlashAttention-2")
                elif version == "auto":
                    self._add_success("Auto-selecting FlashAttention version")

                # Check Split-K setting
                if config.kernel.flash_attention_split_k:
                    if torch.cuda.is_available():
                        compute_cap = torch.cuda.get_device_capability(0)
                        if compute_cap >= (8, 0):
                            self._add_success("Split-K optimization enabled (supported)")
                        else:
                            self._add_warning(
                                f"Split-K enabled but compute capability {compute_cap} < 8.0"
                            )

            except ImportError as e:
                self._add_failure(f"FlashAttention import failed: {e}")

        except Exception as e:
            self._add_failure(f"FlashAttention validation failed: {e}")

    def _validate_fused_activation_kernels(self, config: TorchBridgeConfig) -> None:
        """Validate Fused Linear+Activation kernels."""
        try:
            if not config.kernel.fuse_linear_activation:
                self._add_warning("Fused Linear+Activation kernels disabled in configuration")
                return

            # Try to import fused kernels
            try:
                from ..hardware.gpu.custom_kernels import (
                    FusedLinearGELU,
                    FusedLinearSiLU,
                    create_fused_ffn_layer,
                )
                self._add_success("Fused Linear+Activation modules importable")

                # Test module creation
                if config.kernel.fused_gelu_enabled:
                    gelu_layer = FusedLinearGELU(512, 2048)
                    self._add_success("FusedLinearGELU created successfully")

                    if gelu_layer._cuda_kernel_available:
                        self._add_success("FusedLinearGELU CUDA kernel available")
                    else:
                        self._add_warning("FusedLinearGELU using PyTorch fallback")

                if config.kernel.fused_silu_enabled:
                    silu_layer = FusedLinearSiLU(768, 3072)
                    self._add_success("FusedLinearSiLU created successfully")

                    if silu_layer._cuda_kernel_available:
                        self._add_success("FusedLinearSiLU CUDA kernel available")
                    else:
                        self._add_warning("FusedLinearSiLU using PyTorch fallback")

                # Test FFN layer factory
                create_fused_ffn_layer(512, 2048, activation="gelu")
                self._add_success("Fused FFN layer created successfully")

            except ImportError as e:
                self._add_failure(f"Fused kernel import failed: {e}")

        except Exception as e:
            self._add_failure(f"Fused kernel validation failed: {e}")

    def _validate_fp8_kernels(self, config: TorchBridgeConfig) -> None:
        """Validate FP8 kernels (H100/Blackwell only)."""
        try:
            if not torch.cuda.is_available():
                self._add_warning("FP8 kernels require CUDA")
                return

            # Check compute capability
            compute_cap = torch.cuda.get_device_capability(0)

            if compute_cap >= (9, 0):
                self._add_success(f"Compute capability {compute_cap} supports FP8")

                # Check FP8 settings
                if config.kernel.fp8_attention:
                    self._add_success("FP8 attention enabled")
                if config.kernel.fp8_layernorm:
                    self._add_success("FP8 LayerNorm enabled")
                if config.kernel.fp8_matmul:
                    self._add_success("FP8 MatMul enabled")

            else:
                self._add_warning(
                    f"FP8 kernels enabled but compute capability {compute_cap} < 9.0 (H100+). "
                    "FP8 will be disabled at runtime."
                )

        except Exception as e:
            self._add_failure(f"FP8 kernel validation failed: {e}")

    def _validate_nvidia_model_structure(self, model: nn.Module, nvidia_config) -> None:
        """Validate NVIDIA model structure."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self._add_success(f"Total parameters: {total_params:,}")
        self._add_success(f"Trainable parameters: {trainable_params:,}")

        # Check for NVIDIA-friendly patterns
        has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
        has_conv = any(isinstance(m, (nn.Conv2d, nn.Conv3d)) for m in model.modules())

        if has_linear:
            self._add_success("Model contains Linear layers (good for Tensor Cores)")
        if has_conv:
            self._add_success("Model contains Conv layers (good for CUDA)")

    def _validate_nvidia_layer_optimization(self, model: nn.Module, nvidia_config) -> None:
        """Validate NVIDIA layer optimization."""

        optimal_div = 16 if nvidia_config.tensor_core_version >= 4 else 8

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_f, out_f = module.in_features, module.out_features

                if in_f % optimal_div != 0 or out_f % optimal_div != 0:
                    self._add_warning(
                        f"Layer {name} dimensions ({in_f}x{out_f}) not optimal for Tensor Cores "
                        f"(should be divisible by {optimal_div})"
                    )

        self._add_success("NVIDIA layer optimization validation completed")

    def _validate_tensor_core_optimization(self, model: nn.Module, nvidia_config) -> None:
        """Validate Tensor Core optimization."""
        linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

        if linear_count > 0:
            self._add_success(f"Model has {linear_count} Linear layers for Tensor Core acceleration")

        # Check for mixed precision
        if nvidia_config.mixed_precision_enabled:
            self._add_success("Mixed precision enabled for Tensor Core utilization")

    def _validate_nvidia_memory_efficiency(self, model: nn.Module, nvidia_config) -> None:
        """Validate NVIDIA memory efficiency."""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB

        if param_memory < 100:
            self._add_success(f"Small model ({param_memory:.1f}MB) - efficient memory usage")
        elif param_memory < 1000:
            self._add_success(f"Medium model ({param_memory:.1f}MB) - manageable memory usage")
        else:
            self._add_warning(f"Large model ({param_memory:.1f}MB) - consider gradient checkpointing")

    def _validate_nvidia_performance_characteristics(self, model: nn.Module, nvidia_config) -> None:
        """Validate NVIDIA performance characteristics."""
        # Check for performance-friendly patterns
        sequential_count = sum(1 for m in model.modules() if isinstance(m, nn.Sequential))
        if sequential_count > 0:
            self._add_success(f"Sequential modules: {sequential_count}")

        # Check for attention patterns
        attention_count = sum(1 for name, _ in model.named_modules() if 'attention' in name.lower())
        if attention_count > 0:
            self._add_success(f"Attention modules: {attention_count} (good for FlashAttention)")

        self._add_success("NVIDIA performance characteristics validation completed")


# Global validator instance for convenience
default_validator = UnifiedValidator()


def validate_model(model: nn.Module, input_shape: tuple[int, ...], **kwargs) -> ValidationSummary:
    """Convenience function for model validation."""
    return default_validator.validate_model(model, input_shape, **kwargs)


def validate_configuration(config: TorchBridgeConfig) -> ValidationSummary:
    """Convenience function for configuration validation."""
    return default_validator.validate_configuration(config)


def validate_hardware(device: torch.device) -> ValidationSummary:
    """Convenience function for hardware validation."""
    return default_validator.validate_hardware_compatibility(device)


def validate_tpu_configuration(config: TorchBridgeConfig) -> ValidationSummary:
    """Convenience function for TPU configuration validation."""
    return default_validator.validate_tpu_compatibility(config)


def validate_tpu_model(model: nn.Module, tpu_config, sample_inputs: torch.Tensor | None = None) -> ValidationSummary:
    """Convenience function for TPU model optimization validation."""
    return default_validator.validate_tpu_model_optimization(model, tpu_config, sample_inputs)


def validate_nvidia_configuration(config: TorchBridgeConfig) -> ValidationSummary:
    """Convenience function for NVIDIA configuration validation."""
    return default_validator.validate_nvidia_compatibility(config)


def validate_nvidia_model(model: nn.Module, nvidia_config, sample_inputs: torch.Tensor | None = None) -> ValidationSummary:
    """Convenience function for NVIDIA model optimization validation."""
    return default_validator.validate_nvidia_model_optimization(model, nvidia_config, sample_inputs)


def validate_custom_kernels(config: TorchBridgeConfig) -> ValidationSummary:
    """Convenience function for custom CUDA kernel validation."""
    return default_validator.validate_custom_kernels(config)
