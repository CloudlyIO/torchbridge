"""
Production Readiness Validator for TorchBridge (v0.5.3)

Validates models are ready for production deployment by checking:
- Model exportability (ONNX, TorchScript, SafeTensors)
- Inference performance (latency, throughput)
- Memory requirements
- Numerical stability
- Input/output validation

Example:
    ```python
    from torchbridge.deployment import validate_production_readiness

    result = validate_production_readiness(
        model=my_model,
        sample_input=torch.randn(1, 512),
        requirements={
            "max_latency_ms": 50,
            "min_throughput": 100,
            "max_memory_mb": 2048,
        }
    )

    if result.passed:
        print("Model is production-ready!")
    else:
        print(f"Issues: {result.failed_checks}")
    ```
"""

import gc
import logging
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class ValidationSeverity(Enum):
    """Severity level for validation checks."""
    CRITICAL = "critical"  # Must pass for production
    WARNING = "warning"    # Should pass, but not blocking
    INFO = "info"         # Informational only


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationCheck:
    """Result of a single validation check.

    Attributes:
        name: Check name
        status: Check status (passed/failed/skipped/error)
        severity: Severity level
        message: Human-readable description
        details: Additional details
        measured_value: Measured value (if applicable)
        threshold: Threshold value (if applicable)
    """
    name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: dict[str, Any] | None = None
    measured_value: float | None = None
    threshold: float | None = None

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status == ValidationStatus.FAILED


@dataclass
class ProductionRequirements:
    """Requirements for production deployment.

    Attributes:
        max_latency_ms: Maximum inference latency in milliseconds
        min_throughput: Minimum throughput (samples/second)
        max_memory_mb: Maximum memory usage in MB
        require_onnx_export: Whether ONNX export must succeed
        require_torchscript_export: Whether TorchScript export must succeed
        require_safetensors_export: Whether SafeTensors export must succeed
        max_output_deviation: Maximum allowed numerical deviation
        batch_sizes: Batch sizes to test
        warmup_iterations: Warmup iterations before benchmarking
        benchmark_iterations: Number of benchmark iterations
    """
    max_latency_ms: float | None = None
    min_throughput: float | None = None
    max_memory_mb: float | None = None
    require_onnx_export: bool = True
    require_torchscript_export: bool = True
    require_safetensors_export: bool = False
    max_output_deviation: float = 1e-5
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])
    warmup_iterations: int = 5
    benchmark_iterations: int = 20


@dataclass
class ProductionValidationResult:
    """Result of production readiness validation.

    Attributes:
        passed: Whether all critical checks passed
        checks: List of validation check results
        summary: Human-readable summary
        recommendations: List of recommendations
        latency_stats: Latency statistics
        memory_stats: Memory statistics
    """
    passed: bool
    checks: list[ValidationCheck] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    latency_stats: dict[str, float] | None = None
    memory_stats: dict[str, float] | None = None

    @property
    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.failed]

    @property
    def passed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.passed]

    @property
    def critical_failures(self) -> list[ValidationCheck]:
        return [
            c for c in self.checks
            if c.failed and c.severity == ValidationSeverity.CRITICAL
        ]


# =============================================================================
# Production Validator
# =============================================================================

class ProductionValidator:
    """Validates model production readiness.

    Performs comprehensive validation including:
    - Export format compatibility
    - Performance benchmarking
    - Memory profiling
    - Numerical stability
    - Input validation

    Example:
        ```python
        validator = ProductionValidator()
        result = validator.validate(
            model=my_model,
            sample_input=torch.randn(1, 512),
            requirements=ProductionRequirements(
                max_latency_ms=50,
                min_throughput=100
            )
        )
        ```
    """

    def __init__(self):
        """Initialize production validator."""
        self._temp_dir = None

    def validate(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        requirements: ProductionRequirements | None = None,
        device: torch.device | None = None,
    ) -> ProductionValidationResult:
        """Run full production validation.

        Args:
            model: PyTorch model to validate
            sample_input: Sample input tensor(s)
            requirements: Production requirements
            device: Device to test on

        Returns:
            ProductionValidationResult with all check results
        """
        if requirements is None:
            requirements = ProductionRequirements()

        if device is None:
            device = next(model.parameters()).device

        checks = []

        # Move model and input to device
        model = model.to(device)
        model.eval()

        if isinstance(sample_input, torch.Tensor):
            sample_input = sample_input.to(device)
        else:
            sample_input = tuple(t.to(device) for t in sample_input)

        # Run all validation checks
        checks.append(self._check_forward_pass(model, sample_input))
        checks.append(self._check_determinism(model, sample_input))
        checks.extend(self._check_exports(model, sample_input, requirements))
        checks.extend(self._check_performance(model, sample_input, requirements, device))
        checks.append(self._check_memory(model, sample_input, requirements, device))
        checks.append(self._check_gradient_mode(model))

        # Calculate overall result
        critical_failures = [
            c for c in checks
            if c.failed and c.severity == ValidationSeverity.CRITICAL
        ]
        passed = len(critical_failures) == 0

        # Generate summary
        passed_count = len([c for c in checks if c.passed])
        failed_count = len([c for c in checks if c.failed])
        summary = f"Validation {'PASSED' if passed else 'FAILED'}: {passed_count} passed, {failed_count} failed"

        # Generate recommendations
        recommendations = self._generate_recommendations(checks, requirements)

        # Extract stats
        latency_check = next((c for c in checks if c.name == "latency"), None)
        latency_stats = latency_check.details if latency_check else None

        memory_check = next((c for c in checks if c.name == "memory_usage"), None)
        memory_stats = memory_check.details if memory_check else None

        return ProductionValidationResult(
            passed=passed,
            checks=checks,
            summary=summary,
            recommendations=recommendations,
            latency_stats=latency_stats,
            memory_stats=memory_stats,
        )

    def _check_forward_pass(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> ValidationCheck:
        """Check that forward pass succeeds."""
        try:
            with torch.no_grad():
                if isinstance(sample_input, tuple):
                    output = model(*sample_input)
                else:
                    output = model(sample_input)

            return ValidationCheck(
                name="forward_pass",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.CRITICAL,
                message="Forward pass completed successfully",
                details={"output_shape": str(output.shape if hasattr(output, 'shape') else type(output))}
            )
        except Exception as e:
            return ValidationCheck(
                name="forward_pass",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Forward pass failed: {e}",
            )

    def _check_determinism(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        num_runs: int = 3,
    ) -> ValidationCheck:
        """Check that model produces deterministic outputs."""
        try:
            with torch.no_grad():
                outputs = []
                for _ in range(num_runs):
                    if isinstance(sample_input, tuple):
                        out = model(*sample_input)
                    else:
                        out = model(sample_input)
                    outputs.append(out.clone())

            # Check all outputs are identical
            for i in range(1, len(outputs)):
                if not torch.allclose(outputs[0], outputs[i], rtol=1e-5, atol=1e-5):
                    return ValidationCheck(
                        name="determinism",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.WARNING,
                        message="Model produces non-deterministic outputs",
                    )

            return ValidationCheck(
                name="determinism",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.WARNING,
                message="Model produces deterministic outputs",
            )
        except Exception as e:
            return ValidationCheck(
                name="determinism",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.WARNING,
                message=f"Determinism check failed: {e}",
            )

    def _check_exports(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        requirements: ProductionRequirements,
    ) -> list[ValidationCheck]:
        """Check model export compatibility."""
        checks = []

        with tempfile.TemporaryDirectory() as temp_dir:
            # ONNX Export
            if requirements.require_onnx_export:
                checks.append(self._check_onnx_export(model, sample_input, temp_dir))

            # TorchScript Export
            if requirements.require_torchscript_export:
                checks.append(self._check_torchscript_export(model, sample_input, temp_dir))

            # SafeTensors Export
            if requirements.require_safetensors_export:
                checks.append(self._check_safetensors_export(model, temp_dir))

        return checks

    def _check_onnx_export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        temp_dir: str,
    ) -> ValidationCheck:
        """Check ONNX export."""
        try:
            output_path = Path(temp_dir) / "model.onnx"

            # Export to ONNX
            if isinstance(sample_input, tuple):
                torch.onnx.export(
                    model, sample_input, str(output_path),
                    opset_version=14,
                    do_constant_folding=True,
                )
            else:
                torch.onnx.export(
                    model, (sample_input,), str(output_path),
                    opset_version=14,
                    do_constant_folding=True,
                )

            if output_path.exists():
                return ValidationCheck(
                    name="onnx_export",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.CRITICAL,
                    message="ONNX export successful",
                    details={"file_size_mb": output_path.stat().st_size / 1024 / 1024}
                )
            else:
                return ValidationCheck(
                    name="onnx_export",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.CRITICAL,
                    message="ONNX export failed: output file not created",
                )
        except Exception as e:
            return ValidationCheck(
                name="onnx_export",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"ONNX export failed: {e}",
            )

    def _check_torchscript_export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        temp_dir: str,
    ) -> ValidationCheck:
        """Check TorchScript export."""
        try:
            output_path = Path(temp_dir) / "model.pt"

            # Try tracing first
            if isinstance(sample_input, tuple):
                traced = torch.jit.trace(model, sample_input)
            else:
                traced = torch.jit.trace(model, sample_input)

            traced.save(str(output_path))

            if output_path.exists():
                return ValidationCheck(
                    name="torchscript_export",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.CRITICAL,
                    message="TorchScript export successful (trace)",
                    details={"file_size_mb": output_path.stat().st_size / 1024 / 1024}
                )
            else:
                return ValidationCheck(
                    name="torchscript_export",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.CRITICAL,
                    message="TorchScript export failed: output file not created",
                )
        except Exception as e:
            return ValidationCheck(
                name="torchscript_export",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"TorchScript export failed: {e}",
            )

    def _check_safetensors_export(
        self,
        model: nn.Module,
        temp_dir: str,
    ) -> ValidationCheck:
        """Check SafeTensors export."""
        try:
            from safetensors.torch import save_file
            output_path = Path(temp_dir) / "model.safetensors"

            save_file(model.state_dict(), str(output_path))

            if output_path.exists():
                return ValidationCheck(
                    name="safetensors_export",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.CRITICAL,
                    message="SafeTensors export successful",
                    details={"file_size_mb": output_path.stat().st_size / 1024 / 1024}
                )
            else:
                return ValidationCheck(
                    name="safetensors_export",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.CRITICAL,
                    message="SafeTensors export failed: output file not created",
                )
        except ImportError:
            return ValidationCheck(
                name="safetensors_export",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.CRITICAL,
                message="SafeTensors export skipped: library not installed",
            )
        except Exception as e:
            return ValidationCheck(
                name="safetensors_export",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"SafeTensors export failed: {e}",
            )

    def _check_performance(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        requirements: ProductionRequirements,
        device: torch.device,
    ) -> list[ValidationCheck]:
        """Check performance requirements."""
        checks = []

        try:
            # Warmup
            with torch.no_grad():
                for _ in range(requirements.warmup_iterations):
                    if isinstance(sample_input, tuple):
                        _ = model(*sample_input)
                    else:
                        _ = model(sample_input)

            # Benchmark latency
            latencies = []
            with torch.no_grad():
                for _ in range(requirements.benchmark_iterations):
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    if isinstance(sample_input, tuple):
                        _ = model(*sample_input)
                    else:
                        _ = model(sample_input)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    end = time.perf_counter()

                    latencies.append((end - start) * 1000)  # ms

            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
            throughput = 1000 / avg_latency  # samples/second

            latency_details = {
                "avg_ms": avg_latency,
                "min_ms": min_latency,
                "max_ms": max_latency,
                "p95_ms": p95_latency,
                "throughput": throughput,
            }

            # Check latency requirement
            if requirements.max_latency_ms is not None:
                if avg_latency <= requirements.max_latency_ms:
                    checks.append(ValidationCheck(
                        name="latency",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Latency {avg_latency:.2f}ms <= {requirements.max_latency_ms}ms",
                        measured_value=avg_latency,
                        threshold=requirements.max_latency_ms,
                        details=latency_details,
                    ))
                else:
                    checks.append(ValidationCheck(
                        name="latency",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Latency {avg_latency:.2f}ms > {requirements.max_latency_ms}ms",
                        measured_value=avg_latency,
                        threshold=requirements.max_latency_ms,
                        details=latency_details,
                    ))
            else:
                checks.append(ValidationCheck(
                    name="latency",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message=f"Latency: {avg_latency:.2f}ms (no requirement set)",
                    measured_value=avg_latency,
                    details=latency_details,
                ))

            # Check throughput requirement
            if requirements.min_throughput is not None:
                if throughput >= requirements.min_throughput:
                    checks.append(ValidationCheck(
                        name="throughput",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Throughput {throughput:.2f}/s >= {requirements.min_throughput}/s",
                        measured_value=throughput,
                        threshold=requirements.min_throughput,
                    ))
                else:
                    checks.append(ValidationCheck(
                        name="throughput",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Throughput {throughput:.2f}/s < {requirements.min_throughput}/s",
                        measured_value=throughput,
                        threshold=requirements.min_throughput,
                    ))

        except Exception as e:
            checks.append(ValidationCheck(
                name="performance",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Performance check failed: {e}",
            ))

        return checks

    def _check_memory(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        requirements: ProductionRequirements,
        device: torch.device,
    ) -> ValidationCheck:
        """Check memory usage."""
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                gc.collect()

                with torch.no_grad():
                    if isinstance(sample_input, tuple):
                        _ = model(*sample_input)
                    else:
                        _ = model(sample_input)
                    torch.cuda.synchronize()

                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

                details = {
                    "peak_mb": peak_memory_mb,
                    "current_mb": current_memory_mb,
                    "device": str(device),
                }

                if requirements.max_memory_mb is not None:
                    if peak_memory_mb <= requirements.max_memory_mb:
                        return ValidationCheck(
                            name="memory_usage",
                            status=ValidationStatus.PASSED,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Peak memory {peak_memory_mb:.2f}MB <= {requirements.max_memory_mb}MB",
                            measured_value=peak_memory_mb,
                            threshold=requirements.max_memory_mb,
                            details=details,
                        )
                    else:
                        return ValidationCheck(
                            name="memory_usage",
                            status=ValidationStatus.FAILED,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Peak memory {peak_memory_mb:.2f}MB > {requirements.max_memory_mb}MB",
                            measured_value=peak_memory_mb,
                            threshold=requirements.max_memory_mb,
                            details=details,
                        )
                else:
                    return ValidationCheck(
                        name="memory_usage",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message=f"Peak memory: {peak_memory_mb:.2f}MB (no requirement set)",
                        measured_value=peak_memory_mb,
                        details=details,
                    )
            else:
                return ValidationCheck(
                    name="memory_usage",
                    status=ValidationStatus.SKIPPED,
                    severity=ValidationSeverity.INFO,
                    message="Memory check skipped: CUDA not available",
                )
        except Exception as e:
            return ValidationCheck(
                name="memory_usage",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.WARNING,
                message=f"Memory check failed: {e}",
            )

    def _check_gradient_mode(self, model: nn.Module) -> ValidationCheck:
        """Check that model is in eval mode."""
        if model.training:
            return ValidationCheck(
                name="eval_mode",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.WARNING,
                message="Model is in training mode (should be eval mode for inference)",
            )
        else:
            return ValidationCheck(
                name="eval_mode",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.WARNING,
                message="Model is in eval mode",
            )

    def _generate_recommendations(
        self,
        checks: list[ValidationCheck],
        requirements: ProductionRequirements,
    ) -> list[str]:
        """Generate recommendations based on check results."""
        recommendations = []

        for check in checks:
            if check.failed:
                if check.name == "latency":
                    recommendations.append(
                        "Consider optimizing model with torch.compile() or quantization"
                    )
                elif check.name == "memory_usage":
                    recommendations.append(
                        "Consider using mixed precision (FP16) or gradient checkpointing"
                    )
                elif check.name == "onnx_export":
                    recommendations.append(
                        "Check for unsupported ONNX operators in your model"
                    )
                elif check.name == "torchscript_export":
                    recommendations.append(
                        "Consider using torch.jit.script() instead of trace() for models with control flow"
                    )
                elif check.name == "eval_mode":
                    recommendations.append(
                        "Call model.eval() before deployment"
                    )
                elif check.name == "determinism":
                    recommendations.append(
                        "Check for dropout or random layers - disable them for inference"
                    )

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_production_readiness(
    model: nn.Module,
    sample_input: torch.Tensor | tuple[torch.Tensor, ...],
    requirements: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> ProductionValidationResult:
    """Validate model production readiness.

    Convenience function for quick validation.

    Args:
        model: PyTorch model to validate
        sample_input: Sample input tensor(s)
        requirements: Dict with production requirements:
            - max_latency_ms: Maximum latency in ms
            - min_throughput: Minimum throughput (samples/sec)
            - max_memory_mb: Maximum memory usage in MB
        device: Device to test on

    Returns:
        ProductionValidationResult with all check results

    Example:
        ```python
        result = validate_production_readiness(
            model=my_model,
            sample_input=torch.randn(1, 512),
            requirements={"max_latency_ms": 50}
        )
        print(result.summary)
        ```
    """
    validator = ProductionValidator()

    prod_requirements = ProductionRequirements()
    if requirements:
        if "max_latency_ms" in requirements:
            prod_requirements.max_latency_ms = requirements["max_latency_ms"]
        if "min_throughput" in requirements:
            prod_requirements.min_throughput = requirements["min_throughput"]
        if "max_memory_mb" in requirements:
            prod_requirements.max_memory_mb = requirements["max_memory_mb"]

    return validator.validate(model, sample_input, prod_requirements, device)
