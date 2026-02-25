"""
Quantization Engine

Core engine that auto-selects the optimal quantization format per detected
backend, applies quantization, and provides fallback chains when a requested
format is unavailable.
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchbridge.core.config import (
    AMDConfig,
    HardwareBackend,
    HardwareConfig,
    NVIDIAConfig,
    TPUConfig,
    TrainiumConfig,
)

from .compatibility import Architecture, QuantizationCompatibilityMatrix
from .formats import QuantizationFormat
from .torchao_integration import TORCHAO_AVAILABLE, TorchAOBackend

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Result from a quantization operation."""
    success: bool
    model: nn.Module | None
    format_applied: QuantizationFormat
    format_requested: QuantizationFormat
    used_fallback: bool = False
    fallback_chain: list[QuantizationFormat] = field(default_factory=list)
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def memory_reduction_pct(self) -> float:
        """Actual memory reduction percentage."""
        if self.memory_before_mb <= 0:
            return 0.0
        return (
            (self.memory_before_mb - self.memory_after_mb)
            / self.memory_before_mb
            * 100
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "format_applied": self.format_applied.value,
            "format_requested": self.format_requested.value,
            "used_fallback": self.used_fallback,
            "fallback_chain": [f.value for f in self.fallback_chain],
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_reduction_pct": self.memory_reduction_pct,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 * 1024)


class QuantizationEngine:
    """Backend-aware quantization engine.

    Detects the current hardware backend and architecture, then selects
    the optimal quantization format or applies a user-specified format
    with automatic fallback.

    Args:
        backend: Optional BaseBackend instance. If ``None``, the engine
            detects the backend from ``HardwareConfig``.
    """

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend
        self._hw_backend, self._architecture = self._detect_hardware(backend)

    def quantize(
        self,
        model: nn.Module,
        format: str | QuantizationFormat = "auto",
        calibration_data: Any | None = None,
        in_place: bool = False,
    ) -> QuantizationResult:
        """Quantize a model.

        Args:
            model: PyTorch model to quantize.
            format: Quantization format string or enum. ``"auto"`` selects
                the optimal format for the detected backend.
            calibration_data: Optional calibration data for formats that
                require it (SmoothQuant).
            in_place: If ``False`` (default), quantizes a deep copy.

        Returns:
            QuantizationResult with the quantized model and metadata.
        """
        # Resolve format
        is_auto = False
        if isinstance(format, str):
            if format.lower() == "auto":
                is_auto = True
                requested = QuantizationFormat.NONE  # sentinel for auto
            elif format.lower() == "none":
                requested = QuantizationFormat.NONE
            else:
                requested = QuantizationFormat.from_string(format)
        else:
            requested = format

        # Handle NONE explicitly — return model unchanged
        if not is_auto and requested == QuantizationFormat.NONE:
            memory = _model_size_mb(model)
            return QuantizationResult(
                success=True,
                model=model,
                format_applied=QuantizationFormat.NONE,
                format_requested=QuantizationFormat.NONE,
                memory_before_mb=memory,
                memory_after_mb=memory,
            )

        # Auto-select optimal format
        if is_auto:
            target_format = QuantizationCompatibilityMatrix.get_optimal_format(
                self._hw_backend, self._architecture
            )
            requested_for_result = target_format
        else:
            target_format = requested
            requested_for_result = requested

        # Check support and get fallback chain
        fallback_chain = QuantizationCompatibilityMatrix.get_fallback_chain(
            target_format, self._hw_backend, self._architecture
        )
        used_fallback = fallback_chain[0] != target_format if fallback_chain else False
        actual_format = fallback_chain[0] if fallback_chain else target_format

        if used_fallback:
            msg = (
                f"{target_format.value} not supported on "
                f"{self._hw_backend.value}/{self._architecture}; "
                f"falling back to {actual_format.value}"
            )
            logger.warning(msg)

        # Prepare model
        memory_before = _model_size_mb(model)
        work_model = model if in_place else copy.deepcopy(model)

        result = QuantizationResult(
            success=False,
            model=None,
            format_applied=actual_format,
            format_requested=requested_for_result,
            used_fallback=used_fallback,
            fallback_chain=fallback_chain,
            memory_before_mb=memory_before,
        )

        if used_fallback:
            result.warnings.append(
                f"Requested {target_format.value} unavailable; "
                f"using {actual_format.value}"
            )

        # Apply quantization
        try:
            quantized = self._apply_format(
                work_model, actual_format, calibration_data
            )
            result.success = True
            result.model = quantized
            result.memory_after_mb = _model_size_mb(quantized)
        except Exception as e:
            result.errors.append(str(e))
            result.model = work_model
            result.memory_after_mb = memory_before
            logger.error("Quantization failed: %s", e, exc_info=True)

        return result

    def get_optimal_format(self) -> QuantizationFormat:
        """Return the optimal format for the detected backend."""
        return QuantizationCompatibilityMatrix.get_optimal_format(
            self._hw_backend, self._architecture
        )

    def get_supported_formats(self) -> list[QuantizationFormat]:
        """Return all supported formats for the detected backend."""
        return QuantizationCompatibilityMatrix.get_supported_formats(
            self._hw_backend, self._architecture
        )

    @property
    def backend_name(self) -> str:
        """Human-readable backend name."""
        return self._hw_backend.value

    @property
    def architecture_name(self) -> str:
        """Human-readable architecture name."""
        return str(self._architecture) if self._architecture else "unknown"

    # ── private ───────────────────────────────────────────────────────────

    def _detect_hardware(
        self, backend: Any | None
    ) -> tuple[HardwareBackend, Architecture]:
        """Detect hardware backend and architecture."""
        if backend is not None:
            return self._detect_from_backend_object(backend)

        # Fall back to HardwareConfig auto-detection
        hw = HardwareConfig()
        arch: Architecture = None

        if hw.backend == HardwareBackend.CUDA:
            arch = hw.nvidia.architecture
        elif hw.backend == HardwareBackend.AMD:
            arch = hw.amd.architecture
        elif hw.backend == HardwareBackend.TRAINIUM:
            arch = hw.trainium.architecture
        elif hw.backend == HardwareBackend.TPU:
            arch = hw.tpu.version

        return hw.backend, arch

    def _detect_from_backend_object(
        self, backend: Any
    ) -> tuple[HardwareBackend, Architecture]:
        """Extract backend type and architecture from a BaseBackend instance."""
        name = getattr(backend, "BACKEND_NAME", "cpu").lower()
        backend_map = {
            "nvidia": HardwareBackend.CUDA,
            "cuda": HardwareBackend.CUDA,
            "amd": HardwareBackend.AMD,
            "rocm": HardwareBackend.AMD,
            "trainium": HardwareBackend.TRAINIUM,
            "neuron": HardwareBackend.TRAINIUM,
            "tpu": HardwareBackend.TPU,
            "xla": HardwareBackend.TPU,
        }
        hw_backend = backend_map.get(name, HardwareBackend.CPU)

        # Try to get architecture from backend
        arch: Architecture = None
        if hw_backend == HardwareBackend.CUDA:
            arch = NVIDIAConfig().architecture
        elif hw_backend == HardwareBackend.AMD:
            arch = AMDConfig().architecture
        elif hw_backend == HardwareBackend.TRAINIUM:
            arch = TrainiumConfig().architecture
        elif hw_backend == HardwareBackend.TPU:
            arch = TPUConfig().version

        return hw_backend, arch

    def _torchao_backend_str(self) -> str:
        """Return the torchao backend identifier for the detected hardware."""
        if self._hw_backend == HardwareBackend.AMD:
            return "rocm"
        if self._hw_backend == HardwareBackend.CUDA:
            return "cuda"
        return "cpu"

    def _apply_format(
        self,
        model: nn.Module,
        fmt: QuantizationFormat,
        calibration_data: Any | None,
    ) -> nn.Module:
        """Dispatch to the appropriate quantization implementation."""
        if fmt == QuantizationFormat.NONE:
            return model

        if fmt == QuantizationFormat.BF16:
            return self._apply_bf16(model)

        if fmt == QuantizationFormat.INT8_DYNAMIC:
            return self._apply_int8_dynamic(model)

        if fmt == QuantizationFormat.INT8_SMOOTHQUANT:
            return self._apply_int8_smoothquant(model, calibration_data)

        if fmt == QuantizationFormat.INT4_WEIGHT_ONLY:
            return self._apply_int4_weight_only(model)

        if fmt in (QuantizationFormat.FP8_E4M3, QuantizationFormat.FP8_E5M2):
            return self._apply_fp8(model, fmt)

        if fmt == QuantizationFormat.NVFP4:
            return self._apply_nvfp4(model)

        raise ValueError(f"Unhandled quantization format: {fmt.value}")

    # ── format implementations ────────────────────────────────────────────

    def _apply_bf16(self, model: nn.Module) -> nn.Module:
        """Convert model to BFloat16."""
        return model.to(dtype=torch.bfloat16)

    def _apply_int8_dynamic(self, model: nn.Module) -> nn.Module:
        """INT8 dynamic quantization (PyTorch native, no torchao needed)."""
        if TorchAOBackend.is_available_on_backend(self._torchao_backend_str()):
            try:
                return TorchAOBackend.quantize_int8_dynamic(model)
            except Exception as e:
                logger.debug("torchao INT8 failed, using PyTorch native: %s", e)
        elif TORCHAO_AVAILABLE:
            logger.debug(
                "torchao not supported on %s backend; using PyTorch native INT8",
                self._hw_backend.value,
            )

        # PyTorch native fallback
        try:
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized" in str(e).lower():
                logger.warning(
                    "PyTorch quantization engine not available: %s. "
                    "Falling back to BF16.",
                    e,
                )
                return self._apply_bf16(model)
            raise

    def _apply_int8_smoothquant(
        self, model: nn.Module, calibration_data: Any | None
    ) -> nn.Module:
        """SmoothQuant via torchao."""
        if TorchAOBackend.is_available_on_backend(self._torchao_backend_str()):
            return TorchAOBackend.quantize_smoothquant(model, calibration_data)
        # Fallback to regular INT8
        warnings.warn(
            "torchao not available for SmoothQuant; falling back to INT8 dynamic",
            stacklevel=2,
        )
        return self._apply_int8_dynamic(model)

    def _apply_int4_weight_only(self, model: nn.Module) -> nn.Module:
        """INT4 weight-only via torchao."""
        if TorchAOBackend.is_available_on_backend(self._torchao_backend_str()):
            return TorchAOBackend.quantize_int4_weight_only(model)
        # Fallback to INT8
        warnings.warn(
            "torchao not available for INT4; falling back to INT8 dynamic",
            stacklevel=2,
        )
        return self._apply_int8_dynamic(model)

    def _apply_fp8(
        self, model: nn.Module, fmt: QuantizationFormat
    ) -> nn.Module:
        """FP8 quantization using TorchBridge native precision module."""
        try:
            from torchbridge.precision.fp8_native import convert_model_to_native_fp8

            device = next(model.parameters()).device
            return convert_model_to_native_fp8(model, device=device)
        except Exception as e:
            logger.warning("Native FP8 conversion failed: %s", e)
            # Try torchao FP8 (only on supported backends)
            if TorchAOBackend.is_available_on_backend(self._torchao_backend_str()):
                try:
                    return TorchAOBackend.quantize_fp8(model)
                except Exception as e:
                    logger.debug("TorchAO FP8 fallback also failed: %s", e)
            warnings.warn(
                f"FP8 ({fmt.value}) not available; falling back to INT8 dynamic",
                stacklevel=2,
            )
            return self._apply_int8_dynamic(model)

    def _apply_nvfp4(self, model: nn.Module) -> nn.Module:
        """NVFP4 quantization using TorchBridge native precision module."""
        try:
            from torchbridge.precision.fp4_native import convert_model_to_fp4

            return convert_model_to_fp4(model)
        except Exception as e:
            logger.warning("Native FP4 conversion failed: %s", e)
            warnings.warn(
                "NVFP4 not available; falling back to FP8",
                stacklevel=2,
            )
            return self._apply_fp8(model, QuantizationFormat.FP8_E4M3)

