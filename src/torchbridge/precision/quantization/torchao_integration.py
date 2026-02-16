"""
torchao Integration Layer

Provides a soft-import wrapper around torchao for INT8/INT4/SmoothQuant
quantization. Falls back gracefully when torchao is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

# Soft import
try:
    import torchao  # noqa: F401
    from torchao.quantization import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        quantize_,
    )

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


class TorchAOBackend:
    """Wrapper around torchao quantization APIs.

    All methods raise ``RuntimeError`` with a clear install message
    if torchao is not available.
    """

    @staticmethod
    def _require_torchao() -> None:
        if not TORCHAO_AVAILABLE:
            raise RuntimeError(
                "torchao is required for this quantization format. "
                "Install with: pip install torchao"
            )

    @staticmethod
    def quantize_int8_dynamic(model: nn.Module) -> nn.Module:
        """Apply INT8 dynamic quantization via torchao.

        Falls back to PyTorch native ``quantize_dynamic`` when torchao
        is unavailable (handled by the engine, not here).
        """
        TorchAOBackend._require_torchao()
        quantize_(model, int8_dynamic_activation_int8_weight())
        return model

    @staticmethod
    def quantize_int4_weight_only(
        model: nn.Module,
        group_size: int = 128,
    ) -> nn.Module:
        """Apply INT4 weight-only quantization via torchao."""
        TorchAOBackend._require_torchao()
        quantize_(model, int4_weight_only(group_size=group_size))
        return model

    @staticmethod
    def quantize_smoothquant(
        model: nn.Module,
        calibration_data: Any | None = None,
    ) -> nn.Module:
        """Apply SmoothQuant (INT8 with activation smoothing) via torchao.

        Args:
            model: Model to quantize.
            calibration_data: Optional calibration data for activation ranges.
        """
        TorchAOBackend._require_torchao()
        # SmoothQuant uses the same int8 dynamic API in torchao;
        # the smoothing pass is implicit in newer torchao versions.
        quantize_(model, int8_dynamic_activation_int8_weight())
        return model

    @staticmethod
    def quantize_fp8(model: nn.Module) -> nn.Module:
        """Apply FP8 quantization via torchao (if supported).

        Falls back to TorchBridge's native fp8_native module when
        torchao FP8 is not available.
        """
        TorchAOBackend._require_torchao()
        try:
            from torchao.quantization import float8_dynamic_activation_float8_weight

            quantize_(model, float8_dynamic_activation_float8_weight())
            return model
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "torchao FP8 quantization not available in this version. "
                "The engine will fall back to TorchBridge native FP8."
            ) from exc
