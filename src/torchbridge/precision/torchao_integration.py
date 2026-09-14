# SPDX-License-Identifier: Apache-2.0
"""
torchao Integration Layer

Provides a soft-import wrapper around torchao for INT8/INT4
quantization. Falls back gracefully when torchao is not installed.
"""

from __future__ import annotations

import logging

import torch.nn as nn

# torchao renamed its quantization entry points. Asked through torchao_compat
# rather than imported directly: the soft import that used to sit here caught
# that rename as an ImportError and concluded the package was absent, so
# `pip install torchao` was the advice handed to someone who already had it.
from torchbridge.precision.torchao_compat import (
    TORCHAO_AVAILABLE,
    fp8_dynamic_config,
    int4_config,
    int8_dynamic_config,
    quantize_model,
    unavailable_reason,
)

logger = logging.getLogger(__name__)


class TorchAOBackend:
    """Wrapper around torchao quantization APIs.

    All methods raise ``RuntimeError`` with a clear install message
    if torchao is not available.
    """

    @staticmethod
    def is_available_on_backend(backend: str = "cuda") -> bool:
        """Check if torchao is installed AND supports the given backend.

        Args:
            backend: One of ``"cuda"`` (NVIDIA), ``"rocm"`` (AMD), ``"cpu"``,
                or ``"hip"`` (alias for ROCm).

        Returns:
            ``True`` if torchao is available and the backend is supported.
        """
        if not TORCHAO_AVAILABLE:
            return False
        # Normalize aliases
        backend = backend.lower()
        if backend in ("hip", "amd"):
            backend = "rocm"
        if backend == "rocm":
            # torchao ROCm support: INT8 works; FP8 is experimental.
            # Check that the running PyTorch is a ROCm build.
            #
            # Asked through the shared helper. `is not None` used to be written
            # out here, and an empty HIP version string — which has been seen in
            # the wild — makes that answer True while hardware_detector answers
            # False. PR #119 fixed the same mistake in dispatcher.py and
            # backend_factory.py; this was the third copy, and it was outside
            # that PR's diff.
            from torchbridge.core.hardware_detector import is_rocm_build

            return is_rocm_build()
        # CUDA is the primary supported backend; CPU supports INT8 dynamic
        return backend in ("cuda", "cpu")

    @staticmethod
    def _require_torchao() -> None:
        if not TORCHAO_AVAILABLE:
            raise RuntimeError(unavailable_reason("this quantization format"))

    @staticmethod
    def quantize_int8_dynamic(model: nn.Module) -> nn.Module:
        """Apply INT8 dynamic quantization via torchao.

        Falls back to PyTorch native ``quantize_dynamic`` when torchao
        is unavailable (handled by the engine, not here).
        """
        TorchAOBackend._require_torchao()
        quantize_model(model, int8_dynamic_config())
        return model

    @staticmethod
    def quantize_int4_weight_only(
        model: nn.Module,
        group_size: int = 128,
    ) -> nn.Module:
        """Apply INT4 weight-only quantization via torchao."""
        TorchAOBackend._require_torchao()
        quantize_model(model, int4_config(group_size=group_size))
        return model

    @staticmethod
    def quantize_int8_dynamic_activations(model: nn.Module) -> nn.Module:
        """Apply INT8 dynamic activation + INT8 weight quantization via torchao.

        Uses ``int8_dynamic_activation_int8_weight`` from torchao, which
        quantizes activations dynamically per token and weights statically.
        """
        TorchAOBackend._require_torchao()
        quantize_model(model, int8_dynamic_config())
        return model

    @staticmethod
    def quantize_fp8(model: nn.Module) -> nn.Module:
        """Apply FP8 quantization via torchao (if supported)."""
        TorchAOBackend._require_torchao()
        try:
            return quantize_model(model, fp8_dynamic_config())
        except RuntimeError as exc:
            # fp8_dynamic_config() already distinguishes "torchao unusable"
            # from "this build has no FP8", so its message is carried through
            # rather than replaced by a guess about which one happened.
            raise RuntimeError(
                f"{exc} The engine will fall back to TorchBridge native FP8."
            ) from exc
