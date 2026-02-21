"""
Backend-Aware Quantization Compatibility Matrix

Maps (backend, architecture) pairs to supported quantization formats with
optimal selection and fallback chains. Uses hardware enums from core/config.py.
"""

from __future__ import annotations

import logging

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)

from .formats import QuantizationFormat

logger = logging.getLogger(__name__)

# Type alias for architecture values
Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)

# ── Compatibility tables ─────────────────────────────────────────────────────
# Each entry is an ordered list: first element = optimal, rest = fallback chain.

_NVIDIA_FORMATS: dict[NVIDIAArchitecture, list[QuantizationFormat]] = {
    NVIDIAArchitecture.BLACKWELL_DC: [
        QuantizationFormat.NVFP4,
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.INT8_DYNAMIC,
    ],
    NVIDIAArchitecture.BLACKWELL_CONSUMER: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    NVIDIAArchitecture.HOPPER: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.INT8_SMOOTHQUANT,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    NVIDIAArchitecture.AMPERE: [
        QuantizationFormat.INT8_SMOOTHQUANT,
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    NVIDIAArchitecture.ADA: [
        QuantizationFormat.INT8_SMOOTHQUANT,
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    NVIDIAArchitecture.TURING: [
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    NVIDIAArchitecture.VOLTA: [
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    NVIDIAArchitecture.PASCAL: [
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
}

_AMD_FORMATS: dict[AMDArchitecture, list[QuantizationFormat]] = {
    AMDArchitecture.CDNA4: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.INT8_DYNAMIC,
    ],
    AMDArchitecture.CDNA3: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.INT8_DYNAMIC,
    ],
    AMDArchitecture.CDNA2: [
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    AMDArchitecture.CDNA: [
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    AMDArchitecture.RDNA3: [
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
    AMDArchitecture.RDNA2: [
        QuantizationFormat.INT8_DYNAMIC,
        QuantizationFormat.INT4_WEIGHT_ONLY,
    ],
}

_TRAINIUM_FORMATS: dict[TrainiumArchitecture, list[QuantizationFormat]] = {
    TrainiumArchitecture.TRN3: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.BF16,
    ],
    TrainiumArchitecture.TRN2: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.BF16,
    ],
    TrainiumArchitecture.TRN1: [
        QuantizationFormat.BF16,
    ],
    TrainiumArchitecture.INF2: [
        QuantizationFormat.BF16,
    ],
}

_TPU_FORMATS: dict[TPUVersion, list[QuantizationFormat]] = {
    TPUVersion.V7: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.BF16,
    ],
    TPUVersion.V6E: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.BF16,
    ],
    TPUVersion.V5P: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.BF16,
    ],
    TPUVersion.V5E: [
        QuantizationFormat.FP8_E4M3,
        QuantizationFormat.BF16,
    ],
    TPUVersion.V4: [
        QuantizationFormat.BF16,
    ],
}

_CPU_FORMATS: list[QuantizationFormat] = [
    QuantizationFormat.INT8_DYNAMIC,
    QuantizationFormat.INT4_WEIGHT_ONLY,
    QuantizationFormat.BF16,
]


class QuantizationCompatibilityMatrix:
    """Static methods for querying the backend/architecture compatibility matrix."""

    @staticmethod
    def get_supported_formats(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[QuantizationFormat]:
        """Return ordered list of supported formats (optimal first).

        Args:
            backend: Hardware backend enum.
            architecture: Architecture enum (NVIDIA/AMD/Trainium/TPU specific).

        Returns:
            List of QuantizationFormat, ordered best-first.
        """
        table = QuantizationCompatibilityMatrix._get_table(backend)
        if table is None:
            return list(_CPU_FORMATS)

        if isinstance(table, list):
            return list(table)

        # Resolve AUTO architecture to a default
        arch = QuantizationCompatibilityMatrix._resolve_architecture(
            backend, architecture
        )
        return list(table.get(arch, _CPU_FORMATS))

    @staticmethod
    def get_optimal_format(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> QuantizationFormat:
        """Return the single optimal format for a backend/architecture pair."""
        supported = QuantizationCompatibilityMatrix.get_supported_formats(
            backend, architecture
        )
        return supported[0] if supported else QuantizationFormat.INT8_DYNAMIC

    @staticmethod
    def get_fallback_chain(
        requested: QuantizationFormat,
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[QuantizationFormat]:
        """Return fallback chain when *requested* format is unsupported.

        If *requested* is supported, returns ``[requested]``.
        Otherwise returns the full supported list as fallbacks.
        """
        supported = QuantizationCompatibilityMatrix.get_supported_formats(
            backend, architecture
        )
        if requested in supported:
            return [requested]
        return supported

    @staticmethod
    def is_format_supported(
        fmt: QuantizationFormat,
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> bool:
        """Check whether *fmt* is natively supported on this hardware."""
        supported = QuantizationCompatibilityMatrix.get_supported_formats(
            backend, architecture
        )
        return fmt in supported

    # ── private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _get_table(backend: HardwareBackend):
        """Return the format table for a backend."""
        if backend == HardwareBackend.CUDA:
            return _NVIDIA_FORMATS
        if backend == HardwareBackend.AMD:
            return _AMD_FORMATS
        if backend == HardwareBackend.TRAINIUM:
            return _TRAINIUM_FORMATS
        if backend == HardwareBackend.TPU:
            return _TPU_FORMATS
        if backend == HardwareBackend.CPU:
            return _CPU_FORMATS
        return None

    @staticmethod
    def _resolve_architecture(
        backend: HardwareBackend,
        architecture: Architecture,
    ):
        """Resolve AUTO or None to a sensible default architecture."""
        defaults = {
            HardwareBackend.CUDA: NVIDIAArchitecture.AMPERE,
            HardwareBackend.AMD: AMDArchitecture.CDNA3,
            HardwareBackend.TRAINIUM: TrainiumArchitecture.TRN2,
            HardwareBackend.TPU: TPUVersion.V5E,
        }
        if architecture is None:
            return defaults.get(backend)

        # Handle AUTO values
        auto_values = {
            NVIDIAArchitecture.AUTO,
            AMDArchitecture.AUTO,
            TrainiumArchitecture.AUTO,
            TPUVersion.AUTO,
        }
        if architecture in auto_values:
            return defaults.get(backend)

        return architecture
