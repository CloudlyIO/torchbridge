# SPDX-License-Identifier: Apache-2.0
"""
Backend-Aware KV-Cache Dtype Compatibility Matrix

Maps (backend, architecture) pairs to supported KV-cache dtypes with
optimal selection and fallback chains. Mirrors the pattern in
precision/quantization/compatibility.py.
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

from .cache_dtype import KVCacheDtype

logger = logging.getLogger(__name__)

# Type alias for architecture values
Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)

# -- Compatibility tables ----------------------------------------------------
# Each entry is an ordered list: first element = optimal, rest = fallback chain.

_NVIDIA_KV_DTYPES: dict[NVIDIAArchitecture, list[KVCacheDtype]] = {
    NVIDIAArchitecture.BLACKWELL_DC: [
        KVCacheDtype.NVFP4,
        KVCacheDtype.FP8_E4M3,
        KVCacheDtype.BF16,
    ],
    NVIDIAArchitecture.BLACKWELL_CONSUMER: [
        KVCacheDtype.FP8_E4M3,
        KVCacheDtype.BF16,
        KVCacheDtype.FP16,
    ],
    NVIDIAArchitecture.HOPPER: [
        KVCacheDtype.FP8_E4M3,
        KVCacheDtype.BF16,
        KVCacheDtype.FP16,
    ],
    NVIDIAArchitecture.AMPERE: [
        KVCacheDtype.BF16,
        KVCacheDtype.FP16,
    ],
    NVIDIAArchitecture.ADA: [
        KVCacheDtype.FP8_E4M3,
        KVCacheDtype.BF16,
        KVCacheDtype.FP16,
    ],
    NVIDIAArchitecture.TURING: [
        KVCacheDtype.FP16,
    ],
    NVIDIAArchitecture.VOLTA: [
        KVCacheDtype.FP16,
    ],
    NVIDIAArchitecture.PASCAL: [
        KVCacheDtype.FP16,
    ],
}

_AMD_KV_DTYPES: dict[AMDArchitecture, list[KVCacheDtype]] = {
    AMDArchitecture.CDNA4: [
        KVCacheDtype.FP8_E4M3,
        KVCacheDtype.BF16,
    ],
    AMDArchitecture.CDNA3: [
        KVCacheDtype.FP8_E4M3,
        KVCacheDtype.BF16,
    ],
    AMDArchitecture.CDNA2: [
        KVCacheDtype.BF16,
    ],
    AMDArchitecture.CDNA: [
        KVCacheDtype.FP16,
    ],
    AMDArchitecture.RDNA3: [
        KVCacheDtype.BF16,
    ],
    AMDArchitecture.RDNA2: [
        KVCacheDtype.FP16,
    ],
}

_TRAINIUM_KV_DTYPES: dict[TrainiumArchitecture, list[KVCacheDtype]] = {
    TrainiumArchitecture.TRN3: [
        KVCacheDtype.BF16,
    ],
    TrainiumArchitecture.TRN2: [
        KVCacheDtype.BF16,
    ],
    TrainiumArchitecture.TRN1: [
        KVCacheDtype.BF16,
    ],
    TrainiumArchitecture.INF2: [
        KVCacheDtype.BF16,
    ],
}

_TPU_KV_DTYPES: dict[TPUVersion, list[KVCacheDtype]] = {
    TPUVersion.V7: [
        KVCacheDtype.BF16,
    ],
    TPUVersion.V6E: [
        KVCacheDtype.BF16,
    ],
    TPUVersion.V5P: [
        KVCacheDtype.BF16,
    ],
    TPUVersion.V5E: [
        KVCacheDtype.BF16,
    ],
    TPUVersion.V4: [
        KVCacheDtype.BF16,
    ],
}

_CPU_KV_DTYPES: list[KVCacheDtype] = [
    KVCacheDtype.PASSTHROUGH,
]


class KVCacheCompatibilityMatrix:
    """Static methods for querying the KV-cache dtype compatibility matrix."""

    @staticmethod
    def get_supported_dtypes(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[KVCacheDtype]:
        """Return ordered list of supported KV-cache dtypes (optimal first).

        Args:
            backend: Hardware backend enum.
            architecture: Architecture enum (NVIDIA/AMD/Trainium/TPU specific).

        Returns:
            List of KVCacheDtype, ordered best-first.
        """
        table = KVCacheCompatibilityMatrix._get_table(backend)
        if table is None:
            return list(_CPU_KV_DTYPES)

        if isinstance(table, list):
            return list(table)

        arch = KVCacheCompatibilityMatrix._resolve_architecture(backend, architecture)
        return list(table.get(arch, _CPU_KV_DTYPES))

    @staticmethod
    def get_optimal_dtype(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> KVCacheDtype:
        """Return the single optimal KV-cache dtype for a backend/architecture pair."""
        supported = KVCacheCompatibilityMatrix.get_supported_dtypes(
            backend, architecture
        )
        return supported[0] if supported else KVCacheDtype.PASSTHROUGH

    @staticmethod
    def get_fallback_chain(
        requested: KVCacheDtype,
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[KVCacheDtype]:
        """Return fallback chain when *requested* dtype is unsupported.

        If *requested* is supported, returns ``[requested]``.
        Otherwise returns the full supported list as fallbacks.
        """
        supported = KVCacheCompatibilityMatrix.get_supported_dtypes(
            backend, architecture
        )
        if requested in supported:
            return [requested]
        return supported

    @staticmethod
    def is_dtype_supported(
        dtype: KVCacheDtype,
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> bool:
        """Check whether *dtype* is natively supported on this hardware."""
        supported = KVCacheCompatibilityMatrix.get_supported_dtypes(
            backend, architecture
        )
        return dtype in supported

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _get_table(backend: HardwareBackend):
        """Return the dtype table for a backend."""
        if backend == HardwareBackend.CUDA:
            return _NVIDIA_KV_DTYPES
        if backend == HardwareBackend.AMD:
            return _AMD_KV_DTYPES
        if backend == HardwareBackend.TRAINIUM:
            return _TRAINIUM_KV_DTYPES
        if backend == HardwareBackend.TPU:
            return _TPU_KV_DTYPES
        if backend == HardwareBackend.CPU:
            return _CPU_KV_DTYPES
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

        auto_values = {
            NVIDIAArchitecture.AUTO,
            AMDArchitecture.AUTO,
            TrainiumArchitecture.AUTO,
            TPUVersion.AUTO,
        }
        if architecture in auto_values:
            return defaults.get(backend)

        return architecture
