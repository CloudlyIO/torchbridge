"""
Attention Dispatch Compatibility Matrix

Static tables mapping (backend, architecture) pairs to optimal attention
kernels with ordered fallback chains. Follows the same pattern as
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

from .kernel_types import AttentionKernelType

logger = logging.getLogger(__name__)

# ── NVIDIA kernel tables ─────────────────────────────────────────────

_NVIDIA_KERNELS: dict[NVIDIAArchitecture, list[AttentionKernelType]] = {
    NVIDIAArchitecture.BLACKWELL_DC: [
        AttentionKernelType.FLEX_ATTENTION,
        AttentionKernelType.FLASH_ATTENTION_3,
        AttentionKernelType.FLASH_ATTENTION_2,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    NVIDIAArchitecture.BLACKWELL_CONSUMER: [
        AttentionKernelType.FLEX_ATTENTION,
        AttentionKernelType.FLASH_ATTENTION_3,
        AttentionKernelType.FLASH_ATTENTION_2,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    NVIDIAArchitecture.HOPPER: [
        AttentionKernelType.FLEX_ATTENTION,
        AttentionKernelType.FLASH_ATTENTION_3,
        AttentionKernelType.FLASH_ATTENTION_2,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    NVIDIAArchitecture.ADA: [
        AttentionKernelType.FLASH_ATTENTION_2,
        AttentionKernelType.FLEX_ATTENTION,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    NVIDIAArchitecture.AMPERE: [
        AttentionKernelType.FLASH_ATTENTION_2,
        AttentionKernelType.FLEX_ATTENTION,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    NVIDIAArchitecture.TURING: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
    NVIDIAArchitecture.VOLTA: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
    NVIDIAArchitecture.PASCAL: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
}

# ── AMD kernel tables ────────────────────────────────────────────────

_AMD_KERNELS: dict[AMDArchitecture, list[AttentionKernelType]] = {
    AMDArchitecture.CDNA4: [
        AttentionKernelType.FLASH_ATTENTION_CK,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    AMDArchitecture.CDNA3: [
        AttentionKernelType.FLASH_ATTENTION_CK,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    AMDArchitecture.CDNA2: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
    AMDArchitecture.CDNA: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
    AMDArchitecture.RDNA3: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
    AMDArchitecture.RDNA2: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
}

# ── Trainium kernel tables ──────────────────────────────────────────

_TRAINIUM_KERNELS: dict[TrainiumArchitecture, list[AttentionKernelType]] = {
    TrainiumArchitecture.TRN3: [
        AttentionKernelType.NEURONX_SDPA,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    TrainiumArchitecture.TRN2: [
        AttentionKernelType.NEURONX_SDPA,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    TrainiumArchitecture.TRN1: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
    TrainiumArchitecture.INF2: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
}

# ── TPU kernel tables ───────────────────────────────────────────────

_TPU_KERNELS: dict[TPUVersion, list[AttentionKernelType]] = {
    TPUVersion.V7: [
        AttentionKernelType.PALLAS_ATTENTION,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    TPUVersion.V6E: [
        AttentionKernelType.PALLAS_ATTENTION,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    TPUVersion.V5P: [
        AttentionKernelType.PALLAS_ATTENTION,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    TPUVersion.V5E: [
        AttentionKernelType.PALLAS_ATTENTION,
        AttentionKernelType.PYTORCH_SDPA,
    ],
    TPUVersion.V4: [
        AttentionKernelType.PYTORCH_SDPA,
    ],
}

# ── CPU kernel table ────────────────────────────────────────────────

_CPU_KERNELS: list[AttentionKernelType] = [
    AttentionKernelType.PYTORCH_SDPA,
]


class AttentionDispatchMatrix:
    """Static lookup for per-backend, per-architecture kernel support."""

    @staticmethod
    def get_supported_kernels(
        backend: HardwareBackend,
        architecture: NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None = None,
    ) -> list[AttentionKernelType]:
        """Return ordered list of supported kernels (best first)."""
        if backend == HardwareBackend.CUDA:
            arch = architecture or NVIDIAArchitecture.AMPERE
            if arch not in _NVIDIA_KERNELS:
                logger.warning(
                    "Unknown NVIDIA architecture %r — falling back to Pascal kernel list. "
                    "Update AttentionDispatchMatrix for this architecture.",
                    arch,
                )
            return list(_NVIDIA_KERNELS.get(arch, _NVIDIA_KERNELS[NVIDIAArchitecture.PASCAL]))
        elif backend == HardwareBackend.AMD:
            arch = architecture or AMDArchitecture.CDNA3
            if arch not in _AMD_KERNELS:
                logger.warning(
                    "Unknown AMD architecture %r — falling back to CDNA2 kernel list. "
                    "Update AttentionDispatchMatrix for this architecture.",
                    arch,
                )
            return list(_AMD_KERNELS.get(arch, _AMD_KERNELS[AMDArchitecture.CDNA2]))
        elif backend == HardwareBackend.TRAINIUM:
            arch = architecture or TrainiumArchitecture.TRN2
            if arch not in _TRAINIUM_KERNELS:
                logger.warning(
                    "Unknown Trainium architecture %r — falling back to TRN1 kernel list. "
                    "Update AttentionDispatchMatrix for this architecture.",
                    arch,
                )
            return list(_TRAINIUM_KERNELS.get(arch, _TRAINIUM_KERNELS[TrainiumArchitecture.TRN1]))
        elif backend == HardwareBackend.TPU:
            arch = architecture or TPUVersion.V5E
            if arch not in _TPU_KERNELS:
                logger.warning(
                    "Unknown TPU version %r — falling back to v4 kernel list. "
                    "Update AttentionDispatchMatrix for this TPU version.",
                    arch,
                )
            return list(_TPU_KERNELS.get(arch, _TPU_KERNELS[TPUVersion.V4]))
        else:
            return list(_CPU_KERNELS)

    @staticmethod
    def get_fallback_chain(
        requested: AttentionKernelType,
        backend: HardwareBackend,
        architecture: NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None = None,
    ) -> list[AttentionKernelType]:
        """Return fallback chain starting after *requested* kernel."""
        supported = AttentionDispatchMatrix.get_supported_kernels(backend, architecture)
        if requested in supported:
            idx = supported.index(requested)
            return supported[idx + 1:]
        return supported

