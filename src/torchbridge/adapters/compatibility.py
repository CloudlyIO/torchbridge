# SPDX-License-Identifier: Apache-2.0
"""
Adapter Compatibility Matrix

Maps (backend, architecture) pairs to supported adapter methods with
optimal selection and fallback chains. Uses the same pattern as
QuantizationCompatibilityMatrix and AttentionDispatchMatrix.
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
from torchbridge.precision.formats import QuantizationFormat

from .config import AdapterMethod

logger = logging.getLogger(__name__)

# Type alias for architecture values
Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)


# ── Per-backend adapter method support ────────────────────────────────────────
# Each entry is an ordered list: first = optimal, rest = fallback chain.

_NVIDIA_METHODS: dict[NVIDIAArchitecture | None, list[AdapterMethod]] = {
    NVIDIAArchitecture.BLACKWELL_DC: [
        AdapterMethod.QDORA,
        AdapterMethod.QLORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
    NVIDIAArchitecture.BLACKWELL_CONSUMER: [
        AdapterMethod.QLORA,
        AdapterMethod.QDORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
    NVIDIAArchitecture.HOPPER: [
        AdapterMethod.QLORA,
        AdapterMethod.QDORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
    NVIDIAArchitecture.AMPERE: [
        AdapterMethod.QLORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
    NVIDIAArchitecture.ADA: [
        AdapterMethod.QLORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
    NVIDIAArchitecture.TURING: [
        AdapterMethod.LORA,
        AdapterMethod.DORA,
    ],
    None: [
        AdapterMethod.QLORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
}

_AMD_METHODS: dict[AMDArchitecture | None, list[AdapterMethod]] = {
    AMDArchitecture.CDNA4: [
        AdapterMethod.QLORA,
        AdapterMethod.QDORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
    AMDArchitecture.CDNA3: [
        AdapterMethod.QLORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
    AMDArchitecture.CDNA2: [
        AdapterMethod.LORA,
        AdapterMethod.DORA,
    ],
    AMDArchitecture.RDNA4: [
        # Consumer GPU (RX 9000 series) — LORA only (no rocBLAS BLAS guarantees)
        AdapterMethod.LORA,
    ],
    AMDArchitecture.RDNA3: [
        AdapterMethod.LORA,
    ],
    AMDArchitecture.RDNA2: [
        AdapterMethod.LORA,
    ],
    None: [
        AdapterMethod.QLORA,
        AdapterMethod.DORA,
        AdapterMethod.LORA,
    ],
}

_TRAINIUM_METHODS: dict[TrainiumArchitecture | None, list[AdapterMethod]] = {
    TrainiumArchitecture.TRN3: [AdapterMethod.LORA],
    TrainiumArchitecture.TRN2: [AdapterMethod.LORA],
    TrainiumArchitecture.TRN1: [AdapterMethod.LORA],
    None: [AdapterMethod.LORA],
}

_TPU_METHODS: dict[TPUVersion | None, list[AdapterMethod]] = {
    TPUVersion.V7: [AdapterMethod.LORA, AdapterMethod.DORA],
    TPUVersion.V5E: [AdapterMethod.LORA, AdapterMethod.DORA],
    TPUVersion.V4: [AdapterMethod.LORA, AdapterMethod.DORA],
    None: [AdapterMethod.LORA, AdapterMethod.DORA],
}

_CPU_METHODS: list[AdapterMethod] = [
    AdapterMethod.LORA,
    AdapterMethod.DORA,
    AdapterMethod.QLORA,  # INT8 base on CPU (enables CPU-side unit testing)
]

# ── Base quantization format for QLoRA/QDoRA ──────────────────────────────────
# Which quantization format to use for the base model when doing QLoRA/QDoRA.

_QLORA_BASE_FORMAT: dict[HardwareBackend, QuantizationFormat | None] = {
    HardwareBackend.CUDA: QuantizationFormat.INT4_WEIGHT_ONLY,
    HardwareBackend.AMD: QuantizationFormat.INT4_WEIGHT_ONLY,
    HardwareBackend.CPU: QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,  # for testing
    HardwareBackend.TRAINIUM: None,  # QLoRA not supported
    HardwareBackend.TPU: None,  # QLoRA not supported
}

# ── Top-level dispatch table ──────────────────────────────────────────────────

_BACKEND_METHODS: dict[HardwareBackend, dict] = {
    HardwareBackend.CUDA: _NVIDIA_METHODS,
    HardwareBackend.AMD: _AMD_METHODS,
    HardwareBackend.TRAINIUM: _TRAINIUM_METHODS,
    HardwareBackend.TPU: _TPU_METHODS,
}


class AdapterCompatibilityMatrix:
    """Adapter method compatibility matrix across backends.

    Determines the optimal adapter method and base quantization format
    for a given (backend, architecture) pair.
    """

    @staticmethod
    def get_optimal(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> AdapterMethod:
        """Return the optimal adapter method for the given hardware.

        Args:
            backend: Target hardware backend.
            architecture: Specific hardware architecture (optional).

        Returns:
            The optimal AdapterMethod for this hardware.
        """
        chain = AdapterCompatibilityMatrix.get_fallback_chain(backend, architecture)
        return chain[0]

    @staticmethod
    def get_fallback_chain(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[AdapterMethod]:
        """Return the ordered fallback chain for the given hardware.

        The first element is optimal; subsequent elements are fallbacks
        in decreasing preference order.

        Args:
            backend: Target hardware backend.
            architecture: Specific hardware architecture (optional).

        Returns:
            Ordered list of supported AdapterMethods.
        """
        if backend == HardwareBackend.CPU:
            return list(_CPU_METHODS)

        arch_map = _BACKEND_METHODS.get(backend)
        if arch_map is None:
            logger.warning(
                "Unknown backend %s, falling back to LORA",
                backend.value if hasattr(backend, "value") else backend,
            )
            return [AdapterMethod.LORA]

        # Try specific architecture, then None fallback
        if architecture in arch_map:
            return list(arch_map[architecture])
        if None in arch_map:
            return list(arch_map[None])

        return [AdapterMethod.LORA]

    @staticmethod
    def get_base_quant_format(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> QuantizationFormat | None:
        """Return the base quantization format for QLoRA/QDoRA.

        Args:
            backend: Target hardware backend.
            architecture: Specific hardware architecture (optional).

        Returns:
            QuantizationFormat for base model quantization, or None if
            QLoRA is not supported on this backend.
        """
        return _QLORA_BASE_FORMAT.get(backend)

    @staticmethod
    def supports_method(
        backend: HardwareBackend,
        architecture: Architecture,
        method: AdapterMethod,
    ) -> bool:
        """Check if a specific adapter method is supported.

        Args:
            backend: Target hardware backend.
            architecture: Specific hardware architecture.
            method: Adapter method to check.

        Returns:
            True if the method is in the fallback chain for this hardware.
        """
        chain = AdapterCompatibilityMatrix.get_fallback_chain(backend, architecture)
        return method in chain
