"""
Backend-Aware Speculative Decoding Compatibility Matrix

Maps (backend, architecture) pairs to supported speculative methods with
optimal selection and fallback chains. Mirrors the pattern from
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

from .methods import SpeculativeMethod

logger = logging.getLogger(__name__)

# Type alias for architecture values
Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)

# ── Compatibility tables ─────────────────────────────────────────────────────
# Each entry is an ordered list: first element = optimal, rest = fallback chain.

_NVIDIA_METHODS: dict[NVIDIAArchitecture, list[SpeculativeMethod]] = {
    NVIDIAArchitecture.BLACKWELL_DC: [
        SpeculativeMethod.EAGLE,
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    NVIDIAArchitecture.BLACKWELL_CONSUMER: [
        SpeculativeMethod.EAGLE,
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    NVIDIAArchitecture.HOPPER: [
        SpeculativeMethod.EAGLE,
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    NVIDIAArchitecture.AMPERE: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    NVIDIAArchitecture.ADA: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    NVIDIAArchitecture.TURING: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    NVIDIAArchitecture.VOLTA: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    NVIDIAArchitecture.PASCAL: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
}

_AMD_METHODS: dict[AMDArchitecture, list[SpeculativeMethod]] = {
    AMDArchitecture.CDNA4: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    AMDArchitecture.CDNA3: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    AMDArchitecture.CDNA2: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    AMDArchitecture.CDNA: [
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    AMDArchitecture.RDNA3: [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    AMDArchitecture.RDNA2: [
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
}

_TRAINIUM_METHODS: dict[TrainiumArchitecture, list[SpeculativeMethod]] = {
    TrainiumArchitecture.TRN3: [
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    TrainiumArchitecture.TRN2: [
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    TrainiumArchitecture.TRN1: [
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    TrainiumArchitecture.INF2: [
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
}

_TPU_METHODS: dict[TPUVersion, list[SpeculativeMethod]] = {
    TPUVersion.V7: [
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    TPUVersion.V6E: [
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    TPUVersion.V5P: [
        SpeculativeMethod.LAYER_SKIP,
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    TPUVersion.V5E: [
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
    TPUVersion.V4: [
        SpeculativeMethod.PROMPT_LOOKUP,
    ],
}

_CPU_METHODS: list[SpeculativeMethod] = [
    SpeculativeMethod.PROMPT_LOOKUP,
]


class SpeculationCompatibilityMatrix:
    """Static methods for querying the backend/architecture speculation matrix."""

    @staticmethod
    def get_supported_methods(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[SpeculativeMethod]:
        """Return ordered list of supported methods (optimal first).

        Args:
            backend: Hardware backend enum.
            architecture: Architecture enum (NVIDIA/AMD/Trainium/TPU specific).

        Returns:
            List of SpeculativeMethod, ordered best-first.
        """
        table = SpeculationCompatibilityMatrix._get_table(backend)
        if table is None:
            return list(_CPU_METHODS)

        if isinstance(table, list):
            return list(table)

        # Resolve AUTO architecture to a default
        arch = SpeculationCompatibilityMatrix._resolve_architecture(
            backend, architecture
        )
        return list(table.get(arch, _CPU_METHODS))

    @staticmethod
    def get_optimal_method(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> SpeculativeMethod:
        """Return the single optimal method for a backend/architecture pair."""
        supported = SpeculationCompatibilityMatrix.get_supported_methods(
            backend, architecture
        )
        return supported[0] if supported else SpeculativeMethod.PROMPT_LOOKUP

    @staticmethod
    def get_fallback_chain(
        requested: SpeculativeMethod,
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[SpeculativeMethod]:
        """Return fallback chain when *requested* method is unsupported.

        If *requested* is supported, returns ``[requested]``.
        Otherwise returns the full supported list as fallbacks.
        """
        supported = SpeculationCompatibilityMatrix.get_supported_methods(
            backend, architecture
        )
        if requested in supported:
            return [requested]
        return supported

    @staticmethod
    def is_method_supported(
        method: SpeculativeMethod,
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> bool:
        """Check whether *method* is natively supported on this hardware."""
        supported = SpeculationCompatibilityMatrix.get_supported_methods(
            backend, architecture
        )
        return method in supported

    # ── private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _get_table(backend: HardwareBackend):
        """Return the method table for a backend."""
        if backend == HardwareBackend.CUDA:
            return _NVIDIA_METHODS
        if backend == HardwareBackend.AMD:
            return _AMD_METHODS
        if backend == HardwareBackend.TRAINIUM:
            return _TRAINIUM_METHODS
        if backend == HardwareBackend.TPU:
            return _TPU_METHODS
        if backend == HardwareBackend.CPU:
            return _CPU_METHODS
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
