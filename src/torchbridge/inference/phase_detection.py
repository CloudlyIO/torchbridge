"""
Disaggregated Serving Phase Detection

Detects whether inference is in prefill (compute-bound) or decode
(memory-bound) phase and provides hardware-aware profiling recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)

logger = logging.getLogger(__name__)

# Type alias for architecture values
Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)


class PhaseType(Enum):
    """Inference phase in disaggregated serving."""

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


@dataclass(frozen=True)
class PhaseProfile:
    """Hardware profile for a given inference phase."""

    phase: PhaseType
    is_compute_bound: bool
    is_memory_bound: bool
    recommended_hardware: str
    description: str


# ── Phase detection thresholds ───────────────────────────────────────────────
# If generated_tokens is 0 or very small relative to prompt, we're in prefill.
# Once generated_tokens grows, we transition to decode.
_PREFILL_RATIO_THRESHOLD = 0.1  # gen/prompt < 0.1 → prefill
_DECODE_RATIO_THRESHOLD = 1.0  # gen/prompt >= 1.0 → decode


class PhaseDetector:
    """Detects inference phase and provides hardware profiling.

    Prefill is compute-bound (process all prompt tokens in parallel),
    while decode is memory-bound (sequential token generation with
    KV-cache reads).
    """

    @staticmethod
    def detect_phase(
        prompt_tokens: int,
        generated_tokens: int,
        batch_size: int = 1,
    ) -> PhaseType:
        """Detect the current inference phase.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            generated_tokens: Number of tokens generated so far.
            batch_size: Current batch size.

        Returns:
            PhaseType indicating the current phase.
        """
        if prompt_tokens <= 0:
            return PhaseType.DECODE

        if generated_tokens == 0:
            return PhaseType.PREFILL

        ratio = generated_tokens / prompt_tokens

        if ratio < _PREFILL_RATIO_THRESHOLD:
            return PhaseType.PREFILL
        elif ratio >= _DECODE_RATIO_THRESHOLD:
            return PhaseType.DECODE
        else:
            return PhaseType.MIXED

    @staticmethod
    def get_hardware_profile(
        phase: PhaseType,
        backend: HardwareBackend = HardwareBackend.CPU,
        architecture: Architecture = None,
    ) -> PhaseProfile:
        """Get hardware profile and recommendations for a phase.

        Args:
            phase: Inference phase.
            backend: Hardware backend.
            architecture: Specific architecture.

        Returns:
            PhaseProfile with compute/memory characteristics.
        """
        if phase == PhaseType.PREFILL:
            hw_rec = PhaseDetector._get_prefill_recommendation(backend, architecture)
            return PhaseProfile(
                phase=PhaseType.PREFILL,
                is_compute_bound=True,
                is_memory_bound=False,
                recommended_hardware=hw_rec,
                description=(
                    "Prefill phase processes all prompt tokens in parallel. "
                    "Compute-bound — benefits from higher FLOPS."
                ),
            )
        elif phase == PhaseType.DECODE:
            hw_rec = PhaseDetector._get_decode_recommendation(backend, architecture)
            return PhaseProfile(
                phase=PhaseType.DECODE,
                is_compute_bound=False,
                is_memory_bound=True,
                recommended_hardware=hw_rec,
                description=(
                    "Decode phase generates tokens sequentially with KV-cache. "
                    "Memory-bound — benefits from higher memory bandwidth."
                ),
            )
        else:
            # Mixed phase
            return PhaseProfile(
                phase=PhaseType.MIXED,
                is_compute_bound=True,
                is_memory_bound=True,
                recommended_hardware="Balanced compute + bandwidth",
                description=(
                    "Mixed phase with overlapping prefill and decode. "
                    "Benefits from both compute and memory bandwidth."
                ),
            )

    @staticmethod
    def _get_prefill_recommendation(
        backend: HardwareBackend, architecture: Architecture
    ) -> str:
        """Get hardware recommendation for prefill phase."""
        recs = {
            HardwareBackend.CUDA: "High-FLOPS GPU (H100/B200 preferred for FP8 tensor cores)",
            HardwareBackend.AMD: "MI300X/MI350X with high compute density",
            HardwareBackend.TRAINIUM: "Trn2/Trn3 NeuronCores for parallel compute",
            HardwareBackend.TPU: "TPU v5p/v7 with high matrix multiply throughput",
            HardwareBackend.CPU: "Multi-core CPU with AVX-512/AMX",
        }
        return recs.get(backend, "General-purpose accelerator")

    @staticmethod
    def _get_decode_recommendation(
        backend: HardwareBackend, architecture: Architecture
    ) -> str:
        """Get hardware recommendation for decode phase."""
        recs = {
            HardwareBackend.CUDA: "High-bandwidth GPU (H200/B200 with HBM3e preferred)",
            HardwareBackend.AMD: "MI300X with 5.3 TB/s HBM3 bandwidth",
            HardwareBackend.TRAINIUM: "Trn2/Trn3 with HBM3e bandwidth",
            HardwareBackend.TPU: "TPU v5e for cost-efficient decode serving",
            HardwareBackend.CPU: "CPU with high memory bandwidth (DDR5/HBM)",
        }
        return recs.get(backend, "High memory bandwidth hardware")
