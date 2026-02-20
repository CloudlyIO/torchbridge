"""
Zero-Bubble Pipeline Scheduling

Exposes PyTorch's pipeline schedules (GPipe, 1F1B, zero-bubble) through
TorchBridge with backend-aware optimal schedule selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)

logger = logging.getLogger(__name__)

Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)


class PipelineScheduleType(Enum):
    """Pipeline parallelism schedule types."""

    GPIPE = "gpipe"
    INTERLEAVED_1F1B = "interleaved_1f1b"
    ZERO_BUBBLE = "zero_bubble"
    ZBV_ZERO_BUBBLE = "zbv_zero_bubble"
    LOOPED_BFS = "looped_bfs"


@dataclass(frozen=True)
class PipelineScheduleSpec:
    """Specification for a pipeline schedule."""

    schedule: PipelineScheduleType
    display_name: str
    bubble_ratio: str
    requires_async: bool
    min_stages: int
    description: str


PIPELINE_SCHEDULE_SPECS: dict[PipelineScheduleType, PipelineScheduleSpec] = {
    PipelineScheduleType.GPIPE: PipelineScheduleSpec(
        schedule=PipelineScheduleType.GPIPE,
        display_name="GPipe",
        bubble_ratio="(p-1)/m",
        requires_async=False,
        min_stages=2,
        description="Fill-drain schedule. Simple, high bubble ratio.",
    ),
    PipelineScheduleType.INTERLEAVED_1F1B: PipelineScheduleSpec(
        schedule=PipelineScheduleType.INTERLEAVED_1F1B,
        display_name="Interleaved 1F1B",
        bubble_ratio="(p-1)/(m*(v+1))",
        requires_async=False,
        min_stages=2,
        description="Interleaved one-forward-one-backward. Good balance.",
    ),
    PipelineScheduleType.ZERO_BUBBLE: PipelineScheduleSpec(
        schedule=PipelineScheduleType.ZERO_BUBBLE,
        display_name="Zero Bubble (Interleaved)",
        bubble_ratio="~0",
        requires_async=True,
        min_stages=2,
        description="Near-zero bubble via async P2P. Requires fast interconnect.",
    ),
    PipelineScheduleType.ZBV_ZERO_BUBBLE: PipelineScheduleSpec(
        schedule=PipelineScheduleType.ZBV_ZERO_BUBBLE,
        display_name="ZBV Zero Bubble",
        bubble_ratio="0",
        requires_async=True,
        min_stages=4,
        description="Zero-bubble V-schedule. Requires >= 4 stages.",
    ),
    PipelineScheduleType.LOOPED_BFS: PipelineScheduleSpec(
        schedule=PipelineScheduleType.LOOPED_BFS,
        display_name="Looped BFS",
        bubble_ratio="(p-1)/(m*v)",
        requires_async=False,
        min_stages=2,
        description="Breadth-first looped schedule for memory efficiency.",
    ),
}


# ── Compatibility: (backend, architecture) → ordered schedule list ──────────

_CUDA_HIGH_END: list[PipelineScheduleType] = [
    PipelineScheduleType.ZERO_BUBBLE,
    PipelineScheduleType.ZBV_ZERO_BUBBLE,
    PipelineScheduleType.INTERLEAVED_1F1B,
    PipelineScheduleType.LOOPED_BFS,
    PipelineScheduleType.GPIPE,
]

_CUDA_STANDARD: list[PipelineScheduleType] = [
    PipelineScheduleType.INTERLEAVED_1F1B,
    PipelineScheduleType.LOOPED_BFS,
    PipelineScheduleType.GPIPE,
]

_NON_CUDA: list[PipelineScheduleType] = [
    PipelineScheduleType.INTERLEAVED_1F1B,
    PipelineScheduleType.GPIPE,
]

# Architectures with fast async P2P (NVLink + Hopper+)
_ASYNC_P2P_ARCHS: set[NVIDIAArchitecture] = {
    NVIDIAArchitecture.HOPPER,
    NVIDIAArchitecture.BLACKWELL_DC,
    NVIDIAArchitecture.BLACKWELL_CONSUMER,
}


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""

    schedule: PipelineScheduleType = PipelineScheduleType.INTERLEAVED_1F1B
    num_stages: int = 2
    num_microbatches: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "schedule": self.schedule.value,
            "num_stages": self.num_stages,
            "num_microbatches": self.num_microbatches,
        }


class PipelineScheduleFactory:
    """Select optimal pipeline schedule based on hardware and config."""

    @staticmethod
    def get_supported_schedules(
        backend: HardwareBackend,
        architecture: Architecture = None,
    ) -> list[PipelineScheduleType]:
        """Return supported schedules for the given hardware, best first."""
        if backend == HardwareBackend.CUDA:
            if isinstance(architecture, NVIDIAArchitecture) and architecture in _ASYNC_P2P_ARCHS:
                return list(_CUDA_HIGH_END)
            return list(_CUDA_STANDARD)
        if backend == HardwareBackend.AMD:
            return list(_CUDA_STANDARD)
        return list(_NON_CUDA)

    @staticmethod
    def get_optimal_schedule(
        backend: HardwareBackend,
        architecture: Architecture = None,
        num_stages: int = 2,
    ) -> PipelineScheduleType:
        """Auto-select the best pipeline schedule.

        Args:
            backend: Hardware backend.
            architecture: Specific architecture.
            num_stages: Number of pipeline stages.

        Returns:
            Optimal PipelineScheduleType for the hardware.
        """
        supported = PipelineScheduleFactory.get_supported_schedules(
            backend, architecture
        )
        for schedule in supported:
            spec = PIPELINE_SCHEDULE_SPECS[schedule]
            if num_stages >= spec.min_stages:
                return schedule
        return PipelineScheduleType.GPIPE

    @staticmethod
    def get_schedule_spec(
        schedule: PipelineScheduleType,
    ) -> PipelineScheduleSpec:
        """Return the spec for a given schedule type."""
        return PIPELINE_SCHEDULE_SPECS[schedule]

    @staticmethod
    def get_torch_schedule_class_name(
        schedule: PipelineScheduleType,
    ) -> str:
        """Return the PyTorch class name for the schedule.

        These live in torch.distributed.pipelining.schedules.
        """
        mapping = {
            PipelineScheduleType.GPIPE: "ScheduleGPipe",
            PipelineScheduleType.INTERLEAVED_1F1B: "ScheduleInterleaved1F1B",
            PipelineScheduleType.ZERO_BUBBLE: "ScheduleInterleavedZeroBubble",
            PipelineScheduleType.ZBV_ZERO_BUBBLE: "ScheduleZBVZeroBubble",
            PipelineScheduleType.LOOPED_BFS: "ScheduleLoopedBFS",
        }
        return mapping.get(schedule, "ScheduleGPipe")
