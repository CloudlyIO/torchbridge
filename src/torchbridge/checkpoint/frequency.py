# SPDX-License-Identifier: Apache-2.0
"""
Checkpoint Frequency Advisor & Health Triggers

Recommends optimal checkpoint frequency using Young's formula and triggers
immediate checkpoint saves based on hardware health degradation signals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrequencyRecommendation:
    """Recommended checkpoint frequency with reasoning.

    Attributes:
        interval_minutes: Recommended checkpoint interval in minutes.
        interval_steps: Approximate interval in training steps (if step
            time is known).
        reasoning: Human-readable explanation of the recommendation.
        risk_level: Overall risk assessment ("low", "medium", "high").
        optimal_overhead_pct: Estimated I/O overhead as % of training time.
    """

    interval_minutes: int
    interval_steps: int | None
    reasoning: str
    risk_level: str
    optimal_overhead_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "interval_minutes": self.interval_minutes,
            "interval_steps": self.interval_steps,
            "reasoning": self.reasoning,
            "risk_level": self.risk_level,
            "optimal_overhead_pct": round(self.optimal_overhead_pct, 2),
        }


# ── MTBF estimates by cluster size ──────────────────────────────────────────

_MTBF_HOURS: dict[str, float] = {
    "small": 168.0,  # 1-8 GPUs: ~1 week
    "medium": 48.0,  # 9-64 GPUs: ~2 days
    "large": 12.0,  # 65-256 GPUs: ~12 hours
    "xlarge": 4.0,  # 257+ GPUs: ~4 hours
}


def _cluster_size_category(world_size: int) -> str:
    """Classify cluster size for MTBF estimation."""
    if world_size <= 8:
        return "small"
    if world_size <= 64:
        return "medium"
    if world_size <= 256:
        return "large"
    return "xlarge"


class CheckpointFrequencyAdvisor:
    """Recommend optimal checkpoint frequency based on cluster MTBF.

    Uses Young's formula: optimal interval = sqrt(2 * C * M) where:
    - C = checkpoint cost (time to save a checkpoint, in same units as M)
    - M = mean time between failures (MTBF)

    This minimizes the expected total overhead (checkpoint I/O + lost work
    from failures). At the optimal interval, checkpoint overhead and
    expected lost work are equal, each contributing C / sqrt(2CM) of
    total training time.

    References:
        Young, J. W. (1974). "A first order approximation to the optimal
        checkpoint interval." Communications of the ACM.
    """

    def recommend(
        self,
        world_size: int = 1,
        checkpoint_time_seconds: float = 60.0,
        step_time_seconds: float = 1.0,
        mtbf_hours: float | None = None,
    ) -> FrequencyRecommendation:
        """Compute optimal checkpoint frequency.

        Args:
            world_size: Total number of GPUs/ranks in the cluster.
            checkpoint_time_seconds: Estimated time to save one checkpoint.
            step_time_seconds: Average training step duration.
            mtbf_hours: Mean time between failures in hours. If None,
                estimated from cluster size.

        Returns:
            FrequencyRecommendation with interval and reasoning.
        """
        # Determine MTBF
        category = _cluster_size_category(world_size)
        if mtbf_hours is None:
            mtbf_hours = _MTBF_HOURS[category]

        mtbf_seconds = mtbf_hours * 3600.0
        checkpoint_cost = max(checkpoint_time_seconds, 0.001)

        # Young's formula: optimal interval = sqrt(2 * C * M)
        optimal_seconds = math.sqrt(2.0 * checkpoint_cost * mtbf_seconds)
        optimal_minutes = max(1, int(optimal_seconds / 60.0))

        # Compute overhead percentage
        overhead_pct = (checkpoint_cost / optimal_seconds) * 100.0

        # Steps per interval
        interval_steps: int | None = None
        if step_time_seconds > 0:
            interval_steps = max(1, int(optimal_seconds / step_time_seconds))

        # Risk assessment
        if mtbf_hours >= 72:
            risk = "low"
        elif mtbf_hours >= 12:
            risk = "medium"
        else:
            risk = "high"

        reasoning = (
            f"Young's formula: sqrt(2 * {checkpoint_cost:.0f}s * "
            f"{mtbf_hours:.0f}h MTBF) = {optimal_minutes}min interval. "
            f"Cluster: {world_size} GPUs ({category}), "
            f"estimated MTBF: {mtbf_hours:.0f}h. "
            f"Checkpoint overhead: {overhead_pct:.1f}%."
        )

        return FrequencyRecommendation(
            interval_minutes=optimal_minutes,
            interval_steps=interval_steps,
            reasoning=reasoning,
            risk_level=risk,
            optimal_overhead_pct=overhead_pct,
        )


# ── Health-triggered checkpointing ──────────────────────────────────────────


class CheckpointHealthTrigger:
    """Trigger checkpoint saves based on hardware health signals.

    Monitors device health metrics and recommends immediate checkpoint
    saves when hardware degradation is detected. Integrates with
    TorchBridge's HardwareHealthMonitor output format.

    Triggers on:
    - Temperature exceeding critical threshold (approaching thermal shutdown)
    - Memory error rate increase (ECC corrections trending up)
    - Health trend transitioning to DEGRADING or CRITICAL
    - Sudden GPU utilization drop (potential hardware issue)

    Args:
        health_threshold_temp_c: Temperature threshold for triggering (Celsius).
        trigger_on_degrading: Trigger on DEGRADING health trend.
        utilization_drop_threshold: Trigger if utilization drops below this
            fraction of the recent average.
    """

    def __init__(
        self,
        health_threshold_temp_c: float = 85.0,
        trigger_on_degrading: bool = True,
        utilization_drop_threshold: float = 0.5,
    ):
        if health_threshold_temp_c <= 0:
            raise ValueError(
                f"health_threshold_temp_c must be > 0, got {health_threshold_temp_c}"
            )
        if not 0.0 < utilization_drop_threshold <= 1.0:
            raise ValueError(
                f"utilization_drop_threshold must be in (0, 1], "
                f"got {utilization_drop_threshold}"
            )

        self._temp_threshold = health_threshold_temp_c
        self._trigger_on_degrading = trigger_on_degrading
        self._utilization_drop_threshold = utilization_drop_threshold

    def should_checkpoint(self, device_health: dict[str, Any]) -> tuple[bool, str]:
        """Check if an immediate checkpoint should be triggered.

        Args:
            device_health: Health status dict from HardwareHealthMonitor.
                Expected keys: "temperature_c", "health_trend",
                "memory_errors", "utilization", "avg_utilization".

        Returns:
            Tuple of (should_save, reason). reason is empty if no trigger.
        """
        # Temperature check
        temp = device_health.get("temperature_c", 0.0)
        if temp >= self._temp_threshold:
            return True, (
                f"Temperature {temp:.0f}C exceeds threshold {self._temp_threshold:.0f}C"
            )

        # Health trend check
        trend = device_health.get("health_trend", "stable")
        if isinstance(trend, str):
            trend_lower = trend.lower()
        else:
            # Handle enum values (HealthTrend.DEGRADING → "degrading")
            trend_lower = (
                str(trend.value).lower()
                if hasattr(trend, "value")
                else str(trend).lower()
            )

        if trend_lower == "critical":
            return True, "Health trend is CRITICAL"
        if self._trigger_on_degrading and trend_lower == "degrading":
            return True, "Health trend is DEGRADING"

        # Memory error check
        mem_errors = device_health.get("memory_errors", 0)
        if isinstance(mem_errors, (int, float)) and mem_errors > 0:
            return True, f"Memory errors detected: {mem_errors}"

        # Utilization drop check
        util = device_health.get("utilization", None)
        avg_util = device_health.get("avg_utilization", None)
        if (
            util is not None
            and avg_util is not None
            and avg_util > 0
            and util < avg_util * self._utilization_drop_threshold
        ):
            return True, (
                f"Utilization dropped to {util:.0%} "
                f"(avg: {avg_util:.0%}, threshold: "
                f"{self._utilization_drop_threshold:.0%})"
            )

        return False, ""

    def evaluate_cluster_health(
        self, all_device_health: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """Check cluster-wide health for checkpoint trigger.

        Triggers if ANY device in the cluster exceeds thresholds.

        Args:
            all_device_health: List of health dicts, one per device.

        Returns:
            Tuple of (should_save, reason).
        """
        for i, health in enumerate(all_device_health):
            should_save, reason = self.should_checkpoint(health)
            if should_save:
                device_id = health.get("device_id", f"device_{i}")
                return True, f"{device_id}: {reason}"

        return False, ""
