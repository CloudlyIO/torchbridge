"""Tests for checkpoint frequency advisor and health triggers."""

import math

import pytest

from torchbridge.checkpoint.frequency import (
    _MTBF_HOURS,
    CheckpointFrequencyAdvisor,
    CheckpointHealthTrigger,
    FrequencyRecommendation,
    _cluster_size_category,
)


class TestClusterSizeCategory:
    """Tests for cluster size classification."""

    def test_small_cluster(self):
        assert _cluster_size_category(1) == "small"
        assert _cluster_size_category(8) == "small"

    def test_medium_cluster(self):
        assert _cluster_size_category(9) == "medium"
        assert _cluster_size_category(64) == "medium"

    def test_large_cluster(self):
        assert _cluster_size_category(65) == "large"
        assert _cluster_size_category(256) == "large"

    def test_xlarge_cluster(self):
        assert _cluster_size_category(257) == "xlarge"
        assert _cluster_size_category(1024) == "xlarge"


class TestMTBFEstimates:
    """Tests for MTBF default estimates."""

    def test_small_mtbf(self):
        assert _MTBF_HOURS["small"] == 168.0  # 1 week

    def test_medium_mtbf(self):
        assert _MTBF_HOURS["medium"] == 48.0  # 2 days

    def test_large_mtbf(self):
        assert _MTBF_HOURS["large"] == 12.0

    def test_xlarge_mtbf(self):
        assert _MTBF_HOURS["xlarge"] == 4.0

    def test_decreasing_with_size(self):
        values = [_MTBF_HOURS[k] for k in ["small", "medium", "large", "xlarge"]]
        assert values == sorted(values, reverse=True)


class TestFrequencyRecommendation:
    """Tests for FrequencyRecommendation dataclass."""

    def test_to_dict(self):
        rec = FrequencyRecommendation(
            interval_minutes=30,
            interval_steps=1800,
            reasoning="Test reasoning",
            risk_level="medium",
            optimal_overhead_pct=1.23456,
        )
        d = rec.to_dict()
        assert d["interval_minutes"] == 30
        assert d["interval_steps"] == 1800
        assert d["reasoning"] == "Test reasoning"
        assert d["risk_level"] == "medium"
        assert d["optimal_overhead_pct"] == 1.23  # Rounded

    def test_to_dict_none_steps(self):
        rec = FrequencyRecommendation(
            interval_minutes=10,
            interval_steps=None,
            reasoning="No step info",
            risk_level="high",
            optimal_overhead_pct=5.0,
        )
        d = rec.to_dict()
        assert d["interval_steps"] is None


class TestCheckpointFrequencyAdvisor:
    """Tests for CheckpointFrequencyAdvisor."""

    def test_youngs_formula_correctness(self):
        advisor = CheckpointFrequencyAdvisor()
        # C=60s, MTBF=48h(172800s) → sqrt(2*60*172800) = sqrt(20736000) ≈ 4554s ≈ 75min
        rec = advisor.recommend(
            world_size=16,
            checkpoint_time_seconds=60.0,
            mtbf_hours=48.0,
        )
        expected_seconds = math.sqrt(2 * 60.0 * 48.0 * 3600.0)
        expected_minutes = int(expected_seconds / 60.0)
        assert rec.interval_minutes == expected_minutes

    def test_small_cluster_low_risk(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(world_size=4)
        assert rec.risk_level == "low"

    def test_large_cluster_medium_risk(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(world_size=128)
        assert rec.risk_level == "medium"

    def test_xlarge_cluster_high_risk(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(world_size=512)
        assert rec.risk_level == "high"

    def test_custom_mtbf(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(
            world_size=8,
            mtbf_hours=1.0,  # Very unreliable cluster
        )
        assert rec.risk_level == "high"
        # Short MTBF → more frequent checkpoints
        assert rec.interval_minutes < 20

    def test_interval_steps_computed(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(
            world_size=8,
            step_time_seconds=0.5,
        )
        assert rec.interval_steps is not None
        assert rec.interval_steps > 0

    def test_interval_steps_zero_step_time(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(
            world_size=8,
            step_time_seconds=0.0,
        )
        assert rec.interval_steps is None

    def test_reasoning_contains_formula(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(world_size=32)
        assert "Young's formula" in rec.reasoning
        assert "MTBF" in rec.reasoning

    def test_overhead_percentage_positive(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(world_size=8)
        assert rec.optimal_overhead_pct > 0
        assert rec.optimal_overhead_pct < 100

    def test_minimum_interval_is_one(self):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(
            world_size=1,
            checkpoint_time_seconds=0.001,
            mtbf_hours=0.001,
        )
        assert rec.interval_minutes >= 1

    def test_larger_cluster_shorter_interval(self):
        advisor = CheckpointFrequencyAdvisor()
        small = advisor.recommend(world_size=4)
        large = advisor.recommend(world_size=512)
        # Larger clusters have lower MTBF → shorter intervals
        assert large.interval_minutes < small.interval_minutes

    def test_longer_checkpoint_time_longer_interval(self):
        advisor = CheckpointFrequencyAdvisor()
        fast = advisor.recommend(world_size=64, checkpoint_time_seconds=10.0)
        slow = advisor.recommend(world_size=64, checkpoint_time_seconds=300.0)
        # Slower checkpoints → longer intervals to reduce overhead
        assert slow.interval_minutes > fast.interval_minutes


class TestCheckpointHealthTrigger:
    """Tests for CheckpointHealthTrigger."""

    def test_default_init(self):
        trigger = CheckpointHealthTrigger()
        assert trigger._temp_threshold == 85.0
        assert trigger._trigger_on_degrading is True
        assert trigger._utilization_drop_threshold == 0.5

    def test_custom_init(self):
        trigger = CheckpointHealthTrigger(
            health_threshold_temp_c=90.0,
            trigger_on_degrading=False,
            utilization_drop_threshold=0.3,
        )
        assert trigger._temp_threshold == 90.0
        assert trigger._trigger_on_degrading is False

    def test_invalid_temp_threshold(self):
        with pytest.raises(ValueError, match="health_threshold_temp_c"):
            CheckpointHealthTrigger(health_threshold_temp_c=-10)

    def test_invalid_utilization_threshold(self):
        with pytest.raises(ValueError, match="utilization_drop_threshold"):
            CheckpointHealthTrigger(utilization_drop_threshold=0.0)
        with pytest.raises(ValueError, match="utilization_drop_threshold"):
            CheckpointHealthTrigger(utilization_drop_threshold=1.5)

    def test_temperature_trigger(self):
        trigger = CheckpointHealthTrigger(health_threshold_temp_c=85.0)
        should, reason = trigger.should_checkpoint({"temperature_c": 90.0})
        assert should is True
        assert "90" in reason
        assert "85" in reason

    def test_temperature_below_threshold(self):
        trigger = CheckpointHealthTrigger(health_threshold_temp_c=85.0)
        should, reason = trigger.should_checkpoint({"temperature_c": 70.0})
        assert should is False
        assert reason == ""

    def test_critical_health_trend_string(self):
        trigger = CheckpointHealthTrigger()
        should, reason = trigger.should_checkpoint({"health_trend": "critical"})
        assert should is True
        assert "CRITICAL" in reason

    def test_degrading_health_trend(self):
        trigger = CheckpointHealthTrigger(trigger_on_degrading=True)
        should, reason = trigger.should_checkpoint({"health_trend": "degrading"})
        assert should is True
        assert "DEGRADING" in reason

    def test_degrading_ignored_when_disabled(self):
        trigger = CheckpointHealthTrigger(trigger_on_degrading=False)
        should, _ = trigger.should_checkpoint({"health_trend": "degrading"})
        assert should is False

    def test_stable_trend_no_trigger(self):
        trigger = CheckpointHealthTrigger()
        should, _ = trigger.should_checkpoint({"health_trend": "stable"})
        assert should is False

    def test_memory_errors_trigger(self):
        trigger = CheckpointHealthTrigger()
        should, reason = trigger.should_checkpoint({"memory_errors": 5})
        assert should is True
        assert "Memory errors" in reason

    def test_no_memory_errors_no_trigger(self):
        trigger = CheckpointHealthTrigger()
        should, _ = trigger.should_checkpoint({"memory_errors": 0})
        assert should is False

    def test_utilization_drop_trigger(self):
        trigger = CheckpointHealthTrigger(utilization_drop_threshold=0.5)
        should, reason = trigger.should_checkpoint(
            {
                "utilization": 0.2,
                "avg_utilization": 0.9,
            }
        )
        assert should is True
        assert "dropped" in reason.lower()

    def test_utilization_normal_no_trigger(self):
        trigger = CheckpointHealthTrigger(utilization_drop_threshold=0.5)
        should, _ = trigger.should_checkpoint(
            {
                "utilization": 0.85,
                "avg_utilization": 0.9,
            }
        )
        assert should is False

    def test_healthy_device_no_trigger(self):
        trigger = CheckpointHealthTrigger()
        should, reason = trigger.should_checkpoint(
            {
                "temperature_c": 65.0,
                "health_trend": "stable",
                "memory_errors": 0,
                "utilization": 0.85,
                "avg_utilization": 0.9,
            }
        )
        assert should is False
        assert reason == ""

    def test_enum_health_trend(self):
        """Test with enum-like health trend (has .value attribute)."""

        class FakeTrend:
            def __init__(self, val):
                self.value = val

        trigger = CheckpointHealthTrigger()
        should, reason = trigger.should_checkpoint(
            {"health_trend": FakeTrend("critical")}
        )
        assert should is True


class TestClusterHealthEvaluation:
    """Tests for cluster-wide health evaluation."""

    def test_all_healthy(self):
        trigger = CheckpointHealthTrigger()
        devices = [
            {"temperature_c": 65.0, "health_trend": "stable", "memory_errors": 0},
            {"temperature_c": 70.0, "health_trend": "stable", "memory_errors": 0},
        ]
        should, reason = trigger.evaluate_cluster_health(devices)
        assert should is False

    def test_one_device_triggers(self):
        trigger = CheckpointHealthTrigger()
        devices = [
            {"temperature_c": 65.0, "health_trend": "stable", "memory_errors": 0},
            {"temperature_c": 92.0, "health_trend": "stable", "memory_errors": 0},
        ]
        should, reason = trigger.evaluate_cluster_health(devices)
        assert should is True
        assert "92" in reason

    def test_device_id_in_reason(self):
        trigger = CheckpointHealthTrigger()
        devices = [
            {
                "device_id": "gpu:0",
                "temperature_c": 65.0,
                "health_trend": "stable",
                "memory_errors": 0,
            },
            {
                "device_id": "gpu:1",
                "temperature_c": 95.0,
                "health_trend": "stable",
                "memory_errors": 0,
            },
        ]
        should, reason = trigger.evaluate_cluster_health(devices)
        assert should is True
        assert "gpu:1" in reason

    def test_empty_cluster(self):
        trigger = CheckpointHealthTrigger()
        should, reason = trigger.evaluate_cluster_health([])
        assert should is False
