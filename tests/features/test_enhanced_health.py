"""
Tests for enhanced health monitoring.

Tests cover:
- Actual liveness verification
- System resource monitoring
- Health history tracking
- Predictive health analysis
- Threshold configuration
- Auto-recovery recommendations
"""

import time
from datetime import datetime, timedelta

import pytest
import torch
import torch.nn as nn

from kernel_pytorch.monitoring.enhanced_health import (
    EnhancedHealthMonitor,
    HealthHistoryEntry,
    HealthTrend,
    PredictiveHealthReport,
    ResourceThresholds,
    ResourceType,
    SystemResourceMonitor,
    create_enhanced_health_monitor,
)
from kernel_pytorch.monitoring.health_monitor import HealthStatus


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 5):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class TestResourceThresholds:
    """Tests for ResourceThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = ResourceThresholds()
        assert thresholds.cpu_warning_percent == 70.0
        assert thresholds.cpu_critical_percent == 90.0
        assert thresholds.memory_warning_percent == 75.0
        assert thresholds.memory_critical_percent == 90.0
        assert thresholds.disk_warning_percent == 80.0
        assert thresholds.disk_critical_percent == 95.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = ResourceThresholds(
            cpu_warning_percent=60.0,
            memory_critical_percent=95.0,
        )
        assert thresholds.cpu_warning_percent == 60.0
        assert thresholds.memory_critical_percent == 95.0


class TestSystemResourceMonitor:
    """Tests for SystemResourceMonitor."""

    def test_monitor_creation(self):
        """Test monitor can be created."""
        monitor = SystemResourceMonitor()
        assert monitor is not None

    def test_cpu_percent(self):
        """Test CPU percentage retrieval."""
        monitor = SystemResourceMonitor()
        cpu = monitor.get_cpu_percent(interval=0.01)
        # May be None if psutil not available
        if cpu is not None:
            assert 0 <= cpu <= 100

    def test_memory_info(self):
        """Test memory info retrieval."""
        monitor = SystemResourceMonitor()
        mem = monitor.get_memory_info()
        if mem is not None:
            assert "total_gb" in mem
            assert "available_gb" in mem
            assert "used_gb" in mem
            assert "percent" in mem
            assert mem["total_gb"] > 0

    def test_disk_info(self):
        """Test disk info retrieval."""
        monitor = SystemResourceMonitor()
        disk = monitor.get_disk_info("/")
        if disk is not None:
            assert "total_gb" in disk
            assert "free_gb" in disk
            assert "percent" in disk

    def test_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        monitor = SystemResourceMonitor()
        gpu_info = monitor.get_gpu_memory_info()
        # May be None if no GPU
        if torch.cuda.is_available():
            assert gpu_info is not None
            assert len(gpu_info) > 0

    def test_process_info(self):
        """Test process info retrieval."""
        monitor = SystemResourceMonitor()
        proc = monitor.get_process_info()
        if proc is not None:
            assert "pid" in proc
            assert "memory_mb" in proc
            assert "threads" in proc


class TestEnhancedHealthMonitor:
    """Tests for EnhancedHealthMonitor."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def monitor(self, model):
        """Create a health monitor."""
        return EnhancedHealthMonitor(model=model, model_name="test_model")

    def test_monitor_creation(self, model):
        """Test monitor creation."""
        monitor = EnhancedHealthMonitor(model=model)
        assert monitor is not None
        assert monitor.model is model

    def test_monitor_without_model(self):
        """Test monitor creation without model."""
        monitor = EnhancedHealthMonitor()
        assert monitor.model is None

    def test_set_thresholds(self, monitor):
        """Test threshold configuration."""
        monitor.set_thresholds(
            cpu_warning=50.0,
            cpu_critical=80.0,
            memory_warning=60.0,
        )
        assert monitor._thresholds.cpu_warning_percent == 50.0
        assert monitor._thresholds.cpu_critical_percent == 80.0
        assert monitor._thresholds.memory_warning_percent == 60.0

    def test_is_live_returns_bool(self, monitor):
        """Test liveness check returns boolean."""
        result = monitor.is_live()
        assert isinstance(result, bool)

    def test_is_live_actually_checks(self, monitor):
        """Test liveness check actually performs verification."""
        # Should pass under normal conditions
        assert monitor.is_live() is True

    def test_is_live_caching(self, monitor):
        """Test liveness results are cached."""
        # First call
        result1 = monitor.is_live()
        # Second call (should be cached)
        result2 = monitor.is_live()
        assert result1 == result2

    def test_add_liveness_check(self, monitor):
        """Test adding custom liveness check."""
        check_called = []

        def custom_check():
            check_called.append(True)
            return True

        monitor.add_liveness_check(custom_check)

        # Clear cache
        monitor._last_liveness_check = None

        monitor.is_live()
        assert len(check_called) > 0

    def test_failing_liveness_check(self, monitor):
        """Test that failing check returns False."""
        def always_fail():
            return False

        monitor.add_liveness_check(always_fail)
        monitor._last_liveness_check = None

        assert monitor.is_live() is False

    def test_check_health_returns_health_check(self, monitor):
        """Test health check returns proper type."""
        health = monitor.check_health()
        assert health is not None
        assert hasattr(health, "overall_status")
        assert hasattr(health, "components")

    def test_check_health_includes_system_resources(self, monitor):
        """Test health check includes system resource components."""
        health = monitor.check_health()
        component_names = [c.name for c in health.components]

        # Should include base checks
        assert "model" in component_names
        assert "gpu" in component_names
        assert "inference" in component_names

        # Should include enhanced resource checks
        assert "cpu" in component_names
        assert "memory" in component_names
        assert "disk" in component_names
        assert "process" in component_names

    def test_health_history_recording(self, monitor):
        """Test health checks are recorded in history."""
        # Perform a few health checks
        monitor.check_health()
        monitor.check_health()
        monitor.check_health()

        history = monitor.get_health_history()
        assert len(history) >= 3

    def test_health_history_limit(self, monitor):
        """Test health history respects limit parameter."""
        for _ in range(10):
            monitor.check_health()

        history = monitor.get_health_history(limit=5)
        assert len(history) == 5

    def test_health_history_time_filter(self, monitor):
        """Test health history time filtering."""
        monitor.check_health()

        # Get history from last minute
        history = monitor.get_health_history(minutes=1)
        assert len(history) >= 1

        # All entries should be recent
        cutoff = datetime.now() - timedelta(minutes=1)
        for entry in history:
            assert entry.timestamp >= cutoff


class TestPredictiveHealth:
    """Tests for predictive health analysis."""

    @pytest.fixture
    def monitor(self):
        """Create monitor with predictive enabled."""
        return EnhancedHealthMonitor(enable_predictive=True)

    def test_predictive_with_no_data(self, monitor):
        """Test predictive health with no history."""
        report = monitor.get_predictive_health()
        assert isinstance(report, PredictiveHealthReport)
        assert report.data_points == 0

    def test_predictive_with_insufficient_data(self, monitor):
        """Test predictive health with insufficient history."""
        monitor.check_health()
        report = monitor.get_predictive_health()
        assert "Insufficient data" in report.recommendations[0] or report.data_points < 3

    def test_predictive_with_data(self, monitor):
        """Test predictive health with sufficient history."""
        # Generate some history
        for _ in range(5):
            monitor.check_health()
            time.sleep(0.01)

        report = monitor.get_predictive_health()
        assert report.data_points >= 5
        assert isinstance(report.trend, HealthTrend)

    def test_predictive_report_to_dict(self, monitor):
        """Test report can be converted to dict."""
        monitor.check_health()
        report = monitor.get_predictive_health()
        report_dict = report.to_dict()

        assert "current_status" in report_dict
        assert "predicted_status" in report_dict
        assert "trend" in report_dict
        assert "recommendations" in report_dict

    def test_predictive_disabled(self):
        """Test predictive monitoring when disabled."""
        monitor = EnhancedHealthMonitor(enable_predictive=False)
        report = monitor.get_predictive_health()
        assert "disabled" in report.recommendations[0].lower()


class TestHealthTrend:
    """Tests for HealthTrend enum."""

    def test_trend_values(self):
        """Test trend enum values."""
        assert HealthTrend.IMPROVING.value == "improving"
        assert HealthTrend.STABLE.value == "stable"
        assert HealthTrend.DEGRADING.value == "degrading"
        assert HealthTrend.CRITICAL.value == "critical"


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_resource_types(self):
        """Test resource type values."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.DISK.value == "disk"
        assert ResourceType.GPU_MEMORY.value == "gpu_memory"
        assert ResourceType.NETWORK.value == "network"


class TestHealthHistoryEntry:
    """Tests for HealthHistoryEntry dataclass."""

    def test_entry_creation(self):
        """Test history entry creation."""
        entry = HealthHistoryEntry(
            timestamp=datetime.now(),
            status=HealthStatus.HEALTHY,
            components={"model": HealthStatus.HEALTHY},
            metrics={"cpu_percent": 50.0},
        )
        assert entry.status == HealthStatus.HEALTHY
        assert "model" in entry.components
        assert entry.metrics["cpu_percent"] == 50.0


class TestCreateEnhancedHealthMonitor:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        monitor = create_enhanced_health_monitor()
        assert isinstance(monitor, EnhancedHealthMonitor)
        assert monitor._enable_predictive is True

    def test_create_with_model(self):
        """Test creation with model."""
        model = SimpleModel()
        monitor = create_enhanced_health_monitor(
            model=model,
            model_name="my_model",
        )
        assert monitor.model is model
        assert monitor.model_name == "my_model"

    def test_create_without_predictive(self):
        """Test creation with predictive disabled."""
        monitor = create_enhanced_health_monitor(enable_predictive=False)
        assert monitor._enable_predictive is False


class TestRecommendations:
    """Tests for auto-recovery recommendations."""

    @pytest.fixture
    def monitor(self):
        """Create monitor with low thresholds for testing."""
        monitor = EnhancedHealthMonitor()
        monitor.set_thresholds(
            cpu_warning=1.0,  # Very low threshold
            memory_warning=1.0,
        )
        return monitor

    def test_recommendations_generated(self, monitor):
        """Test that recommendations are generated."""
        # Generate history
        for _ in range(5):
            monitor.check_health()

        report = monitor.get_predictive_health()
        # Should have some recommendations due to low thresholds
        assert isinstance(report.recommendations, list)


class TestIntegration:
    """Integration tests for enhanced health monitoring."""

    def test_full_health_check_workflow(self):
        """Test complete health monitoring workflow."""
        model = SimpleModel()
        monitor = create_enhanced_health_monitor(
            model=model,
            model_name="integration_test",
            enable_predictive=True,
        )

        # Configure thresholds
        monitor.set_thresholds(cpu_warning=80.0, memory_warning=85.0)

        # Check liveness
        assert monitor.is_live() is True

        # Check readiness
        assert monitor.is_ready() is True

        # Full health check
        health = monitor.check_health()
        assert health is not None

        # Get predictive report
        report = monitor.get_predictive_health()
        assert report is not None

        # Get history
        history = monitor.get_health_history(limit=10)
        assert len(history) >= 1

    def test_model_health_with_inference(self):
        """Test health monitoring with actual inference."""
        model = SimpleModel()
        monitor = create_enhanced_health_monitor(model=model)

        # Simulate successful inference
        monitor.record_inference(success=True)

        health = monitor.check_health()
        inference_component = next(
            (c for c in health.components if c.name == "inference"),
            None
        )
        assert inference_component is not None
        assert inference_component.status == HealthStatus.HEALTHY

    def test_model_health_with_errors(self):
        """Test health monitoring with inference errors."""
        model = SimpleModel()
        monitor = create_enhanced_health_monitor(model=model)

        # Simulate many errors
        for _ in range(15):
            monitor.record_inference(success=False)

        health = monitor.check_health()
        inference_component = next(
            (c for c in health.components if c.name == "inference"),
            None
        )
        assert inference_component is not None
        assert inference_component.status == HealthStatus.UNHEALTHY


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_health_checks(self):
        """Test concurrent health check calls."""
        import threading

        monitor = create_enhanced_health_monitor()
        results = []
        errors = []

        def check_health():
            try:
                health = monitor.check_health()
                results.append(health.overall_status)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=check_health)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_concurrent_liveness_checks(self):
        """Test concurrent liveness check calls."""
        import threading

        monitor = create_enhanced_health_monitor()
        results = []

        def check_liveness():
            results.append(monitor.is_live())

        threads = [
            threading.Thread(target=check_liveness)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(isinstance(r, bool) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
