"""
Tests for the TorchBridge Monitoring Module (v0.3.10)

Tests cover:
- Prometheus metrics exporter
- Grafana dashboard generation
- Health monitoring
- Integration between components
"""

import json
import os
import tempfile
import time

import pytest
import torch
import torch.nn as nn

# ============================================================================
# Test Fixtures
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 5):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Prometheus Exporter Tests
# ============================================================================


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from torchbridge.monitoring import prometheus_exporter

        config = prometheus_exporter.MetricsConfig()
        assert config.port == 9090
        assert config.model_name == "model"
        assert config.namespace == "torchbridge"

    def test_custom_config(self):
        """Test custom configuration."""
        from torchbridge.monitoring import prometheus_exporter

        config = prometheus_exporter.MetricsConfig(
            port=8080,
            model_name="transformer",
            enable_gpu_metrics=False,
        )
        assert config.port == 8080
        assert config.model_name == "transformer"
        assert config.enable_gpu_metrics is False


class TestMetricsExporter:
    """Tests for MetricsExporter."""

    def test_exporter_creation(self):
        """Test exporter instantiation."""
        from torchbridge.monitoring import MetricsExporter

        exporter = MetricsExporter()
        assert exporter is not None
        assert exporter.config.model_name == "model"

    def test_record_inference(self):
        """Test recording inference metrics."""
        from torchbridge.monitoring import MetricsExporter

        exporter = MetricsExporter()
        exporter.record_inference(latency_ms=5.0, batch_size=32)

        metrics = exporter.get_inference_metrics()
        assert metrics.total_requests == 1
        assert metrics.total_samples == 32

    def test_record_multiple_inferences(self):
        """Test recording multiple inferences."""
        from torchbridge.monitoring import MetricsExporter

        exporter = MetricsExporter()

        for _ in range(10):
            exporter.record_inference(latency_ms=5.0, batch_size=8)

        metrics = exporter.get_inference_metrics()
        assert metrics.total_requests == 10
        assert metrics.total_samples == 80

    def test_track_inference_context(self):
        """Test tracking inference with context manager."""
        from torchbridge.monitoring import MetricsExporter

        exporter = MetricsExporter()

        with exporter.track_inference(batch_size=16):
            time.sleep(0.01)  # Simulate inference

        metrics = exporter.get_inference_metrics()
        assert metrics.total_requests == 1
        assert metrics.total_samples == 16
        assert metrics.average_latency_ms > 0

    def test_get_metrics_text(self):
        """Test getting metrics in Prometheus format."""
        from torchbridge.monitoring import MetricsExporter

        exporter = MetricsExporter()
        exporter.record_inference(latency_ms=5.0, batch_size=8)

        text = exporter.get_metrics_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_reset_metrics(self):
        """Test resetting metrics."""
        from torchbridge.monitoring import MetricsExporter

        exporter = MetricsExporter()
        exporter.record_inference(latency_ms=5.0, batch_size=8)
        exporter.reset()

        metrics = exporter.get_inference_metrics()
        assert metrics.total_requests == 0
        assert metrics.total_samples == 0

    def test_system_metrics(self):
        """Test getting system metrics."""
        from torchbridge.monitoring import MetricsExporter

        exporter = MetricsExporter()
        metrics = exporter.get_system_metrics()

        assert hasattr(metrics, "cpu_percent")
        assert hasattr(metrics, "memory_percent")
        assert hasattr(metrics, "gpu_memory_used_mb")


class TestCreateMetricsExporter:
    """Tests for create_metrics_exporter utility."""

    def test_create_with_defaults(self):
        """Test creating exporter with defaults."""
        from torchbridge.monitoring import create_metrics_exporter

        exporter = create_metrics_exporter()
        assert exporter.config.model_name == "model"

    def test_create_with_options(self):
        """Test creating exporter with options."""
        from torchbridge.monitoring import create_metrics_exporter

        exporter = create_metrics_exporter(
            model_name="my_model",
            port=8080,
            enable_gpu_metrics=False,
        )

        assert exporter.config.model_name == "my_model"
        assert exporter.config.port == 8080


# ============================================================================
# Grafana Dashboard Tests
# ============================================================================


class TestDashboardPanel:
    """Tests for DashboardPanel."""

    def test_panel_creation(self):
        """Test panel instantiation."""
        from torchbridge.monitoring import DashboardPanel

        panel = DashboardPanel(title="Test Panel")
        assert panel.title == "Test Panel"
        assert panel.panel_type == "graph"

    def test_panel_to_dict(self):
        """Test panel serialization."""
        from torchbridge.monitoring import DashboardPanel

        panel = DashboardPanel(
            title="Latency",
            panel_type="timeseries",
            targets=[{"expr": "test_metric"}],
        )

        panel_dict = panel.to_dict()
        assert panel_dict["title"] == "Latency"
        assert panel_dict["type"] == "timeseries"
        assert len(panel_dict["targets"]) == 1


class TestGrafanaDashboard:
    """Tests for GrafanaDashboard."""

    def test_dashboard_creation(self):
        """Test dashboard instantiation."""
        from torchbridge.monitoring import GrafanaDashboard

        dashboard = GrafanaDashboard(title="Test Dashboard")
        assert dashboard.title == "Test Dashboard"
        assert dashboard.uid == "test-dashboard"

    def test_add_panel(self):
        """Test adding panels to dashboard."""
        from torchbridge.monitoring import DashboardPanel, GrafanaDashboard

        dashboard = GrafanaDashboard(title="Test")
        dashboard.add_panel(DashboardPanel(title="Panel 1"))
        dashboard.add_panel(DashboardPanel(title="Panel 2"))

        assert len(dashboard.panels) == 2

    def test_dashboard_to_dict(self):
        """Test dashboard serialization."""
        from torchbridge.monitoring import DashboardPanel, GrafanaDashboard

        dashboard = GrafanaDashboard(title="Test")
        dashboard.add_panel(DashboardPanel(title="Panel 1"))

        dashboard_dict = dashboard.to_dict()
        assert "dashboard" in dashboard_dict
        assert dashboard_dict["dashboard"]["title"] == "Test"
        assert len(dashboard_dict["dashboard"]["panels"]) == 1

    def test_dashboard_to_json(self):
        """Test dashboard JSON export."""
        from torchbridge.monitoring import GrafanaDashboard

        dashboard = GrafanaDashboard(title="Test")
        json_str = dashboard.to_json()

        parsed = json.loads(json_str)
        assert parsed["dashboard"]["title"] == "Test"


class TestCreateDashboards:
    """Tests for dashboard creation utilities."""

    def test_create_inference_dashboard(self):
        """Test creating inference dashboard."""
        from torchbridge.monitoring import create_inference_dashboard

        dashboard = create_inference_dashboard(model_name="transformer")

        assert "transformer" in dashboard.title
        assert len(dashboard.panels) > 0

    def test_create_system_dashboard(self):
        """Test creating system dashboard."""
        from torchbridge.monitoring import create_system_dashboard

        dashboard = create_system_dashboard()

        assert "System" in dashboard.title
        assert len(dashboard.panels) > 0

    def test_create_full_dashboard(self):
        """Test creating full dashboard."""
        from torchbridge.monitoring import create_full_dashboard

        dashboard = create_full_dashboard(model_name="bert")

        assert "bert" in dashboard.title
        # Should have panels from both inference and system dashboards
        assert len(dashboard.panels) > 5

    def test_export_dashboard_json(self, temp_dir):
        """Test exporting dashboard to JSON file."""
        from torchbridge.monitoring import (
            create_inference_dashboard,
            export_dashboard_json,
        )

        dashboard = create_inference_dashboard()
        output_path = os.path.join(temp_dir, "dashboard.json")

        export_dashboard_json(dashboard, output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
            assert "dashboard" in data


# ============================================================================
# Health Monitor Tests
# ============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test health status values."""
        from torchbridge.monitoring import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentHealth:
    """Tests for ComponentHealth."""

    def test_component_health_creation(self):
        """Test component health creation."""
        from torchbridge.monitoring import health_monitor

        health = health_monitor.ComponentHealth(
            name="test",
            status=health_monitor.HealthStatus.HEALTHY,
            message="All good",
        )

        assert health.name == "test"
        assert health.status == health_monitor.HealthStatus.HEALTHY

    def test_component_health_to_dict(self):
        """Test component health serialization."""
        from torchbridge.monitoring import health_monitor

        health = health_monitor.ComponentHealth(
            name="test",
            status=health_monitor.HealthStatus.HEALTHY,
        )

        health_dict = health.to_dict()
        assert health_dict["name"] == "test"
        assert health_dict["status"] == "healthy"


class TestHealthCheck:
    """Tests for HealthCheck."""

    def test_health_check_creation(self):
        """Test health check creation."""
        from torchbridge.monitoring import health_monitor

        check = health_monitor.HealthCheck(
            overall_status=health_monitor.HealthStatus.HEALTHY,
            components=[],
        )

        assert check.is_healthy()
        assert check.is_ready()

    def test_health_check_degraded(self):
        """Test degraded health check."""
        from torchbridge.monitoring import health_monitor

        check = health_monitor.HealthCheck(
            overall_status=health_monitor.HealthStatus.DEGRADED,
            components=[],
        )

        assert not check.is_healthy()
        assert check.is_ready()  # Degraded is still ready

    def test_health_check_to_dict(self):
        """Test health check serialization."""
        from torchbridge.monitoring import health_monitor

        check = health_monitor.HealthCheck(
            overall_status=health_monitor.HealthStatus.HEALTHY,
            components=[],
        )

        check_dict = check.to_dict()
        assert check_dict["overall_status"] == "healthy"
        assert "timestamp" in check_dict


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_monitor_creation(self):
        """Test health monitor creation."""
        from torchbridge.monitoring import HealthMonitor

        monitor = HealthMonitor()
        assert monitor is not None

    def test_monitor_with_model(self, simple_model):
        """Test health monitor with model."""
        from torchbridge.monitoring import HealthMonitor

        monitor = HealthMonitor(model=simple_model, model_name="simple")
        assert monitor.model is not None
        assert monitor.model_name == "simple"

    def test_check_health(self, simple_model):
        """Test full health check."""
        from torchbridge.monitoring import HealthMonitor

        monitor = HealthMonitor(model=simple_model)
        health = monitor.check_health()

        assert health.overall_status is not None
        assert len(health.components) >= 3  # model, gpu, inference

    def test_liveness_probe(self):
        """Test liveness probe."""
        from torchbridge.monitoring import HealthMonitor

        monitor = HealthMonitor()
        assert monitor.is_live() is True

    def test_readiness_probe(self, simple_model):
        """Test readiness probe."""
        from torchbridge.monitoring import HealthMonitor

        monitor = HealthMonitor(model=simple_model)
        # Should be ready with a valid model
        assert monitor.is_ready() is True

    def test_record_inference(self):
        """Test recording inference events."""
        from torchbridge.monitoring import HealthMonitor

        monitor = HealthMonitor()
        monitor.record_inference(success=True)
        monitor.record_inference(success=False)

        # Check through health check
        health = monitor.check_health()
        inference_health = next(
            c for c in health.components if c.name == "inference"
        )
        assert inference_health.details.get("error_count", 0) == 1

    def test_uptime(self):
        """Test uptime tracking."""
        from torchbridge.monitoring import HealthMonitor

        monitor = HealthMonitor()
        time.sleep(0.1)

        uptime = monitor.get_uptime()
        assert uptime >= 0.1

    def test_custom_health_check(self):
        """Test registering custom health check."""
        from torchbridge.monitoring import HealthMonitor, health_monitor

        monitor = HealthMonitor()

        def custom_check():
            return health_monitor.ComponentHealth(
                name="custom",
                status=health_monitor.HealthStatus.HEALTHY,
                message="Custom check passed",
            )

        monitor.register_check("custom", custom_check)
        health = monitor.check_health()

        custom_component = next(
            (c for c in health.components if c.name == "custom"), None
        )
        assert custom_component is not None
        assert custom_component.status == health_monitor.HealthStatus.HEALTHY


class TestCreateHealthMonitor:
    """Tests for create_health_monitor utility."""

    def test_create_without_model(self):
        """Test creating monitor without model."""
        from torchbridge.monitoring import create_health_monitor

        monitor = create_health_monitor()
        assert monitor is not None

    def test_create_with_model(self, simple_model):
        """Test creating monitor with model."""
        from torchbridge.monitoring import create_health_monitor

        monitor = create_health_monitor(model=simple_model, model_name="test")
        assert monitor.model is not None
        assert monitor.model_name == "test"


# ============================================================================
# Integration Tests
# ============================================================================


class TestMonitoringIntegration:
    """Integration tests for monitoring module."""

    def test_import_all(self):
        """Test importing all monitoring components."""
        from torchbridge.monitoring import (
            GrafanaDashboard,
            # Health
            HealthMonitor,
            MetricsExporter,
        )

        assert MetricsExporter is not None
        assert GrafanaDashboard is not None
        assert HealthMonitor is not None

    def test_full_monitoring_workflow(self, simple_model, temp_dir):
        """Test full monitoring workflow."""
        from torchbridge.monitoring import (
            HealthMonitor,
            MetricsExporter,
            create_inference_dashboard,
            export_dashboard_json,
        )

        # Create metrics exporter
        exporter = MetricsExporter()

        # Create health monitor
        monitor = HealthMonitor(model=simple_model)

        # Record some inferences
        for _ in range(10):
            with exporter.track_inference(batch_size=8):
                time.sleep(0.001)
            monitor.record_inference(success=True)

        # Check metrics
        metrics = exporter.get_inference_metrics()
        assert metrics.total_requests == 10
        assert metrics.total_samples == 80

        # Check health
        health = monitor.check_health()
        assert health.is_healthy()

        # Export dashboard
        dashboard = create_inference_dashboard(model_name="test")
        output_path = os.path.join(temp_dir, "dashboard.json")
        export_dashboard_json(dashboard, output_path)

        assert os.path.exists(output_path)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
