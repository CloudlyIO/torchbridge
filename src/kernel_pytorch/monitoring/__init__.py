"""
Production Monitoring Module for KernelPyTorch

This module provides comprehensive monitoring capabilities for production
deployments of KernelPyTorch-optimized models.

Key Components:
- Prometheus Metrics: Export metrics for Prometheus scraping
- Grafana Dashboards: Pre-configured dashboard definitions
- Health Monitoring: System and model health checks
- Performance Tracking: Latency, throughput, and resource monitoring

Example:
    ```python
    from kernel_pytorch.monitoring import (
        MetricsExporter,
        start_metrics_server,
        create_grafana_dashboard
    )

    # Start Prometheus metrics server
    exporter = MetricsExporter(model_name="my_model")
    start_metrics_server(exporter, port=9090)

    # Record inference metrics
    exporter.record_inference(latency_ms=5.2, batch_size=32)
    ```

Version: 0.3.10
"""

from .prometheus_exporter import (
    MetricsExporter,
    MetricsConfig,
    InferenceMetrics,
    SystemMetrics,
    ModelMetrics,
    start_metrics_server,
    create_metrics_exporter,
)

from .grafana_dashboards import (
    GrafanaDashboard,
    DashboardPanel,
    create_inference_dashboard,
    create_system_dashboard,
    create_full_dashboard,
    export_dashboard_json,
)

from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    HealthCheck,
    ComponentHealth,
    create_health_monitor,
)

__version__ = "0.4.2"

__all__ = [
    # Prometheus
    "MetricsExporter",
    "MetricsConfig",
    "InferenceMetrics",
    "SystemMetrics",
    "ModelMetrics",
    "start_metrics_server",
    "create_metrics_exporter",
    # Grafana
    "GrafanaDashboard",
    "DashboardPanel",
    "create_inference_dashboard",
    "create_system_dashboard",
    "create_full_dashboard",
    "export_dashboard_json",
    # Health
    "HealthMonitor",
    "HealthStatus",
    "HealthCheck",
    "ComponentHealth",
    "create_health_monitor",
]
