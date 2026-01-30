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

from .enhanced_health import (
    EnhancedHealthMonitor,
    HealthHistoryEntry,
    HealthTrend,
    PredictiveHealthReport,
    ResourceThresholds,
    ResourceType,
    SystemResourceMonitor,
    create_enhanced_health_monitor,
)
from .grafana_alerts import (
    AlertAnnotation,
    AlertCondition,
    AlertLabel,
    AlertRule,
    AlertRuleBuilder,
    AlertRuleGroup,
    AlertSeverity,
    AlertState,
    ComparisonOperator,
    create_default_alert_rules,
    export_alert_rules_json,
    export_alert_rules_yaml,
)
from .grafana_dashboards import (
    DashboardPanel,
    GrafanaDashboard,
    create_full_dashboard,
    create_inference_dashboard,
    create_system_dashboard,
    export_dashboard_json,
)
from .health_monitor import (
    ComponentHealth,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    create_health_monitor,
)
from .prometheus_exporter import (
    InferenceMetrics,
    MetricsConfig,
    MetricsExporter,
    ModelMetrics,
    SystemMetrics,
    create_metrics_exporter,
    start_metrics_server,
)
from .slo_framework import (
    BudgetStatus,
    ComplianceReport,
    ComplianceStatus,
    SLICollector,
    SLIMeasurement,
    SLIType,
    SLOConfig,
    SLOManager,
    SLOStatus,
    create_slo_manager,
)
from .structured_logging import (
    CorrelationContext,
    LogConfig,
    LogContext,
    LogLevel,
    PerformanceLogger,
    StructuredLogger,
    configure_logging,
    correlation_context,
    get_correlation_id,
    get_logger,
    log_context,
    log_function_call,
    performance_log,
    set_correlation_id,
)

__version__ = "0.4.34"

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
    # Structured Logging
    "LogConfig",
    "LogLevel",
    "StructuredLogger",
    "CorrelationContext",
    "LogContext",
    "PerformanceLogger",
    "configure_logging",
    "get_logger",
    "correlation_context",
    "log_context",
    "get_correlation_id",
    "set_correlation_id",
    "log_function_call",
    "performance_log",
    # Enhanced Health Monitoring
    "ResourceType",
    "HealthTrend",
    "ResourceThresholds",
    "HealthHistoryEntry",
    "PredictiveHealthReport",
    "SystemResourceMonitor",
    "EnhancedHealthMonitor",
    "create_enhanced_health_monitor",
    # SLO/SLI Framework
    "SLIType",
    "ComplianceStatus",
    "BudgetStatus",
    "SLOConfig",
    "SLIMeasurement",
    "SLOStatus",
    "ComplianceReport",
    "SLICollector",
    "SLOManager",
    "create_slo_manager",
    # Grafana Alert Rules
    "AlertSeverity",
    "AlertState",
    "ComparisonOperator",
    "AlertCondition",
    "AlertAnnotation",
    "AlertLabel",
    "AlertRule",
    "AlertRuleGroup",
    "AlertRuleBuilder",
    "create_default_alert_rules",
    "export_alert_rules_json",
    "export_alert_rules_yaml",
]
