"""
Grafana Dashboard Generator for TorchBridge

This module provides pre-configured Grafana dashboard definitions for
monitoring TorchBridge inference servers.

Features:
- Inference performance dashboard (latency, throughput)
- System resources dashboard (GPU, CPU, memory)
- Full operational dashboard
- JSON export for Grafana import

Example:
    ```python
    from torchbridge.monitoring import (
        create_full_dashboard,
        export_dashboard_json
    )

    # Create dashboard
    dashboard = create_full_dashboard(model_name="transformer")

    # Export to JSON file
    export_dashboard_json(dashboard, "dashboard.json")
    ```

"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DashboardPanel:
    """Grafana dashboard panel definition."""

    title: str
    panel_type: str = "graph"  # graph, gauge, stat, table, heatmap
    datasource: str = "Prometheus"
    targets: list[dict[str, Any]] = field(default_factory=list)
    grid_pos: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 12, "h": 8})
    options: dict[str, Any] = field(default_factory=dict)
    field_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Grafana panel JSON format."""
        panel = {
            "title": self.title,
            "type": self.panel_type,
            "datasource": {"type": "prometheus", "uid": self.datasource},
            "targets": self.targets,
            "gridPos": self.grid_pos,
        }

        if self.options:
            panel["options"] = self.options

        if self.field_config:
            panel["fieldConfig"] = self.field_config

        return panel

@dataclass
class GrafanaDashboard:
    """Grafana dashboard definition."""

    title: str
    uid: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=lambda: ["torchbridge", "inference"])
    panels: list[DashboardPanel] = field(default_factory=list)
    refresh: str = "5s"
    time_from: str = "now-1h"
    time_to: str = "now"
    variables: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.uid:
            self.uid = self.title.lower().replace(" ", "-").replace("_", "-")

    def add_panel(self, panel: DashboardPanel) -> None:
        """Add a panel to the dashboard."""
        self.panels.append(panel)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Grafana dashboard JSON format."""
        return {
            "dashboard": {
                "id": None,
                "uid": self.uid,
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "timezone": "browser",
                "schemaVersion": 38,
                "version": 1,
                "refresh": self.refresh,
                "time": {
                    "from": self.time_from,
                    "to": self.time_to,
                },
                "templating": {
                    "list": self.variables,
                },
                "panels": [
                    {**p.to_dict(), "id": i + 1}
                    for i, p in enumerate(self.panels)
                ],
                "annotations": {
                    "list": [
                        {
                            "builtIn": 1,
                            "datasource": "-- Grafana --",
                            "enable": True,
                            "hide": True,
                            "iconColor": "rgba(0, 211, 255, 1)",
                            "name": "Annotations & Alerts",
                            "type": "dashboard",
                        }
                    ]
                },
            },
            "overwrite": True,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

def _create_prometheus_target(
    expr: str,
    legend: str = "",
    ref_id: str = "A",
) -> dict[str, Any]:
    """Create a Prometheus query target."""
    return {
        "expr": expr,
        "legendFormat": legend,
        "refId": ref_id,
        "datasource": {"type": "prometheus", "uid": "Prometheus"},
    }

def create_inference_dashboard(
    model_name: str = "model",
    namespace: str = "torchbridge",
) -> GrafanaDashboard:
    """
    Create an inference performance dashboard.

    Args:
        model_name: Name of the model
        namespace: Prometheus metrics namespace

    Returns:
        GrafanaDashboard for inference metrics
    """
    dashboard = GrafanaDashboard(
        title=f"TorchBridge Inference - {model_name}",
        description="Inference performance monitoring for TorchBridge models",
        tags=["torchbridge", "inference", model_name],
    )

    # Request Rate Panel
    dashboard.add_panel(DashboardPanel(
        title="Request Rate",
        panel_type="timeseries",
        targets=[
            _create_prometheus_target(
                f'rate({namespace}_inference_requests_total{{model="{model_name}"}}[5m])',
                "Requests/sec",
            )
        ],
        grid_pos={"x": 0, "y": 0, "w": 8, "h": 6},
        options={"legend": {"displayMode": "list", "placement": "bottom"}},
    ))

    # Throughput Panel
    dashboard.add_panel(DashboardPanel(
        title="Throughput",
        panel_type="stat",
        targets=[
            _create_prometheus_target(
                f'{namespace}_inference_throughput_samples_per_second{{model="{model_name}"}}',
                "samples/sec",
            )
        ],
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 6},
        options={
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "colorMode": "value",
            "graphMode": "area",
        },
    ))

    # Error Rate Panel
    dashboard.add_panel(DashboardPanel(
        title="Error Rate",
        panel_type="stat",
        targets=[
            _create_prometheus_target(
                f'rate({namespace}_inference_requests_total{{model="{model_name}",status="error"}}[5m]) / '
                f'rate({namespace}_inference_requests_total{{model="{model_name}"}}[5m]) * 100',
                "Error %",
            )
        ],
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 6},
        options={
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "colorMode": "value",
        },
        field_config={
            "defaults": {
                "unit": "percent",
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 1},
                        {"color": "red", "value": 5},
                    ],
                },
            },
        },
    ))

    # Total Requests Panel
    dashboard.add_panel(DashboardPanel(
        title="Total Requests",
        panel_type="stat",
        targets=[
            _create_prometheus_target(
                f'{namespace}_inference_requests_total{{model="{model_name}"}}',
                "Total",
            )
        ],
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 6},
        options={
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "colorMode": "value",
            "graphMode": "none",
        },
    ))

    # Latency Distribution Panel
    dashboard.add_panel(DashboardPanel(
        title="Latency Distribution",
        panel_type="heatmap",
        targets=[
            _create_prometheus_target(
                f'rate({namespace}_inference_latency_milliseconds_bucket{{model="{model_name}"}}[5m])',
                "{{le}}",
            )
        ],
        grid_pos={"x": 0, "y": 6, "w": 12, "h": 8},
        options={
            "calculate": False,
            "yAxis": {"unit": "ms"},
            "color": {"scheme": "Spectral"},
        },
    ))

    # Latency Percentiles Panel
    dashboard.add_panel(DashboardPanel(
        title="Latency Percentiles",
        panel_type="timeseries",
        targets=[
            _create_prometheus_target(
                f'histogram_quantile(0.50, rate({namespace}_inference_latency_milliseconds_bucket{{model="{model_name}"}}[5m]))',
                "p50",
                "A",
            ),
            _create_prometheus_target(
                f'histogram_quantile(0.95, rate({namespace}_inference_latency_milliseconds_bucket{{model="{model_name}"}}[5m]))',
                "p95",
                "B",
            ),
            _create_prometheus_target(
                f'histogram_quantile(0.99, rate({namespace}_inference_latency_milliseconds_bucket{{model="{model_name}"}}[5m]))',
                "p99",
                "C",
            ),
        ],
        grid_pos={"x": 12, "y": 6, "w": 12, "h": 8},
        options={"legend": {"displayMode": "list", "placement": "bottom"}},
        field_config={"defaults": {"unit": "ms"}},
    ))

    # Batch Size Distribution Panel
    dashboard.add_panel(DashboardPanel(
        title="Batch Size Distribution",
        panel_type="histogram",
        targets=[
            _create_prometheus_target(
                f'{namespace}_inference_batch_size_bucket{{model="{model_name}"}}',
                "{{le}}",
            )
        ],
        grid_pos={"x": 0, "y": 14, "w": 12, "h": 6},
    ))

    return dashboard

def create_system_dashboard(
    namespace: str = "torchbridge",
) -> GrafanaDashboard:
    """
    Create a system resources dashboard.

    Args:
        namespace: Prometheus metrics namespace

    Returns:
        GrafanaDashboard for system metrics
    """
    dashboard = GrafanaDashboard(
        title="TorchBridge System Resources",
        description="System resource monitoring for TorchBridge deployments",
        tags=["torchbridge", "system", "gpu"],
    )

    # GPU Memory Usage Panel
    dashboard.add_panel(DashboardPanel(
        title="GPU Memory Usage",
        panel_type="timeseries",
        targets=[
            _create_prometheus_target(
                f'{namespace}_gpu_memory_used_bytes / 1024 / 1024 / 1024',
                "Used (GB)",
                "A",
            ),
            _create_prometheus_target(
                f'{namespace}_gpu_memory_total_bytes / 1024 / 1024 / 1024',
                "Total (GB)",
                "B",
            ),
        ],
        grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
        options={"legend": {"displayMode": "list", "placement": "bottom"}},
        field_config={"defaults": {"unit": "decgbytes"}},
    ))

    # GPU Memory Percentage Panel
    dashboard.add_panel(DashboardPanel(
        title="GPU Memory %",
        panel_type="gauge",
        targets=[
            _create_prometheus_target(
                f'{namespace}_gpu_memory_used_bytes / {namespace}_gpu_memory_total_bytes * 100',
                "Memory %",
            )
        ],
        grid_pos={"x": 12, "y": 0, "w": 6, "h": 8},
        options={
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "showThresholdLabels": False,
            "showThresholdMarkers": True,
        },
        field_config={
            "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 100,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 70},
                        {"color": "red", "value": 90},
                    ],
                },
            },
        },
    ))

    # Model Info Panel
    dashboard.add_panel(DashboardPanel(
        title="Active Models",
        panel_type="table",
        targets=[
            _create_prometheus_target(
                f'{namespace}_model_info',
                "{{model}} - {{version}}",
            )
        ],
        grid_pos={"x": 18, "y": 0, "w": 6, "h": 8},
    ))

    return dashboard

def create_full_dashboard(
    model_name: str = "model",
    namespace: str = "torchbridge",
) -> GrafanaDashboard:
    """
    Create a comprehensive dashboard with all metrics.

    Args:
        model_name: Name of the model
        namespace: Prometheus metrics namespace

    Returns:
        Full GrafanaDashboard with inference and system metrics
    """
    inference_dash = create_inference_dashboard(model_name, namespace)
    system_dash = create_system_dashboard(namespace)

    # Merge dashboards
    full_dashboard = GrafanaDashboard(
        title=f"TorchBridge Full Dashboard - {model_name}",
        description="Complete monitoring dashboard for TorchBridge inference",
        tags=["torchbridge", "inference", "system", model_name],
    )

    # Add inference panels
    for panel in inference_dash.panels:
        full_dashboard.add_panel(panel)

    # Add system panels with adjusted positions
    y_offset = 20  # Offset for system panels
    for panel in system_dash.panels:
        adjusted_panel = DashboardPanel(
            title=panel.title,
            panel_type=panel.panel_type,
            datasource=panel.datasource,
            targets=panel.targets,
            grid_pos={**panel.grid_pos, "y": panel.grid_pos["y"] + y_offset},
            options=panel.options,
            field_config=panel.field_config,
        )
        full_dashboard.add_panel(adjusted_panel)

    return full_dashboard

def export_dashboard_json(
    dashboard: GrafanaDashboard,
    output_path: str,
    indent: int = 2,
) -> str:
    """
    Export dashboard to JSON file.

    Args:
        dashboard: Dashboard to export
        output_path: Path for output file
        indent: JSON indentation

    Returns:
        Path to the exported file
    """
    with open(output_path, "w") as f:
        f.write(dashboard.to_json(indent=indent))

    return output_path
