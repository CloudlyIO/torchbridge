"""
Grafana Alert Rules for TorchBridge

Provides comprehensive alert rule definitions for Grafana alerting.

Alert Categories:
- Latency alerts (p50, p95, p99 thresholds)
- Error rate alerts
- GPU memory alerts
- System resource alerts
- SLO violation alerts
- Availability alerts

Example:
    ```python
    from torchbridge.monitoring import (
        AlertRuleGroup,
        create_default_alert_rules,
        export_alert_rules_json,
    )

    # Create alert rules
    rules = create_default_alert_rules(
        latency_p99_threshold=200,
        error_rate_threshold=1.0,
        gpu_memory_threshold=90,
    )

    # Export for Grafana provisioning
    export_alert_rules_json(rules, "alerts.json")
    ```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .structured_logging import get_logger

logger = get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"  # Paging/on-call level

class AlertState(Enum):
    """Possible alert states."""
    OK = "ok"
    PENDING = "pending"
    ALERTING = "alerting"
    NO_DATA = "nodata"
    ERROR = "error"

class ComparisonOperator(Enum):
    """Comparison operators for alert conditions."""
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "neq"

@dataclass
class AlertCondition:
    """Alert condition definition."""
    query: str
    threshold: float
    operator: ComparisonOperator = ComparisonOperator.GREATER_THAN
    for_duration: str = "5m"  # Time the condition must be true
    datasource_uid: str = "prometheus"

    def to_dict(self) -> dict[str, Any]:
        """Convert to Grafana alert condition format."""
        return {
            "evaluator": {
                "type": self.operator.value,
                "params": [self.threshold],
            },
            "query": {
                "params": [self.query, self.for_duration, "now"],
            },
            "reducer": {
                "type": "avg",
                "params": [],
            },
            "type": "query",
        }

@dataclass
class AlertAnnotation:
    """Alert annotation (displayed in alerts)."""
    summary: str
    description: str = ""
    runbook_url: str = ""
    dashboard_uid: str = ""
    panel_id: int = 0

    def to_dict(self) -> dict[str, str]:
        """Convert to annotation dict."""
        annotations = {
            "summary": self.summary,
            "description": self.description,
        }
        if self.runbook_url:
            annotations["runbook_url"] = self.runbook_url
        if self.dashboard_uid:
            annotations["__dashboardUid__"] = self.dashboard_uid
        if self.panel_id:
            annotations["__panelId__"] = str(self.panel_id)
        return annotations

@dataclass
class AlertLabel:
    """Alert label for routing and grouping."""
    name: str
    value: str

    def to_tuple(self) -> tuple[str, str]:
        """Convert to tuple."""
        return (self.name, self.value)

@dataclass
class AlertRule:
    """Grafana alert rule definition."""
    name: str
    condition: AlertCondition
    severity: AlertSeverity = AlertSeverity.WARNING
    annotations: AlertAnnotation | None = None
    labels: list[AlertLabel] = field(default_factory=list)
    evaluation_interval: str = "1m"
    no_data_state: AlertState = AlertState.NO_DATA
    exec_err_state: AlertState = AlertState.ERROR
    is_paused: bool = False

    def __post_init__(self):
        """Set default annotations if not provided."""
        if self.annotations is None:
            self.annotations = AlertAnnotation(
                summary=f"Alert: {self.name}",
                description=f"Alert triggered for {self.name}",
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to Grafana alert rule format."""
        labels = {label.name: label.value for label in self.labels}
        labels["severity"] = self.severity.value

        return {
            "title": self.name,
            "condition": "C",  # Condition reference
            "data": [
                {
                    "refId": "A",
                    "queryType": "",
                    "relativeTimeRange": {
                        "from": 600,  # 10 minutes
                        "to": 0,
                    },
                    "datasourceUid": self.condition.datasource_uid,
                    "model": {
                        "expr": self.condition.query,
                        "intervalMs": 1000,
                        "maxDataPoints": 43200,
                        "refId": "A",
                    },
                },
                {
                    "refId": "B",
                    "queryType": "",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0,
                    },
                    "datasourceUid": "__expr__",
                    "model": {
                        "conditions": [
                            {
                                "evaluator": {
                                    "params": [],
                                    "type": "gt",
                                },
                                "operator": {
                                    "type": "and",
                                },
                                "query": {
                                    "params": ["B"],
                                },
                                "reducer": {
                                    "params": [],
                                    "type": "last",
                                },
                                "type": "query",
                            }
                        ],
                        "datasource": {
                            "type": "__expr__",
                            "uid": "__expr__",
                        },
                        "expression": "A",
                        "intervalMs": 1000,
                        "maxDataPoints": 43200,
                        "reducer": "last",
                        "refId": "B",
                        "type": "reduce",
                    },
                },
                {
                    "refId": "C",
                    "queryType": "",
                    "relativeTimeRange": {
                        "from": 600,
                        "to": 0,
                    },
                    "datasourceUid": "__expr__",
                    "model": {
                        "conditions": [
                            {
                                "evaluator": {
                                    "params": [self.condition.threshold],
                                    "type": self.condition.operator.value,
                                },
                                "operator": {
                                    "type": "and",
                                },
                                "query": {
                                    "params": ["C"],
                                },
                                "reducer": {
                                    "params": [],
                                    "type": "last",
                                },
                                "type": "query",
                            }
                        ],
                        "datasource": {
                            "type": "__expr__",
                            "uid": "__expr__",
                        },
                        "expression": "B",
                        "intervalMs": 1000,
                        "maxDataPoints": 43200,
                        "refId": "C",
                        "type": "threshold",
                    },
                },
            ],
            "noDataState": self.no_data_state.value.upper(),
            "execErrState": self.exec_err_state.value.upper() if self.exec_err_state != AlertState.ERROR else "Error",
            "for": self.condition.for_duration,
            "annotations": self.annotations.to_dict() if self.annotations else {},
            "labels": labels,
            "isPaused": self.is_paused,
        }

@dataclass
class AlertRuleGroup:
    """Group of related alert rules."""
    name: str
    folder: str = "TorchBridge"
    evaluation_interval: str = "1m"
    rules: list[AlertRule] = field(default_factory=list)

    def add_rule(self, rule: AlertRule) -> None:
        """Add a rule to the group."""
        self.rules.append(rule)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Grafana alert rule group format."""
        return {
            "name": self.name,
            "folder": self.folder,
            "interval": self.evaluation_interval,
            "rules": [rule.to_dict() for rule in self.rules],
        }

class AlertRuleBuilder:
    """Builder for creating alert rules with sensible defaults."""

    def __init__(self, datasource_uid: str = "prometheus"):
        self.datasource_uid = datasource_uid

    def latency_p99_alert(
        self,
        threshold_ms: float = 200.0,
        severity: AlertSeverity = AlertSeverity.WARNING,
        for_duration: str = "5m",
    ) -> AlertRule:
        """Create a P99 latency alert."""
        return AlertRule(
            name="High P99 Latency",
            condition=AlertCondition(
                query="histogram_quantile(0.99, rate(torchbridge_inference_latency_milliseconds_bucket[5m]))",
                threshold=threshold_ms,
                operator=ComparisonOperator.GREATER_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="High inference latency detected",
                description=f"P99 latency is above {threshold_ms}ms threshold. "
                           "Consider scaling up or optimizing the model.",
                runbook_url="https://docs.torchbridge.io/runbooks/high-latency",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "latency"),
            ],
        )

    def latency_p95_alert(
        self,
        threshold_ms: float = 150.0,
        severity: AlertSeverity = AlertSeverity.WARNING,
        for_duration: str = "5m",
    ) -> AlertRule:
        """Create a P95 latency alert."""
        return AlertRule(
            name="Elevated P95 Latency",
            condition=AlertCondition(
                query="histogram_quantile(0.95, rate(torchbridge_inference_latency_milliseconds_bucket[5m]))",
                threshold=threshold_ms,
                operator=ComparisonOperator.GREATER_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="Elevated P95 latency",
                description=f"P95 latency exceeds {threshold_ms}ms. Monitor for degradation.",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "latency"),
            ],
        )

    def error_rate_alert(
        self,
        threshold_percent: float = 1.0,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
        for_duration: str = "5m",
    ) -> AlertRule:
        """Create an error rate alert."""
        return AlertRule(
            name="High Error Rate",
            condition=AlertCondition(
                query='sum(rate(torchbridge_inference_requests_total{status="error"}[5m])) / '
                      'sum(rate(torchbridge_inference_requests_total[5m])) * 100',
                threshold=threshold_percent,
                operator=ComparisonOperator.GREATER_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="High error rate detected",
                description=f"Error rate exceeds {threshold_percent}%. "
                           "Check logs for error details.",
                runbook_url="https://docs.torchbridge.io/runbooks/high-error-rate",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "error"),
            ],
        )

    def gpu_memory_alert(
        self,
        threshold_percent: float = 90.0,
        severity: AlertSeverity = AlertSeverity.WARNING,
        for_duration: str = "5m",
    ) -> AlertRule:
        """Create a GPU memory usage alert."""
        return AlertRule(
            name="High GPU Memory Usage",
            condition=AlertCondition(
                query="torchbridge_gpu_memory_used_bytes / torchbridge_gpu_memory_total_bytes * 100",
                threshold=threshold_percent,
                operator=ComparisonOperator.GREATER_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="GPU memory usage is high",
                description=f"GPU memory usage exceeds {threshold_percent}%. "
                           "Consider reducing batch size or enabling memory optimization.",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "resource"),
                AlertLabel("resource", "gpu_memory"),
            ],
        )

    def gpu_memory_critical_alert(
        self,
        threshold_percent: float = 95.0,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
        for_duration: str = "2m",
    ) -> AlertRule:
        """Create a critical GPU memory alert."""
        return AlertRule(
            name="Critical GPU Memory Usage",
            condition=AlertCondition(
                query="torchbridge_gpu_memory_used_bytes / torchbridge_gpu_memory_total_bytes * 100",
                threshold=threshold_percent,
                operator=ComparisonOperator.GREATER_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="CRITICAL: GPU memory nearly exhausted",
                description=f"GPU memory usage exceeds {threshold_percent}%. "
                           "OOM errors imminent. Immediate action required.",
                runbook_url="https://docs.torchbridge.io/runbooks/gpu-oom",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "resource"),
                AlertLabel("resource", "gpu_memory"),
            ],
        )

    def throughput_alert(
        self,
        min_throughput: float = 10.0,
        severity: AlertSeverity = AlertSeverity.WARNING,
        for_duration: str = "10m",
    ) -> AlertRule:
        """Create a low throughput alert."""
        return AlertRule(
            name="Low Throughput",
            condition=AlertCondition(
                query="torchbridge_inference_throughput_samples_per_second",
                threshold=min_throughput,
                operator=ComparisonOperator.LESS_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="Low inference throughput",
                description=f"Throughput below {min_throughput} samples/second. "
                           "Check for bottlenecks or service issues.",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "performance"),
            ],
        )

    def availability_alert(
        self,
        threshold_percent: float = 99.9,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
        for_duration: str = "5m",
    ) -> AlertRule:
        """Create an availability SLO alert."""
        return AlertRule(
            name="Availability SLO Violation",
            condition=AlertCondition(
                query='sum(rate(torchbridge_inference_requests_total{status="success"}[1h])) / '
                      'sum(rate(torchbridge_inference_requests_total[1h])) * 100',
                threshold=threshold_percent,
                operator=ComparisonOperator.LESS_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="Availability SLO violation",
                description=f"Availability dropped below {threshold_percent}% SLO. "
                           "Investigate and remediate immediately.",
                runbook_url="https://docs.torchbridge.io/runbooks/availability-slo",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "slo"),
                AlertLabel("slo", "availability"),
            ],
        )

    def error_budget_alert(
        self,
        burn_rate_threshold: float = 2.0,
        severity: AlertSeverity = AlertSeverity.WARNING,
        for_duration: str = "15m",
    ) -> AlertRule:
        """Create an error budget burn rate alert."""
        return AlertRule(
            name="High Error Budget Burn Rate",
            condition=AlertCondition(
                query='(1 - sum(rate(torchbridge_inference_requests_total{status="success"}[1h])) / '
                      'sum(rate(torchbridge_inference_requests_total[1h]))) / (1 - 0.999) * 100',
                threshold=burn_rate_threshold * 100,  # Convert to percentage
                operator=ComparisonOperator.GREATER_THAN,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="Error budget being consumed rapidly",
                description=f"Error budget burn rate is {burn_rate_threshold}x normal. "
                           "At this rate, budget will be exhausted soon.",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "slo"),
                AlertLabel("slo", "error_budget"),
            ],
        )

    def no_requests_alert(
        self,
        severity: AlertSeverity = AlertSeverity.WARNING,
        for_duration: str = "10m",
    ) -> AlertRule:
        """Create an alert for no incoming requests."""
        return AlertRule(
            name="No Incoming Requests",
            condition=AlertCondition(
                query="sum(rate(torchbridge_inference_requests_total[5m]))",
                threshold=0,
                operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                for_duration=for_duration,
                datasource_uid=self.datasource_uid,
            ),
            severity=severity,
            annotations=AlertAnnotation(
                summary="No inference requests received",
                description="No requests received in the last 10 minutes. "
                           "Check if service is reachable or if traffic routing is correct.",
            ),
            labels=[
                AlertLabel("service", "torchbridge"),
                AlertLabel("type", "traffic"),
            ],
        )

def create_default_alert_rules(
    latency_p99_threshold: float = 200.0,
    latency_p95_threshold: float = 150.0,
    error_rate_threshold: float = 1.0,
    gpu_memory_warning_threshold: float = 85.0,
    gpu_memory_critical_threshold: float = 95.0,
    availability_slo: float = 99.9,
    datasource_uid: str = "prometheus",
) -> list[AlertRuleGroup]:
    """
    Create default alert rule groups.

    Args:
        latency_p99_threshold: P99 latency threshold in ms
        latency_p95_threshold: P95 latency threshold in ms
        error_rate_threshold: Error rate threshold in percent
        gpu_memory_warning_threshold: GPU memory warning threshold in percent
        gpu_memory_critical_threshold: GPU memory critical threshold in percent
        availability_slo: Availability SLO target in percent
        datasource_uid: Prometheus datasource UID

    Returns:
        List of AlertRuleGroups
    """
    builder = AlertRuleBuilder(datasource_uid)

    # Latency alerts group
    latency_group = AlertRuleGroup(
        name="TorchBridge Latency Alerts",
        folder="TorchBridge",
    )
    latency_group.add_rule(builder.latency_p99_alert(latency_p99_threshold))
    latency_group.add_rule(builder.latency_p95_alert(latency_p95_threshold))

    # Error alerts group
    error_group = AlertRuleGroup(
        name="TorchBridge Error Alerts",
        folder="TorchBridge",
    )
    error_group.add_rule(builder.error_rate_alert(error_rate_threshold))

    # Resource alerts group
    resource_group = AlertRuleGroup(
        name="TorchBridge Resource Alerts",
        folder="TorchBridge",
    )
    resource_group.add_rule(builder.gpu_memory_alert(gpu_memory_warning_threshold))
    resource_group.add_rule(builder.gpu_memory_critical_alert(gpu_memory_critical_threshold))

    # SLO alerts group
    slo_group = AlertRuleGroup(
        name="TorchBridge SLO Alerts",
        folder="TorchBridge",
    )
    slo_group.add_rule(builder.availability_alert(availability_slo))
    slo_group.add_rule(builder.error_budget_alert())

    # Traffic alerts group
    traffic_group = AlertRuleGroup(
        name="TorchBridge Traffic Alerts",
        folder="TorchBridge",
    )
    traffic_group.add_rule(builder.throughput_alert())
    traffic_group.add_rule(builder.no_requests_alert())

    return [latency_group, error_group, resource_group, slo_group, traffic_group]

def export_alert_rules_json(
    rule_groups: list[AlertRuleGroup],
    output_path: str | Path,
) -> None:
    """
    Export alert rules to JSON file for Grafana provisioning.

    Args:
        rule_groups: List of alert rule groups
        output_path: Output file path
    """
    output = {
        "apiVersion": 1,
        "groups": [group.to_dict() for group in rule_groups],
    }

    path = Path(output_path)
    path.write_text(json.dumps(output, indent=2))

    logger.info(
        "Alert rules exported",
        output_path=str(path),
        group_count=len(rule_groups),
        rule_count=sum(len(g.rules) for g in rule_groups),
    )

def export_alert_rules_yaml(
    rule_groups: list[AlertRuleGroup],
    output_path: str | Path,
) -> None:
    """
    Export alert rules to YAML file for Grafana provisioning.

    Args:
        rule_groups: List of alert rule groups
        output_path: Output file path

    Raises:
        ImportError: If PyYAML is not installed
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML export. Install with: pip install pyyaml") from None

    output = {
        "apiVersion": 1,
        "groups": [group.to_dict() for group in rule_groups],
    }

    path = Path(output_path)
    path.write_text(yaml.dump(output, default_flow_style=False))

    logger.info(
        "Alert rules exported (YAML)",
        output_path=str(path),
        group_count=len(rule_groups),
    )

# Export all public APIs
__all__ = [
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
