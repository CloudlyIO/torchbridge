"""
Tests for Grafana Alert Rules.

Tests cover:
- Alert rule creation
- Alert conditions
- Alert annotations and labels
- Alert rule groups
- Alert rule builder
- JSON/YAML export
"""

import json
import tempfile
from pathlib import Path

import pytest

from kernel_pytorch.monitoring.grafana_alerts import (
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
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Test severity level values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.PAGE.value == "page"


class TestAlertState:
    """Tests for AlertState enum."""

    def test_state_values(self):
        """Test alert state values."""
        assert AlertState.OK.value == "ok"
        assert AlertState.PENDING.value == "pending"
        assert AlertState.ALERTING.value == "alerting"
        assert AlertState.NO_DATA.value == "nodata"
        assert AlertState.ERROR.value == "error"


class TestComparisonOperator:
    """Tests for ComparisonOperator enum."""

    def test_operator_values(self):
        """Test comparison operator values."""
        assert ComparisonOperator.GREATER_THAN.value == "gt"
        assert ComparisonOperator.GREATER_THAN_OR_EQUAL.value == "gte"
        assert ComparisonOperator.LESS_THAN.value == "lt"
        assert ComparisonOperator.LESS_THAN_OR_EQUAL.value == "lte"
        assert ComparisonOperator.EQUAL.value == "eq"
        assert ComparisonOperator.NOT_EQUAL.value == "neq"


class TestAlertCondition:
    """Tests for AlertCondition dataclass."""

    def test_basic_condition(self):
        """Test basic condition creation."""
        condition = AlertCondition(
            query="rate(requests_total[5m])",
            threshold=100.0,
        )
        assert condition.query == "rate(requests_total[5m])"
        assert condition.threshold == 100.0
        assert condition.operator == ComparisonOperator.GREATER_THAN

    def test_condition_with_all_params(self):
        """Test condition with all parameters."""
        condition = AlertCondition(
            query="cpu_usage",
            threshold=80.0,
            operator=ComparisonOperator.LESS_THAN,
            for_duration="10m",
            datasource_uid="my-prometheus",
        )
        assert condition.operator == ComparisonOperator.LESS_THAN
        assert condition.for_duration == "10m"
        assert condition.datasource_uid == "my-prometheus"

    def test_condition_to_dict(self):
        """Test condition serialization."""
        condition = AlertCondition(
            query="test_metric",
            threshold=50.0,
            operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        )
        cond_dict = condition.to_dict()

        assert "evaluator" in cond_dict
        assert cond_dict["evaluator"]["type"] == "gte"
        assert cond_dict["evaluator"]["params"] == [50.0]


class TestAlertAnnotation:
    """Tests for AlertAnnotation dataclass."""

    def test_basic_annotation(self):
        """Test basic annotation creation."""
        annotation = AlertAnnotation(
            summary="Test alert",
            description="This is a test alert",
        )
        assert annotation.summary == "Test alert"
        assert annotation.description == "This is a test alert"

    def test_annotation_with_runbook(self):
        """Test annotation with runbook URL."""
        annotation = AlertAnnotation(
            summary="Test",
            description="Description",
            runbook_url="https://example.com/runbook",
        )
        anno_dict = annotation.to_dict()
        assert anno_dict["runbook_url"] == "https://example.com/runbook"

    def test_annotation_to_dict(self):
        """Test annotation serialization."""
        annotation = AlertAnnotation(
            summary="Summary",
            description="Description",
            dashboard_uid="dash-123",
            panel_id=5,
        )
        anno_dict = annotation.to_dict()

        assert anno_dict["summary"] == "Summary"
        assert anno_dict["description"] == "Description"
        assert anno_dict["__dashboardUid__"] == "dash-123"
        assert anno_dict["__panelId__"] == "5"


class TestAlertLabel:
    """Tests for AlertLabel dataclass."""

    def test_label_creation(self):
        """Test label creation."""
        label = AlertLabel(name="team", value="ml-platform")
        assert label.name == "team"
        assert label.value == "ml-platform"

    def test_label_to_tuple(self):
        """Test label to tuple conversion."""
        label = AlertLabel(name="env", value="production")
        assert label.to_tuple() == ("env", "production")


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_basic_rule(self):
        """Test basic alert rule creation."""
        rule = AlertRule(
            name="Test Alert",
            condition=AlertCondition(
                query="test_metric",
                threshold=100.0,
            ),
        )
        assert rule.name == "Test Alert"
        assert rule.severity == AlertSeverity.WARNING  # default

    def test_rule_with_all_params(self):
        """Test rule with all parameters."""
        rule = AlertRule(
            name="Critical Alert",
            condition=AlertCondition(query="test", threshold=10.0),
            severity=AlertSeverity.CRITICAL,
            annotations=AlertAnnotation(
                summary="Critical",
                description="Very critical",
            ),
            labels=[
                AlertLabel("team", "sre"),
                AlertLabel("env", "prod"),
            ],
            evaluation_interval="30s",
            no_data_state=AlertState.OK,
            is_paused=True,
        )
        assert rule.severity == AlertSeverity.CRITICAL
        assert len(rule.labels) == 2
        assert rule.evaluation_interval == "30s"
        assert rule.no_data_state == AlertState.OK
        assert rule.is_paused is True

    def test_rule_auto_annotation(self):
        """Test automatic annotation creation."""
        rule = AlertRule(
            name="Auto Annotation Test",
            condition=AlertCondition(query="test", threshold=1.0),
        )
        assert rule.annotations is not None
        assert "Auto Annotation Test" in rule.annotations.summary

    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = AlertRule(
            name="Serialization Test",
            condition=AlertCondition(
                query="test_query",
                threshold=50.0,
            ),
            severity=AlertSeverity.WARNING,
            labels=[AlertLabel("service", "test")],
        )
        rule_dict = rule.to_dict()

        assert rule_dict["title"] == "Serialization Test"
        assert "data" in rule_dict
        assert len(rule_dict["data"]) == 3  # Query, reducer, threshold
        assert rule_dict["labels"]["severity"] == "warning"
        assert rule_dict["labels"]["service"] == "test"


class TestAlertRuleGroup:
    """Tests for AlertRuleGroup dataclass."""

    def test_basic_group(self):
        """Test basic group creation."""
        group = AlertRuleGroup(name="Test Group")
        assert group.name == "Test Group"
        assert group.folder == "KernelPyTorch"  # default
        assert len(group.rules) == 0

    def test_add_rule(self):
        """Test adding rules to group."""
        group = AlertRuleGroup(name="Test")
        rule = AlertRule(
            name="Rule 1",
            condition=AlertCondition(query="test", threshold=1.0),
        )
        group.add_rule(rule)

        assert len(group.rules) == 1
        assert group.rules[0].name == "Rule 1"

    def test_group_to_dict(self):
        """Test group serialization."""
        group = AlertRuleGroup(
            name="Test Group",
            folder="MyFolder",
            evaluation_interval="2m",
        )
        group.add_rule(AlertRule(
            name="Test Rule",
            condition=AlertCondition(query="test", threshold=1.0),
        ))

        group_dict = group.to_dict()
        assert group_dict["name"] == "Test Group"
        assert group_dict["folder"] == "MyFolder"
        assert group_dict["interval"] == "2m"
        assert len(group_dict["rules"]) == 1


class TestAlertRuleBuilder:
    """Tests for AlertRuleBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a builder instance."""
        return AlertRuleBuilder()

    def test_latency_p99_alert(self, builder):
        """Test P99 latency alert creation."""
        rule = builder.latency_p99_alert(threshold_ms=200.0)
        assert rule.name == "High P99 Latency"
        assert rule.condition.threshold == 200.0
        assert "0.99" in rule.condition.query

    def test_latency_p95_alert(self, builder):
        """Test P95 latency alert creation."""
        rule = builder.latency_p95_alert(threshold_ms=150.0)
        assert rule.name == "Elevated P95 Latency"
        assert rule.condition.threshold == 150.0
        assert "0.95" in rule.condition.query

    def test_error_rate_alert(self, builder):
        """Test error rate alert creation."""
        rule = builder.error_rate_alert(threshold_percent=1.0)
        assert rule.name == "High Error Rate"
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.condition.threshold == 1.0

    def test_gpu_memory_alert(self, builder):
        """Test GPU memory alert creation."""
        rule = builder.gpu_memory_alert(threshold_percent=85.0)
        assert rule.name == "High GPU Memory Usage"
        assert rule.condition.threshold == 85.0

    def test_gpu_memory_critical_alert(self, builder):
        """Test critical GPU memory alert creation."""
        rule = builder.gpu_memory_critical_alert(threshold_percent=95.0)
        assert rule.name == "Critical GPU Memory Usage"
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.condition.for_duration == "2m"

    def test_throughput_alert(self, builder):
        """Test throughput alert creation."""
        rule = builder.throughput_alert(min_throughput=10.0)
        assert rule.name == "Low Throughput"
        assert rule.condition.operator == ComparisonOperator.LESS_THAN

    def test_availability_alert(self, builder):
        """Test availability SLO alert creation."""
        rule = builder.availability_alert(threshold_percent=99.9)
        assert "Availability" in rule.name
        assert rule.condition.operator == ComparisonOperator.LESS_THAN

    def test_error_budget_alert(self, builder):
        """Test error budget alert creation."""
        rule = builder.error_budget_alert(burn_rate_threshold=2.0)
        assert "Error Budget" in rule.name

    def test_no_requests_alert(self, builder):
        """Test no requests alert creation."""
        rule = builder.no_requests_alert()
        assert rule.name == "No Incoming Requests"
        assert rule.condition.threshold == 0

    def test_custom_datasource(self):
        """Test builder with custom datasource."""
        builder = AlertRuleBuilder(datasource_uid="custom-prometheus")
        rule = builder.latency_p99_alert()
        assert rule.condition.datasource_uid == "custom-prometheus"


class TestCreateDefaultAlertRules:
    """Tests for create_default_alert_rules function."""

    def test_creates_all_groups(self):
        """Test that all default groups are created."""
        groups = create_default_alert_rules()
        assert len(groups) == 5

        group_names = [g.name for g in groups]
        assert "KernelPyTorch Latency Alerts" in group_names
        assert "KernelPyTorch Error Alerts" in group_names
        assert "KernelPyTorch Resource Alerts" in group_names
        assert "KernelPyTorch SLO Alerts" in group_names
        assert "KernelPyTorch Traffic Alerts" in group_names

    def test_custom_thresholds(self):
        """Test creating rules with custom thresholds."""
        groups = create_default_alert_rules(
            latency_p99_threshold=100.0,
            error_rate_threshold=0.5,
            gpu_memory_warning_threshold=80.0,
        )

        # Find latency group and check threshold
        latency_group = next(g for g in groups if "Latency" in g.name)
        p99_rule = next(r for r in latency_group.rules if "P99" in r.name)
        assert p99_rule.condition.threshold == 100.0

    def test_total_rule_count(self):
        """Test total number of rules created."""
        groups = create_default_alert_rules()
        total_rules = sum(len(g.rules) for g in groups)
        assert total_rules >= 8  # At least 8 default rules


class TestExportAlertRulesJson:
    """Tests for JSON export functionality."""

    def test_export_creates_file(self):
        """Test that JSON export creates a file."""
        groups = create_default_alert_rules()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_alert_rules_json(groups, output_path)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_valid_json(self):
        """Test that exported JSON is valid."""
        groups = create_default_alert_rules()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_alert_rules_json(groups, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert "apiVersion" in data
            assert "groups" in data
            assert len(data["groups"]) == len(groups)
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_structure(self):
        """Test exported JSON structure."""
        groups = [AlertRuleGroup(
            name="Test",
            rules=[AlertRule(
                name="Test Rule",
                condition=AlertCondition(query="test", threshold=1.0),
            )]
        )]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_alert_rules_json(groups, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert data["apiVersion"] == 1
            assert len(data["groups"]) == 1
            assert data["groups"][0]["name"] == "Test"
            assert len(data["groups"][0]["rules"]) == 1
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestAlertRuleValidation:
    """Tests for alert rule validation."""

    def test_rule_has_required_fields(self):
        """Test that rules have required fields for Grafana."""
        builder = AlertRuleBuilder()
        rule = builder.latency_p99_alert()
        rule_dict = rule.to_dict()

        # Required fields for Grafana
        assert "title" in rule_dict
        assert "condition" in rule_dict
        assert "data" in rule_dict
        assert "for" in rule_dict
        assert "labels" in rule_dict

    def test_rule_data_has_query(self):
        """Test that rule data includes the PromQL query."""
        builder = AlertRuleBuilder()
        rule = builder.error_rate_alert()
        rule_dict = rule.to_dict()

        # First data entry should be the query
        query_data = rule_dict["data"][0]
        assert "model" in query_data
        assert "expr" in query_data["model"]
        assert "error" in query_data["model"]["expr"].lower()


class TestIntegration:
    """Integration tests for alert rules."""

    def test_full_workflow(self):
        """Test complete alert rules workflow."""
        # Create custom builder
        builder = AlertRuleBuilder(datasource_uid="prod-prometheus")

        # Create custom groups
        latency_group = AlertRuleGroup(name="Custom Latency")
        latency_group.add_rule(builder.latency_p99_alert(threshold_ms=100))
        latency_group.add_rule(builder.latency_p95_alert(threshold_ms=75))

        resource_group = AlertRuleGroup(name="Custom Resources")
        resource_group.add_rule(builder.gpu_memory_alert(threshold_percent=80))

        groups = [latency_group, resource_group]

        # Export to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_alert_rules_json(groups, output_path)

            # Verify export
            with open(output_path) as f:
                data = json.load(f)

            assert len(data["groups"]) == 2
            assert data["groups"][0]["name"] == "Custom Latency"
            assert len(data["groups"][0]["rules"]) == 2
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_alert_labels_routing(self):
        """Test alert labels for routing."""
        builder = AlertRuleBuilder()

        # Different severity alerts
        warning_rule = builder.gpu_memory_alert()
        critical_rule = builder.gpu_memory_critical_alert()

        warning_dict = warning_rule.to_dict()
        critical_dict = critical_rule.to_dict()

        assert warning_dict["labels"]["severity"] == "warning"
        assert critical_dict["labels"]["severity"] == "critical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
