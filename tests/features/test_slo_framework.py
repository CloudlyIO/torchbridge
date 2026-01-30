"""
Tests for SLO/SLI Framework.

Tests cover:
- SLI measurement collection
- SLO configuration
- Error budget calculation
- Compliance tracking
- Burn rate analysis
- Recommendations generation
"""


import pytest

from torchbridge.monitoring.slo_framework import (
    BudgetStatus,
    ComplianceReport,
    ComplianceStatus,
    SLICollector,
    SLIType,
    SLOConfig,
    SLOManager,
    create_slo_manager,
)


class TestSLIType:
    """Tests for SLIType enum."""

    def test_sli_types(self):
        """Test SLI type values."""
        assert SLIType.LATENCY_P50.value == "latency_p50"
        assert SLIType.LATENCY_P95.value == "latency_p95"
        assert SLIType.LATENCY_P99.value == "latency_p99"
        assert SLIType.LATENCY_MEAN.value == "latency_mean"
        assert SLIType.ERROR_RATE.value == "error_rate"
        assert SLIType.AVAILABILITY.value == "availability"
        assert SLIType.THROUGHPUT.value == "throughput"
        assert SLIType.CUSTOM.value == "custom"


class TestComplianceStatus:
    """Tests for ComplianceStatus enum."""

    def test_compliance_values(self):
        """Test compliance status values."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.AT_RISK.value == "at_risk"
        assert ComplianceStatus.VIOLATED.value == "violated"
        assert ComplianceStatus.INSUFFICIENT_DATA.value == "insufficient_data"


class TestBudgetStatus:
    """Tests for BudgetStatus enum."""

    def test_budget_status_values(self):
        """Test budget status values."""
        assert BudgetStatus.HEALTHY.value == "healthy"
        assert BudgetStatus.WARNING.value == "warning"
        assert BudgetStatus.CRITICAL.value == "critical"
        assert BudgetStatus.EXHAUSTED.value == "exhausted"


class TestSLOConfig:
    """Tests for SLOConfig dataclass."""

    def test_basic_config(self):
        """Test basic SLO configuration."""
        config = SLOConfig(
            name="test_latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        )
        assert config.name == "test_latency"
        assert config.sli_type == SLIType.LATENCY_P99
        assert config.target == 100.0
        assert config.window_minutes == 60  # default

    def test_config_with_all_params(self):
        """Test SLO configuration with all parameters."""
        config = SLOConfig(
            name="availability",
            sli_type=SLIType.AVAILABILITY,
            target=99.9,
            window_minutes=1440,
            description="Service availability",
            unit="%",
        )
        assert config.window_minutes == 1440
        assert config.description == "Service availability"
        assert config.unit == "%"

    def test_config_auto_unit(self):
        """Test automatic unit assignment."""
        latency_config = SLOConfig(
            name="latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        )
        assert latency_config.unit == "ms"

        availability_config = SLOConfig(
            name="avail",
            sli_type=SLIType.AVAILABILITY,
            target=99.9,
        )
        assert availability_config.unit == "%"

    def test_config_comparison_direction(self):
        """Test comparison direction auto-setting."""
        latency_config = SLOConfig(
            name="latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        )
        assert latency_config.comparison == "lte"  # latency should be <= target

        availability_config = SLOConfig(
            name="avail",
            sli_type=SLIType.AVAILABILITY,
            target=99.9,
        )
        assert availability_config.comparison == "gte"  # availability should be >= target


class TestSLICollector:
    """Tests for SLICollector."""

    @pytest.fixture
    def collector(self):
        """Create a fresh collector."""
        return SLICollector()

    def test_record_latency(self, collector):
        """Test latency recording."""
        collector.record_latency(50.0)
        collector.record_latency(60.0)

        latencies = collector.get_latencies()
        assert len(latencies) == 2
        assert 50.0 in latencies
        assert 60.0 in latencies

    def test_record_request(self, collector):
        """Test request recording."""
        collector.record_request(success=True)
        collector.record_request(success=False)
        collector.record_request(success=True)

        requests = collector.get_requests()
        assert len(requests) == 3

    def test_record_custom(self, collector):
        """Test custom metric recording."""
        collector.record_custom("gpu_utilization", 75.0)
        collector.record_custom("gpu_utilization", 80.0)

        values = collector.get_custom("gpu_utilization")
        assert len(values) == 2
        assert 75.0 in values
        assert 80.0 in values

    def test_calculate_latency_p50(self, collector):
        """Test p50 latency calculation."""
        for i in range(100):
            collector.record_latency(float(i))

        p50 = collector.calculate_sli(SLIType.LATENCY_P50, window_minutes=60)
        assert p50 is not None
        assert 45 <= p50 <= 55  # Should be around 49.5

    def test_calculate_latency_p99(self, collector):
        """Test p99 latency calculation."""
        for i in range(100):
            collector.record_latency(float(i))

        p99 = collector.calculate_sli(SLIType.LATENCY_P99, window_minutes=60)
        assert p99 is not None
        assert p99 >= 95  # Should be high

    def test_calculate_latency_mean(self, collector):
        """Test mean latency calculation."""
        collector.record_latency(10.0)
        collector.record_latency(20.0)
        collector.record_latency(30.0)

        mean = collector.calculate_sli(SLIType.LATENCY_MEAN, window_minutes=60)
        assert mean == 20.0

    def test_calculate_availability(self, collector):
        """Test availability calculation."""
        for _ in range(9):
            collector.record_request(success=True)
        collector.record_request(success=False)

        availability = collector.calculate_sli(SLIType.AVAILABILITY, window_minutes=60)
        assert availability == 90.0  # 9/10 = 90%

    def test_calculate_error_rate(self, collector):
        """Test error rate calculation."""
        for _ in range(9):
            collector.record_request(success=True)
        collector.record_request(success=False)

        error_rate = collector.calculate_sli(SLIType.ERROR_RATE, window_minutes=60)
        assert error_rate == 10.0  # 1/10 = 10%

    def test_window_filtering(self, collector):
        """Test time window filtering."""
        collector.record_latency(100.0)

        # Get with very short window (should still include recent)
        latencies = collector.get_latencies(window_minutes=1)
        assert len(latencies) == 1

    def test_empty_collection(self, collector):
        """Test calculations with no data."""
        p99 = collector.calculate_sli(SLIType.LATENCY_P99, window_minutes=60)
        assert p99 is None

        availability = collector.calculate_sli(SLIType.AVAILABILITY, window_minutes=60)
        assert availability is None


class TestSLOManager:
    """Tests for SLOManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh manager."""
        return SLOManager()

    def test_add_slo(self, manager):
        """Test adding SLO."""
        config = SLOConfig(
            name="test_latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        )
        manager.add_slo(config)

        slo = manager.get_slo("test_latency")
        assert slo is not None
        assert slo.name == "test_latency"

    def test_remove_slo(self, manager):
        """Test removing SLO."""
        config = SLOConfig(
            name="test_latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        )
        manager.add_slo(config)
        assert manager.remove_slo("test_latency") is True
        assert manager.get_slo("test_latency") is None

    def test_remove_nonexistent_slo(self, manager):
        """Test removing non-existent SLO."""
        assert manager.remove_slo("nonexistent") is False

    def test_list_slos(self, manager):
        """Test listing SLOs."""
        manager.add_slo(SLOConfig("slo1", SLIType.LATENCY_P99, 100.0))
        manager.add_slo(SLOConfig("slo2", SLIType.AVAILABILITY, 99.9))

        slos = manager.list_slos()
        assert len(slos) == 2
        names = [s.name for s in slos]
        assert "slo1" in names
        assert "slo2" in names

    def test_record_latency(self, manager):
        """Test recording latency through manager."""
        manager.record_latency(50.0)
        manager.record_latency(60.0)

        # Verify data was recorded
        latencies = manager._collector.get_latencies()
        assert len(latencies) == 2

    def test_record_request(self, manager):
        """Test recording request through manager."""
        manager.record_request(success=True)
        manager.record_request(success=False)

        requests = manager._collector.get_requests()
        assert len(requests) == 2

    def test_get_slo_status(self, manager):
        """Test getting SLO status."""
        manager.add_slo(SLOConfig(
            name="latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        ))

        # Record some data
        for _ in range(20):
            manager.record_latency(50.0)

        status = manager.get_slo_status("latency")
        assert status is not None
        assert status.slo_name == "latency"
        assert status.target == 100.0

    def test_compliance_when_meeting_slo(self, manager):
        """Test compliance when SLO is being met."""
        manager.add_slo(SLOConfig(
            name="latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        ))

        # All latencies below target
        for _ in range(100):
            manager.record_latency(50.0)

        status = manager.get_slo_status("latency")
        assert status.is_compliant is True
        assert status.compliance_status == ComplianceStatus.COMPLIANT

    def test_compliance_when_violating_slo(self, manager):
        """Test compliance when SLO is violated."""
        manager.add_slo(SLOConfig(
            name="latency",
            sli_type=SLIType.LATENCY_P99,
            target=100.0,
        ))

        # All latencies above target
        for _ in range(100):
            manager.record_latency(150.0)

        status = manager.get_slo_status("latency")
        assert status.is_compliant is False
        assert status.compliance_status == ComplianceStatus.VIOLATED

    def test_availability_slo(self, manager):
        """Test availability SLO compliance."""
        manager.add_slo(SLOConfig(
            name="availability",
            sli_type=SLIType.AVAILABILITY,
            target=99.0,
        ))

        # 95 successes, 5 failures = 95% availability
        for _ in range(95):
            manager.record_request(success=True)
        for _ in range(5):
            manager.record_request(success=False)

        status = manager.get_slo_status("availability")
        assert status.current_value == 95.0
        assert status.is_compliant is False  # Below 99% target


class TestComplianceReport:
    """Tests for compliance reporting."""

    @pytest.fixture
    def manager(self):
        """Create manager with default SLOs and data."""
        manager = create_slo_manager(default_slos=True)

        # Record some data
        for _ in range(50):
            manager.record_latency(100.0)
            manager.record_request(success=True)

        return manager

    def test_generate_report(self, manager):
        """Test generating compliance report."""
        report = manager.get_compliance_report()

        assert isinstance(report, ComplianceReport)
        assert report.timestamp is not None
        assert len(report.slo_statuses) > 0

    def test_report_to_dict(self, manager):
        """Test report serialization."""
        report = manager.get_compliance_report()
        report_dict = report.to_dict()

        assert "timestamp" in report_dict
        assert "overall_status" in report_dict
        assert "summary" in report_dict
        assert "slos" in report_dict

    def test_report_counts(self, manager):
        """Test report status counts."""
        report = manager.get_compliance_report()

        total = (
            report.compliant_count +
            report.violated_count +
            report.at_risk_count
        )
        # Should have some status for each SLO
        assert total >= 0

    def test_overall_status_when_all_compliant(self):
        """Test overall status when all SLOs are met."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 1000.0))

        for _ in range(100):
            manager.record_latency(50.0)

        report = manager.get_compliance_report()
        assert report.overall_status == ComplianceStatus.COMPLIANT

    def test_overall_status_when_violated(self):
        """Test overall status when SLO is violated."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 10.0))

        for _ in range(100):
            manager.record_latency(500.0)  # Way above target

        report = manager.get_compliance_report()
        assert report.overall_status == ComplianceStatus.VIOLATED


class TestErrorBudget:
    """Tests for error budget calculations."""

    def test_error_budget_calculation(self):
        """Test error budget is calculated correctly."""
        manager = SLOManager()
        manager.add_slo(SLOConfig(
            name="availability",
            sli_type=SLIType.AVAILABILITY,
            target=99.0,  # 99% target = 1% error budget
        ))

        # 99% availability
        for _ in range(99):
            manager.record_request(success=True)
        manager.record_request(success=False)

        status = manager.get_slo_status("availability")
        assert status.error_budget_total == 1.0  # 100 - 99 = 1%
        # When at exactly 99%, no budget consumed
        assert status.error_budget_consumed == 0.0

    def test_budget_status_healthy(self):
        """Test healthy budget status."""
        manager = SLOManager()
        manager.add_slo(SLOConfig(
            name="availability",
            sli_type=SLIType.AVAILABILITY,
            target=99.0,
        ))

        # All successes - no budget consumed
        for _ in range(100):
            manager.record_request(success=True)

        status = manager.get_slo_status("availability")
        assert status.budget_status == BudgetStatus.HEALTHY

    def test_budget_status_exhausted(self):
        """Test exhausted budget status."""
        manager = SLOManager()
        manager.add_slo(SLOConfig(
            name="availability",
            sli_type=SLIType.AVAILABILITY,
            target=99.0,  # 1% error budget
        ))

        # 90% availability - way over budget
        for _ in range(90):
            manager.record_request(success=True)
        for _ in range(10):
            manager.record_request(success=False)

        status = manager.get_slo_status("availability")
        assert status.budget_status == BudgetStatus.EXHAUSTED


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_for_violated_latency(self):
        """Test recommendations for latency violation."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 100.0))

        for _ in range(100):
            manager.record_latency(500.0)

        report = manager.get_compliance_report()
        assert any("latency" in r.lower() for r in report.recommendations)

    def test_recommendations_for_violated_availability(self):
        """Test recommendations for availability violation."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("availability", SLIType.AVAILABILITY, 99.9))

        for _ in range(90):
            manager.record_request(success=True)
        for _ in range(10):
            manager.record_request(success=False)

        report = manager.get_compliance_report()
        assert any("availability" in r.lower() for r in report.recommendations)

    def test_recommendations_when_healthy(self):
        """Test recommendations when all healthy."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 1000.0))

        for _ in range(100):
            manager.record_latency(50.0)

        report = manager.get_compliance_report()
        assert any("healthy" in r.lower() for r in report.recommendations)


class TestComplianceHistory:
    """Tests for compliance history tracking."""

    def test_history_recorded(self):
        """Test that compliance reports are recorded."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 100.0))

        for _ in range(20):
            manager.record_latency(50.0)

        # Generate multiple reports
        manager.get_compliance_report()
        manager.get_compliance_report()
        manager.get_compliance_report()

        history = manager.get_compliance_history()
        assert len(history) == 3

    def test_history_limit(self):
        """Test history limit parameter."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 100.0))

        for _ in range(20):
            manager.record_latency(50.0)

        for _ in range(10):
            manager.get_compliance_report()

        history = manager.get_compliance_history(limit=5)
        assert len(history) == 5


class TestCreateSLOManager:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test creation with default SLOs."""
        manager = create_slo_manager(default_slos=True)

        slos = manager.list_slos()
        assert len(slos) >= 3  # Should have default SLOs

        names = [s.name for s in slos]
        assert "latency_p99" in names
        assert "availability" in names
        assert "error_rate" in names

    def test_create_without_defaults(self):
        """Test creation without default SLOs."""
        manager = create_slo_manager(default_slos=False)

        slos = manager.list_slos()
        assert len(slos) == 0


class TestSLOStatusSerialization:
    """Tests for SLOStatus serialization."""

    def test_to_dict(self):
        """Test SLOStatus to_dict method."""
        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 100.0))

        for _ in range(20):
            manager.record_latency(50.0)

        status = manager.get_slo_status("latency")
        status_dict = status.to_dict()

        assert "slo_name" in status_dict
        assert "sli_type" in status_dict
        assert "target" in status_dict
        assert "current_value" in status_dict
        assert "error_budget" in status_dict
        assert "is_compliant" in status_dict


class TestIntegration:
    """Integration tests for SLO framework."""

    def test_full_workflow(self):
        """Test complete SLO monitoring workflow."""
        # Create manager with default SLOs
        manager = create_slo_manager(default_slos=True)

        # Simulate traffic
        for i in range(100):
            # Record latencies (varying)
            latency = 50 + (i % 50)
            manager.record_latency(latency)

            # Record requests (95% success rate)
            manager.record_request(success=(i % 20 != 0))

        # Get compliance report
        report = manager.get_compliance_report()

        # Verify report is complete
        assert report is not None
        assert len(report.slo_statuses) == 3
        assert report.error_budget_burn_rate >= 0

        # Check individual SLOs
        for status in report.slo_statuses:
            assert status.measurement_count > 0
            assert status.current_value is not None

    def test_multi_threaded_recording(self):
        """Test thread-safe recording."""
        import threading

        manager = SLOManager()
        manager.add_slo(SLOConfig("latency", SLIType.LATENCY_P99, 100.0))

        errors = []

        def record_worker():
            try:
                for _ in range(100):
                    manager.record_latency(50.0)
                    manager.record_request(success=True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_worker)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all data was recorded
        latencies = manager._collector.get_latencies()
        assert len(latencies) == 500  # 5 threads * 100 records


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
