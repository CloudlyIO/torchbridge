#!/usr/bin/env python3
"""
Tests for RegressionDetector functionality
"""

from datetime import datetime

import pytest
from benchmarks.framework.benchmark_runner import PerformanceMetrics
from benchmarks.regression.baseline_manager import BaselineMetrics
from benchmarks.regression.regression_detector import (
    RegressionDetector,
    RegressionResult,
    RegressionSeverity,
)


class TestRegressionResult:
    """Test RegressionResult dataclass functionality"""

    def test_regression_result_creation(self):
        """Test creating RegressionResult"""
        current = PerformanceMetrics(
            latency_ms=15.0,
            throughput_samples_per_sec=60.0,
            peak_memory_mb=200.0,
            memory_efficiency=0.8,
            accuracy_loss=0.02,
            statistical_significance=True,
            confidence_interval_95=(14.5, 15.5)
        )

        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=1.0,
            mean_throughput=100.0,
            std_throughput=5.0,
            mean_memory_mb=150.0,
            std_memory_mb=10.0,
            sample_count=20,
            confidence_interval_95=(9.0, 11.0),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        result = RegressionResult(
            model_name="test_model",
            current_performance=current,
            baseline_performance=baseline,
            performance_delta_percent=50.0,
            throughput_delta_percent=40.0,
            memory_delta_percent=33.3,
            statistical_significance=True,
            severity=RegressionSeverity.CRITICAL,
            confidence_level=0.95,
            recommendation="Critical regression detected",
            timestamp=datetime.now()
        )

        assert result.model_name == "test_model"
        assert result.performance_delta_percent == 50.0
        assert result.severity == RegressionSeverity.CRITICAL

    def test_is_regression(self):
        """Test regression detection logic"""
        # Create regression result with no severity
        no_regression = RegressionResult(
            model_name="test",
            current_performance=None,
            baseline_performance=None,
            performance_delta_percent=1.0,
            throughput_delta_percent=0.5,
            memory_delta_percent=0.8,
            statistical_significance=False,
            severity=RegressionSeverity.NONE,
            confidence_level=0.95,
            recommendation="No regression",
            timestamp=datetime.now()
        )
        assert not no_regression.is_regression()

        # Create regression result with minor severity
        minor_regression = RegressionResult(
            model_name="test",
            current_performance=None,
            baseline_performance=None,
            performance_delta_percent=3.0,
            throughput_delta_percent=2.5,
            memory_delta_percent=2.8,
            statistical_significance=True,
            severity=RegressionSeverity.MINOR,
            confidence_level=0.95,
            recommendation="Minor regression",
            timestamp=datetime.now()
        )
        assert minor_regression.is_regression()

    def test_is_blocking(self):
        """Test blocking regression detection"""
        # Minor regression should not be blocking
        minor_regression = RegressionResult(
            model_name="test",
            current_performance=None,
            baseline_performance=None,
            performance_delta_percent=3.0,
            throughput_delta_percent=2.5,
            memory_delta_percent=2.8,
            statistical_significance=True,
            severity=RegressionSeverity.MINOR,
            confidence_level=0.95,
            recommendation="Minor regression",
            timestamp=datetime.now()
        )
        assert not minor_regression.is_blocking()

        # Major regression should be blocking
        major_regression = RegressionResult(
            model_name="test",
            current_performance=None,
            baseline_performance=None,
            performance_delta_percent=8.0,
            throughput_delta_percent=7.5,
            memory_delta_percent=7.8,
            statistical_significance=True,
            severity=RegressionSeverity.MAJOR,
            confidence_level=0.95,
            recommendation="Major regression",
            timestamp=datetime.now()
        )
        assert major_regression.is_blocking()

        # Critical regression should be blocking
        critical_regression = RegressionResult(
            model_name="test",
            current_performance=None,
            baseline_performance=None,
            performance_delta_percent=15.0,
            throughput_delta_percent=12.5,
            memory_delta_percent=14.8,
            statistical_significance=True,
            severity=RegressionSeverity.CRITICAL,
            confidence_level=0.95,
            recommendation="Critical regression",
            timestamp=datetime.now()
        )
        assert critical_regression.is_blocking()

    def test_to_dict(self):
        """Test regression result serialization"""
        current = PerformanceMetrics(
            latency_ms=15.0,
            throughput_samples_per_sec=60.0,
            peak_memory_mb=200.0,
            memory_efficiency=0.8,
            accuracy_loss=0.02,
            statistical_significance=True,
            confidence_interval_95=(14.5, 15.5)
        )

        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=1.0,
            mean_throughput=100.0,
            std_throughput=5.0,
            mean_memory_mb=150.0,
            std_memory_mb=10.0,
            sample_count=20,
            confidence_interval_95=(9.0, 11.0),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        result = RegressionResult(
            model_name="test_model",
            current_performance=current,
            baseline_performance=baseline,
            performance_delta_percent=50.0,
            throughput_delta_percent=40.0,
            memory_delta_percent=33.3,
            statistical_significance=True,
            severity=RegressionSeverity.CRITICAL,
            confidence_level=0.95,
            recommendation="Critical regression detected",
            timestamp=datetime.now()
        )

        result_dict = result.to_dict()

        assert result_dict['model_name'] == "test_model"
        assert result_dict['performance_delta_percent'] == 50.0
        assert result_dict['severity'] == "critical"
        assert result_dict['current_latency_ms'] == 15.0
        assert result_dict['baseline_latency_ms'] == 10.0
        assert isinstance(result_dict['timestamp'], str)


class TestRegressionDetector:
    """Test RegressionDetector functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.detector = RegressionDetector(
            minor_threshold_percent=2.0,
            major_threshold_percent=5.0,
            critical_threshold_percent=10.0,
            confidence_level=0.95,
            min_sample_size=5
        )

    def test_detector_initialization(self):
        """Test RegressionDetector initialization"""
        assert self.detector.minor_threshold == 2.0
        assert self.detector.major_threshold == 5.0
        assert self.detector.critical_threshold == 10.0
        assert self.detector.confidence_level == 0.95
        assert self.detector.min_sample_size == 5

    def test_detect_no_regression(self):
        """Test detecting no performance regression"""
        current = PerformanceMetrics(
            latency_ms=10.1,  # Very slight increase
            throughput_samples_per_sec=99.5,
            peak_memory_mb=128.5,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(9.8, 10.4)
        )

        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.5,
            mean_throughput=100.0,
            std_throughput=2.0,
            mean_memory_mb=128.0,
            std_memory_mb=5.0,
            sample_count=20,
            confidence_interval_95=(9.5, 10.5),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        result = self.detector.detect_regression(current, baseline)

        assert result.severity == RegressionSeverity.NONE
        assert not result.is_regression()
        assert not result.is_blocking()
        assert "No significant performance regression" in result.recommendation

    def test_detect_minor_regression(self):
        """Test detecting minor performance regression"""
        current = PerformanceMetrics(
            latency_ms=10.3,  # 3% increase
            throughput_samples_per_sec=97.0,  # 3% decrease
            peak_memory_mb=131.0,  # 2.3% increase
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(10.0, 10.6)
        )

        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.2,
            mean_throughput=100.0,
            std_throughput=2.0,
            mean_memory_mb=128.0,
            std_memory_mb=3.0,
            sample_count=20,
            confidence_interval_95=(9.8, 10.2),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        result = self.detector.detect_regression(current, baseline)

        assert result.severity == RegressionSeverity.MINOR
        assert result.is_regression()
        assert not result.is_blocking()
        assert "Minor regression detected" in result.recommendation

    def test_detect_major_regression(self):
        """Test detecting major performance regression"""
        current = PerformanceMetrics(
            latency_ms=10.7,  # 7% increase
            throughput_samples_per_sec=92.0,  # 8% decrease
            peak_memory_mb=140.0,  # 9.4% increase
            memory_efficiency=0.80,
            accuracy_loss=0.02,
            statistical_significance=True,
            confidence_interval_95=(10.3, 11.1)
        )

        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.2,
            mean_throughput=100.0,
            std_throughput=2.0,
            mean_memory_mb=128.0,
            std_memory_mb=3.0,
            sample_count=20,
            confidence_interval_95=(9.8, 10.2),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        result = self.detector.detect_regression(current, baseline)

        assert result.severity == RegressionSeverity.MAJOR
        assert result.is_regression()
        assert result.is_blocking()
        assert "MAJOR regression detected" in result.recommendation

    def test_detect_critical_regression(self):
        """Test detecting critical performance regression"""
        current = PerformanceMetrics(
            latency_ms=12.0,  # 20% increase
            throughput_samples_per_sec=80.0,  # 20% decrease
            peak_memory_mb=160.0,  # 25% increase
            memory_efficiency=0.70,
            accuracy_loss=0.05,
            statistical_significance=True,
            confidence_interval_95=(11.5, 12.5)
        )

        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.3,
            mean_throughput=100.0,
            std_throughput=3.0,
            mean_memory_mb=128.0,
            std_memory_mb=5.0,
            sample_count=20,
            confidence_interval_95=(9.7, 10.3),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        result = self.detector.detect_regression(current, baseline)

        assert result.severity == RegressionSeverity.CRITICAL
        assert result.is_regression()
        assert result.is_blocking()
        assert "CRITICAL regression detected" in result.recommendation

    def test_calculate_performance_delta(self):
        """Test performance delta calculation"""
        # Test latency increase (worse performance)
        latency_delta = self.detector._calculate_performance_delta(12.0, 10.0, higher_is_better=False)
        assert latency_delta == 20.0

        # Test throughput decrease (worse performance, but positive delta due to inversion)
        throughput_delta = self.detector._calculate_performance_delta(80.0, 100.0, higher_is_better=True)
        assert throughput_delta == 20.0

        # Test throughput increase (better performance, negative delta due to inversion)
        throughput_improvement = self.detector._calculate_performance_delta(120.0, 100.0, higher_is_better=True)
        assert throughput_improvement == -20.0

        # Test zero baseline
        zero_delta = self.detector._calculate_performance_delta(10.0, 0.0)
        assert zero_delta == 0.0

    def test_calculate_z_score(self):
        """Test z-score calculation"""
        z_score = self.detector._calculate_z_score(12.0, 10.0, 1.0)
        assert z_score == 2.0

        z_score_negative = self.detector._calculate_z_score(8.0, 10.0, 1.0)
        assert z_score_negative == 2.0  # Absolute value

        # Test zero standard deviation
        z_score_zero_std = self.detector._calculate_z_score(12.0, 10.0, 0.0)
        assert z_score_zero_std == 0.0

    def test_analyze_statistical_significance(self):
        """Test statistical significance analysis"""
        # Current performance within confidence interval (not significant)
        current_within = PerformanceMetrics(
            latency_ms=10.0,
            throughput_samples_per_sec=100.0,
            peak_memory_mb=128.0,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(9.8, 10.2)
        )

        baseline_good = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.5,
            mean_throughput=100.0,
            std_throughput=2.0,
            mean_memory_mb=128.0,
            std_memory_mb=3.0,
            sample_count=20,
            confidence_interval_95=(9.5, 10.5),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        significance = self.detector._analyze_statistical_significance(current_within, baseline_good)
        assert not significance

        # Current performance outside confidence interval (significant)
        current_outside = PerformanceMetrics(
            latency_ms=12.0,
            throughput_samples_per_sec=85.0,
            peak_memory_mb=150.0,
            memory_efficiency=0.80,
            accuracy_loss=0.02,
            statistical_significance=True,
            confidence_interval_95=(11.8, 12.2)
        )

        significance_outside = self.detector._analyze_statistical_significance(current_outside, baseline_good)
        assert significance_outside

        # Insufficient baseline samples
        baseline_insufficient = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.5,
            mean_throughput=100.0,
            std_throughput=2.0,
            mean_memory_mb=128.0,
            std_memory_mb=3.0,
            sample_count=3,  # Below minimum
            confidence_interval_95=(9.5, 10.5),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        significance_insufficient = self.detector._analyze_statistical_significance(current_outside, baseline_insufficient)
        assert not significance_insufficient

    def test_determine_severity(self):
        """Test severity determination"""
        # Test no regression
        severity_none = self.detector._determine_severity(1.5)
        assert severity_none == RegressionSeverity.NONE

        # Test minor regression
        severity_minor = self.detector._determine_severity(3.0)
        assert severity_minor == RegressionSeverity.MINOR

        # Test major regression
        severity_major = self.detector._determine_severity(7.0)
        assert severity_major == RegressionSeverity.MAJOR

        # Test critical regression
        severity_critical = self.detector._determine_severity(15.0)
        assert severity_critical == RegressionSeverity.CRITICAL

        # Test negative values (still use absolute value)
        severity_negative = self.detector._determine_severity(-7.0)
        assert severity_negative == RegressionSeverity.MAJOR

    def test_analyze_trend(self):
        """Test performance trend analysis"""
        # Create trending performance metrics (degrading)
        degrading_metrics = []
        for i in range(10):
            metrics = PerformanceMetrics(
                latency_ms=10.0 + i * 0.1,  # Increasing latency
                throughput_samples_per_sec=100.0 - i * 0.5,  # Decreasing throughput
                peak_memory_mb=128.0,
                memory_efficiency=0.85,
                accuracy_loss=0.01,
                statistical_significance=True,
                confidence_interval_95=(9.0, 11.0)
            )
            degrading_metrics.append(metrics)

        trend_analysis = self.detector.analyze_trend(degrading_metrics)

        assert trend_analysis["latency_trend"]["direction"] == "degrading"
        assert trend_analysis["throughput_trend"]["direction"] == "degrading"
        assert trend_analysis["measurements_analyzed"] == 10

    def test_analyze_trend_insufficient_data(self):
        """Test trend analysis with insufficient data"""
        insufficient_metrics = [
            PerformanceMetrics(
                latency_ms=10.0,
                throughput_samples_per_sec=100.0,
                peak_memory_mb=128.0,
                memory_efficiency=0.85,
                accuracy_loss=0.01,
                statistical_significance=True,
                confidence_interval_95=(9.0, 11.0)
            )
        ]

        trend_analysis = self.detector.analyze_trend(insufficient_metrics)
        assert "error" in trend_analysis
        assert "Insufficient data" in trend_analysis["error"]

    def test_batch_analyze(self):
        """Test batch analysis of multiple models"""
        # Create test measurements
        measurements = []
        baselines = {}

        for i in range(3):
            model_name = f"model_{i}"

            # Create measurement
            metrics = PerformanceMetrics(
                latency_ms=10.0 + i * 2,  # Varying performance
                throughput_samples_per_sec=100.0 - i * 10,
                peak_memory_mb=128.0 + i * 20,
                memory_efficiency=0.85,
                accuracy_loss=0.01,
                statistical_significance=True,
                confidence_interval_95=(9.0, 11.0)
            )
            measurements.append((model_name, metrics))

            # Create baseline
            baseline = BaselineMetrics(
                model_name=model_name,
                mean_latency_ms=10.0,
                std_latency_ms=0.5,
                mean_throughput=100.0,
                std_throughput=3.0,
                mean_memory_mb=128.0,
                std_memory_mb=5.0,
                sample_count=20,
                confidence_interval_95=(9.5, 10.5),
                established_date=datetime.now(),
                last_validated_date=datetime.now()
            )
            baselines[model_name] = baseline

        results = self.detector.batch_analyze(measurements, baselines)

        assert len(results) == 3
        assert all(isinstance(r, RegressionResult) for r in results)
        assert results[0].model_name == "model_0"
        assert results[1].model_name == "model_1"
        assert results[2].model_name == "model_2"

        # Check that different severities are detected based on performance differences
        severities = [r.severity for r in results]
        assert RegressionSeverity.NONE in severities
        assert any(s != RegressionSeverity.NONE for s in severities)

    def test_batch_analyze_missing_baseline(self):
        """Test batch analysis with missing baseline"""
        measurements = [("missing_model", PerformanceMetrics(
            latency_ms=10.0,
            throughput_samples_per_sec=100.0,
            peak_memory_mb=128.0,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(9.0, 11.0)
        ))]

        baselines = {}  # Empty baselines

        with pytest.warns(UserWarning, match="No baseline found for model"):
            results = self.detector.batch_analyze(measurements, baselines)

        assert len(results) == 0

    def test_generate_recommendation(self):
        """Test recommendation generation"""
        # Test no regression
        rec_none = self.detector._generate_recommendation(
            1.0, 0.5, 0.8, RegressionSeverity.NONE, True
        )
        assert "No significant performance regression" in rec_none

        # Test critical regression with specific metrics
        rec_critical = self.detector._generate_recommendation(
            15.0, 12.0, 8.0, RegressionSeverity.CRITICAL, True
        )
        assert "CRITICAL regression detected" in rec_critical
        assert "Latency increased by 15.0%" in rec_critical
        assert "Throughput decreased by 12.0%" in rec_critical
        assert "Memory usage increased by 8.0%" in rec_critical

        # Test without statistical significance
        rec_not_significant = self.detector._generate_recommendation(
            3.0, 2.5, 2.0, RegressionSeverity.MINOR, False
        )
        assert "Note: Change may not be statistically significant" in rec_not_significant
