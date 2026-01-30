"""
Tests for performance tracking and regression detection.

Tests Stage 3B: Performance Regression Detection
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchbridge.core.performance_tracker import (
    MetricType,
    PerformanceMetrics,
    PerformanceTracker,
    RegressionResult,
    RegressionSeverity,
    detect_regression,
    get_performance_tracker,
    track_performance,
)


# Test fixtures
@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir) / 'metrics.json'
    shutil.rmtree(temp_dir)


@pytest.fixture
def tracker(temp_storage):
    """Create performance tracker with temp storage."""
    return PerformanceTracker(storage_path=temp_storage)


@pytest.fixture
def simple_model():
    """Simple test model."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleModel()


@pytest.fixture
def sample_inputs():
    """Sample inputs for testing."""
    return torch.randn(8, 128)


# PerformanceTracker Tests
class TestPerformanceTracker:
    """Test performance tracker functionality."""

    def test_tracker_initialization(self, temp_storage):
        """Test tracker initialization."""
        tracker = PerformanceTracker(storage_path=temp_storage)

        assert tracker.storage_path == temp_storage
        assert isinstance(tracker.metrics_history, dict)
        assert len(tracker.metrics_history) == 0

    def test_benchmark_model(self, tracker, simple_model, sample_inputs):
        """Test model benchmarking."""
        latency_ms, throughput, memory_mb = tracker.benchmark_model(
            simple_model,
            sample_inputs,
            num_iterations=10,
            warmup_iterations=2
        )

        assert latency_ms > 0
        assert throughput > 0
        assert memory_mb >= 0

    def test_record_performance(self, tracker, simple_model, sample_inputs):
        """Test recording performance metrics."""
        metrics = tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model",
            backend="cpu",
            optimization_level="conservative"
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.model_name == "test_model"
        assert metrics.backend == "cpu"
        assert metrics.optimization_level == "conservative"
        assert metrics.latency_ms > 0
        assert metrics.throughput > 0

    def test_record_multiple_performance(self, tracker, simple_model, sample_inputs):
        """Test recording multiple performance metrics."""
        # Record baseline
        tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model",
            backend="cpu",
            optimization_level="conservative"
        )

        # Record optimized
        tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model",
            backend="cpu",
            optimization_level="aggressive"
        )

        # Both should be recorded
        model_hash = tracker._compute_model_hash(simple_model)
        assert len(tracker.metrics_history[model_hash]) == 2

    def test_get_baseline(self, tracker, simple_model, sample_inputs):
        """Test getting baseline performance."""
        # No baseline initially
        baseline = tracker.get_baseline(simple_model)
        assert baseline is None

        # Record performance
        tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model"
        )

        # Should have baseline now
        baseline = tracker.get_baseline(simple_model)
        assert baseline is not None
        assert baseline.model_name == "test_model"

    def test_get_baseline_with_filters(self, tracker, simple_model, sample_inputs):
        """Test getting baseline with filters."""
        # Record multiple metrics
        tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model",
            backend="cpu",
            optimization_level="conservative"
        )

        tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model",
            backend="nvidia",
            optimization_level="aggressive"
        )

        # Get baseline with filters
        baseline_cpu = tracker.get_baseline(simple_model, backend="cpu")
        baseline_nvidia = tracker.get_baseline(simple_model, backend="nvidia")

        assert baseline_cpu.backend == "cpu"
        assert baseline_nvidia.backend == "nvidia"

    def test_metrics_persistence(self, temp_storage, simple_model, sample_inputs):
        """Test that metrics are persisted to disk."""
        # Create tracker and record metrics
        tracker1 = PerformanceTracker(storage_path=temp_storage)
        tracker1.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model"
        )

        # Create new tracker with same storage
        tracker2 = PerformanceTracker(storage_path=temp_storage)

        # Should load existing metrics
        baseline = tracker2.get_baseline(simple_model)
        assert baseline is not None
        assert baseline.model_name == "test_model"


# Regression Detection Tests
class TestRegressionDetection:
    """Test regression detection functionality."""

    def test_no_regression_with_improvement(self, tracker, simple_model):
        """Test that no regression is detected when performance improves."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=8.0,  # Improved (lower)
            throughput=120.0,  # Improved (higher)
            memory_mb=95.0,  # Improved (lower)
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        regressions = tracker.detect_regression(simple_model, current, baseline)

        # No regressions should be detected
        assert len(regressions) == 0

    def test_latency_regression_minor(self, tracker, simple_model):
        """Test minor latency regression detection."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=10.5,  # 5% slower (minor regression)
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        regressions = tracker.detect_regression(simple_model, current, baseline)

        # Should detect minor latency regression
        assert len(regressions) >= 1
        latency_regression = next(r for r in regressions if r.metric_type == MetricType.LATENCY)
        assert latency_regression.severity == RegressionSeverity.MINOR

    def test_latency_regression_moderate(self, tracker, simple_model):
        """Test moderate latency regression detection."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=12.0,  # 20% slower (moderate regression)
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        regressions = tracker.detect_regression(simple_model, current, baseline)

        # Should detect moderate latency regression
        latency_regression = next(r for r in regressions if r.metric_type == MetricType.LATENCY)
        assert latency_regression.severity == RegressionSeverity.MODERATE

    def test_latency_regression_severe(self, tracker, simple_model):
        """Test severe latency regression detection."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=15.0,  # 50% slower (severe regression)
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        regressions = tracker.detect_regression(simple_model, current, baseline)

        # Should detect severe latency regression
        latency_regression = next(r for r in regressions if r.metric_type == MetricType.LATENCY)
        assert latency_regression.severity == RegressionSeverity.SEVERE

    def test_throughput_regression(self, tracker, simple_model):
        """Test throughput regression detection."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=10.0,
            throughput=70.0,  # 30% lower throughput (severe regression)
            memory_mb=100.0,
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        regressions = tracker.detect_regression(simple_model, current, baseline)

        # Should detect throughput regression
        throughput_regression = next(r for r in regressions if r.metric_type == MetricType.THROUGHPUT)
        assert throughput_regression.severity == RegressionSeverity.SEVERE

    def test_memory_regression(self, tracker, simple_model):
        """Test memory regression detection."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=140.0,  # 40% more memory (severe regression)
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        regressions = tracker.detect_regression(simple_model, current, baseline)

        # Should detect memory regression
        memory_regression = next(r for r in regressions if r.metric_type == MetricType.MEMORY)
        assert memory_regression.severity == RegressionSeverity.SEVERE

    def test_warn_if_regression(self, tracker, simple_model):
        """Test warning on regression."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=15.0,  # Severe regression
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        # Should issue warning
        with pytest.warns(UserWarning):
            tracker.warn_if_regression(simple_model, current, baseline)


# Performance History Tests
class TestPerformanceHistory:
    """Test performance history functionality."""

    def test_get_performance_history(self, tracker, simple_model, sample_inputs):
        """Test getting performance history."""
        # Record multiple metrics
        for i in range(5):
            tracker.record_performance(
                model=simple_model,
                sample_inputs=sample_inputs,
                model_name=f"test_model_{i}"
            )

        history = tracker.get_performance_history(simple_model, limit=3)

        # Should return last 3 in reverse order
        assert len(history) == 3
        assert history[0].model_name == "test_model_4"  # Most recent first

    def test_clear_history_specific_model(self, tracker, simple_model, sample_inputs):
        """Test clearing history for specific model."""
        tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model"
        )

        # Clear history
        tracker.clear_history(simple_model)

        # Should have no history
        history = tracker.get_performance_history(simple_model)
        assert len(history) == 0

    def test_clear_all_history(self, tracker, simple_model, sample_inputs):
        """Test clearing all history."""
        tracker.record_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model"
        )

        # Clear all history
        tracker.clear_history()

        # Should have no history
        assert len(tracker.metrics_history) == 0


# Convenience Function Tests
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_performance_tracker(self):
        """Test get_performance_tracker."""
        tracker = get_performance_tracker()
        assert isinstance(tracker, PerformanceTracker)

    def test_track_performance(self, simple_model, sample_inputs):
        """Test track_performance convenience function."""
        metrics = track_performance(
            model=simple_model,
            sample_inputs=sample_inputs,
            model_name="test_model"
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.model_name == "test_model"

    def test_detect_regression_function(self, simple_model):
        """Test detect_regression convenience function."""
        baseline = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-01T00:00:00",
            latency_ms=10.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="conservative",
            device="cpu",
            batch_size=8
        )

        current = PerformanceMetrics(
            model_name="test_model",
            model_hash="test_hash",
            timestamp="2024-01-02T00:00:00",
            latency_ms=15.0,
            throughput=100.0,
            memory_mb=100.0,
            backend="cpu",
            optimization_level="aggressive",
            device="cpu",
            batch_size=8
        )

        regressions = detect_regression(simple_model, current, baseline=baseline)

        assert len(regressions) > 0
        assert all(isinstance(r, RegressionResult) for r in regressions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
