#!/usr/bin/env python3
"""
Tests for BaselineManager functionality
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from benchmarks.framework.benchmark_runner import PerformanceMetrics
from benchmarks.regression.baseline_manager import BaselineManager, BaselineMetrics


class TestBaselineMetrics:
    """Test BaselineMetrics dataclass functionality"""

    def test_baseline_metrics_creation(self):
        """Test creating BaselineMetrics"""
        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.5,
            std_latency_ms=1.2,
            mean_throughput=95.2,
            std_throughput=5.1,
            mean_memory_mb=128.0,
            std_memory_mb=10.0,
            sample_count=20,
            confidence_interval_95=(9.8, 11.2),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        assert baseline.model_name == "test_model"
        assert baseline.mean_latency_ms == 10.5
        assert baseline.sample_count == 20

    def test_baseline_metrics_serialization(self):
        """Test BaselineMetrics to/from dict conversion"""
        now = datetime.now()
        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.5,
            std_latency_ms=1.2,
            mean_throughput=95.2,
            std_throughput=5.1,
            mean_memory_mb=128.0,
            std_memory_mb=10.0,
            sample_count=20,
            confidence_interval_95=(9.8, 11.2),
            established_date=now,
            last_validated_date=now
        )

        # Test serialization
        data = baseline.to_dict()
        assert isinstance(data['established_date'], str)
        assert data['model_name'] == "test_model"

        # Test deserialization
        restored = BaselineMetrics.from_dict(data)
        assert restored.model_name == baseline.model_name
        assert restored.mean_latency_ms == baseline.mean_latency_ms
        assert isinstance(restored.established_date, datetime)


class TestBaselineManager:
    """Test BaselineManager functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = BaselineManager(baselines_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_baseline_manager_initialization(self):
        """Test BaselineManager initialization"""
        assert self.manager.baselines_dir.exists()
        assert self.manager.models_dir.exists()
        assert isinstance(self.manager.registry, dict)
        assert 'version' in self.manager.registry

    def test_establish_baseline_from_single_metrics(self):
        """Test establishing baseline from single PerformanceMetrics"""
        metrics = PerformanceMetrics(
            latency_ms=15.5,
            throughput_samples_per_sec=64.3,
            peak_memory_mb=256.0,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(14.8, 16.2)
        )

        success = self.manager.establish_baseline("test_model", metrics)
        assert success

        # Verify baseline was stored
        baseline = self.manager.get_baseline("test_model")
        assert baseline is not None
        assert baseline.model_name == "test_model"
        assert baseline.mean_latency_ms == 15.5
        assert baseline.sample_count == 1

    def test_get_nonexistent_baseline(self):
        """Test getting baseline that doesn't exist"""
        baseline = self.manager.get_baseline("nonexistent_model")
        assert baseline is None

    def test_update_baseline(self):
        """Test updating existing baseline"""
        # Establish initial baseline
        initial_metrics = PerformanceMetrics(
            latency_ms=15.5,
            throughput_samples_per_sec=64.3,
            peak_memory_mb=256.0,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(14.8, 16.2)
        )
        self.manager.establish_baseline("test_model", initial_metrics)

        # Update with new metrics
        updated_metrics = PerformanceMetrics(
            latency_ms=14.2,
            throughput_samples_per_sec=70.5,
            peak_memory_mb=240.0,
            memory_efficiency=0.90,
            accuracy_loss=0.005,
            statistical_significance=True,
            confidence_interval_95=(13.8, 14.6)
        )
        success = self.manager.update_baseline("test_model", updated_metrics)
        assert success

        # Verify baseline was updated
        baseline = self.manager.get_baseline("test_model")
        assert baseline.mean_latency_ms == 14.2

    def test_validate_baseline_quality(self):
        """Test baseline quality validation"""
        # Good baseline
        good_baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.5,
            mean_throughput=100.0,
            std_throughput=5.0,
            mean_memory_mb=128.0,
            std_memory_mb=10.0,
            sample_count=15,
            confidence_interval_95=(9.5, 10.5),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )
        assert self.manager.validate_baseline_quality(good_baseline)

        # Poor baseline - too few samples
        poor_baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=0.5,
            mean_throughput=100.0,
            std_throughput=5.0,
            mean_memory_mb=128.0,
            std_memory_mb=10.0,
            sample_count=3,  # Too few samples
            confidence_interval_95=(9.5, 10.5),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )
        assert not self.manager.validate_baseline_quality(poor_baseline)

        # Poor baseline - high variance
        high_variance_baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=8.0,  # Very high standard deviation
            mean_throughput=100.0,
            std_throughput=5.0,
            mean_memory_mb=128.0,
            std_memory_mb=10.0,
            sample_count=15,
            confidence_interval_95=(2.0, 18.0),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )
        assert not self.manager.validate_baseline_quality(high_variance_baseline)

    def test_establish_baseline_from_historical_data(self):
        """Test establishing baseline from historical benchmark files"""
        # Create mock historical data files
        results_dir = Path(self.temp_dir) / "results"
        results_dir.mkdir()

        # Create mock benchmark result files
        for i in range(10):
            timestamp = datetime.now() - timedelta(days=i)
            filename = f"test_model_inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            result_data = {
                "config": {"name": "test_model"},
                "results": {
                    "PyTorch Native": {
                        "latency_ms": 10.0 + i * 0.1,  # Slight variation
                        "throughput_samples_per_sec": 100.0 - i * 0.5,
                        "peak_memory_mb": 128.0,
                        "memory_efficiency": 0.85,
                        "accuracy_loss": 0.01,
                        "statistical_significance": True,
                        "confidence_interval_95": [9.5, 10.5]
                    }
                }
            }

            with open(results_dir / filename, 'w') as f:
                json.dump(result_data, f)

        # Test establishing baseline from historical data
        baseline = self.manager.establish_baseline_from_historical_data(
            "test_model",
            str(results_dir),
            window_days=30,
            min_samples=5
        )

        assert baseline is not None
        assert baseline.model_name == "test_model"
        assert baseline.sample_count == 10
        assert 10.0 <= baseline.mean_latency_ms <= 11.0
        assert baseline.std_latency_ms > 0  # Should have some variance

    def test_establish_baseline_insufficient_data(self):
        """Test baseline establishment with insufficient historical data"""
        results_dir = Path(self.temp_dir) / "results"
        results_dir.mkdir()

        # Create only 2 files (below minimum)
        for i in range(2):
            timestamp = datetime.now() - timedelta(days=i)
            filename = f"test_model_inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            result_data = {
                "config": {"name": "test_model"},
                "results": {
                    "PyTorch Native": {
                        "latency_ms": 10.0,
                        "throughput_samples_per_sec": 100.0,
                        "peak_memory_mb": 128.0
                    }
                }
            }

            with open(results_dir / filename, 'w') as f:
                json.dump(result_data, f)

        baseline = self.manager.establish_baseline_from_historical_data(
            "test_model",
            str(results_dir),
            min_samples=5
        )

        assert baseline is None

    def test_list_available_models(self):
        """Test listing models with baselines"""
        # Initially empty
        models = self.manager.list_available_models()
        assert len(models) == 0

        # Add some baselines
        metrics = PerformanceMetrics(
            latency_ms=15.5,
            throughput_samples_per_sec=64.3,
            peak_memory_mb=256.0,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(14.8, 16.2)
        )

        self.manager.establish_baseline("model_1", metrics)
        self.manager.establish_baseline("model_2", metrics)

        models = self.manager.list_available_models()
        assert len(models) == 2
        assert "model_1" in models
        assert "model_2" in models

    def test_get_baseline_summary(self):
        """Test getting baseline summary"""
        # Add a baseline
        metrics = PerformanceMetrics(
            latency_ms=15.5,
            throughput_samples_per_sec=64.3,
            peak_memory_mb=256.0,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(14.8, 16.2)
        )
        self.manager.establish_baseline("test_model", metrics)

        summary = self.manager.get_baseline_summary()
        assert summary["total_models"] == 1
        assert "test_model" in summary["models"]
        assert "registry_version" in summary
        assert summary["models"]["test_model"]["mean_latency_ms"] == 15.5

    def test_extract_timestamp_from_filename(self):
        """Test timestamp extraction from benchmark filenames"""
        # Test with valid filename format
        test_data = {"config": {"name": "test"}}
        timestamp = self.manager._extract_timestamp(
            "test_model_inference_20251204_143522.json",
            test_data
        )
        assert timestamp is not None
        assert timestamp.year == 2025
        assert timestamp.month == 12
        assert timestamp.day == 4

    def test_extract_metrics_from_result(self):
        """Test extracting metrics from benchmark result data"""
        result_data = {
            "results": {
                "PyTorch Native": {
                    "latency_ms": 12.5,
                    "throughput_samples_per_sec": 80.0,
                    "peak_memory_mb": 200.0,
                    "memory_efficiency": 0.9,
                    "accuracy_loss": 0.005,
                    "statistical_significance": True,
                    "confidence_interval_95": [12.0, 13.0]
                }
            }
        }

        metrics = self.manager._extract_metrics_from_result(result_data, "test_model")
        assert metrics is not None
        assert metrics.latency_ms == 12.5
        assert metrics.throughput_samples_per_sec == 80.0
        assert metrics.peak_memory_mb == 200.0

    def test_calculate_baseline_statistics(self):
        """Test baseline statistics calculation"""
        # Create sample metrics
        metrics_list = []
        for i in range(10):
            metrics = PerformanceMetrics(
                latency_ms=10.0 + i * 0.1,
                throughput_samples_per_sec=100.0 - i * 0.5,
                peak_memory_mb=128.0 + i,
                memory_efficiency=0.85,
                accuracy_loss=0.01,
                statistical_significance=True,
                confidence_interval_95=(9.5, 10.5)
            )
            metrics_list.append(metrics)

        baseline = self.manager._calculate_baseline_statistics("test_model", metrics_list)

        assert baseline.model_name == "test_model"
        assert baseline.sample_count == 10
        assert baseline.mean_latency_ms > 10.0
        assert baseline.std_latency_ms > 0
        assert baseline.confidence_interval_95[0] < baseline.confidence_interval_95[1]
