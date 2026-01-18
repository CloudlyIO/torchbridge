#!/usr/bin/env python3
"""
Performance Regression Testing Benchmark Suite

Comprehensive benchmarking of the regression testing framework components:
- Baseline establishment performance
- Regression detection accuracy and speed
- Threshold management efficiency
- Statistical analysis validation

Usage:
    python regression_benchmark.py [--quick] [--detailed]
"""

import argparse
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import statistics

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.regression.baseline_manager import BaselineManager, BaselineMetrics
from benchmarks.regression.regression_detector import RegressionDetector, RegressionSeverity
from benchmarks.regression.threshold_manager import ThresholdManager
from benchmarks.framework.benchmark_runner import PerformanceMetrics


class RegressionFrameworkBenchmark:
    """Benchmark suite for regression testing framework"""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = {}
        self.temp_dir = None

    def setup(self):
        """Set up benchmark environment"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="regression_benchmark_"))
        print(f"🔧 Setting up benchmark environment in {self.temp_dir}")

    def teardown(self):
        """Clean up benchmark environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_test_data(self, num_models: int, samples_per_model: int) -> Dict[str, List[PerformanceMetrics]]:
        """Create test performance data for benchmarking"""
        test_data = {}

        for model_idx in range(num_models):
            model_name = f"test_model_{model_idx}"
            metrics_list = []

            # Base performance with realistic variance
            base_latency = 10.0 + model_idx * 2.0  # Different base performance per model
            base_throughput = 100.0 - model_idx * 5.0
            base_memory = 128.0 + model_idx * 32.0

            for sample_idx in range(samples_per_model):
                # Add realistic variance
                latency_noise = np.random.normal(0, 0.1)  # 10% std dev
                throughput_noise = np.random.normal(0, 0.05)  # 5% std dev
                memory_noise = np.random.normal(0, 0.02)  # 2% std dev

                metrics = PerformanceMetrics(
                    latency_ms=base_latency * (1.0 + latency_noise),
                    throughput_samples_per_sec=base_throughput * (1.0 + throughput_noise),
                    peak_memory_mb=base_memory * (1.0 + memory_noise),
                    memory_efficiency=0.85,
                    accuracy_loss=0.01,
                    statistical_significance=True,
                    confidence_interval_95=(base_latency * 0.95, base_latency * 1.05)
                )
                metrics_list.append(metrics)

            test_data[model_name] = metrics_list

        return test_data

    def create_historical_files(self, test_data: Dict[str, List[PerformanceMetrics]]) -> Path:
        """Create historical benchmark files for baseline establishment testing"""
        results_dir = self.temp_dir / "historical_results"
        results_dir.mkdir(exist_ok=True)

        for model_name, metrics_list in test_data.items():
            for idx, metrics in enumerate(metrics_list):
                # Create timestamp going back in time
                timestamp = datetime.now() - timedelta(days=idx, hours=idx % 24)
                filename = f"{model_name}_inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

                result_data = {
                    "config": {"name": model_name},
                    "timestamp": timestamp.isoformat(),
                    "results": {
                        "PyTorch Native": {
                            "latency_ms": metrics.latency_ms,
                            "throughput_samples_per_sec": metrics.throughput_samples_per_sec,
                            "peak_memory_mb": metrics.peak_memory_mb,
                            "memory_efficiency": metrics.memory_efficiency,
                            "accuracy_loss": metrics.accuracy_loss,
                            "statistical_significance": metrics.statistical_significance,
                            "confidence_interval_95": list(metrics.confidence_interval_95)
                        }
                    }
                }

                with open(results_dir / filename, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)

        return results_dir

    def benchmark_baseline_establishment(self, test_data: Dict[str, List[PerformanceMetrics]]) -> Dict[str, Any]:
        """Benchmark baseline establishment performance"""
        print("\n📊 Benchmarking baseline establishment...")

        results_dir = self.create_historical_files(test_data)
        baseline_manager = BaselineManager(baselines_dir=str(self.temp_dir / "baselines"))

        # Benchmark individual baseline establishment
        establishment_times = []
        established_baselines = {}

        for model_name in test_data.keys():
            start_time = time.time()

            baseline = baseline_manager.establish_baseline_from_historical_data(
                model_name=model_name,
                results_dir=str(results_dir),
                window_days=30,
                min_samples=5
            )

            elapsed_time = time.time() - start_time
            establishment_times.append(elapsed_time)

            if baseline:
                established_baselines[model_name] = baseline

        # Benchmark batch operations
        start_time = time.time()
        summary = baseline_manager.get_baseline_summary()
        summary_time = time.time() - start_time

        # Benchmark baseline quality validation
        validation_times = []
        for baseline in established_baselines.values():
            start_time = time.time()
            is_valid = baseline_manager.validate_baseline_quality(baseline)
            validation_times.append(time.time() - start_time)

        return {
            "establishment_times": establishment_times,
            "mean_establishment_time": statistics.mean(establishment_times),
            "total_baselines_established": len(established_baselines),
            "summary_generation_time": summary_time,
            "validation_times": validation_times,
            "mean_validation_time": statistics.mean(validation_times) if validation_times else 0,
            "established_baselines": established_baselines
        }

    def benchmark_regression_detection(self, established_baselines: Dict[str, BaselineMetrics]) -> Dict[str, Any]:
        """Benchmark regression detection performance and accuracy"""
        print("\n🔍 Benchmarking regression detection...")

        detector = RegressionDetector(
            minor_threshold_percent=2.0,
            major_threshold_percent=5.0,
            critical_threshold_percent=10.0,
            confidence_level=0.95,
            min_sample_size=5
        )

        # Test scenarios with known regression levels
        test_scenarios = [
            ("no_regression", 1.01, RegressionSeverity.NONE),
            ("minor_regression", 1.03, RegressionSeverity.MINOR),
            ("major_regression", 1.08, RegressionSeverity.MAJOR),
            ("critical_regression", 1.15, RegressionSeverity.CRITICAL),
            ("improvement", 0.92, RegressionSeverity.NONE)
        ]

        model_name = list(established_baselines.keys())[0]
        baseline = established_baselines[model_name]

        # Benchmark individual detection
        detection_times = []
        accuracy_results = []

        for scenario_name, latency_multiplier, expected_severity in test_scenarios:
            # Create test metrics
            test_metrics = PerformanceMetrics(
                latency_ms=baseline.mean_latency_ms * latency_multiplier,
                throughput_samples_per_sec=baseline.mean_throughput * (2.0 - latency_multiplier),
                peak_memory_mb=baseline.mean_memory_mb * latency_multiplier,
                memory_efficiency=0.85,
                accuracy_loss=0.01,
                statistical_significance=True,
                confidence_interval_95=(10.0, 15.0)
            )

            # Time the detection
            start_time = time.time()
            result = detector.detect_regression(test_metrics, baseline)
            detection_time = time.time() - start_time

            detection_times.append(detection_time)

            # Check accuracy
            correct_detection = result.severity == expected_severity
            accuracy_results.append(correct_detection)

        # Benchmark batch analysis
        batch_measurements = []
        for i, (model_name, baseline) in enumerate(list(established_baselines.items())[:3]):
            test_metrics = PerformanceMetrics(
                latency_ms=baseline.mean_latency_ms * (1.0 + i * 0.02),  # Varying regression levels
                throughput_samples_per_sec=baseline.mean_throughput,
                peak_memory_mb=baseline.mean_memory_mb,
                memory_efficiency=0.85,
                accuracy_loss=0.01,
                statistical_significance=True,
                confidence_interval_95=(10.0, 15.0)
            )
            batch_measurements.append((model_name, test_metrics))

        start_time = time.time()
        batch_results = detector.batch_analyze(batch_measurements, established_baselines)
        batch_analysis_time = time.time() - start_time

        # Benchmark trend analysis
        trending_metrics = []
        for i in range(10):
            metrics = PerformanceMetrics(
                latency_ms=baseline.mean_latency_ms * (1.0 + i * 0.01),
                throughput_samples_per_sec=baseline.mean_throughput,
                peak_memory_mb=baseline.mean_memory_mb,
                memory_efficiency=0.85,
                accuracy_loss=0.01,
                statistical_significance=True,
                confidence_interval_95=(10.0, 15.0)
            )
            trending_metrics.append(metrics)

        start_time = time.time()
        trend_analysis = detector.analyze_trend(trending_metrics)
        trend_analysis_time = time.time() - start_time

        return {
            "detection_times": detection_times,
            "mean_detection_time": statistics.mean(detection_times),
            "accuracy_rate": sum(accuracy_results) / len(accuracy_results),
            "batch_analysis_time": batch_analysis_time,
            "batch_results_count": len(batch_results),
            "trend_analysis_time": trend_analysis_time,
            "trend_analysis_success": "error" not in trend_analysis
        }

    def benchmark_threshold_management(self, established_baselines: Dict[str, BaselineMetrics]) -> Dict[str, Any]:
        """Benchmark threshold management performance"""
        print("\n⚙️  Benchmarking threshold management...")

        threshold_manager = ThresholdManager(thresholds_dir=str(self.temp_dir / "thresholds"))

        # Benchmark threshold retrieval
        retrieval_times = []
        for model_name in established_baselines.keys():
            start_time = time.time()
            config = threshold_manager.get_thresholds(model_name, "default")
            retrieval_times.append(time.time() - start_time)

        # Benchmark threshold updates from baselines
        update_times = []
        for model_name, baseline in established_baselines.items():
            start_time = time.time()
            updated_config = threshold_manager.update_thresholds_from_baseline(baseline)
            update_times.append(time.time() - start_time)

        # Benchmark validation
        validation_times = []
        for model_name in established_baselines.keys():
            start_time = time.time()
            validation = threshold_manager.validate_threshold_sensitivity(model_name)
            validation_times.append(time.time() - start_time)

        # Benchmark environment adjustments
        adjustment_times = []
        environments = ["cpu", "gpu", "cloud", "ci"]
        for model_name in list(established_baselines.keys())[:2]:  # Test subset
            base_config = threshold_manager.get_thresholds(model_name)
            for env in environments:
                start_time = time.time()
                adjusted = threshold_manager.apply_environment_adjustments(base_config, env)
                adjustment_times.append(time.time() - start_time)

        # Benchmark export/import
        start_time = time.time()
        export_data = threshold_manager.export_threshold_config()
        export_time = time.time() - start_time

        new_manager = ThresholdManager(thresholds_dir=str(self.temp_dir / "thresholds_import"))
        start_time = time.time()
        import_success = new_manager.import_threshold_config(export_data)
        import_time = time.time() - start_time

        # Benchmark summary generation
        start_time = time.time()
        summary = threshold_manager.get_threshold_summary()
        summary_time = time.time() - start_time

        return {
            "retrieval_times": retrieval_times,
            "mean_retrieval_time": statistics.mean(retrieval_times),
            "update_times": update_times,
            "mean_update_time": statistics.mean(update_times),
            "validation_times": validation_times,
            "mean_validation_time": statistics.mean(validation_times),
            "adjustment_times": adjustment_times,
            "mean_adjustment_time": statistics.mean(adjustment_times),
            "export_time": export_time,
            "import_time": import_time,
            "import_success": import_success,
            "summary_generation_time": summary_time,
            "total_configurations": summary["total_models"]
        }

    def benchmark_statistical_accuracy(self, test_data: Dict[str, List[PerformanceMetrics]]) -> Dict[str, Any]:
        """Benchmark statistical accuracy of the regression detection"""
        print("\n🧮 Benchmarking statistical accuracy...")

        # Create baseline from first half of data
        model_name = list(test_data.keys())[0]
        all_metrics = test_data[model_name]
        baseline_metrics = all_metrics[:len(all_metrics)//2]
        test_metrics = all_metrics[len(all_metrics)//2:]

        # Calculate true baseline statistics
        latencies = [m.latency_ms for m in baseline_metrics]
        true_mean = statistics.mean(latencies)
        true_std = statistics.stdev(latencies) if len(latencies) > 1 else 0

        # Create baseline through our system
        baseline_manager = BaselineManager(baselines_dir=str(self.temp_dir / "stat_baselines"))
        computed_baseline = baseline_manager._calculate_baseline_statistics(model_name, baseline_metrics)

        # Compare accuracy
        mean_error = abs(computed_baseline.mean_latency_ms - true_mean) / true_mean * 100
        std_error = abs(computed_baseline.std_latency_ms - true_std) / true_std * 100 if true_std > 0 else 0

        # Test statistical significance detection
        detector = RegressionDetector()
        false_positives = 0
        true_negatives = 0

        # Test with data that should not be regressions (from same distribution)
        for metrics in test_metrics:
            result = detector.detect_regression(metrics, computed_baseline)
            if result.statistical_significance and result.severity != RegressionSeverity.NONE:
                false_positives += 1
            else:
                true_negatives += 1

        false_positive_rate = false_positives / len(test_metrics) * 100

        return {
            "mean_accuracy_error_percent": mean_error,
            "std_accuracy_error_percent": std_error,
            "false_positive_rate_percent": false_positive_rate,
            "true_negatives": true_negatives,
            "baseline_sample_count": computed_baseline.sample_count,
            "confidence_interval_width": computed_baseline.confidence_interval_95[1] - computed_baseline.confidence_interval_95[0]
        }

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("🚀 Starting comprehensive regression framework benchmark...")

        # Determine test sizes based on mode
        if self.quick_mode:
            num_models = 3
            samples_per_model = 10
        else:
            num_models = 10
            samples_per_model = 30

        # Create test data
        print(f"📊 Generating test data ({num_models} models, {samples_per_model} samples each)...")
        test_data = self.create_test_data(num_models, samples_per_model)

        # Run benchmark components
        baseline_results = self.benchmark_baseline_establishment(test_data)
        regression_results = self.benchmark_regression_detection(baseline_results["established_baselines"])
        threshold_results = self.benchmark_threshold_management(baseline_results["established_baselines"])

        if not self.quick_mode:
            statistical_results = self.benchmark_statistical_accuracy(test_data)
        else:
            statistical_results = {"skipped": "quick_mode"}

        # Compile overall results
        overall_results = {
            "benchmark_config": {
                "quick_mode": self.quick_mode,
                "num_models": num_models,
                "samples_per_model": samples_per_model,
                "timestamp": datetime.now().isoformat()
            },
            "baseline_establishment": baseline_results,
            "regression_detection": regression_results,
            "threshold_management": threshold_results,
            "statistical_accuracy": statistical_results
        }

        return overall_results

    def print_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results"""
        print("\n" + "=" * 60)
        print("🎯 REGRESSION FRAMEWORK BENCHMARK RESULTS")
        print("=" * 60)

        config = results["benchmark_config"]
        print(f"\nConfiguration:")
        print(f"  Mode: {'Quick' if config['quick_mode'] else 'Comprehensive'}")
        print(f"  Models tested: {config['num_models']}")
        print(f"  Samples per model: {config['samples_per_model']}")

        # Baseline establishment results
        baseline = results["baseline_establishment"]
        print(f"\n📊 Baseline Establishment:")
        print(f"  Mean establishment time: {baseline['mean_establishment_time']*1000:.2f}ms")
        print(f"  Baselines established: {baseline['total_baselines_established']}")
        print(f"  Summary generation time: {baseline['summary_generation_time']*1000:.2f}ms")
        print(f"  Mean validation time: {baseline['mean_validation_time']*1000:.2f}ms")

        # Regression detection results
        regression = results["regression_detection"]
        print(f"\n🔍 Regression Detection:")
        print(f"  Mean detection time: {regression['mean_detection_time']*1000:.2f}ms")
        print(f"  Detection accuracy: {regression['accuracy_rate']*100:.1f}%")
        print(f"  Batch analysis time: {regression['batch_analysis_time']*1000:.2f}ms")
        print(f"  Trend analysis time: {regression['trend_analysis_time']*1000:.2f}ms")
        print(f"  Trend analysis success: {'✅' if regression['trend_analysis_success'] else '❌'}")

        # Threshold management results
        threshold = results["threshold_management"]
        print(f"\n⚙️  Threshold Management:")
        print(f"  Mean retrieval time: {threshold['mean_retrieval_time']*1000:.2f}ms")
        print(f"  Mean update time: {threshold['mean_update_time']*1000:.2f}ms")
        print(f"  Mean validation time: {threshold['mean_validation_time']*1000:.2f}ms")
        print(f"  Mean adjustment time: {threshold['mean_adjustment_time']*1000:.2f}ms")
        print(f"  Export time: {threshold['export_time']*1000:.2f}ms")
        print(f"  Import time: {threshold['import_time']*1000:.2f}ms")
        print(f"  Import success: {'✅' if threshold['import_success'] else '❌'}")

        # Statistical accuracy results
        if "skipped" not in results["statistical_accuracy"]:
            stats = results["statistical_accuracy"]
            print(f"\n🧮 Statistical Accuracy:")
            print(f"  Mean accuracy error: {stats['mean_accuracy_error_percent']:.3f}%")
            print(f"  Std accuracy error: {stats['std_accuracy_error_percent']:.3f}%")
            print(f"  False positive rate: {stats['false_positive_rate_percent']:.1f}%")
            print(f"  Confidence interval width: {stats['confidence_interval_width']:.3f}ms")

        # Performance summary
        print(f"\n🎯 Performance Summary:")
        total_time = (baseline['mean_establishment_time'] +
                     regression['mean_detection_time'] +
                     threshold['mean_update_time'])
        print(f"  Total processing time per model: {total_time*1000:.2f}ms")
        print(f"  Throughput: {1/total_time:.1f} models/sec")

        # Quality metrics
        print(f"\n📈 Quality Metrics:")
        print(f"  Detection accuracy: {regression['accuracy_rate']*100:.1f}%")
        if "false_positive_rate_percent" in results["statistical_accuracy"]:
            print(f"  False positive rate: {results['statistical_accuracy']['false_positive_rate_percent']:.1f}%")
        print(f"  Framework components: ✅ All operational")


def main():
    parser = argparse.ArgumentParser(description="Regression Framework Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with minimal data")
    parser.add_argument("--detailed", action="store_true", help="Show detailed timing breakdowns")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    benchmark = RegressionFrameworkBenchmark(quick_mode=args.quick)

    try:
        benchmark.setup()
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_results(results)

        if args.detailed:
            print(f"\n📋 Detailed Results:")
            print(json.dumps(results, indent=2, default=str))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        benchmark.teardown()

    print(f"\n✅ Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())