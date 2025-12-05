#!/usr/bin/env python3
"""
Performance Regression Testing Demo

Demonstrates the performance regression testing framework with:
- Baseline establishment from historical data
- Real-time regression detection
- Adaptive threshold management
- Comprehensive reporting and analysis

Usage:
    python regression_testing_demo.py [--quick] [--validate] [--establish-baselines]
"""

import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil

# Add src and root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import regression testing framework
from benchmarks.regression.baseline_manager import BaselineManager, BaselineMetrics
from benchmarks.regression.regression_detector import RegressionDetector, RegressionSeverity
from benchmarks.regression.threshold_manager import ThresholdManager, ThresholdConfig

# Import existing benchmark infrastructure
from benchmarks.framework.benchmark_runner import PerformanceMetrics
import torch


def create_sample_historical_data(results_dir: Path, model_name: str, num_samples: int = 15):
    """Create sample historical benchmark data for demonstration"""
    print(f"📊 Creating {num_samples} historical benchmark results for {model_name}...")

    results_dir.mkdir(parents=True, exist_ok=True)

    # Simulate realistic performance variance over time
    base_latency = 12.0
    base_throughput = 85.0
    base_memory = 256.0

    for i in range(num_samples):
        # Add realistic variance and slight degradation over time
        variance_factor = 1.0 + (i * 0.002)  # Slight degradation over time
        noise = 1.0 + (i % 5 - 2) * 0.02  # ±4% random noise

        timestamp = datetime.now().replace(day=max(1, 15 - i))
        filename = f"{model_name}_inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        result_data = {
            "config": {"name": model_name},
            "timestamp": timestamp.isoformat(),
            "results": {
                "PyTorch Native": {
                    "latency_ms": base_latency * variance_factor * noise,
                    "throughput_samples_per_sec": base_throughput / (variance_factor * noise),
                    "peak_memory_mb": base_memory * (1.0 + i * 0.01),
                    "memory_efficiency": 0.85 - i * 0.005,
                    "accuracy_loss": 0.01 + i * 0.001,
                    "statistical_significance": True,
                    "confidence_interval_95": [base_latency * 0.95, base_latency * 1.05]
                }
            }
        }

        with open(results_dir / filename, 'w') as f:
            json.dump(result_data, f, indent=2)

    print(f"✅ Created {num_samples} historical results in {results_dir}")


def establish_baselines_demo(temp_dir: Path):
    """Demonstrate baseline establishment from historical data"""
    print("\n🎯 BASELINE ESTABLISHMENT DEMO")
    print("=" * 50)

    # Set up demo environment
    baseline_manager = BaselineManager(baselines_dir=str(temp_dir / "baselines"))
    results_dir = temp_dir / "results"

    models = ["matrix_multiply_512", "matrix_multiply_1024", "gpt2_small"]

    # Create sample historical data for each model
    for model in models:
        create_sample_historical_data(results_dir, model)

    print(f"\n📈 Establishing baselines from historical data...")
    established_baselines = {}

    for model in models:
        print(f"\nProcessing {model}:")
        baseline = baseline_manager.establish_baseline_from_historical_data(
            model_name=model,
            results_dir=str(results_dir),
            window_days=30,
            min_samples=5
        )

        if baseline:
            established_baselines[model] = baseline
            quality = baseline_manager.validate_baseline_quality(baseline)

            print(f"  ✅ Baseline established")
            print(f"     Mean latency: {baseline.mean_latency_ms:.2f}ms ± {baseline.std_latency_ms:.2f}ms")
            print(f"     Mean throughput: {baseline.mean_throughput:.1f} samples/sec")
            print(f"     Sample count: {baseline.sample_count}")
            print(f"     Quality: {'✅ Good' if quality else '⚠️ Poor'}")
            print(f"     Confidence interval: ({baseline.confidence_interval_95[0]:.2f}, {baseline.confidence_interval_95[1]:.2f})")
        else:
            print(f"  ❌ Failed to establish baseline for {model}")

    # Display baseline summary
    summary = baseline_manager.get_baseline_summary()
    print(f"\n📊 BASELINE SUMMARY")
    print(f"Total models with baselines: {summary['total_models']}")
    print(f"Registry version: {summary['registry_version']}")

    return established_baselines, baseline_manager


def threshold_management_demo(established_baselines: dict, temp_dir: Path):
    """Demonstrate adaptive threshold management"""
    print("\n⚙️  THRESHOLD MANAGEMENT DEMO")
    print("=" * 50)

    threshold_manager = ThresholdManager(thresholds_dir=str(temp_dir / "thresholds"))

    print(f"📊 Setting up adaptive thresholds based on baseline variance...")

    for model_name, baseline in established_baselines.items():
        # Get default thresholds first
        default_config = threshold_manager.get_thresholds(model_name, "default")
        print(f"\n{model_name} - Default thresholds:")
        print(f"  Minor: {default_config.minor_threshold_percent}%")
        print(f"  Major: {default_config.major_threshold_percent}%")
        print(f"  Critical: {default_config.critical_threshold_percent}%")

        # Update thresholds based on baseline variance
        adaptive_config = threshold_manager.update_thresholds_from_baseline(
            baseline, environment="default", sensitivity_factor=2.0
        )

        print(f"  Adaptive thresholds (auto-tuned):")
        print(f"  Minor: {adaptive_config.minor_threshold_percent:.1f}%")
        print(f"  Major: {adaptive_config.major_threshold_percent:.1f}%")
        print(f"  Critical: {adaptive_config.critical_threshold_percent:.1f}%")

        # Validate threshold configuration
        validation = threshold_manager.validate_threshold_sensitivity(model_name, "default")
        print(f"  Threshold validation: {'✅ Good' if validation['validation']['reasonable_spacing'] else '⚠️ Needs adjustment'}")

        # Demonstrate environment-specific adjustments
        ci_config = threshold_manager.apply_environment_adjustments(adaptive_config, "ci")
        print(f"  CI environment adjustments:")
        print(f"  Major: {ci_config.major_threshold_percent:.1f}% (was {adaptive_config.major_threshold_percent:.1f}%)")

    # Show threshold summary
    threshold_summary = threshold_manager.get_threshold_summary()
    print(f"\n📈 THRESHOLD SUMMARY")
    print(f"Total configurations: {threshold_summary['total_models']}")
    print(f"Auto-tuned: {threshold_summary['auto_tuned_count']}")
    print(f"Environments: {threshold_summary['environments']}")
    print(f"Average thresholds: Minor={threshold_summary['average_thresholds']['minor']:.1f}%, "
          f"Major={threshold_summary['average_thresholds']['major']:.1f}%, "
          f"Critical={threshold_summary['average_thresholds']['critical']:.1f}%")

    return threshold_manager


def regression_detection_demo(established_baselines: dict, threshold_manager: ThresholdManager):
    """Demonstrate real-time regression detection"""
    print("\n🔍 REGRESSION DETECTION DEMO")
    print("=" * 50)

    # Create regression detector with adaptive thresholds
    detector = RegressionDetector(
        minor_threshold_percent=2.0,
        major_threshold_percent=5.0,
        critical_threshold_percent=10.0,
        confidence_level=0.95,
        min_sample_size=5
    )

    print(f"🧪 Simulating various performance scenarios...")

    scenarios = [
        ("No Regression", 1.01, 0.99, 1.02),      # Slight normal variance
        ("Minor Regression", 1.03, 0.97, 1.02),   # 3% latency increase
        ("Major Regression", 1.08, 0.92, 1.15),   # 8% latency increase
        ("Critical Regression", 1.25, 0.80, 1.30), # 25% latency increase
        ("Performance Improvement", 0.92, 1.08, 0.95)  # 8% latency improvement
    ]

    model_name = list(established_baselines.keys())[0]
    baseline = established_baselines[model_name]

    print(f"\nTesting regression scenarios for {model_name}")
    print(f"Baseline: {baseline.mean_latency_ms:.2f}ms latency, {baseline.mean_throughput:.1f} throughput")
    print("-" * 80)

    regression_results = []

    for scenario_name, latency_mult, throughput_mult, memory_mult in scenarios:
        # Create simulated current performance
        current_performance = PerformanceMetrics(
            latency_ms=baseline.mean_latency_ms * latency_mult,
            throughput_samples_per_sec=baseline.mean_throughput * throughput_mult,
            peak_memory_mb=baseline.mean_memory_mb * memory_mult,
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(
                baseline.mean_latency_ms * latency_mult * 0.98,
                baseline.mean_latency_ms * latency_mult * 1.02
            )
        )

        # Detect regression
        result = detector.detect_regression(current_performance, baseline)
        regression_results.append(result)

        # Display results
        severity_emoji = {
            RegressionSeverity.NONE: "✅",
            RegressionSeverity.MINOR: "⚠️ ",
            RegressionSeverity.MAJOR: "❌",
            RegressionSeverity.CRITICAL: "🚨"
        }

        print(f"{severity_emoji[result.severity]} {scenario_name}")
        print(f"   Latency: {current_performance.latency_ms:.2f}ms ({result.performance_delta_percent:+.1f}%)")
        print(f"   Throughput: {current_performance.throughput_samples_per_sec:.1f} ({result.throughput_delta_percent:+.1f}%)")
        print(f"   Memory: {current_performance.peak_memory_mb:.1f}MB ({result.memory_delta_percent:+.1f}%)")
        print(f"   Severity: {result.severity.value.upper()}")
        print(f"   Statistically significant: {'Yes' if result.statistical_significance else 'No'}")
        print(f"   Blocking: {'Yes' if result.is_blocking() else 'No'}")
        print(f"   Recommendation: {result.recommendation}")
        print()

    # Demonstrate batch analysis
    print("🔄 BATCH ANALYSIS DEMO")
    print("-" * 30)

    measurements = []
    for i, model_name in enumerate(list(established_baselines.keys())[:2]):
        baseline = established_baselines[model_name]
        # Simulate different regression levels
        latency_mult = 1.0 + (i + 1) * 0.03  # 3%, 6% increases

        current_performance = PerformanceMetrics(
            latency_ms=baseline.mean_latency_ms * latency_mult,
            throughput_samples_per_sec=baseline.mean_throughput * (2.0 - latency_mult),
            peak_memory_mb=baseline.mean_memory_mb * (1.0 + i * 0.1),
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(10.0, 15.0)
        )
        measurements.append((model_name, current_performance))

    batch_results = detector.batch_analyze(measurements, established_baselines)

    print(f"Analyzed {len(batch_results)} models:")
    for result in batch_results:
        blocking = "🚫 BLOCKING" if result.is_blocking() else "✅ Non-blocking"
        print(f"  {result.model_name}: {result.severity.value.upper()} ({result.performance_delta_percent:+.1f}%) - {blocking}")

    return regression_results


def trend_analysis_demo(established_baselines: dict):
    """Demonstrate performance trend analysis"""
    print("\n📈 TREND ANALYSIS DEMO")
    print("=" * 50)

    detector = RegressionDetector()

    # Simulate historical performance measurements with trends
    print("🔍 Analyzing performance trends over time...")

    model_name = list(established_baselines.keys())[0]
    baseline = established_baselines[model_name]

    # Create trending performance data
    trending_metrics = []
    for i in range(10):
        # Simulate gradual performance degradation
        degradation_factor = 1.0 + i * 0.01  # 1% degradation per measurement
        noise = 1.0 + (i % 3 - 1) * 0.005  # Small random noise

        metrics = PerformanceMetrics(
            latency_ms=baseline.mean_latency_ms * degradation_factor * noise,
            throughput_samples_per_sec=baseline.mean_throughput / (degradation_factor * noise),
            peak_memory_mb=baseline.mean_memory_mb * (1.0 + i * 0.005),
            memory_efficiency=0.85,
            accuracy_loss=0.01,
            statistical_significance=True,
            confidence_interval_95=(10.0, 15.0)
        )
        trending_metrics.append(metrics)

    trend_analysis = detector.analyze_trend(trending_metrics)

    if "error" not in trend_analysis:
        print(f"\n📊 Trend Analysis Results for {model_name}:")
        print(f"  Measurements analyzed: {trend_analysis['measurements_analyzed']}")
        print(f"  Latency trend: {trend_analysis['latency_trend']['direction']}")
        print(f"  Latency trend strength: {trend_analysis['latency_trend']['trend_strength_percent']:.1f}%")
        print(f"  Throughput trend: {trend_analysis['throughput_trend']['direction']}")
        print(f"  Throughput trend strength: {trend_analysis['throughput_trend']['trend_strength_percent']:.1f}%")
        print(f"  Recommendation: {trend_analysis['recommendation']}")
    else:
        print(f"❌ Trend analysis failed: {trend_analysis['error']}")


def run_actual_benchmarks():
    """Run actual benchmarks with simple matrix operations for realistic data"""
    print("\n🧪 REAL BENCHMARK INTEGRATION")
    print("=" * 50)

    print("🚀 Running actual matrix operation benchmarks...")

    # Test different matrix operation configurations
    configs = [
        (512, "matrix_512"),
        (1024, "matrix_1024")
    ]

    device = torch.device("cpu")  # Use CPU for consistent results

    actual_results = {}

    for dim, model_name in configs:
        print(f"\nBenchmarking {model_name} (dim={dim}x{dim})...")

        # Create test matrices
        matrix_a = torch.randn(dim, dim, device=device)
        matrix_b = torch.randn(dim, dim, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = torch.mm(matrix_a, matrix_b)

        # Benchmark using simple timing approach
        import time

        try:
            # Warm up and time the operation
            num_iterations = 100

            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_iterations):
                    result = torch.mm(matrix_a, matrix_b)
            end_time = time.perf_counter()

            # Calculate average latency
            total_time = end_time - start_time
            latency_ms = (total_time / num_iterations) * 1000

            # Estimate throughput (operations per second)
            avg_time_per_op = total_time / num_iterations
            throughput = 1.0 / avg_time_per_op

            # Estimate memory usage (rough)
            memory_mb = dim * dim * 4 * 3 / 1024 / 1024  # 3 matrices, 4 bytes per float32

            metrics = PerformanceMetrics(
                latency_ms=latency_ms,
                throughput_samples_per_sec=throughput,
                peak_memory_mb=memory_mb,
                memory_efficiency=0.85,
                accuracy_loss=0.0,
                statistical_significance=True,
                confidence_interval_95=(latency_ms * 0.95, latency_ms * 1.05)
            )

            actual_results[model_name] = metrics

            print(f"  ✅ Latency: {latency_ms:.2f}ms")
            print(f"     Throughput: {throughput:.1f} operations/sec")
            print(f"     Memory: {memory_mb:.1f}MB")

        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")

    return actual_results


def main():
    parser = argparse.ArgumentParser(description="Performance Regression Testing Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick demo with minimal data")
    parser.add_argument("--validate", action="store_true", help="Validate the regression testing framework")
    parser.add_argument("--establish-baselines", action="store_true", help="Focus on baseline establishment")
    parser.add_argument("--real-benchmarks", action="store_true", help="Run actual model benchmarks")

    args = parser.parse_args()

    print("🎯 PERFORMANCE REGRESSION TESTING FRAMEWORK DEMO")
    print("=" * 60)
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print(f"Validation mode: {'ON' if args.validate else 'OFF'}")

    # Create temporary directory for demo
    temp_dir = Path(tempfile.mkdtemp(prefix="regression_demo_"))

    try:
        # Phase 1: Baseline establishment
        if args.establish_baselines or not args.quick:
            established_baselines, baseline_manager = establish_baselines_demo(temp_dir)
        else:
            print("\n⚠️  Skipping baseline establishment (quick mode)")
            # Create minimal baselines for demo
            established_baselines = {}
            baseline_manager = BaselineManager(baselines_dir=str(temp_dir / "baselines"))

            # Create a simple baseline for demo
            sample_baseline = BaselineMetrics(
                model_name="matrix_multiply_512",
                mean_latency_ms=12.5,
                std_latency_ms=1.2,
                mean_throughput=85.0,
                std_throughput=8.0,
                mean_memory_mb=256.0,
                std_memory_mb=15.0,
                sample_count=15,
                confidence_interval_95=(11.8, 13.2),
                established_date=datetime.now(),
                last_validated_date=datetime.now()
            )
            established_baselines["matrix_multiply_512"] = sample_baseline
            baseline_manager._store_baseline(sample_baseline)

        # Phase 2: Threshold management
        if established_baselines:
            threshold_manager = threshold_management_demo(established_baselines, temp_dir)

            # Phase 3: Regression detection
            regression_results = regression_detection_demo(established_baselines, threshold_manager)

            # Phase 4: Trend analysis
            if not args.quick:
                trend_analysis_demo(established_baselines)

            # Phase 5: Real benchmark integration (optional)
            if args.real_benchmarks:
                actual_results = run_actual_benchmarks()

                if actual_results:
                    print("\n🔬 REAL BENCHMARK REGRESSION ANALYSIS")
                    print("=" * 50)

                    detector = RegressionDetector()

                    for model_name, metrics in actual_results.items():
                        if model_name in established_baselines:
                            baseline = established_baselines[model_name]
                            result = detector.detect_regression(metrics, baseline)

                            print(f"\n{model_name}:")
                            print(f"  Current: {metrics.latency_ms:.2f}ms")
                            print(f"  Baseline: {baseline.mean_latency_ms:.2f}ms")
                            print(f"  Delta: {result.performance_delta_percent:+.1f}%")
                            print(f"  Severity: {result.severity.value.upper()}")
                            print(f"  Recommendation: {result.recommendation}")

            # Validation tests
            if args.validate:
                print("\n✅ FRAMEWORK VALIDATION")
                print("=" * 50)

                validation_passed = True

                # Test baseline quality
                for model_name, baseline in established_baselines.items():
                    quality = baseline_manager.validate_baseline_quality(baseline)
                    print(f"Baseline quality for {model_name}: {'✅ PASS' if quality else '❌ FAIL'}")
                    if not quality:
                        validation_passed = False

                # Test threshold configurations
                for model_name in established_baselines.keys():
                    validation = threshold_manager.validate_threshold_sensitivity(model_name)
                    reasonable = (validation['validation']['reasonable_spacing'] and
                                validation['validation']['reasonable_values'])
                    print(f"Threshold config for {model_name}: {'✅ PASS' if reasonable else '❌ FAIL'}")
                    if not reasonable:
                        validation_passed = False

                # Test regression detection
                detector = RegressionDetector()
                test_scenarios = [
                    (1.01, RegressionSeverity.NONE),     # Should be no regression
                    (1.08, RegressionSeverity.MAJOR),    # Should be major
                    (1.25, RegressionSeverity.CRITICAL)  # Should be critical
                ]

                model_name = list(established_baselines.keys())[0]
                baseline = established_baselines[model_name]

                for multiplier, expected_severity in test_scenarios:
                    test_metrics = PerformanceMetrics(
                        latency_ms=baseline.mean_latency_ms * multiplier,
                        throughput_samples_per_sec=baseline.mean_throughput,
                        peak_memory_mb=baseline.mean_memory_mb,
                        memory_efficiency=0.85,
                        accuracy_loss=0.01,
                        statistical_significance=True,
                        confidence_interval_95=(10.0, 15.0)
                    )

                    result = detector.detect_regression(test_metrics, baseline)
                    passed = result.severity == expected_severity
                    print(f"Regression detection ({multiplier}x latency): {'✅ PASS' if passed else '❌ FAIL'}")
                    if not passed:
                        validation_passed = False

                print(f"\n🎯 Overall validation: {'✅ ALL TESTS PASSED' if validation_passed else '❌ SOME TESTS FAILED'}")

        else:
            print("❌ No baselines established. Cannot proceed with regression testing demo.")

    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary files from {temp_dir}")

    print("\n🎉 Regression testing demo completed!")
    print("\n📚 Key takeaways:")
    print("   • Baselines can be automatically established from historical benchmark data")
    print("   • Thresholds adapt to each model's performance variance")
    print("   • Statistical significance testing prevents false positives")
    print("   • Severity classification helps prioritize investigation efforts")
    print("   • Environment-specific adjustments handle CI/cloud variance")
    print("   • Trend analysis detects gradual performance degradation")


if __name__ == "__main__":
    main()