"""
Performance Regression Detection Demo - Stage 3B

Demonstrates performance tracking and automatic regression detection
to ensure optimizations don't degrade performance.

This demo shows:
1. Recording baseline performance metrics
2. Recording optimized performance metrics
3. Detecting performance regressions
4. Automatic warning on regressions
5. Performance history tracking
"""

import torch
import torch.nn as nn
from typing import List
import warnings

from torchbridge.core.performance_tracker import (
    get_performance_tracker,
    PerformanceMetrics,
    RegressionSeverity,
)
from torchbridge.core.management import get_manager

# Use shared utilities
from demos.shared.utils import print_section


# Test models
class SimpleLinearModel(nn.Module):
    """Simple linear model for testing."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class SlowLinearModel(nn.Module):
    """Deliberately slower model to simulate regression."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        # More layers = slower
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)


def print_metrics(metrics: PerformanceMetrics, label: str = "Metrics"):
    """Print performance metrics."""
    print(f"{label}:")
    print(f"  Model: {metrics.model_name}")
    print(f"  Backend: {metrics.backend}")
    print(f"  Optimization Level: {metrics.optimization_level}")
    print(f"  Latency: {metrics.latency_ms:.3f} ms")
    print(f"  Throughput: {metrics.throughput:.1f} samples/sec")
    print(f"  Memory: {metrics.memory_mb:.2f} MB")
    print(f"  Timestamp: {metrics.timestamp}")


def print_regression_results(regressions: List):
    """Print regression detection results."""
    if not regressions:
        print("✅ No regressions detected!")
        return

    print(f"⚠️  {len(regressions)} regression(s) detected:\n")

    for i, regression in enumerate(regressions, 1):
        severity_emoji = {
            RegressionSeverity.MINOR: "ℹ️",
            RegressionSeverity.MODERATE: "⚠️",
            RegressionSeverity.SEVERE: "🔴"
        }

        emoji = severity_emoji.get(regression.severity, "⚠️")
        print(f"{emoji} Regression {i}: {regression.severity.value.upper()}")
        print(f"   {regression.message}")
        print()


def demo_1_record_baseline():
    """Demo 1: Record Baseline Performance."""
    print_section("Demo 1: Record Baseline Performance")

    print("Creating a simple model...")
    model = SimpleLinearModel(input_dim=512, hidden_dim=1024, output_dim=512)
    sample_inputs = torch.randn(32, 512)

    print("Recording baseline performance...\n")

    tracker = get_performance_tracker()
    baseline_metrics = tracker.record_performance(
        model=model,
        sample_inputs=sample_inputs,
        model_name="SimpleLinearModel",
        backend="cpu",
        optimization_level="none"
    )

    print_metrics(baseline_metrics, "Baseline Performance")
    print("\n✅ Baseline performance recorded!")


def demo_2_detect_improvement():
    """Demo 2: Detect Performance Improvement."""
    print_section("Demo 2: Detect Performance Improvement")

    print("Creating the same model...")
    model = SimpleLinearModel(input_dim=512, hidden_dim=1024, output_dim=512)
    sample_inputs = torch.randn(32, 512)

    tracker = get_performance_tracker()

    # Get baseline
    baseline = tracker.get_baseline(model)
    if baseline:
        print("Found existing baseline:")
        print(f"  Latency: {baseline.latency_ms:.3f} ms\n")
    else:
        print("No baseline found, recording new baseline...\n")
        baseline = tracker.record_performance(
            model=model,
            sample_inputs=sample_inputs,
            model_name="SimpleLinearModel",
            backend="cpu",
            optimization_level="none"
        )

    print("Applying optimizations...")
    optimized_model = get_manager().auto_optimize(
        model,
        sample_inputs=sample_inputs,
        for_inference=True
    )

    print("Recording optimized performance...\n")
    optimized_metrics = tracker.record_performance(
        model=optimized_model,
        sample_inputs=sample_inputs,
        model_name="SimpleLinearModel_Optimized",
        backend="cpu",
        optimization_level="aggressive"
    )

    print_metrics(optimized_metrics, "Optimized Performance")

    # Note: We compare against baseline manually since models are different
    improvement_pct = (baseline.latency_ms - optimized_metrics.latency_ms) / baseline.latency_ms * 100

    if improvement_pct > 0:
        print(f"\n✅ Performance improved by {improvement_pct:.1f}%!")
    else:
        print(f"\nℹ️  Performance similar (expected on CPU)")


def demo_3_detect_regression():
    """Demo 3: Detect Performance Regression."""
    print_section("Demo 3: Detect Performance Regression")

    print("Simulating a performance regression...\n")

    # Use same model to track regressions
    tracker = get_performance_tracker()

    print("Creating baseline model...")
    model = SimpleLinearModel(input_dim=256, hidden_dim=512, output_dim=256)
    sample_inputs = torch.randn(16, 256)

    # Clear previous history for this model
    tracker.clear_history(model)

    print("Recording baseline performance...")
    baseline_metrics = tracker.record_performance(
        model=model,
        sample_inputs=sample_inputs,
        model_name="RegressionTest",
        backend="cpu",
        optimization_level="conservative"
    )

    print_metrics(baseline_metrics, "Baseline")

    # Simulate regression by using a slower model with same structure
    print("\nApplying 'optimization' (actually slower)...")
    slow_model = SimpleLinearModel(input_dim=256, hidden_dim=512, output_dim=256)

    # Make it slower by running it multiple times
    def slow_forward_wrapper(original_forward):
        def wrapper(*args, **kwargs):
            # Call original forward multiple times to simulate slowdown
            result = original_forward(*args, **kwargs)
            _ = original_forward(*args, **kwargs)  # Extra call
            return result
        return wrapper

    slow_model.forward = slow_forward_wrapper(slow_model.forward)

    print("Recording 'optimized' (actually regressed) performance...")
    regressed_metrics = tracker.record_performance(
        model=slow_model,
        sample_inputs=sample_inputs,
        model_name="RegressionTest_Slow",
        backend="cpu",
        optimization_level="aggressive"
    )

    print_metrics(regressed_metrics, "After 'Optimization'")

    # Detect regression (compare using same model structure)
    print("\nChecking for regressions...")
    regressions = tracker.detect_regression(model, regressed_metrics, baseline_metrics)

    print_regression_results(regressions)


def demo_4_automatic_warning():
    """Demo 4: Automatic Warning on Regression."""
    print_section("Demo 4: Automatic Warning on Regression")

    print("Demonstrating automatic regression warnings...\n")

    tracker = get_performance_tracker()
    model = SimpleLinearModel(input_dim=128, hidden_dim=256, output_dim=128)
    sample_inputs = torch.randn(8, 128)

    # Clear history
    tracker.clear_history(model)

    print("Recording baseline...")
    baseline = tracker.record_performance(
        model=model,
        sample_inputs=sample_inputs,
        model_name="WarningTest",
        backend="cpu",
        optimization_level="conservative"
    )

    # Create a slow version
    slow_model = SimpleLinearModel(input_dim=128, hidden_dim=256, output_dim=128)

    def very_slow_forward(original_forward):
        def wrapper(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            # Make it much slower
            for _ in range(3):
                _ = original_forward(*args, **kwargs)
            return result
        return wrapper

    slow_model.forward = very_slow_forward(slow_model.forward)

    print("Recording regressed performance...")
    regressed = tracker.record_performance(
        model=slow_model,
        sample_inputs=sample_inputs,
        model_name="WarningTest_Slow",
        backend="cpu",
        optimization_level="aggressive"
    )

    print("\nChecking for regressions (will issue warnings)...\n")

    # This should issue warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tracker.warn_if_regression(model, regressed, baseline)

        if w:
            print(f"Caught {len(w)} warning(s):")
            for warning in w:
                print(f"  {warning.message}")
        else:
            print("No warnings issued (performance may be similar)")


def demo_5_performance_history():
    """Demo 5: Performance History Tracking."""
    print_section("Demo 5: Performance History Tracking")

    print("Tracking performance over time...\n")

    tracker = get_performance_tracker()
    model = SimpleLinearModel(input_dim=64, hidden_dim=128, output_dim=64)
    sample_inputs = torch.randn(4, 64)

    # Clear history
    tracker.clear_history(model)

    print("Recording multiple performance metrics...")

    optimization_levels = ["none", "conservative", "balanced", "aggressive"]

    for level in optimization_levels:
        print(f"  Recording with {level} optimization...")
        tracker.record_performance(
            model=model,
            sample_inputs=sample_inputs,
            model_name=f"HistoryTest_{level}",
            backend="cpu",
            optimization_level=level
        )

    print("\nRetrieving performance history:")
    history = tracker.get_performance_history(model, limit=10)

    print(f"\nFound {len(history)} records (most recent first):\n")

    for i, metrics in enumerate(history, 1):
        print(f"{i}. {metrics.model_name}")
        print(f"   Level: {metrics.optimization_level}")
        print(f"   Latency: {metrics.latency_ms:.3f} ms")
        print(f"   Throughput: {metrics.throughput:.1f} samples/sec")
        print()

    print("✅ Performance history retrieved!")


def demo_6_comparison_chart():
    """Demo 6: Performance Comparison Chart."""
    print_section("Demo 6: Performance Comparison")

    print("Comparing multiple optimization levels...\n")

    tracker = get_performance_tracker()
    model = SimpleLinearModel(input_dim=256, hidden_dim=512, output_dim=256)
    sample_inputs = torch.randn(16, 256)

    # Clear history
    tracker.clear_history(model)

    results = {}

    for level in ["conservative", "balanced", "aggressive"]:
        print(f"Benchmarking {level} optimization...")
        metrics = tracker.record_performance(
            model=model,
            sample_inputs=sample_inputs,
            model_name=f"Comparison_{level}",
            backend="cpu",
            optimization_level=level
        )
        results[level] = metrics

    # Print comparison table
    print("\n" + "="*70)
    print("Performance Comparison Table")
    print("="*70)
    print(f"{'Level':<15} {'Latency (ms)':<15} {'Throughput':<20} {'Memory (MB)':<15}")
    print("-"*70)

    for level, metrics in results.items():
        print(f"{level:<15} {metrics.latency_ms:>12.3f}   {metrics.throughput:>17.1f}   {metrics.memory_mb:>12.2f}")

    print("="*70)
    print("\n✅ Comparison complete!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  Performance Regression Detection Demo - Stage 3B")
    print("  Automatic Performance Tracking and Regression Detection")
    print("="*70)

    try:
        # Run all demos
        demo_1_record_baseline()
        demo_2_detect_improvement()
        demo_3_detect_regression()
        demo_4_automatic_warning()
        demo_5_performance_history()
        demo_6_comparison_chart()

        # Final summary
        print_section("Summary")
        print("✅ All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  1. Baseline performance recording")
        print("  2. Performance improvement detection")
        print("  3. Automatic regression detection")
        print("  4. Automatic warning on regressions")
        print("  5. Performance history tracking")
        print("  6. Performance comparison across optimization levels")

        print("\nUsage Summary:")
        print("  # Record performance")
        print("  tracker = get_performance_tracker()")
        print("  metrics = tracker.record_performance(model, inputs, 'model_name')")
        print()
        print("  # Detect regressions")
        print("  regressions = tracker.detect_regression(model, current_metrics)")
        print()
        print("  # Automatic warning")
        print("  tracker.warn_if_regression(model, current_metrics)")

        print("\nBenefit:")
        print("  Automatically catch performance regressions before they reach production!")

        print("\n" + "="*70)
        print("  Demo Complete!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
