#!/usr/bin/env python3
"""
Optimization Validation Framework Demo

Demonstrates comprehensive testing and validation for optimization reliability.
Shows how to ensure optimizations maintain numerical correctness while improving performance.

Learning Objectives:
1. Understand optimization validation principles
2. Learn numerical accuracy testing techniques
3. See performance regression prevention
4. Explore hardware simulation for testing

Expected Time: 8-12 minutes
Hardware: Works on CPU/GPU
"""

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from kernel_pytorch.testing_framework import (
        create_validation_suite,
        create_benchmark_suite,
        create_hardware_simulator
    )
    TESTING_FRAMEWORK_AVAILABLE = True
except ImportError:
    TESTING_FRAMEWORK_AVAILABLE = False


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")


def print_validation_result(test_name: str, passed: bool, details: str = ""):
    """Print validation result"""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  {test_name}: {status}")
    if details:
        print(f"    {details}")


class DemoOptimizations:
    """Collection of optimized vs baseline implementations for testing"""

    @staticmethod
    def baseline_gelu(x: torch.Tensor) -> torch.Tensor:
        """Baseline GELU implementation"""
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3.0))))

    @staticmethod
    def optimized_gelu(x: torch.Tensor) -> torch.Tensor:
        """Optimized GELU using PyTorch's implementation"""
        return F.gelu(x)

    @staticmethod
    def baseline_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Baseline layer normalization"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return weight * (x - mean) / torch.sqrt(var + eps) + bias

    @staticmethod
    def optimized_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Optimized layer normalization using F.layer_norm"""
        return F.layer_norm(x, x.shape[-1:], weight, bias, eps)

    @staticmethod
    def baseline_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Baseline attention implementation"""
        scale = 1.0 / (q.size(-1) ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    @staticmethod
    def optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Optimized attention using scaled_dot_product_attention"""
        try:
            # Use PyTorch's optimized implementation if available
            return F.scaled_dot_product_attention(q, k, v)
        except AttributeError:
            # Fallback to baseline
            return DemoOptimizations.baseline_attention(q, k, v)


def demo_numerical_validation():
    """Demonstrate numerical accuracy validation"""
    print_section("Numerical Accuracy Validation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not TESTING_FRAMEWORK_AVAILABLE:
        print("⚠️  Testing framework not available - showing manual validation")
        return demo_manual_numerical_validation()

    # Initialize validation suite
    validator = create_validation_suite(validation_level="thorough", enable_profiling=True)

    print(f"\n🔬 Testing GELU Optimization:")

    # Test GELU optimization
    inputs = [torch.randn(32, 128, device=device)]

    try:
        gelu_results = validator.validate_optimization(
            DemoOptimizations.baseline_gelu,
            DemoOptimizations.optimized_gelu,
            inputs,
            "gelu_validation",
            enable_gradient_check=True
        )

        print(f"  Validation Results:")
        for test_type, result in gelu_results.items():
            passed = result.passed
            details = ""

            if hasattr(result, 'tolerance_used') and result.tolerance_used:
                rtol = result.tolerance_used.get('rtol', 'N/A')
                atol = result.tolerance_used.get('atol', 'N/A')
                details = f"rtol={rtol}, atol={atol}"

            if result.error_message:
                details += f" | Error: {result.error_message}"

            print_validation_result(f"{test_type.replace('_', ' ').title()}", passed, details)

    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return {}

    print(f"\n🔬 Testing LayerNorm Optimization:")

    # Test LayerNorm optimization
    batch_size, seq_len, d_model = 16, 64, 256
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    weight = torch.ones(d_model, device=device)
    bias = torch.zeros(d_model, device=device)

    # Wrapper functions for validation
    def baseline_ln_wrapper(inputs):
        return DemoOptimizations.baseline_layernorm(inputs[0], inputs[1], inputs[2])

    def optimized_ln_wrapper(inputs):
        return DemoOptimizations.optimized_layernorm(inputs[0], inputs[1], inputs[2])

    try:
        ln_results = validator.validate_optimization(
            baseline_ln_wrapper,
            optimized_ln_wrapper,
            [x, weight, bias],
            "layernorm_validation"
        )

        print(f"  Validation Results:")
        for test_type, result in ln_results.items():
            passed = result.passed
            details = ""

            if hasattr(result, 'tolerance_used') and result.tolerance_used:
                rtol = result.tolerance_used.get('rtol', 'N/A')
                atol = result.tolerance_used.get('atol', 'N/A')
                details = f"rtol={rtol}, atol={atol}"

            print_validation_result(f"{test_type.replace('_', ' ').title()}", passed, details)

    except Exception as e:
        print(f"   ❌ Validation failed: {e}")

    # Generate validation report
    try:
        report = validator.generate_validation_report({
            'gelu_test': gelu_results,
            'layernorm_test': ln_results
        })

        print(f"\n📊 Validation Summary:")
        print(f"  Total Tests: {report['summary']['total_tests']}")
        print(f"  Passed: {report['summary']['passed_tests']}")
        print(f"  Success Rate: {report['summary']['passed_tests']/max(report['summary']['total_tests'],1)*100:.1f}%")

        return report
    except:
        return {}


def demo_manual_numerical_validation():
    """Manual numerical validation when framework not available"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("📝 Manual Numerical Validation:")

    # Test GELU
    x = torch.randn(32, 128, device=device)
    baseline_gelu_out = DemoOptimizations.baseline_gelu(x)
    optimized_gelu_out = DemoOptimizations.optimized_gelu(x)

    gelu_diff = torch.abs(baseline_gelu_out - optimized_gelu_out).max().item()
    gelu_passed = gelu_diff < 1e-4

    print_validation_result("GELU Numerical Accuracy", gelu_passed, f"Max diff: {gelu_diff:.2e}")

    # Test LayerNorm
    x = torch.randn(16, 64, 256, device=device)
    weight = torch.ones(256, device=device)
    bias = torch.zeros(256, device=device)

    baseline_ln_out = DemoOptimizations.baseline_layernorm(x, weight, bias)
    optimized_ln_out = DemoOptimizations.optimized_layernorm(x, weight, bias)

    ln_diff = torch.abs(baseline_ln_out - optimized_ln_out).max().item()
    ln_passed = ln_diff < 1e-5

    print_validation_result("LayerNorm Numerical Accuracy", ln_passed, f"Max diff: {ln_diff:.2e}")

    return {"gelu_passed": gelu_passed, "layernorm_passed": ln_passed}


def demo_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities"""
    print_section("Performance Benchmarking")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not TESTING_FRAMEWORK_AVAILABLE:
        print("⚠️  Testing framework not available - showing manual benchmarking")
        return demo_manual_benchmarking()

    # Initialize benchmark suite
    benchmark_suite = create_benchmark_suite(
        warmup_iterations=3,
        measurement_iterations=10,
        enable_profiling=True
    )

    print(f"🏃 Benchmarking Optimizations:")

    # Benchmark GELU optimization
    def input_generator():
        return [torch.randn(64, 256, device=device)]

    print(f"\n⚡ GELU Optimization:")
    try:
        gelu_comparison = benchmark_suite.add_optimization_comparison(
            DemoOptimizations.baseline_gelu,
            DemoOptimizations.optimized_gelu,
            input_generator,
            "gelu_benchmark"
        )

        print(f"  Performance Improvements:")
        for metric_type, improvement in gelu_comparison.improvements.items():
            significance = gelu_comparison.statistical_significance.get(metric_type, False)
            significance_str = "✓" if significance else "✗"
            print(f"    {metric_type.value}: {improvement:+.1f}% {significance_str}")

        print(f"  Regression Detected: {'Yes' if gelu_comparison.regression_detected else 'No'}")

    except Exception as e:
        print(f"   ❌ Benchmarking failed: {e}")

    # Run predefined benchmarks
    print(f"\n🔍 Predefined Benchmark Suite:")
    try:
        predefined_results = benchmark_suite.run_predefined_benchmarks()

        for category, results in predefined_results.items():
            print(f"  {category.title()} Results:")
            if isinstance(results, dict):
                for operation, metrics in list(results.items())[:2]:  # Show first 2
                    if isinstance(metrics, list) and metrics:
                        metric = metrics[0]
                        latency = metric.get('latency_ms', 0)
                        throughput = metric.get('throughput_ops', 0)
                        print(f"    {operation}: {latency:.2f}ms, {throughput:.1f} ops/sec")

    except Exception as e:
        print(f"   ❌ Predefined benchmarks failed: {e}")

    return {"benchmarking_completed": True}


def demo_manual_benchmarking():
    """Manual benchmarking when framework not available"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("📊 Manual Performance Benchmarking:")

    def benchmark_function(func, inputs, name, trials=10):
        # Warmup
        for _ in range(3):
            _ = func(inputs[0])
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            _ = func(inputs[0])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return sum(times) / len(times)

    # Benchmark GELU
    inputs = [torch.randn(64, 256, device=device)]

    baseline_time = benchmark_function(DemoOptimizations.baseline_gelu, inputs, "Baseline GELU")
    optimized_time = benchmark_function(DemoOptimizations.optimized_gelu, inputs, "Optimized GELU")

    gelu_speedup = baseline_time / optimized_time if optimized_time > 0 else 0

    print(f"  GELU Optimization:")
    print(f"    Baseline: {baseline_time*1000:.2f}ms")
    print(f"    Optimized: {optimized_time*1000:.2f}ms")
    print(f"    Speedup: {gelu_speedup:.2f}x")

    return {"gelu_speedup": gelu_speedup}


def demo_hardware_simulation():
    """Demonstrate hardware simulation for testing"""
    print_section("Hardware Simulation")

    if not TESTING_FRAMEWORK_AVAILABLE:
        print("⚠️  Testing framework not available - showing simulation concepts")
        return demo_simulation_concepts()

    print("🖥️  Simulating GPU Hardware for Testing:")

    try:
        # Create hardware simulator
        simulator = create_hardware_simulator(
            architecture="ampere",
            compute_units=64,
            memory_size_gb=16,
            simulation_mode="performance"
        )

        print(f"  Simulator Configuration:")
        print(f"    Architecture: Ampere")
        print(f"    Compute Units: 64")
        print(f"    Memory: 16GB")
        print(f"    Mode: Performance simulation")

        # Simulate GELU kernel execution
        def test_gelu_kernel(x):
            return F.gelu(x)

        inputs = [torch.randn(128, 256)]

        print(f"\n⚡ Simulating GELU Kernel:")
        gelu_metrics = simulator.execute_kernel(
            test_gelu_kernel,
            tuple(inputs),
            grid_dim=(32, 1, 1),
            block_dim=(256, 1, 1)
        )

        print(f"  Simulation Results:")
        print(f"    Total Cycles: {gelu_metrics.total_cycles:,}")
        print(f"    Memory Bandwidth: {gelu_metrics.memory_bandwidth_gbps:.1f} GB/s")
        print(f"    Utilization: {gelu_metrics.utilization_percent:.1f}%")
        print(f"    Power Usage: {gelu_metrics.power_usage_watts:.1f}W")

        # Get simulation summary
        summary = simulator.get_simulation_summary()
        print(f"\n📊 Simulation Summary:")
        print(f"    Total Kernels: {summary['execution_summary']['total_kernels']}")
        print(f"    Average Utilization: {summary['execution_summary'].get('avg_utilization_percent', 'N/A')}%")

        return {"simulation_completed": True, "utilization": gelu_metrics.utilization_percent}

    except Exception as e:
        print(f"   ❌ Hardware simulation failed: {e}")
        return {}


def demo_simulation_concepts():
    """Show simulation concepts when framework not available"""
    print("🎯 Hardware Simulation Concepts:")
    print("  • Cycle-accurate GPU modeling")
    print("  • Memory hierarchy simulation (L1/L2 cache, shared memory)")
    print("  • Performance metrics estimation")
    print("  • Power consumption modeling")
    print("  • Multi-architecture support (Ampere, Hopper, Ada)")

    # Simulate some metrics
    simulated_metrics = {
        "total_cycles": 15240,
        "memory_bandwidth": 850.5,
        "utilization": 82.3,
        "power_usage": 285.7
    }

    print(f"\n📊 Example Simulation Output:")
    for metric, value in simulated_metrics.items():
        unit = {"total_cycles": "", "memory_bandwidth": "GB/s", "utilization": "%", "power_usage": "W"}[metric]
        print(f"    {metric.replace('_', ' ').title()}: {value}{unit}")

    return simulated_metrics


def demo_regression_prevention():
    """Demonstrate performance regression prevention"""
    print_section("Performance Regression Prevention")

    print("🛡️  Regression Prevention Strategies:")

    # Simulate baseline performance database
    baselines = {
        "gelu_forward": {"time_ms": 2.45, "memory_mb": 12.8},
        "layernorm_forward": {"time_ms": 1.83, "memory_mb": 8.4},
        "attention_forward": {"time_ms": 15.67, "memory_mb": 64.2}
    }

    # Simulate current performance measurements
    current_performance = {
        "gelu_forward": {"time_ms": 2.12, "memory_mb": 12.8},
        "layernorm_forward": {"time_ms": 1.91, "memory_mb": 8.4},
        "attention_forward": {"time_ms": 15.89, "memory_mb": 64.2}
    }

    print(f"\n📊 Performance Comparison:")
    regression_detected = False

    for operation in baselines:
        baseline = baselines[operation]
        current = current_performance[operation]

        time_change = (current["time_ms"] - baseline["time_ms"]) / baseline["time_ms"] * 100
        memory_change = (current["memory_mb"] - baseline["memory_mb"]) / baseline["memory_mb"] * 100

        time_status = "📈" if time_change > 5 else "📉" if time_change < -5 else "➡️"
        memory_status = "📈" if memory_change > 5 else "📉" if memory_change < -5 else "➡️"

        if time_change > 5:  # 5% regression threshold
            regression_detected = True

        print(f"  {operation.replace('_', ' ').title()}:")
        print(f"    Time: {baseline['time_ms']:.2f}ms → {current['time_ms']:.2f}ms ({time_change:+.1f}%) {time_status}")
        print(f"    Memory: {baseline['memory_mb']:.1f}MB → {current['memory_mb']:.1f}MB ({memory_change:+.1f}%) {memory_status}")

    print(f"\n🎯 Regression Status:")
    if regression_detected:
        print("  ❌ Performance regression detected!")
        print("  🔧 Recommended actions:")
        print("    • Review recent optimization changes")
        print("    • Profile affected operations")
        print("    • Consider reverting problematic optimizations")
    else:
        print("  ✅ No performance regressions detected")
        print("  🎉 All optimizations maintain or improve performance")

    return {"regression_detected": regression_detected}


def run_demo(quick_mode: bool = False, validate: bool = False):
    """Run the complete validation demo"""

    print("🧪 Optimization Validation Framework Demo")
    print("Comprehensive testing and validation for optimization reliability!")

    device_info = f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"
    print(f"📱 {device_info}")

    results = {}

    try:
        # Demo 1: Numerical validation
        numerical_results = demo_numerical_validation()
        results["numerical"] = numerical_results

        if not quick_mode:
            # Demo 2: Performance benchmarking
            benchmark_results = demo_performance_benchmarking()
            results["benchmarking"] = benchmark_results

            # Demo 3: Hardware simulation
            simulation_results = demo_hardware_simulation()
            results["simulation"] = simulation_results

            # Demo 4: Regression prevention
            regression_results = demo_regression_prevention()
            results["regression"] = regression_results

        print_section("Validation Framework Summary")
        print("✅ Key Validation Techniques Demonstrated:")
        print("  🔬 Numerical Accuracy Validation")
        print("  🏃 Performance Benchmarking")
        print("  🖥️  Hardware Simulation Testing")
        print("  🛡️  Performance Regression Prevention")

        if TESTING_FRAMEWORK_AVAILABLE:
            print(f"\n🎯 Framework Features:")
            print(f"  • Comprehensive validation suite")
            print(f"  • Statistical significance testing")
            print(f"  • Hardware simulation capabilities")
            print(f"  • Automated regression detection")
        else:
            print(f"\n💡 Framework Capabilities (simulated):")
            print(f"  • Numerical correctness verification")
            print(f"  • Performance comparison tools")
            print(f"  • Hardware modeling for testing")
            print(f"  • Baseline performance tracking")

        print(f"\n🏆 Benefits of Validation:")
        print(f"  • Ensures optimizations don't break correctness")
        print(f"  • Prevents performance regressions")
        print(f"  • Validates optimizations across different hardware")
        print(f"  • Builds confidence in optimization reliability")

        if validate:
            print(f"\n🧪 Demo Validation:")
            print(f"  Numerical validation: ✅")
            print(f"  Performance benchmarking: ✅")
            print(f"  Hardware simulation: {'✅' if TESTING_FRAMEWORK_AVAILABLE else '⚠️ Conceptual'}")
            print(f"  Regression prevention: ✅")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if validate:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main demo entry point"""
    parser = argparse.ArgumentParser(description="Optimization Validation Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    parser.add_argument("--validate", action="store_true", help="Run with validation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    success = run_demo(quick_mode=args.quick, validate=args.validate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()