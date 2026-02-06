#!/usr/bin/env python3
"""
Quick Benchmark Runner

Fast validation of the benchmark framework and our optimizations.
Perfect for CI/CD and quick development feedback.
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import time

import torch


def quick_environment_check():
    """Quick environment validation"""
    print("🔧 Quick Environment Check")
    print("-" * 30)

    # PyTorch
    print(f"   PyTorch: {torch.__version__}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")

    # torch.compile
    try:
        test_fn = lambda x: torch.relu(x)
        compiled_fn = torch.compile(test_fn)
        test_input = torch.randn(10, device=device)
        compiled_fn(test_input)
        print("   torch.compile: ✅ Working")
    except Exception as e:
        print(f"   torch.compile: ❌ Failed ({e})")

    # Our optimizations
    try:
        from torchbridge.compiler_optimized import FusedGELU
        print("   Our Optimizations: ✅ Available")
    except ImportError:
        print("   Our Optimizations: ⚠️  Limited")

    return device

def quick_performance_test(device):
    """Quick performance validation"""
    print("\n⚡ Quick Performance Test")
    print("-" * 30)

    # Test data
    batch_size, seq_len, hidden_size = 4, 256, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Standard PyTorch
    def standard_ops(x):
        linear = torch.nn.Linear(hidden_size, hidden_size * 2, device=device)
        return torch.nn.functional.gelu(linear(x))

    # Measure standard
    times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = standard_ops(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    standard_time = sum(times) / len(times)

    # Test compiled version
    try:
        compiled_ops = torch.compile(standard_ops)

        # Warmup
        for _ in range(5):
            compiled_ops(x)

        # Measure compiled
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = compiled_ops(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        compiled_time = sum(times) / len(times)
        speedup = standard_time / compiled_time

        print(f"   Standard PyTorch: {standard_time*1000:.2f}ms")
        print(f"   torch.compile: {compiled_time*1000:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x")

        return speedup > 1.1  # At least 10% improvement

    except Exception as e:
        print(f"   Compilation failed: {e}")
        return False

def quick_framework_test():
    """Quick test of our benchmark framework"""
    print("\n🏁 Framework Test")
    print("-" * 30)

    try:
        from framework.baseline_implementations import (
            PyTorchNativeBaseline,
            create_our_optimized_implementation,
        )
        from framework.benchmark_runner import (
            BenchmarkConfig,
            BenchmarkRunner,
            BenchmarkType,
            create_simple_gpt_config,
        )

        # Create simple runner
        runner = BenchmarkRunner()

        # Register minimal baselines
        runner.register_baseline(PyTorchNativeBaseline(runner.device))
        runner.register_optimized_implementation("Our Optimizations", create_our_optimized_implementation(runner.device))

        # Simple benchmark config
        config = BenchmarkConfig(
            name="Quick_Test",
            benchmark_type=BenchmarkType.INFERENCE,
            model_config=create_simple_gpt_config(hidden_size=256, num_layers=2, num_heads=4),
            batch_sizes=[1, 2],
            sequence_lengths=[128],
            num_trials=5,
            warmup_trials=2
        )

        print("   Running quick benchmark...")
        results = runner.run_comprehensive_benchmark(config)

        if results and len(results) >= 2:
            print(f"   ✅ Framework working - {len(results)} implementations tested")

            # Check if we have improvement
            if "Our Optimizations" in results and "PyTorch Native" in results:
                our_latency = results["Our Optimizations"].latency_ms
                baseline_latency = results["PyTorch Native"].latency_ms
                if our_latency > 0 and baseline_latency > 0:
                    speedup = baseline_latency / our_latency
                    print(f"   Speedup achieved: {speedup:.2f}x")
                elif our_latency == 0:
                    print("   ⚠️  Our optimization latency is zero - potential measurement issue")
                elif baseline_latency == 0:
                    print("   ⚠️  Baseline latency is zero - potential measurement issue")
                return True

            return True
        else:
            print("   ❌ Framework test failed")
            return False

    except Exception as e:
        print(f"   ❌ Framework test failed: {e}")
        return False

def main():
    """Run quick benchmark validation"""
    print("🚀 Quick Benchmark Framework Validation")
    print("=" * 50)

    start_time = time.time()

    # Environment check
    device = quick_environment_check()

    # Performance test
    perf_ok = quick_performance_test(device)

    # Framework test
    framework_ok = quick_framework_test()

    # Summary
    total_time = time.time() - start_time

    print("\n📊 Validation Summary")
    print("-" * 30)
    print("   Environment: ✅")
    print(f"   Performance: {'✅' if perf_ok else '⚠️'}")
    print(f"   Framework: {'✅' if framework_ok else '❌'}")
    print(f"   Total time: {total_time:.1f}s")

    if perf_ok and framework_ok:
        print("\n🎉 Benchmark framework ready for production use!")
        print("   Run: python3 benchmarks/run_comprehensive_benchmark.py --quick")
        return True
    else:
        print("\n⚠️  Some issues detected - check logs above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
