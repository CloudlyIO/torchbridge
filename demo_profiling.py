#!/usr/bin/env python3
"""
Performance Profiling Demo

Demonstrates comprehensive profiling and benchmarking tools for
analyzing kernel performance, memory usage, and optimization effectiveness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from kernel_pytorch.components.basic_optimized import (
    OptimizedTransformerBlock,
    OptimizedMultiHeadAttention,
    OptimizedLayerNorm,
    OptimizedMLP
)

from kernel_pytorch.utils.profiling import KernelProfiler, quick_benchmark, compare_functions

print("📊 Performance Profiling and Benchmarking Demo")
print("="*60)

def demonstrate_basic_profiling():
    """Show basic profiling capabilities"""
    print("\n🔍 BASIC PROFILING")
    print("-" * 30)

    # Create profiler
    profiler = KernelProfiler(device="cpu")

    # Test operation
    def matrix_multiply(x, y):
        return torch.matmul(x, y)

    # Sample data
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)

    print("   Profiling matrix multiplication...")

    # Profile with context manager
    with profiler.profile("matrix_multiply"):
        result = matrix_multiply(a, b)

    # Get results
    results = profiler.results["matrix_multiply"][0]
    print(f"   ⏱️  Time taken: {results['time']*1000:.2f}ms")
    print(f"   💾 Memory used: {results['memory_used']/(1024*1024):.2f}MB")
    print(f"   ✅ Output shape: {result.shape}")


def demonstrate_function_comparison():
    """Compare different implementations"""
    print("\n⚖️  FUNCTION COMPARISON")
    print("-" * 30)

    # Define different layer norm implementations
    def pytorch_layernorm(x):
        return F.layer_norm(x, (x.shape[-1],))

    def manual_layernorm(x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)

    def optimized_layernorm(x):
        norm = OptimizedLayerNorm(x.shape[-1])
        return norm(x)

    implementations = {
        "PyTorch Built-in": pytorch_layernorm,
        "Manual Implementation": manual_layernorm,
        "Optimized Component": optimized_layernorm
    }

    # Test data
    x = torch.randn(32, 128, 512)

    print(f"   Comparing LayerNorm implementations...")
    print(f"   Input shape: {x.shape}")

    # Compare implementations
    results = compare_functions(implementations, x)

    print(f"\n   📈 Performance Results:")
    baseline_time = None
    for name, stats in results.items():
        if baseline_time is None:
            baseline_time = stats['mean_time']
        speedup = baseline_time / stats['mean_time']
        print(f"      {name:20s}: {stats['mean_time']*1000:6.2f}ms ({speedup:.2f}x speedup)")


def demonstrate_model_benchmarking():
    """Benchmark complete model performance"""
    print("\n🏎️  MODEL BENCHMARKING")
    print("-" * 30)

    profiler = KernelProfiler()

    # Create models
    small_model = OptimizedTransformerBlock(256, 8)
    large_model = OptimizedTransformerBlock(512, 16)

    models = {
        "Small Model (256d)": small_model,
        "Large Model (512d)": large_model
    }

    # Test inputs
    inputs = {
        "Small Model (256d)": torch.randn(4, 64, 256),
        "Large Model (512d)": torch.randn(4, 64, 512)
    }

    print("   Benchmarking transformer models...")

    for name, model in models.items():
        input_tensor = inputs[name]

        print(f"\n   {name}:")
        print(f"      Input shape: {input_tensor.shape}")

        # Benchmark
        stats = profiler.benchmark_function(
            model,
            args=(input_tensor,),
            num_iterations=30,
            name=name.lower().replace(" ", "_")
        )

        # Calculate throughput
        tokens_per_sec = input_tensor.numel() / stats['mean_time']
        params = sum(p.numel() for p in model.parameters())

        print(f"      ⏱️  Time: {stats['mean_time']*1000:.2f}ms")
        print(f"      🚀 Throughput: {tokens_per_sec/1e6:.1f}M tokens/sec")
        print(f"      🔢 Parameters: {params/1e6:.2f}M")


def demonstrate_scaling_analysis():
    """Analyze performance scaling with input size"""
    print("\n📈 SCALING ANALYSIS")
    print("-" * 30)

    profiler = KernelProfiler()
    model = OptimizedTransformerBlock(256, 8)

    def run_model(x):
        return model(x)

    # Test different input sizes
    input_sizes = [
        (2, 32, 256),   # Small
        (4, 64, 256),   # Medium
        (8, 128, 256),  # Large
        (16, 256, 256)  # Extra large
    ]

    print("   Analyzing performance scaling...")

    results = profiler.analyze_kernel_efficiency(
        run_model,
        input_sizes,
        "transformer_scaling"
    )

    print(f"\n   📊 Scaling Results:")
    for data in results['scaling_data']:
        batch, seq, dim = data['input_size']
        print(f"      [{batch:2d}, {seq:3d}, {dim}]: {data['mean_time']*1000:6.2f}ms "
              f"({data['throughput_gelements_per_sec']:.2f} GElem/s)")

    # Show scaling pattern
    analysis = results['analysis']
    print(f"\n   🔍 Analysis:")
    print(f"      Pattern: {analysis['pattern']}")
    print(f"      Efficiency: {analysis['efficiency']}")
    print(f"      Time scaling slope: {analysis['time_scaling_slope']:.2f}")


def demonstrate_memory_profiling():
    """Show memory usage patterns"""
    print("\n💾 MEMORY PROFILING")
    print("-" * 30)

    # Memory-intensive operations
    def create_large_tensor():
        return torch.randn(1000, 1000)

    def memory_operations():
        tensors = []
        for i in range(5):
            tensor = torch.randn(200, 200)
            tensors.append(tensor)
        return torch.stack(tensors)

    print("   Analyzing memory patterns...")

    profiler = KernelProfiler()

    # Profile memory usage
    with profiler.profile("large_tensor_creation"):
        large_tensor = create_large_tensor()

    with profiler.profile("multiple_allocations"):
        stacked_tensor = memory_operations()

    # Show results
    for operation in ["large_tensor_creation", "multiple_allocations"]:
        if operation in profiler.results:
            result = profiler.results[operation][0]
            print(f"   {operation.replace('_', ' ').title()}:")
            print(f"      Time: {result['time']*1000:.2f}ms")
            print(f"      Memory: {result['memory_used']/(1024*1024):.2f}MB")


def demonstrate_optimization_recommendations():
    """Show automated optimization recommendations"""
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 30)

    profiler = KernelProfiler()

    # Run some operations to generate data
    model = OptimizedTransformerBlock(128, 4)

    for i in range(10):
        x = torch.randn(2, 32, 128)
        with profiler.profile(f"test_operation_{i}"):
            _ = model(x)

    # Generate report
    report = profiler.generate_report()

    # Extract just the recommendations part
    lines = report.split('\n')
    recommendations_start = False
    recommendations = []

    for line in lines:
        if "OPTIMIZATION RECOMMENDATIONS" in line:
            recommendations_start = True
            recommendations.append(line)
        elif recommendations_start:
            recommendations.append(line)
            if line.strip() == "" and len(recommendations) > 5:
                break

    print('\n'.join(recommendations))


def demonstrate_advanced_concepts():
    """Show advanced profiling concepts"""
    print("\n🧪 ADVANCED CONCEPTS")
    print("-" * 30)

    print("   1. Compute vs Memory Bound Analysis:")

    # Compute-bound operation
    def compute_heavy(x):
        for _ in range(100):
            x = torch.matmul(x, x.transpose(-2, -1))
        return x

    # Memory-bound operation
    def memory_heavy(x):
        results = []
        for i in range(100):
            results.append(x + i)
        return torch.stack(results)

    x = torch.randn(64, 64)

    compute_stats = quick_benchmark(compute_heavy, x)
    memory_stats = quick_benchmark(memory_heavy, x)

    print(f"      Compute-heavy: {compute_stats['mean_time']*1000:.2f}ms")
    print(f"      Memory-heavy:  {memory_stats['mean_time']*1000:.2f}ms")

    print("\n   2. Batch Size Impact:")

    model = OptimizedMultiHeadAttention(256, 8)

    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 64, 256)
        stats = quick_benchmark(model, x)
        per_sample_time = stats['mean_time'] / batch_size
        print(f"      Batch {batch_size:2d}: {stats['mean_time']*1000:6.2f}ms "
              f"({per_sample_time*1000:.2f}ms/sample)")


def demonstrate_best_practices():
    """Show profiling best practices"""
    print("\n✅ PROFILING BEST PRACTICES")
    print("-" * 30)

    print("   📋 Key Principles:")
    print("      1. Always use warmup iterations before timing")
    print("      2. Run multiple iterations for statistical significance")
    print("      3. Profile both time and memory usage")
    print("      4. Test with realistic input sizes")
    print("      5. Compare against baselines")
    print("      6. Consider both peak and sustained performance")
    print("      7. Profile the full model, not just components")

    print("\n   🎯 Optimization Priorities:")
    print("      1. Correctness first - verify outputs match")
    print("      2. Memory bandwidth often more limiting than compute")
    print("      3. Batch size scaling is critical for efficiency")
    print("      4. Kernel fusion reduces memory round-trips")
    print("      5. Memory layout affects cache performance")


def main():
    """Main demonstration function"""

    # Basic profiling
    demonstrate_basic_profiling()

    # Function comparisons
    demonstrate_function_comparison()

    # Model benchmarking
    demonstrate_model_benchmarking()

    # Scaling analysis
    demonstrate_scaling_analysis()

    # Memory profiling
    demonstrate_memory_profiling()

    # Recommendations
    demonstrate_optimization_recommendations()

    # Advanced concepts
    demonstrate_advanced_concepts()

    # Best practices
    demonstrate_best_practices()

    # Summary
    print(f"\n🎓 PROFILING SUMMARY")
    print("="*30)
    print("   🔍 Profiling Tools Demonstrated:")
    print("      • Basic operation timing and memory tracking")
    print("      • Function implementation comparisons")
    print("      • Complete model benchmarking")
    print("      • Performance scaling analysis")
    print("      • Memory usage pattern detection")
    print("      • Automated optimization recommendations")

    print(f"\n🎯 Key Insights:")
    print("   • Profiling is essential for understanding performance bottlenecks")
    print("   • Different operations have different optimization opportunities")
    print("   • Memory bandwidth often limits performance more than compute")
    print("   • Batch size significantly affects efficiency")
    print("   • Systematic measurement guides optimization decisions")

    print(f"\n✅ Performance Profiling Demo Complete!")


if __name__ == "__main__":
    main()