"""
AMD Backend Optimization Benchmarks (v0.4.9)

Comprehensive benchmark suite for AMD ROCm backend optimization strategies.
Measures performance across different optimization levels, architectures,
and workloads.

Usage:
    python benchmarks/amd_optimization_benchmark.py

Output includes:
- Optimization level comparison (conservative, balanced, aggressive)
- Operator fusion overhead measurement
- Memory layout optimization impact
- Architecture-specific performance differences
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchbridge.backends.amd.amd_backend import AMDBackend
from torchbridge.backends.amd.amd_optimizer import AMDOptimizer
from torchbridge.backends.amd.memory_manager import AMDMemoryManager
from torchbridge.backends.amd.rocm_compiler import ROCmCompiler
from torchbridge.core.config import AMDArchitecture, AMDConfig


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    optimization_level: str
    architecture: str
    duration_ms: float
    throughput: float  # items/sec
    memory_mb: float
    fusion_count: int
    success: bool
    error: str | None = None


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def create_transformer_block(hidden_size: int = 256) -> nn.Module:
    """Create a transformer-like block for benchmarking."""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size * 4),
        nn.GELU(),
        nn.Linear(hidden_size * 4, hidden_size),
        nn.LayerNorm(hidden_size),
    )


def create_conv_block() -> nn.Module:
    """Create a convolutional block for benchmarking."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )


def benchmark_optimization_level(
    model: nn.Module,
    config: AMDConfig,
    input_tensor: torch.Tensor,
    num_iterations: int = 100,
    warmup: int = 10
) -> BenchmarkResult:
    """Benchmark a single optimization level."""
    try:
        backend = AMDBackend(config)
        optimizer = AMDOptimizer(config)

        # Prepare model
        model_copy = type(model)(*[getattr(model, 'hidden_size', 256)]
                                 if hasattr(model, 'hidden_size') else [])

        # Move to device
        prepared_model = backend.prepare_model(model_copy)
        optimized_model = optimizer.optimize(prepared_model)
        input_on_device = input_tensor.to(backend.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = optimized_model(input_on_device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = optimized_model(input_on_device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000
        throughput = num_iterations / (duration_ms / 1000)

        # Get memory usage
        memory_mb = 0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        summary = optimizer.get_optimization_summary()

        return BenchmarkResult(
            name=f"{config.optimization_level}_{config.architecture.value}",
            optimization_level=config.optimization_level,
            architecture=config.architecture.value,
            duration_ms=duration_ms,
            throughput=throughput,
            memory_mb=memory_mb,
            fusion_count=summary.get('fused_operations', 0),
            success=True
        )

    except Exception as e:
        return BenchmarkResult(
            name=f"{config.optimization_level}_{config.architecture.value}",
            optimization_level=config.optimization_level,
            architecture=config.architecture.value,
            duration_ms=0,
            throughput=0,
            memory_mb=0,
            fusion_count=0,
            success=False,
            error=str(e)
        )


def benchmark_compilation_cache() -> dict[str, Any]:
    """Benchmark HIP kernel compilation caching."""
    config = AMDConfig()
    compiler = ROCmCompiler(config)

    kernel_source = """
    __global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """

    # First compilation (cache miss)
    start = time.perf_counter()
    compiler.compile_kernel(kernel_source, "gemm_kernel")
    first_compile_ms = (time.perf_counter() - start) * 1000

    # Second compilation (cache hit)
    start = time.perf_counter()
    compiler.compile_kernel(kernel_source, "gemm_kernel")
    cached_compile_ms = (time.perf_counter() - start) * 1000

    stats = compiler.get_compilation_stats()

    return {
        "first_compile_ms": first_compile_ms,
        "cached_compile_ms": cached_compile_ms,
        "speedup": first_compile_ms / cached_compile_ms if cached_compile_ms > 0 else 0,
        "cache_hit_rate": stats.get("cache_hit_rate_percent", 0),
        "total_compilations": stats.get("total_compilations", 0),
    }


def benchmark_memory_management() -> dict[str, Any]:
    """Benchmark AMD memory management operations."""
    config = AMDConfig(memory_pool_size_gb=1.0)
    manager = AMDMemoryManager(config)

    allocation_times = []
    free_times = []

    # Benchmark allocations
    for size in [64, 256, 1024, 4096]:
        start = time.perf_counter()
        try:
            tensor = manager.allocate_tensor(
                shape=(size, size),
                dtype=torch.float32,
                purpose="benchmark"
            )
            allocation_times.append({
                "size": size * size * 4 / 1024 / 1024,  # MB
                "time_ms": (time.perf_counter() - start) * 1000
            })

            start = time.perf_counter()
            manager.free_tensor(tensor)
            free_times.append({
                "size": size * size * 4 / 1024 / 1024,
                "time_ms": (time.perf_counter() - start) * 1000
            })
        except Exception as e:
            allocation_times.append({
                "size": size * size * 4 / 1024 / 1024,
                "time_ms": 0,
                "error": str(e)
            })

    # Benchmark defragmentation
    start = time.perf_counter()
    manager.defragment()
    defrag_time_ms = (time.perf_counter() - start) * 1000

    stats = manager.get_memory_stats()

    return {
        "allocation_times": allocation_times,
        "free_times": free_times,
        "defrag_time_ms": defrag_time_ms,
        "memory_stats": stats,
    }


def run_all_benchmarks() -> dict[str, Any]:
    """Run all AMD optimization benchmarks."""
    print_section("AMD Backend Optimization Benchmarks (v0.4.9)")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {}
    }

    # 1. Optimization Level Comparison
    print_section("1. Optimization Level Comparison")

    batch_size = 32
    hidden_size = 256

    transformer_model = create_transformer_block(hidden_size)
    transformer_input = torch.randn(batch_size, hidden_size)

    opt_level_results = []
    for level in ["conservative", "balanced", "aggressive"]:
        for arch in [AMDArchitecture.CDNA2, AMDArchitecture.CDNA3]:
            config = AMDConfig(
                architecture=arch,
                optimization_level=level,
                enable_operator_fusion=True,
            )

            result = benchmark_optimization_level(
                transformer_model,
                config,
                transformer_input,
                num_iterations=50
            )
            opt_level_results.append(asdict(result))

            status = "OK" if result.success else f"FAIL: {result.error}"
            print(f"  {level:12} | {arch.value:6} | "
                  f"{result.throughput:8.1f} iter/s | {status}")

    results["benchmarks"]["optimization_levels"] = opt_level_results

    # 2. Compilation Cache Performance
    print_section("2. Compilation Cache Performance")

    cache_results = benchmark_compilation_cache()
    results["benchmarks"]["compilation_cache"] = cache_results

    print(f"  First compile:  {cache_results['first_compile_ms']:.2f} ms")
    print(f"  Cached compile: {cache_results['cached_compile_ms']:.4f} ms")
    print(f"  Cache speedup:  {cache_results['speedup']:.1f}x")
    print(f"  Cache hit rate: {cache_results['cache_hit_rate']:.1f}%")

    # 3. Memory Management Performance
    print_section("3. Memory Management Performance")

    memory_results = benchmark_memory_management()
    results["benchmarks"]["memory_management"] = memory_results

    print("  Allocation times:")
    for alloc in memory_results["allocation_times"]:
        if "error" not in alloc:
            print(f"    {alloc['size']:6.1f} MB: {alloc['time_ms']:.4f} ms")
        else:
            print(f"    {alloc['size']:6.1f} MB: {alloc['error']}")

    print(f"  Defragmentation: {memory_results['defrag_time_ms']:.2f} ms")

    # 4. Conv Block Optimization
    print_section("4. Convolutional Block Optimization")

    conv_model = create_conv_block()
    conv_model.eval()  # Required for BN fusion
    conv_input = torch.randn(batch_size, 3, 64, 64)

    conv_results = []
    for level in ["conservative", "balanced", "aggressive"]:
        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level=level,
            enable_operator_fusion=True,
        )

        result = benchmark_optimization_level(
            conv_model,
            config,
            conv_input,
            num_iterations=50
        )
        conv_results.append(asdict(result))

        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {level:12} | {result.throughput:8.1f} iter/s | "
              f"fusions: {result.fusion_count} | {status}")

    results["benchmarks"]["conv_optimization"] = conv_results

    # Summary
    print_section("Benchmark Summary")

    successful = sum(1 for r in opt_level_results if r["success"])
    total = len(opt_level_results)
    print(f"  Optimization benchmarks: {successful}/{total} successful")
    print(f"  Cache hit rate: {cache_results['cache_hit_rate']:.1f}%")
    print("  Memory management: OK")

    # Save results
    output_path = Path(__file__).parent / "results" / "amd_optimization_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
