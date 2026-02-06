#!/usr/bin/env python3
"""
AMD ROCm Integration Benchmark

Comprehensive performance benchmarking for AMD backend including
backend operations, optimizer performance, compiler efficiency,
memory management, and cross-backend comparison.

Phase 4C-Pre Week 5: AMD Testing & Integration (v0.3.5)

Usage:
    PYTHONPATH=src python3 benchmarks/amd_integration_benchmark.py
    PYTHONPATH=src python3 benchmarks/amd_integration_benchmark.py --quick
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from torchbridge.backends.amd import (
    AMDBackend,
    AMDMemoryManager,
    AMDOptimizer,
    HIPUtilities,
    ROCmCompiler,
)
from torchbridge.core.config import AMDArchitecture, AMDConfig


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_result(result: BenchmarkResult) -> None:
    """Print a benchmark result."""
    print(f"  {result.name}:")
    print(f"    Iterations: {result.iterations}")
    print(f"    Average:    {result.avg_time_ms:.4f} ms")
    print(f"    Min/Max:    {result.min_time_ms:.4f} / {result.max_time_ms:.4f} ms")
    print(f"    Std Dev:    {result.std_dev_ms:.4f} ms")


def run_timed_iterations(func, iterations: int, warmup: int = 3) -> BenchmarkResult:
    """Run a function multiple times and collect timing statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return BenchmarkResult(
        name="",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
    )


# ============================================================================
# Test Models
# ============================================================================

class SmallModel(nn.Module):
    """Small model for quick benchmarks."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MediumModel(nn.Module):
    """Medium model for standard benchmarks."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """Transformer block for attention benchmarks."""
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ============================================================================
# AMD Backend Benchmarks
# ============================================================================

def benchmark_amd_backend(iterations: int = 100) -> dict[str, BenchmarkResult]:
    """Benchmark AMD backend operations."""
    print_section("AMD Backend Benchmarks")

    results = {}

    # Backend creation
    result = run_timed_iterations(
        lambda: AMDBackend(),
        iterations=iterations,
    )
    result.name = "Backend Creation"
    results["backend_creation"] = result
    print_result(result)

    # Model preparation
    backend = AMDBackend()
    model = SmallModel()

    result = run_timed_iterations(
        lambda: backend.prepare_model(model),
        iterations=iterations,
    )
    result.name = "Model Preparation (Small)"
    results["model_preparation_small"] = result
    print_result(result)

    # Medium model preparation
    model_medium = MediumModel()
    result = run_timed_iterations(
        lambda: backend.prepare_model(model_medium),
        iterations=iterations // 2,
    )
    result.name = "Model Preparation (Medium)"
    results["model_preparation_medium"] = result
    print_result(result)

    # Device info retrieval
    result = run_timed_iterations(
        lambda: backend.get_device_info(),
        iterations=iterations,
    )
    result.name = "Device Info Retrieval"
    results["device_info"] = result
    print_result(result)

    return results


def benchmark_amd_optimizer(iterations: int = 50) -> dict[str, BenchmarkResult]:
    """Benchmark AMD optimizer performance."""
    print_section("AMD Optimizer Benchmarks")

    results = {}
    model = MediumModel()

    for level in ["conservative", "balanced", "aggressive"]:
        config = AMDConfig(optimization_level=level)
        optimizer = AMDOptimizer(config)

        result = run_timed_iterations(
            lambda opt=optimizer: opt.optimize(model),
            iterations=iterations,
        )
        result.name = f"{level.capitalize()} Optimization"
        results[f"optimization_{level}"] = result
        print_result(result)

    # Matrix Cores benchmark (CDNA2/CDNA3)
    for arch in [AMDArchitecture.CDNA2, AMDArchitecture.CDNA3]:
        config = AMDConfig(
            architecture=arch,
            optimization_level="aggressive",
            enable_matrix_cores=True,
        )
        optimizer = AMDOptimizer(config)

        result = run_timed_iterations(
            lambda opt=optimizer: opt.optimize(model),
            iterations=iterations // 2,
        )
        result.name = f"Matrix Cores ({arch.value})"
        results[f"matrix_cores_{arch.value}"] = result
        print_result(result)

    return results


def benchmark_rocm_compiler(iterations: int = 50) -> dict[str, BenchmarkResult]:
    """Benchmark ROCm compiler performance."""
    print_section("ROCm Compiler Benchmarks")

    results = {}

    # Simple kernel compilation
    config = AMDConfig()
    compiler = ROCmCompiler(config)

    simple_kernel = "__global__ void add(float* a, float* b, float* c) { int i = threadIdx.x; c[i] = a[i] + b[i]; }"

    # First compilation (cold cache)
    result = run_timed_iterations(
        lambda: compiler.compile_kernel(simple_kernel, f"add_{time.time_ns()}"),
        iterations=iterations // 5,
        warmup=0,  # No warmup for cold cache test
    )
    result.name = "Cold Cache Compilation"
    results["cold_cache"] = result
    print_result(result)

    # Warm cache compilation
    compiler.clear_cache()
    _ = compiler.compile_kernel(simple_kernel, "add_cached")

    result = run_timed_iterations(
        lambda: compiler.compile_kernel(simple_kernel, "add_cached"),
        iterations=iterations,
    )
    result.name = "Warm Cache Compilation"
    results["warm_cache"] = result
    print_result(result)

    # Complex kernel compilation
    complex_kernel = """
    __global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """

    result = run_timed_iterations(
        lambda: compiler.compile_kernel(complex_kernel, f"matmul_{time.time_ns()}"),
        iterations=iterations // 5,
        warmup=0,
    )
    result.name = "Complex Kernel Compilation"
    results["complex_kernel"] = result
    print_result(result)

    # Print cache statistics
    stats = compiler.get_compilation_stats()
    print("\n  Cache Statistics:")
    print(f"    Total compilations: {stats['total_compilations']}")
    print(f"    Cache hits: {stats['cache_hits']}")
    print(f"    Cache hit rate: {stats['cache_hit_rate_percent']:.1f}%")

    return results


def benchmark_hip_utilities(iterations: int = 100) -> dict[str, BenchmarkResult]:
    """Benchmark HIP utilities performance."""
    print_section("HIP Utilities Benchmarks")

    results = {}
    config = AMDConfig(enable_profiling=True)
    utils = HIPUtilities(config)

    # Stream creation
    stream_idx = [0]

    def create_stream():
        stream_idx[0] += 1
        return utils.create_stream(f"stream_{stream_idx[0]}")

    result = run_timed_iterations(
        create_stream,
        iterations=iterations // 2,
    )
    result.name = "Stream Creation"
    results["stream_creation"] = result
    print_result(result)
    utils.cleanup()

    # Event creation
    utils = HIPUtilities(config)
    event_idx = [0]

    def create_event():
        event_idx[0] += 1
        return utils.create_event(f"event_{event_idx[0]}")

    result = run_timed_iterations(
        create_event,
        iterations=iterations // 2,
    )
    result.name = "Event Creation"
    results["event_creation"] = result
    print_result(result)
    utils.cleanup()

    # Profiling overhead
    utils = HIPUtilities(config)

    def profile_operation():
        with utils.profile_region("test"):
            pass

    result = run_timed_iterations(
        profile_operation,
        iterations=iterations,
    )
    result.name = "Profiling Overhead"
    results["profiling_overhead"] = result
    print_result(result)

    return results


def benchmark_memory_manager(iterations: int = 50) -> dict[str, BenchmarkResult]:
    """Benchmark AMD memory manager performance."""
    print_section("Memory Manager Benchmarks")

    results = {}

    try:
        config = AMDConfig(memory_pool_size_gb=4.0)
        manager = AMDMemoryManager(config)

        # Stats retrieval
        result = run_timed_iterations(
            lambda: manager.get_memory_stats(),
            iterations=iterations,
        )
        result.name = "Memory Stats Retrieval"
        results["stats_retrieval"] = result
        print_result(result)

        # Allocation summary
        result = run_timed_iterations(
            lambda: manager.get_allocation_summary(),
            iterations=iterations,
        )
        result.name = "Allocation Summary"
        results["allocation_summary"] = result
        print_result(result)

    except (AssertionError, RuntimeError) as e:
        print(f"  Skipped (no GPU): {e}")
        # Return dummy results for summary
        results["stats_retrieval"] = BenchmarkResult(
            name="Memory Stats Retrieval",
            iterations=0, total_time_ms=0, avg_time_ms=0,
            min_time_ms=0, max_time_ms=0, std_dev_ms=0,
        )
        results["allocation_summary"] = BenchmarkResult(
            name="Allocation Summary",
            iterations=0, total_time_ms=0, avg_time_ms=0,
            min_time_ms=0, max_time_ms=0, std_dev_ms=0,
        )

    return results


# ============================================================================
# Cross-Architecture Comparison
# ============================================================================

def benchmark_architecture_comparison(iterations: int = 30) -> dict[str, Any]:
    """Compare performance across AMD architectures."""
    print_section("Architecture Comparison")

    results = {}
    model = MediumModel()

    architectures = [
        AMDArchitecture.CDNA2,
        AMDArchitecture.CDNA3,
        AMDArchitecture.RDNA3,
    ]

    for arch in architectures:
        config = AMDConfig(
            architecture=arch,
            optimization_level="balanced",
        )
        optimizer = AMDOptimizer(config)

        result = run_timed_iterations(
            lambda opt=optimizer: opt.optimize(model),
            iterations=iterations,
        )
        result.name = f"{arch.value.upper()} Optimization"
        results[arch.value] = result
        print_result(result)

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all AMD integration benchmarks."""
    parser = argparse.ArgumentParser(description="AMD Integration Benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with fewer iterations",
    )
    args = parser.parse_args()

    # Adjust iterations based on mode
    if args.quick:
        backend_iters = 20
        optimizer_iters = 10
        compiler_iters = 10
        utils_iters = 20
        memory_iters = 10
        arch_iters = 5
    else:
        backend_iters = 100
        optimizer_iters = 50
        compiler_iters = 50
        utils_iters = 100
        memory_iters = 50
        arch_iters = 30

    print("\n" + "=" * 70)
    print("  AMD ROCm Integration Benchmark (v0.3.5)")
    print("  Mode:", "Quick" if args.quick else "Full")
    print("=" * 70)

    all_results = {}

    # Run all benchmarks
    all_results["backend"] = benchmark_amd_backend(backend_iters)
    all_results["optimizer"] = benchmark_amd_optimizer(optimizer_iters)
    all_results["compiler"] = benchmark_rocm_compiler(compiler_iters)
    all_results["utilities"] = benchmark_hip_utilities(utils_iters)
    all_results["memory"] = benchmark_memory_manager(memory_iters)
    all_results["architecture"] = benchmark_architecture_comparison(arch_iters)

    # Summary
    print_section("Summary")

    total_benchmarks = sum(len(r) for r in all_results.values())
    print(f"  Total benchmarks run: {total_benchmarks}")

    # Key metrics
    print("\n  Key Performance Metrics:")
    print(f"    Backend creation:     {all_results['backend']['backend_creation'].avg_time_ms:.4f} ms")
    print(f"    Model preparation:    {all_results['backend']['model_preparation_small'].avg_time_ms:.4f} ms")
    print(f"    Balanced optimization:{all_results['optimizer']['optimization_balanced'].avg_time_ms:.4f} ms")
    print(f"    Warm cache compile:   {all_results['compiler']['warm_cache'].avg_time_ms:.4f} ms")

    print("\n  Benchmark complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
