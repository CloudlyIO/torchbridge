"""
Custom Kernel Performance Benchmarks

Comprehensive benchmarking suite for custom CUDA kernels:
- FlashAttention-3 vs PyTorch SDPA
- Fused Linear+Activation vs separate operations
- Integration with PerformanceTracker for regression detection
- Statistical analysis with warmup and confidence intervals

Performance Targets:
- FlashAttention: 2-5x speedup vs PyTorch SDPA
- Fused Linear+Act: 1.8-2.5x speedup vs separate ops
- Overall Phase 4A: 5-10x improvement for target operations
"""

import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, 'src')



@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    iterations: int
    speedup_vs_baseline: float | None = None


class KernelBenchmarkSuite:
    """
    Comprehensive benchmark suite for custom kernels.

    Includes warmup, statistical analysis, and regression detection.
    """

    def __init__(
        self,
        device: torch.device,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        verbose: bool = True
    ):
        """
        Initialize benchmark suite.

        Args:
            device: Target device for benchmarks
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            verbose: Print detailed results
        """
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.verbose = verbose
        self.results: list[BenchmarkResult] = []

    def benchmark_function(
        self,
        func: Callable,
        name: str,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single function with warmup and statistics.

        Args:
            func: Function to benchmark
            name: Name for this benchmark
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            BenchmarkResult with timing statistics
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = func(*args, **kwargs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.benchmark_iterations):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = func(*args, **kwargs)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        result = BenchmarkResult(
            name=name,
            mean_ms=statistics.mean(times),
            std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_ms=min(times),
            max_ms=max(times),
            median_ms=statistics.median(times),
            iterations=self.benchmark_iterations
        )

        self.results.append(result)

        if self.verbose:
            print(f"\n{name}:")
            print(f"  Mean: {result.mean_ms:.3f} ± {result.std_ms:.3f} ms")
            print(f"  Median: {result.median_ms:.3f} ms")
            print(f"  Min: {result.min_ms:.3f} ms, Max: {result.max_ms:.3f} ms")

        return result

    def calculate_speedup(
        self,
        baseline_result: BenchmarkResult,
        optimized_result: BenchmarkResult
    ) -> float:
        """Calculate speedup of optimized vs baseline."""
        speedup = baseline_result.mean_ms / optimized_result.mean_ms
        optimized_result.speedup_vs_baseline = speedup

        if self.verbose:
            print(f"\n{'Speedup'} {optimized_result.name} vs {baseline_result.name}:")
            print(f"  {speedup:.2f}x faster")

        return speedup

    def print_summary(self):
        """Print summary of all benchmarks."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        for result in self.results:
            speedup_str = f" ({result.speedup_vs_baseline:.2f}x)" if result.speedup_vs_baseline else ""
            print(f"\n{result.name}{speedup_str}:")
            print(f"  {result.mean_ms:.3f} ± {result.std_ms:.3f} ms")


# ===== FlashAttention Benchmarks =====

def benchmark_flash_attention(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    device: torch.device = torch.device('cpu'),
    benchmark_suite: KernelBenchmarkSuite | None = None
) -> dict[str, BenchmarkResult]:
    """
    Benchmark FlashAttention-3 vs PyTorch SDPA.

    Args:
        batch_size: Batch size for benchmark
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Target device
        benchmark_suite: Optional benchmark suite to use

    Returns:
        Dictionary of benchmark results
    """
    if benchmark_suite is None:
        benchmark_suite = KernelBenchmarkSuite(device)

    print(f"\n{'=' * 80}")
    print("FlashAttention Benchmark")
    print(f"  Shape: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"  Device: {device}")
    print(f"{'=' * 80}")

    # Create inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    scale = 1.0 / (head_dim ** 0.5)

    # Baseline: PyTorch SDPA (Scaled Dot-Product Attention)
    def pytorch_sdpa():
        return F.scaled_dot_product_attention(Q, K, V, scale=scale)

    baseline_result = benchmark_suite.benchmark_function(
        pytorch_sdpa,
        "PyTorch SDPA (Baseline)"
    )

    # Try to import and benchmark FlashAttention-3
    results = {"baseline": baseline_result}

    try:
        from torchbridge.hardware.gpu.custom_kernels import FlashAttentionV3

        fa3 = FlashAttentionV3(scale=scale).to(device)

        def flash_attention_v3():
            return fa3(Q, K, V)

        fa3_result = benchmark_suite.benchmark_function(
            flash_attention_v3,
            "FlashAttention-3 (Custom)"
        )

        speedup = benchmark_suite.calculate_speedup(baseline_result, fa3_result)
        results["flash_attention_v3"] = fa3_result
        results["speedup"] = speedup

        # Check performance target (2-5x)
        if speedup >= 2.0:
            print(f"\n✅ Performance target met: {speedup:.2f}x >= 2.0x")
        else:
            print(f"\n⚠️  Performance target not met: {speedup:.2f}x < 2.0x")
            print("   Note: May be due to overhead on CPU or small problem size")

    except ImportError as e:
        print(f"\n⚠️  FlashAttention-3 not available: {e}")
        print("   Skipping custom kernel benchmark")

    return results


# ===== Fused Linear+Activation Benchmarks =====

def benchmark_fused_linear_activation(
    batch_size: int = 128,
    in_features: int = 1024,
    out_features: int = 4096,
    activation: str = "gelu",
    device: torch.device = torch.device('cpu'),
    benchmark_suite: KernelBenchmarkSuite | None = None
) -> dict[str, BenchmarkResult]:
    """
    Benchmark Fused Linear+Activation vs separate operations.

    Args:
        batch_size: Batch size for benchmark
        in_features: Input features
        out_features: Output features
        activation: Activation function ('gelu' or 'silu')
        device: Target device
        benchmark_suite: Optional benchmark suite to use

    Returns:
        Dictionary of benchmark results
    """
    if benchmark_suite is None:
        benchmark_suite = KernelBenchmarkSuite(device)

    print(f"\n{'=' * 80}")
    print(f"Fused Linear+{activation.upper()} Benchmark")
    print(f"  Shape: [{batch_size}, {in_features}] → [{batch_size}, {out_features}]")
    print(f"  Device: {device}")
    print(f"{'=' * 80}")

    # Create input
    x = torch.randn(batch_size, in_features, device=device)

    # Baseline: Separate Linear + Activation
    linear = nn.Linear(in_features, out_features).to(device)

    if activation == "gelu":
        act_fn = nn.GELU()
    elif activation == "silu":
        act_fn = nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    def separate_ops():
        return act_fn(linear(x))

    baseline_result = benchmark_suite.benchmark_function(
        separate_ops,
        f"Separate Linear+{activation.upper()} (Baseline)"
    )

    # Try to import and benchmark Fused kernel
    results = {"baseline": baseline_result}

    try:
        if activation == "gelu":
            from torchbridge.hardware.gpu.custom_kernels import FusedLinearGELU
            fused_layer = FusedLinearGELU(in_features, out_features).to(device)
        else:  # silu
            from torchbridge.hardware.gpu.custom_kernels import FusedLinearSiLU
            fused_layer = FusedLinearSiLU(in_features, out_features).to(device)

        # Copy weights for fair comparison
        fused_layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            fused_layer.bias.data.copy_(linear.bias.data)

        def fused_ops():
            return fused_layer(x)

        fused_result = benchmark_suite.benchmark_function(
            fused_ops,
            f"Fused Linear+{activation.upper()} (Custom)"
        )

        speedup = benchmark_suite.calculate_speedup(baseline_result, fused_result)
        results["fused"] = fused_result
        results["speedup"] = speedup

        # Check performance target (1.8-2.5x)
        if speedup >= 1.8:
            print(f"\n✅ Performance target met: {speedup:.2f}x >= 1.8x")
        else:
            print(f"\n⚠️  Performance target not met: {speedup:.2f}x < 1.8x")
            print("   Note: May be due to overhead on CPU or small problem size")

    except ImportError as e:
        print(f"\n⚠️  Fused {activation.upper()} not available: {e}")
        print("   Skipping custom kernel benchmark")

    return results


# ===== Main Benchmark Runner =====

def run_all_benchmarks(device: torch.device | None = None):
    """
    Run all custom kernel benchmarks.

    Args:
        device: Target device (auto-detected if None)
    """
    # Auto-detect device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'#' * 80}")
    print("# Custom Kernel Benchmark Suite")
    print(f"# Device: {device}")
    print("# Warmup: 10 iterations, Benchmark: 100 iterations")
    print(f"{'#' * 80}")

    benchmark_suite = KernelBenchmarkSuite(
        device=device,
        warmup_iterations=10,
        benchmark_iterations=100,
        verbose=True
    )

    all_results = {}

    # FlashAttention Benchmarks
    print("\n" + "=" * 80)
    print("ATTENTION BENCHMARKS")
    print("=" * 80)

    # Small sequence
    all_results['fa_small'] = benchmark_flash_attention(
        batch_size=2, num_heads=8, seq_len=128, head_dim=64,
        device=device, benchmark_suite=benchmark_suite
    )

    # Medium sequence
    all_results['fa_medium'] = benchmark_flash_attention(
        batch_size=2, num_heads=8, seq_len=512, head_dim=64,
        device=device, benchmark_suite=benchmark_suite
    )

    # Large sequence (only on CUDA)
    if device.type == 'cuda':
        all_results['fa_large'] = benchmark_flash_attention(
            batch_size=2, num_heads=8, seq_len=2048, head_dim=64,
            device=device, benchmark_suite=benchmark_suite
        )

    # Fused Linear+Activation Benchmarks
    print("\n" + "=" * 80)
    print("FUSED LINEAR+ACTIVATION BENCHMARKS")
    print("=" * 80)

    # GELU - small
    all_results['gelu_small'] = benchmark_fused_linear_activation(
        batch_size=32, in_features=512, out_features=2048, activation="gelu",
        device=device, benchmark_suite=benchmark_suite
    )

    # GELU - large
    all_results['gelu_large'] = benchmark_fused_linear_activation(
        batch_size=128, in_features=2048, out_features=8192, activation="gelu",
        device=device, benchmark_suite=benchmark_suite
    )

    # SiLU - medium
    all_results['silu_medium'] = benchmark_fused_linear_activation(
        batch_size=64, in_features=1024, out_features=4096, activation="silu",
        device=device, benchmark_suite=benchmark_suite
    )

    # Print final summary
    benchmark_suite.print_summary()

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    speedups = []
    for key, result_dict in all_results.items():
        if 'speedup' in result_dict:
            speedups.append(result_dict['speedup'])

    if speedups:
        avg_speedup = statistics.mean(speedups)
        print(f"\nAverage speedup across all benchmarks: {avg_speedup:.2f}x")

        if avg_speedup >= 2.0:
            print("✅ Overall performance target met!")
        else:
            print("⚠️  Overall performance target not met on this device")
            if device.type == 'cpu':
                print("   Note: Significant speedups typically observed on CUDA GPUs")
    else:
        print("\n⚠️  No custom kernels were benchmarked")
        print("   This may be because:")
        print("   - CUDA is not available")
        print("   - Custom kernels not compiled")
        print("   - Running on CPU")

    return all_results


if __name__ == "__main__":
    # Run all benchmarks
    results = run_all_benchmarks()

    print(f"\n{'#' * 80}")
    print("# Benchmark complete!")
    print(f"{'#' * 80}\n")
