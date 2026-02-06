#!/usr/bin/env python3
"""
Attention Efficiency Benchmarks

Benchmarks for v0.4.23 attention implementations:
- ViT Attention Slicing
- Sparse Attention (Block, Strided, Dynamic)
- Memory-Efficient Attention (Chunked, LongSequence)

Measures:
- Throughput (samples/second)
- Memory usage (MB)
- Scaling with sequence length

v0.4.23 - Complete Placeholder Implementations
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    seq_len: int
    batch_size: int
    embed_dim: int
    throughput: float  # samples/sec
    latency_ms: float  # milliseconds
    memory_mb: float | None  # MB, only on CUDA
    success: bool
    error: str | None = None


def measure_memory():
    """Get current GPU memory usage in MB."""
    if not CUDA_AVAILABLE:
        return None
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


def reset_memory():
    """Reset GPU memory stats."""
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> BenchmarkResult:
    """Benchmark a model's forward pass."""
    try:
        x = torch.randn(batch_size, seq_len, embed_dim, device=DEVICE)

        reset_memory()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if hasattr(model, 'forward'):
                    # Check if model expects tuple return (like MultiheadAttention)
                    try:
                        _ = model(x, x, x)
                    except TypeError:
                        _ = model(x)

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                try:
                    _ = model(x, x, x)
                except TypeError:
                    _ = model(x)

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        throughput = (batch_size * num_iterations) / elapsed
        latency = (elapsed / num_iterations) * 1000  # ms

        memory = None
        if CUDA_AVAILABLE:
            memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return BenchmarkResult(
            name=model.__class__.__name__,
            seq_len=seq_len,
            batch_size=batch_size,
            embed_dim=embed_dim,
            throughput=throughput,
            latency_ms=latency,
            memory_mb=memory,
            success=True,
        )

    except Exception as e:
        return BenchmarkResult(
            name=model.__class__.__name__,
            seq_len=seq_len,
            batch_size=batch_size,
            embed_dim=embed_dim,
            throughput=0,
            latency_ms=0,
            memory_mb=None,
            success=False,
            error=str(e),
        )


def benchmark_sliced_attention(
    seq_lengths: list[int],
    batch_size: int = 4,
    embed_dim: int = 768,
    num_heads: int = 12,
) -> list[BenchmarkResult]:
    """Benchmark SlicedMultiheadAttention."""
    results = []

    try:
        from torchbridge.models.vision.vit import SlicedMultiheadAttention
    except ImportError as e:
        print(f"Could not import SlicedMultiheadAttention: {e}")
        return results

    for seq_len in seq_lengths:
        # Test different slice sizes
        for slice_size in [16, 32, 64]:
            model = SlicedMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                slice_size=slice_size,
            ).to(DEVICE)

            result = benchmark_model(model, batch_size, seq_len, embed_dim)
            result.name = f"SlicedAttn_s{slice_size}"
            results.append(result)

            print(f"  {result.name} @ seq={seq_len}: "
                  f"{result.throughput:.1f} samples/s, "
                  f"{result.latency_ms:.2f} ms")

    return results


def benchmark_sparse_attention(
    seq_lengths: list[int],
    batch_size: int = 4,
    embed_dim: int = 768,
    num_heads: int = 12,
) -> list[BenchmarkResult]:
    """Benchmark sparse attention implementations."""
    results = []

    try:
        from torchbridge.attention.core.config import AttentionConfig
        from torchbridge.attention.implementations.sparse import (
            BlockSparseAttention,
            StridedSparseAttention,
        )
    except ImportError as e:
        print(f"Could not import sparse attention: {e}")
        return results

    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_sequence_length=max(seq_lengths),
    )

    for seq_len in seq_lengths:
        # Block sparse
        for block_size in [32, 64]:
            model = BlockSparseAttention(
                config,
                block_size=block_size,
            ).to(DEVICE)

            result = benchmark_model(model, batch_size, seq_len, embed_dim)
            result.name = f"BlockSparse_b{block_size}"
            results.append(result)

            print(f"  {result.name} @ seq={seq_len}: "
                  f"{result.throughput:.1f} samples/s, "
                  f"{result.latency_ms:.2f} ms")

        # Strided sparse
        model = StridedSparseAttention(
            config,
            local_window=64,
            stride=32,
        ).to(DEVICE)

        result = benchmark_model(model, batch_size, seq_len, embed_dim)
        result.name = "StridedSparse"
        results.append(result)

        print(f"  {result.name} @ seq={seq_len}: "
              f"{result.throughput:.1f} samples/s, "
              f"{result.latency_ms:.2f} ms")

    return results


def benchmark_memory_efficient_attention(
    seq_lengths: list[int],
    batch_size: int = 4,
    embed_dim: int = 768,
    num_heads: int = 12,
) -> list[BenchmarkResult]:
    """Benchmark memory-efficient attention implementations."""
    results = []

    try:
        from torchbridge.attention.core.config import AttentionConfig
        from torchbridge.attention.implementations.memory_efficient import (
            ChunkedAttention,
            LongSequenceAttention,
            MemoryEfficientAttention,
        )
    except ImportError as e:
        print(f"Could not import memory-efficient attention: {e}")
        return results

    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_sequence_length=max(seq_lengths),
    )

    for seq_len in seq_lengths:
        # Memory efficient
        for chunk_size in [32, 64]:
            model = MemoryEfficientAttention(
                config,
                chunk_size=chunk_size,
            ).to(DEVICE)

            result = benchmark_model(model, batch_size, seq_len, embed_dim)
            result.name = f"MemEfficient_c{chunk_size}"
            results.append(result)

            print(f"  {result.name} @ seq={seq_len}: "
                  f"{result.throughput:.1f} samples/s, "
                  f"{result.latency_ms:.2f} ms")

        # Chunked (double)
        model = ChunkedAttention(
            config,
            query_chunk_size=32,
            kv_chunk_size=32,
        ).to(DEVICE)

        result = benchmark_model(model, batch_size, seq_len, embed_dim)
        result.name = "ChunkedDouble"
        results.append(result)

        print(f"  {result.name} @ seq={seq_len}: "
              f"{result.throughput:.1f} samples/s, "
              f"{result.latency_ms:.2f} ms")

        # Long sequence
        model = LongSequenceAttention(
            config,
            window_size=64,
            global_stride=32,
        ).to(DEVICE)

        result = benchmark_model(model, batch_size, seq_len, embed_dim)
        result.name = "LongSequence"
        results.append(result)

        print(f"  {result.name} @ seq={seq_len}: "
              f"{result.throughput:.1f} samples/s, "
              f"{result.latency_ms:.2f} ms")

    return results


def benchmark_standard_attention(
    seq_lengths: list[int],
    batch_size: int = 4,
    embed_dim: int = 768,
    num_heads: int = 12,
) -> list[BenchmarkResult]:
    """Benchmark standard PyTorch attention for comparison."""
    results = []

    for seq_len in seq_lengths:
        model = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        ).to(DEVICE)

        result = benchmark_model(model, batch_size, seq_len, embed_dim)
        result.name = "StandardMHA"
        results.append(result)

        print(f"  {result.name} @ seq={seq_len}: "
              f"{result.throughput:.1f} samples/s, "
              f"{result.latency_ms:.2f} ms")

    return results


def run_benchmarks(
    seq_lengths: list[int] = None,
    batch_size: int = 4,
    output_file: str = None,
):
    """Run all benchmarks."""
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512]

    print("=" * 60)
    print("Attention Efficiency Benchmarks (v0.4.23)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths: {seq_lengths}")
    print()

    all_results = []

    # Standard attention (baseline)
    print("Standard PyTorch MultiheadAttention:")
    print("-" * 40)
    all_results.extend(benchmark_standard_attention(seq_lengths, batch_size))
    print()

    # Sliced attention
    print("Sliced Attention (ViT optimization):")
    print("-" * 40)
    all_results.extend(benchmark_sliced_attention(seq_lengths, batch_size))
    print()

    # Sparse attention
    print("Sparse Attention:")
    print("-" * 40)
    all_results.extend(benchmark_sparse_attention(seq_lengths, batch_size))
    print()

    # Memory-efficient attention
    print("Memory-Efficient Attention:")
    print("-" * 40)
    all_results.extend(benchmark_memory_efficient_attention(seq_lengths, batch_size))
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    # Group by attention type and find best throughput
    by_name = {}
    for r in all_results:
        if r.name not in by_name:
            by_name[r.name] = []
        by_name[r.name].append(r)

    print("\nAverage throughput by attention type:")
    for name, results in sorted(by_name.items()):
        avg_throughput = sum(r.throughput for r in results) / len(results)
        print(f"  {name}: {avg_throughput:.1f} samples/s")

    # Save results
    if output_file:
        results_dict = {
            "device": str(DEVICE),
            "batch_size": batch_size,
            "seq_lengths": seq_lengths,
            "results": [asdict(r) for r in all_results],
        }
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Attention Efficiency Benchmarks")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for benchmarks",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()
    run_benchmarks(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
