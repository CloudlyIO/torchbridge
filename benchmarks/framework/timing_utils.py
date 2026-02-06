"""
Shared timing utilities for TorchBridge benchmarks.

This module provides standardized timing and measurement utilities
for consistent benchmarking across all backends (NVIDIA, TPU, AMD).

Version: 0.3.6
"""

import functools
import statistics
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TimingResult:
    """Container for timing measurement results."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    percentiles: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Average:    {self.avg_time_ms:.4f} ms\n"
            f"  Min/Max:    {self.min_time_ms:.4f} / {self.max_time_ms:.4f} ms\n"
            f"  Std Dev:    {self.std_dev_ms:.4f} ms"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "std_dev_ms": self.std_dev_ms,
            "percentiles": self.percentiles,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    iterations: int = 100
    warmup: int = 10
    sync_cuda: bool = True
    compute_percentiles: bool = True
    percentile_values: tuple[int, ...] = (50, 90, 95, 99)


# ============================================================================
# Core Timing Functions
# ============================================================================

def run_timed_iterations(
    func: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 10,
    name: str = "benchmark",
    sync_cuda: bool = True,
    compute_percentiles: bool = True,
) -> TimingResult:
    """
    Run a function multiple times and collect timing statistics.

    Args:
        func: Function to benchmark (should take no arguments)
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)
        name: Name for the benchmark result
        sync_cuda: Whether to sync CUDA after each iteration
        compute_percentiles: Whether to compute p50/p90/p95/p99

    Returns:
        TimingResult with comprehensive timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Calculate statistics
    total_time = sum(times)
    avg_time = statistics.mean(times) if times else 0.0
    min_time = min(times) if times else 0.0
    max_time = max(times) if times else 0.0
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

    # Calculate percentiles
    percentiles = {}
    if compute_percentiles and times:
        sorted_times = sorted(times)
        for p in (50, 90, 95, 99):
            idx = int(len(sorted_times) * p / 100)
            idx = min(idx, len(sorted_times) - 1)
            percentiles[f"p{p}"] = sorted_times[idx]

    return TimingResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time,
        avg_time_ms=avg_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_dev_ms=std_dev,
        percentiles=percentiles,
    )


def benchmark_function(
    func: Callable[..., Any],
    *args,
    iterations: int = 100,
    warmup: int = 10,
    name: str | None = None,
    **kwargs
) -> TimingResult:
    """
    Benchmark a function with arguments.

    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        iterations: Number of timed iterations
        warmup: Number of warmup iterations
        name: Name for the result (defaults to function name)
        **kwargs: Keyword arguments for the function

    Returns:
        TimingResult with timing statistics
    """
    if name is None:
        name = func.__name__

    # Wrap function with args/kwargs
    wrapped = lambda: func(*args, **kwargs)

    return run_timed_iterations(
        wrapped,
        iterations=iterations,
        warmup=warmup,
        name=name,
    )


# ============================================================================
# Context Manager for Timing
# ============================================================================

@contextmanager
def timer(name: str = "operation", print_result: bool = False):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("my_operation") as t:
            # code to time
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")

    Args:
        name: Name of the operation
        print_result: Whether to print result on exit
    """
    class TimerContext:
        def __init__(self):
            self.start_time: float = 0.0
            self.end_time: float = 0.0
            self.elapsed_ms: float = 0.0
            self.name = name

    ctx = TimerContext()
    ctx.start_time = time.perf_counter()

    try:
        yield ctx
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ctx.end_time = time.perf_counter()
        ctx.elapsed_ms = (ctx.end_time - ctx.start_time) * 1000

        if print_result:
            print(f"⏱️  {name}: {ctx.elapsed_ms:.4f} ms")


# ============================================================================
# Decorator for Timing
# ============================================================================

def timed(iterations: int = 1, warmup: int = 0, print_result: bool = True):
    """
    Decorator to time function execution.

    Usage:
        @timed(iterations=100, warmup=10)
        def my_function():
            # code to time
            pass

    Args:
        iterations: Number of times to run the function
        warmup: Number of warmup runs
        print_result: Whether to print timing results
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Single execution mode
            if iterations == 1 and warmup == 0:
                with timer(func.__name__, print_result=print_result) as t:
                    result = func(*args, **kwargs)
                return result

            # Multi-iteration benchmark mode
            wrapped = lambda: func(*args, **kwargs)
            timing_result = run_timed_iterations(
                wrapped,
                iterations=iterations,
                warmup=warmup,
                name=func.__name__,
            )

            if print_result:
                print(timing_result)

            # Return last execution result
            return func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# Memory Tracking
# ============================================================================

@dataclass
class MemorySnapshot:
    """Snapshot of memory usage."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    device: str


def get_cuda_memory_snapshot(device: int = 0) -> MemorySnapshot | None:
    """
    Get current CUDA memory usage snapshot.

    Args:
        device: CUDA device index

    Returns:
        MemorySnapshot or None if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize(device)

    return MemorySnapshot(
        allocated_mb=torch.cuda.memory_allocated(device) / 1024 / 1024,
        reserved_mb=torch.cuda.memory_reserved(device) / 1024 / 1024,
        max_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        device=f"cuda:{device}",
    )


@contextmanager
def track_memory(device: int = 0, reset_peak: bool = True):
    """
    Context manager to track memory usage of a code block.

    Usage:
        with track_memory() as mem:
            # memory-intensive code
        print(f"Peak memory: {mem.peak_allocated_mb:.2f} MB")

    Args:
        device: CUDA device index
        reset_peak: Whether to reset peak memory before measuring
    """
    class MemoryTracker:
        def __init__(self):
            self.before: MemorySnapshot | None = None
            self.after: MemorySnapshot | None = None
            self.peak_allocated_mb: float = 0.0
            self.delta_mb: float = 0.0

    tracker = MemoryTracker()

    if torch.cuda.is_available():
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device)
        tracker.before = get_cuda_memory_snapshot(device)

    try:
        yield tracker
    finally:
        if torch.cuda.is_available():
            tracker.after = get_cuda_memory_snapshot(device)
            if tracker.after and tracker.before:
                tracker.peak_allocated_mb = tracker.after.max_allocated_mb
                tracker.delta_mb = tracker.after.allocated_mb - tracker.before.allocated_mb


# ============================================================================
# Throughput Calculation
# ============================================================================

def calculate_throughput(
    result: TimingResult,
    batch_size: int,
    unit: str = "samples/sec"
) -> float:
    """
    Calculate throughput from timing result.

    Args:
        result: TimingResult from benchmark
        batch_size: Batch size per iteration
        unit: Output unit (samples/sec, tokens/sec, images/sec)

    Returns:
        Throughput value
    """
    if result.avg_time_ms <= 0:
        return 0.0

    samples_per_ms = batch_size / result.avg_time_ms
    return samples_per_ms * 1000  # Convert to per-second


def calculate_tokens_per_second(
    result: TimingResult,
    batch_size: int,
    sequence_length: int,
) -> float:
    """
    Calculate token throughput for transformer models.

    Args:
        result: TimingResult from benchmark
        batch_size: Batch size per iteration
        sequence_length: Sequence length

    Returns:
        Tokens per second
    """
    total_tokens = batch_size * sequence_length
    return calculate_throughput(result, total_tokens, unit="tokens/sec")


# ============================================================================
# Comparison Utilities
# ============================================================================

def compare_results(
    baseline: TimingResult,
    optimized: TimingResult,
    print_comparison: bool = True,
) -> dict[str, float]:
    """
    Compare two timing results.

    Args:
        baseline: Baseline timing result
        optimized: Optimized timing result
        print_comparison: Whether to print comparison

    Returns:
        Dictionary with speedup and other metrics
    """
    speedup = baseline.avg_time_ms / optimized.avg_time_ms if optimized.avg_time_ms > 0 else 0.0
    improvement_pct = (1 - optimized.avg_time_ms / baseline.avg_time_ms) * 100 if baseline.avg_time_ms > 0 else 0.0

    comparison = {
        "speedup": speedup,
        "improvement_pct": improvement_pct,
        "baseline_avg_ms": baseline.avg_time_ms,
        "optimized_avg_ms": optimized.avg_time_ms,
        "time_saved_ms": baseline.avg_time_ms - optimized.avg_time_ms,
    }

    if print_comparison:
        print(f"\n📊 Comparison: {baseline.name} vs {optimized.name}")
        print(f"   Baseline:    {baseline.avg_time_ms:.4f} ms")
        print(f"   Optimized:   {optimized.avg_time_ms:.4f} ms")
        print(f"   Speedup:     {speedup:.2f}x")
        print(f"   Improvement: {improvement_pct:.1f}%")

    return comparison


# ============================================================================
# Reporting Utilities
# ============================================================================

def print_section_header(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def print_result(result: TimingResult, indent: int = 2) -> None:
    """Print a formatted benchmark result."""
    prefix = " " * indent
    print(f"{prefix}{result.name}:")
    print(f"{prefix}  Iterations: {result.iterations}")
    print(f"{prefix}  Average:    {result.avg_time_ms:.4f} ms")
    print(f"{prefix}  Min/Max:    {result.min_time_ms:.4f} / {result.max_time_ms:.4f} ms")
    print(f"{prefix}  Std Dev:    {result.std_dev_ms:.4f} ms")

    if result.percentiles:
        percentile_str = ", ".join(f"{k}: {v:.4f}ms" for k, v in result.percentiles.items())
        print(f"{prefix}  Percentiles: {percentile_str}")


def create_summary_table(
    results: list[TimingResult],
    title: str = "Benchmark Results"
) -> str:
    """
    Create a formatted summary table of results.

    Args:
        results: List of TimingResult objects
        title: Table title

    Returns:
        Formatted string table
    """
    if not results:
        return "No results to display"

    # Calculate column widths
    name_width = max(len(r.name) for r in results)
    name_width = max(name_width, 10)

    # Header
    lines = [
        f"\n{title}",
        "-" * (name_width + 50),
        f"{'Name':<{name_width}} | {'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10} | {'Std Dev':>10}",
        "-" * (name_width + 50),
    ]

    # Data rows
    for r in results:
        lines.append(
            f"{r.name:<{name_width}} | {r.avg_time_ms:>10.4f} | {r.min_time_ms:>10.4f} | {r.max_time_ms:>10.4f} | {r.std_dev_ms:>10.4f}"
        )

    lines.append("-" * (name_width + 50))

    return "\n".join(lines)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Data classes
    "TimingResult",
    "BenchmarkConfig",
    "MemorySnapshot",
    # Core functions
    "run_timed_iterations",
    "benchmark_function",
    # Context managers
    "timer",
    "track_memory",
    # Decorators
    "timed",
    # Memory utilities
    "get_cuda_memory_snapshot",
    # Throughput
    "calculate_throughput",
    "calculate_tokens_per_second",
    # Comparison
    "compare_results",
    # Reporting
    "print_section_header",
    "print_result",
    "create_summary_table",
]
