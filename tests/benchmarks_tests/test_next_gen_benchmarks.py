#!/usr/bin/env python3
"""
Next-Generation Optimizations Benchmark Test Suite

Benchmark tests for next-generation optimizations:
- Performance regression detection
- Optimization effectiveness validation
- Hardware compatibility testing

BENCHMARK TARGETS:
- PyGraph CUDA Graph optimization performance
"""

import os
import sys
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torchbridge.optimizations.next_gen import (
    AutoGraphCapture,
    SelectiveCUDAGraphs,
    create_pygraph_optimizer,
)


class BenchmarkTimer:
    """Utility class for accurate benchmarking."""

    def __init__(self, device: torch.device, warmup_steps: int = 3, benchmark_steps: int = 5):
        self.device = device
        self.warmup_steps = warmup_steps
        self.benchmark_steps = benchmark_steps

    def time_operation(self, operation, *args, **kwargs) -> dict[str, float]:
        """Time operation with proper warmup and statistics."""
        # Warmup
        for _ in range(self.warmup_steps):
            _ = operation(*args, **kwargs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.benchmark_steps):
            start_time = time.perf_counter()
            operation(*args, **kwargs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }


@pytest.fixture
def device():
    """Test device fixture."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def timer(device):
    """Benchmark timer fixture."""
    return BenchmarkTimer(device, warmup_steps=2, benchmark_steps=3)


class TestPyGraphBenchmarks:
    """Benchmark tests for PyGraph CUDA Graph optimizations."""

    def test_pygraph_optimizer_creation_performance(self, device, timer):
        """Benchmark PyGraph optimizer creation."""
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)

        def create_optimizer():
            return create_pygraph_optimizer(model, device=device, optimization_level="balanced")

        timing = timer.time_operation(create_optimizer)

        assert timing['mean_ms'] > 0, "Invalid timing"
        assert timing['mean_ms'] < 5000.0, f"Optimizer creation too slow: {timing['mean_ms']:.2f}ms"

    def test_auto_graph_capture_performance(self, device, timer):
        """Benchmark AutoGraphCapture tracking overhead."""
        auto_capture = AutoGraphCapture(device, capture_threshold=3)

        x = torch.randn(4, 256, device=device)

        def simple_tracked_func():
            def fn(inp):
                return inp * 2 + 1
            return auto_capture.track_execution(fn, (x,), "bench_pattern")

        timing = timer.time_operation(simple_tracked_func)

        assert timing['mean_ms'] > 0, "Invalid timing"
        # AutoGraphCapture tracking should have low overhead
        assert timing['mean_ms'] < 1000.0, f"Tracking overhead too high: {timing['mean_ms']:.2f}ms"

    def test_selective_cuda_graphs_profiling(self, device, timer):
        """Benchmark SelectiveCUDAGraphs profiling."""
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device).eval()

        optimizer = SelectiveCUDAGraphs(model, device)
        x = torch.randn(4, 256, device=device)

        def profile_op():
            return optimizer.profile_operation("bench_op", lambda inp: model(inp), (x,))

        timing = timer.time_operation(profile_op)

        assert timing['mean_ms'] > 0, "Invalid timing"
        assert timing['mean_ms'] < 5000.0, f"Profiling too slow: {timing['mean_ms']:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
