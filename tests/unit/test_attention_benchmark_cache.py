"""Tests for B4 fix: benchmark cache lazy-warming in AttentionDispatcher.

Verifies that select_kernel() populates the benchmark cache on the first
call (lazy warm), so benchmark_latency_ms is no longer always None.
"""

import tempfile

from torchbridge.attention.dispatch.benchmark_cache import KernelBenchmarkCache
from torchbridge.attention.dispatch.dispatcher import (
    AttentionDispatcher,
    AttentionKernelType,
)
from torchbridge.core.config import HardwareBackend


class TestBenchmarkCacheLazyWarm:
    """B4: Dispatcher populates cache on first select_kernel() call."""

    def test_latency_populated_after_select_kernel(self):
        """benchmark_latency_ms must be a positive float after select_kernel()."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = KernelBenchmarkCache(cache_dir=tmp)
            dispatcher = AttentionDispatcher(
                backend=HardwareBackend.CPU, use_benchmark_cache=True
            )
            # Inject a fresh cache pointing to the temp dir
            dispatcher._cache = cache

            result = dispatcher.select_kernel(
                seq_length=128, num_heads=4, head_dim=64
            )
            assert result.benchmark_latency_ms is not None, (
                "benchmark_latency_ms should be populated by lazy warm after B4 fix"
            )
            assert result.benchmark_latency_ms > 0.0

    def test_second_call_uses_cached_value(self):
        """Second select_kernel() call must reuse the cached latency without re-benchmarking."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = KernelBenchmarkCache(cache_dir=tmp)
            dispatcher = AttentionDispatcher(
                backend=HardwareBackend.CPU, use_benchmark_cache=True
            )
            dispatcher._cache = cache

            result1 = dispatcher.select_kernel(seq_length=128, num_heads=4, head_dim=64)
            result2 = dispatcher.select_kernel(seq_length=128, num_heads=4, head_dim=64)

            # Both should have the same latency (read from cache the second time)
            assert result1.benchmark_latency_ms is not None
            assert result2.benchmark_latency_ms is not None
            assert abs(result1.benchmark_latency_ms - result2.benchmark_latency_ms) < 1e-6, (
                "Second call should return identical cached latency"
            )

    def test_cache_disabled_gives_none_latency(self):
        """When use_benchmark_cache=False, benchmark_latency_ms must be None."""
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        result = dispatcher.select_kernel(seq_length=128, num_heads=4, head_dim=64)
        assert result.benchmark_latency_ms is None

    def test_run_benchmark_returns_entry_with_positive_latency(self):
        """KernelBenchmarkCache.run_benchmark() must return a positive latency."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = KernelBenchmarkCache(cache_dir=tmp)
            entry = cache.run_benchmark(
                AttentionKernelType.PYTORCH_SDPA,
                seq_length=64,
                num_heads=2,
                head_dim=32,
                warmup=1,
                iterations=3,
            )
            assert entry.latency_ms > 0.0
            assert entry.seq_length == 64
            assert entry.num_heads == 2
            assert entry.head_dim == 32
