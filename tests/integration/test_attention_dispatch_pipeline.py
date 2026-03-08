"""Integration tests for the attention dispatch pipeline."""

from torchbridge.attention.dispatch import (
    AttentionKernelType,
    KernelBenchmarkCache,
)


class TestBenchmarkCacheIntegration:
    """Benchmark cache warm_cache on CPU."""

    def test_warm_cache_cpu(self, tmp_path):
        cache = KernelBenchmarkCache(cache_dir=str(tmp_path))
        results = cache.warm_cache(
            kernel_types=[AttentionKernelType.PYTORCH_SDPA],
            seq_length=64,
            num_heads=4,
            head_dim=16,
        )
        assert "pytorch_sdpa" in results
        assert results["pytorch_sdpa"] > 0.0

        # Verify cached value
        latency = cache.get_cached_latency(
            AttentionKernelType.PYTORCH_SDPA, 64, 4, 16
        )
        assert latency is not None
        assert latency == results["pytorch_sdpa"]

    def test_cache_persistence(self, tmp_path):
        """Cache survives re-instantiation."""
        cache1 = KernelBenchmarkCache(cache_dir=str(tmp_path))
        cache1.run_benchmark(
            AttentionKernelType.PYTORCH_SDPA, 64, 4, 16
        )

        cache2 = KernelBenchmarkCache(cache_dir=str(tmp_path))
        latency = cache2.get_cached_latency(
            AttentionKernelType.PYTORCH_SDPA, 64, 4, 16
        )
        assert latency is not None
