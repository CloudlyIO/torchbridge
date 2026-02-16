"""Integration tests for the attention dispatch pipeline."""

import torch

from torchbridge.attention.core.config import AttentionModuleConfig
from torchbridge.attention.dispatch import (
    AttentionDispatcher,
    AttentionKernelType,
    KernelBenchmarkCache,
)
from torchbridge.core.config import HardwareBackend


class TestEndToEndCPU:
    """End-to-end dispatch on CPU."""

    def test_dispatcher_select_and_create(self):
        """Dispatcher selects kernel then creates a working attention layer."""
        config = AttentionModuleConfig(embed_dim=64, num_heads=4)
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )

        result = dispatcher.select_kernel(
            seq_length=config.max_sequence_length,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
        )
        assert result.kernel_type == AttentionKernelType.PYTORCH_SDPA

        attn = dispatcher.create_attention(config)
        x = torch.randn(1, 16, 64)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (1, 16, 64)

    def test_gqa_forward_pass(self):
        """GQA attention: fewer KV heads, output shape unchanged."""
        config = AttentionModuleConfig(
            embed_dim=64, num_heads=8, num_kv_heads=2
        )
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        attn = dispatcher.create_attention(config)

        x = torch.randn(2, 32, 64)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (2, 32, 64)

    def test_mqa_forward_pass(self):
        """MQA: single KV head."""
        config = AttentionModuleConfig(
            embed_dim=64, num_heads=8, num_kv_heads=1
        )
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        attn = dispatcher.create_attention(config)

        x = torch.randn(1, 16, 64)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (1, 16, 64)


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
