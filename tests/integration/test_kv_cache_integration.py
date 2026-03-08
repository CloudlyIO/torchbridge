"""
Integration Tests for KV-Cache Pipeline

End-to-end tests combining KV-cache quantization, prefix caching,
and LLM metrics collection.
"""

import pytest
import torch

from torchbridge.models.llm.kv.cache_compatibility import KVCacheCompatibilityMatrix
from torchbridge.models.llm.kv.cache_dtype import KV_DTYPE_SPECS, KVCacheDtype
from torchbridge.models.llm.kv.quantized_cache import (
    QuantizedCacheConfig,
    QuantizedKVCache,
)
from torchbridge.models.llm.kv_cache import CacheConfig


class TestKVCachePipeline:
    """Integration tests for the KV-cache optimization pipeline."""

    @pytest.fixture
    def cache_config(self):
        return CacheConfig(
            max_length=128, num_layers=4, num_heads=8, head_dim=32,
            dtype=torch.float16, device="cpu",
        )

    def test_compatibility_to_cache_pipeline(self, cache_config):
        """Compatibility matrix -> QuantizedKVCache -> create + update."""
        from torchbridge.core.config import HardwareBackend

        optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(HardwareBackend.CPU)
        config = QuantizedCacheConfig(
            cache_config=cache_config,
            kv_dtype=optimal,
        )
        qcache = QuantizedKVCache(config, backend_name="cpu")
        kv = qcache.create_cache(batch_size=1)
        keys = torch.randn(1, 8, 1, 32)
        values = torch.randn(1, 8, 1, 32)
        updated = qcache.update_cache(kv, keys, values, layer_idx=0)
        assert updated is not None

    def test_prefix_cache_with_quantized_kv(self, cache_config):
        """Prefix cache stores and retrieves within QuantizedKVCache."""
        config = QuantizedCacheConfig(
            cache_config=cache_config,
            enable_prefix_caching=True,
        )
        qcache = QuantizedKVCache(config, backend_name="cpu")
        tokens = (100, 200, 300, 400)
        kv_tensors = [
            (torch.randn(1, 8, 4, 32), torch.randn(1, 8, 4, 32))
            for _ in range(4)
        ]

        assert qcache.store_prefix(tokens, kv_tensors) is True
        entry = qcache.lookup_prefix(tokens)
        assert entry is not None
        assert len(entry.kv_tensors) == 4

    def test_prefix_cache_hit_rate_across_requests(self, cache_config):
        """Hit rate should increase as repeated prefixes are served."""
        config = QuantizedCacheConfig(
            cache_config=cache_config,
            enable_prefix_caching=True,
        )
        qcache = QuantizedKVCache(config, backend_name="cpu")

        shared_prefix = tuple(range(50))
        kv = [(torch.randn(2, 4), torch.randn(2, 4)) for _ in range(4)]
        qcache.store_prefix(shared_prefix, kv)

        # Simulate 10 requests, 7 sharing the prefix
        for _ in range(7):
            qcache.lookup_prefix(shared_prefix)
        for _ in range(3):
            qcache.lookup_prefix(tuple(range(50, 100)))

        stats = qcache.get_prefix_cache_stats()
        assert stats is not None
        assert stats["hits"] == 7
        assert stats["misses"] == 3

    def test_all_kv_dtypes_have_consistent_specs(self):
        """Every dtype in every compatibility table should have a KV_DTYPE_SPECS entry."""
        from torchbridge.core.config import (
            AMDArchitecture,
            HardwareBackend,
            NVIDIAArchitecture,
            TPUVersion,
            TrainiumArchitecture,
        )

        backends = [
            (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
            (HardwareBackend.TPU, TPUVersion.V5E),
            (HardwareBackend.CPU, None),
        ]
        for backend, arch in backends:
            for dtype in KVCacheCompatibilityMatrix.get_supported_dtypes(backend, arch):
                assert dtype in KV_DTYPE_SPECS, (
                    f"dtype {dtype.value} from {backend.value} not in KV_DTYPE_SPECS"
                )

    def test_memory_factor_reduces_with_quantization(self):
        """FP8 should have lower memory factor than FP16/BF16."""
        fp16_spec = KV_DTYPE_SPECS[KVCacheDtype.FP16]
        fp8_spec = KV_DTYPE_SPECS[KVCacheDtype.FP8_E4M3]
        nvfp4_spec = KV_DTYPE_SPECS[KVCacheDtype.NVFP4]
        assert fp8_spec.memory_factor < fp16_spec.memory_factor
        assert nvfp4_spec.memory_factor < fp8_spec.memory_factor

    def test_quantized_cache_memory_usage_reports_dtype(self, cache_config):
        """Memory usage should report the resolved dtype."""
        config = QuantizedCacheConfig(cache_config=cache_config)
        qcache = QuantizedKVCache(config, backend_name="cpu")
        kv = qcache.create_cache(batch_size=1)
        usage = qcache.get_memory_usage(kv)
        assert usage["kv_dtype"] == KVCacheDtype.PASSTHROUGH.value
