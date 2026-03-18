"""
Tests for QuantizedKVCache dtype resolution (v0.5.77+)

Tests backend-aware KV dtype selection via the compatibility matrix.
PrefixCache and cache management wrapper methods were deleted in v0.5.77.
"""

from torchbridge.models.llm.kv.cache_dtype import KVCacheDtype
from torchbridge.models.llm.kv.quantized_cache import (
    QuantizedCacheConfig,
    QuantizedKVCache,
)


class TestQuantizedKVCacheDtypeResolution:
    """Tests for QuantizedKVCache dtype resolution via compatibility matrix."""

    def test_cpu_resolves_passthrough(self):
        """CPU backend should resolve to PASSTHROUGH dtype."""
        config = QuantizedCacheConfig()
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_explicit_supported_dtype_honoured(self):
        """Explicitly requested dtype should be used if supported on backend."""
        config = QuantizedCacheConfig(kv_cache_dtype=KVCacheDtype.PASSTHROUGH)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_unsupported_dtype_falls_back_to_optimal(self):
        """Requesting NVFP4 on CPU must fall back to the matrix optimal."""
        config = QuantizedCacheConfig(kv_cache_dtype=KVCacheDtype.NVFP4)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_unknown_backend_defaults_to_cpu_behaviour(self):
        """Unknown backend string should map to CPU fallback."""
        config = QuantizedCacheConfig()
        cache = QuantizedKVCache(config, backend_name="unknown_hw")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_kv_dtype_property_returns_enum(self):
        """kv_dtype property must return a KVCacheDtype enum member."""
        config = QuantizedCacheConfig()
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert isinstance(cache.kv_dtype, KVCacheDtype)
