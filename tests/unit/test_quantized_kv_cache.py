"""
Tests for Quantized KV-Cache and Prefix Cache

Tests PrefixCache LRU eviction, hit/miss tracking, and QuantizedKVCache
dtype resolution, quantization, and prefix cache integration.
"""

import pytest
import torch

from torchbridge.models.llm.kv.cache_dtype import KVCacheDtype
from torchbridge.models.llm.kv.quantized_cache import (
    PrefixCache,
    QuantizedCacheConfig,
    QuantizedKVCache,
)
from torchbridge.models.llm.kv_cache import CacheConfig

# =============================================================================
# PrefixCache Tests
# =============================================================================


class TestPrefixCache:
    """Tests for the PrefixCache."""

    def test_insert_and_lookup(self):
        """Inserted prefix should be retrievable."""
        cache = PrefixCache(max_entries=10, max_tokens=1000)
        tokens = (1, 2, 3, 4, 5)
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]
        assert cache.insert(tokens, kv) is True
        entry = cache.lookup(tokens)
        assert entry is not None
        assert entry.token_ids == tokens
        assert entry.num_tokens == 5

    def test_cache_miss(self):
        """Looking up non-existent prefix should return None."""
        cache = PrefixCache()
        assert cache.lookup((99, 100)) is None

    def test_hit_miss_tracking(self):
        """Hit/miss counters should be accurate."""
        cache = PrefixCache()
        tokens = (1, 2, 3)
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]
        cache.insert(tokens, kv)

        cache.lookup(tokens)  # hit
        cache.lookup((4, 5, 6))  # miss
        cache.lookup(tokens)  # hit

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_lru_eviction_by_entries(self):
        """Oldest entry should be evicted when max_entries reached."""
        cache = PrefixCache(max_entries=2, max_tokens=10000)
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]

        cache.insert((1,), kv)
        cache.insert((2,), kv)
        cache.insert((3,), kv)  # should evict (1,)

        assert cache.lookup((1,)) is None
        assert cache.lookup((2,)) is not None
        assert cache.lookup((3,)) is not None

    def test_lru_eviction_by_tokens(self):
        """Entries should be evicted when max_tokens reached."""
        cache = PrefixCache(max_entries=100, max_tokens=5)
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]

        cache.insert((1, 2, 3), kv)  # 3 tokens
        cache.insert((4, 5, 6), kv)  # 3 tokens, should evict first

        assert cache.lookup((1, 2, 3)) is None
        assert cache.lookup((4, 5, 6)) is not None

    def test_duplicate_insert_updates_access(self):
        """Re-inserting same tokens should update access time."""
        cache = PrefixCache()
        tokens = (1, 2, 3)
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]

        cache.insert(tokens, kv)
        result = cache.insert(tokens, kv)
        assert result is True
        stats = cache.get_stats()
        assert stats["cached_entries"] == 1

    def test_entry_too_large_rejected(self):
        """Entry exceeding max_tokens should be rejected."""
        cache = PrefixCache(max_tokens=3)
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]
        result = cache.insert((1, 2, 3, 4), kv)
        assert result is False

    def test_clear(self):
        """Clear should remove all entries."""
        cache = PrefixCache()
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]
        cache.insert((1, 2), kv)
        cache.clear()
        stats = cache.get_stats()
        assert stats["cached_entries"] == 0
        assert stats["cached_tokens"] == 0

    def test_stats_initial(self):
        """Initial stats should have zero hits/misses."""
        cache = PrefixCache()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_stats_max_fields(self):
        """Stats should include max_entries and max_tokens."""
        cache = PrefixCache(max_entries=128, max_tokens=8192)
        stats = cache.get_stats()
        assert stats["max_entries"] == 128
        assert stats["max_tokens"] == 8192


# =============================================================================
# QuantizedKVCache Tests
# =============================================================================


class TestQuantizedKVCache:
    """Tests for the QuantizedKVCache."""

    @pytest.fixture
    def base_config(self):
        return CacheConfig(
            max_length=128, num_layers=2, num_heads=4, head_dim=16,
            dtype=torch.float16, device="cpu",
        )

    def test_cpu_resolves_passthrough(self, base_config):
        """CPU backend should resolve to PASSTHROUGH dtype."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_explicit_dtype_used_when_supported(self, base_config):
        """Explicitly requested dtype should be used if supported."""
        config = QuantizedCacheConfig(
            cache_config=base_config,
            kv_dtype=KVCacheDtype.PASSTHROUGH,
        )
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_unsupported_dtype_falls_back(self, base_config):
        """Unsupported dtype should fall back to optimal."""
        config = QuantizedCacheConfig(
            cache_config=base_config,
            kv_dtype=KVCacheDtype.NVFP4,
        )
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH

    def test_create_cache(self, base_config):
        """create_cache should delegate to inner manager."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="cpu")
        kv_cache = cache.create_cache(batch_size=1)
        assert kv_cache is not None

    def test_update_cache(self, base_config):
        """update_cache should work with quantized tensors."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="cpu")
        kv_cache = cache.create_cache(batch_size=1)
        new_keys = torch.randn(1, 4, 1, 16)
        new_values = torch.randn(1, 4, 1, 16)
        updated = cache.update_cache(kv_cache, new_keys, new_values, layer_idx=0)
        assert updated is not None

    def test_prefix_cache_disabled_by_default(self, base_config):
        """Prefix cache should be disabled by default."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.get_prefix_cache_stats() is None

    def test_prefix_cache_enabled(self, base_config):
        """Prefix cache should work when enabled."""
        config = QuantizedCacheConfig(
            cache_config=base_config,
            enable_prefix_caching=True,
            prefix_cache_max_entries=64,
        )
        cache = QuantizedKVCache(config, backend_name="cpu")
        stats = cache.get_prefix_cache_stats()
        assert stats is not None
        assert stats["max_entries"] == 64

    def test_store_and_lookup_prefix(self, base_config):
        """store_prefix and lookup_prefix should work together."""
        config = QuantizedCacheConfig(
            cache_config=base_config,
            enable_prefix_caching=True,
        )
        cache = QuantizedKVCache(config, backend_name="cpu")
        tokens = (10, 20, 30)
        kv = [(torch.randn(2, 4), torch.randn(2, 4))]

        assert cache.store_prefix(tokens, kv) is True
        entry = cache.lookup_prefix(tokens)
        assert entry is not None
        assert entry.token_ids == tokens

    def test_lookup_prefix_disabled(self, base_config):
        """lookup_prefix should return None when prefix caching disabled."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.lookup_prefix((1, 2)) is None

    def test_store_prefix_disabled(self, base_config):
        """store_prefix should return False when prefix caching disabled."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="cpu")
        assert cache.store_prefix((1, 2), []) is False

    def test_get_memory_usage(self, base_config):
        """get_memory_usage should include kv_dtype info."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="cpu")
        usage = cache.get_memory_usage()
        assert "kv_dtype" in usage
        assert "memory_factor" in usage

    def test_get_memory_usage_with_cache(self, base_config):
        """get_memory_usage should work with an actual cache object."""
        config = QuantizedCacheConfig(cache_config=base_config)
        qcache = QuantizedKVCache(config, backend_name="cpu")
        kv_cache = qcache.create_cache(batch_size=1)
        usage = qcache.get_memory_usage(kv_cache)
        assert "kv_dtype" in usage
        assert "num_layers" in usage

    def test_unknown_backend_defaults_to_cpu(self, base_config):
        """Unknown backend string should resolve to CPU behavior."""
        config = QuantizedCacheConfig(cache_config=base_config)
        cache = QuantizedKVCache(config, backend_name="unknown_hw")
        assert cache.kv_dtype == KVCacheDtype.PASSTHROUGH
