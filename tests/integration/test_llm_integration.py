"""
Test Suite for LLM KV-Cache Integration

Tests for KVCacheManager, PagedKVCache, SlidingWindowCache, and
the KV-cache compatibility matrix.
"""

import pytest
import torch


class TestKVCacheManager:
    """Tests for KVCacheManager class."""

    def test_cache_manager_creation(self):
        """Test cache manager creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            max_length=2048,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            dtype=torch.float16,
            device="cpu"
        )

        manager = KVCacheManager(config)
        assert manager.config == config

    def test_create_cache(self):
        """Test cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu"
        )

        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=2)

        assert len(cache) == 4  # num_layers
        for key_cache, value_cache in cache:
            assert key_cache.shape == (2, 8, 0, 64)
            assert value_cache.shape == (2, 8, 0, 64)

    def test_update_cache(self):
        """Test cache update."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            max_length=100,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        new_keys = torch.randn(1, 4, 10, 32)
        new_values = torch.randn(1, 4, 10, 32)

        cache = manager.update_cache(cache, new_keys, new_values, layer_idx=0)

        assert cache[0][0].shape[2] == 10
        assert cache[0][1].shape[2] == 10

    def test_cache_truncation(self):
        """Test that cache truncates when exceeding max length."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(
            max_length=20,
            num_layers=1,
            num_heads=2,
            head_dim=16,
            device="cpu"
        )

        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        for _ in range(5):
            new_keys = torch.randn(1, 2, 10, 16)
            new_values = torch.randn(1, 2, 10, 16)
            cache = manager.update_cache(cache, new_keys, new_values, 0)

        assert cache[0][0].shape[2] == 20

    def test_get_cache_length(self):
        """Test cache length retrieval."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(num_layers=1, num_heads=2, head_dim=16, device="cpu")
        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        assert manager.get_cache_length(cache) == 0

        new_keys = torch.randn(1, 2, 5, 16)
        new_values = torch.randn(1, 2, 5, 16)
        cache = manager.update_cache(cache, new_keys, new_values, 0)

        assert manager.get_cache_length(cache) == 5

    def test_get_memory_usage(self):
        """Test memory usage calculation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

        config = CacheConfig(num_layers=2, num_heads=4, head_dim=32, device="cpu")
        manager = KVCacheManager(config)
        cache = manager.create_cache(batch_size=1)

        new_keys = torch.randn(1, 4, 10, 32)
        new_values = torch.randn(1, 4, 10, 32)
        cache = manager.update_cache(cache, new_keys, new_values, 0)

        usage = manager.get_memory_usage(cache)

        assert "cache_memory_mb" in usage
        assert "cache_length" in usage
        assert "num_layers" in usage
        assert usage["cache_memory_mb"] > 0


class TestPagedKVCache:
    """Tests for PagedKVCache class."""

    def test_paged_cache_creation(self):
        """Test paged cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=16,
            page_size=8,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        assert paged_cache.config == config

    def test_initialize_pages(self):
        """Test page initialization."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=8,
            page_size=4,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        paged_cache.initialize_pages()

        assert paged_cache.physical_pages is not None
        assert len(paged_cache.free_pages) == 8

    def test_allocate_pages(self):
        """Test page allocation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=16,
            page_size=4,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        page_table = paged_cache.allocate_pages(batch_size=2, num_pages_per_seq=4)

        assert page_table.shape == (2, 4)
        assert len(paged_cache.free_pages) == 8  # 16 - 8 allocated

    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        from torchbridge.models.llm.kv_cache import CacheConfig, PagedKVCache

        config = CacheConfig(
            num_pages=8,
            page_size=4,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        paged_cache = PagedKVCache(config)
        paged_cache.initialize_pages()
        paged_cache.allocate_pages(batch_size=1, num_pages_per_seq=2)

        usage = paged_cache.get_memory_usage()

        assert "total_mb" in usage
        assert "used_pages" in usage
        assert "free_pages" in usage
        assert "utilization" in usage
        assert usage["used_pages"] == 2
        assert usage["free_pages"] == 6


class TestSlidingWindowCache:
    """Tests for SlidingWindowCache class."""

    def test_sliding_window_creation(self):
        """Test sliding window cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=256,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        assert sw_cache.window_size == 256

    def test_create_cache(self):
        """Test cache creation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=128,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        cache = sw_cache.create_cache(batch_size=1)

        assert len(cache) == 2
        assert cache[0][0].shape[2] == 0

    def test_update_within_window(self):
        """Test update within window size."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=100,
            num_layers=1,
            num_heads=2,
            head_dim=16,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        cache = sw_cache.create_cache(batch_size=1)

        new_keys = torch.randn(1, 2, 50, 16)
        new_values = torch.randn(1, 2, 50, 16)

        cache = sw_cache.update_cache(cache, new_keys, new_values, 0)

        assert cache[0][0].shape[2] == 50

    def test_sliding_window_truncation(self):
        """Test that cache slides when exceeding window size."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(
            window_size=50,
            num_layers=1,
            num_heads=2,
            head_dim=16,
            device="cpu"
        )

        sw_cache = SlidingWindowCache(config)
        cache = sw_cache.create_cache(batch_size=1)

        for _ in range(3):
            new_keys = torch.randn(1, 2, 30, 16)
            new_values = torch.randn(1, 2, 30, 16)
            cache = sw_cache.update_cache(cache, new_keys, new_values, 0)

        assert cache[0][0].shape[2] == 50

    def test_get_window_mask(self):
        """Test window mask generation."""
        from torchbridge.models.llm.kv_cache import CacheConfig, SlidingWindowCache

        config = CacheConfig(window_size=10, device="cpu")
        sw_cache = SlidingWindowCache(config)

        mask = sw_cache.get_window_mask(seq_len=5, cache_len=20)

        assert mask.shape == (5, 25)  # seq_len x (cache_len + seq_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
