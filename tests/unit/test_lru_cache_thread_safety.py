"""
Thread-safety tests for LRUCache and TTLCache.

These tests verify that concurrent access does not corrupt cache state.
All tests will FAIL until threading.RLock is added to LRUCache and TTLCache.
"""

import threading


class TestLRUCacheThreadSafety:
    """Concurrent access must not corrupt LRUCache internal state."""

    def test_concurrent_get_set_no_corruption(self):
        """10 threads doing 100 get/set each; final state must be consistent."""
        from torchbridge.utils.cache import LRUCache

        cache: LRUCache[str, int] = LRUCache(max_size=50)
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            try:
                for i in range(100):
                    key = f"k{i % 20}"
                    cache.set(key, tid * 1000 + i)
                    _ = cache.get(key)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised during concurrent access: {errors}"
        assert len(cache) <= 50, "Cache exceeded max_size under concurrency"

    def test_concurrent_eviction_no_crash(self):
        """Threads hitting eviction boundary simultaneously must not crash."""
        from torchbridge.utils.cache import LRUCache

        cache: LRUCache[str, int] = LRUCache(max_size=10)
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            try:
                for i in range(50):
                    cache.set(f"t{tid}_k{i}", i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Crash during concurrent eviction: {errors}"
        assert len(cache) <= 10

    def test_stats_are_consistent_under_concurrency(self):
        """hits + misses must equal total get() calls after concurrent access."""
        from torchbridge.utils.cache import LRUCache

        cache: LRUCache[str, int] = LRUCache(max_size=20)
        # Pre-populate some keys
        for i in range(10):
            cache.set(f"key{i}", i)

        n_threads = 8
        gets_per_thread = 100
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(gets_per_thread):
                    cache.get(f"key{i % 15}")  # half will miss
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        s = cache.stats()
        total_gets = n_threads * gets_per_thread
        assert s["hits"] + s["misses"] == total_gets, (
            f"hits ({s['hits']}) + misses ({s['misses']}) != {total_gets}"
        )

    def test_ttl_cache_concurrent_get_set(self):
        """TTLCache must not corrupt its internal OrderedDict under concurrent writes."""
        from torchbridge.utils.cache import TTLCache

        cache: TTLCache[str, int] = TTLCache(max_size=30, ttl_seconds=60.0)
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            try:
                for i in range(100):
                    key = f"k{i % 25}"
                    cache.set(key, tid * 100 + i)
                    _ = cache.get(key)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions in TTLCache concurrent access: {errors}"
        assert len(cache._cache) <= 30
