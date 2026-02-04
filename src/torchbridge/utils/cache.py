"""
Shared Cache Utilities for TorchBridge

This module provides shared caching implementations used across backends:
- LRUCache: Generic Least Recently Used cache with statistics tracking

Consolidated from:
- backends/amd/rocm_compiler.py
- backends/tpu/cache_utils.py
"""

import logging
from collections import OrderedDict
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """
    Least Recently Used (LRU) cache with size limits and statistics.

    Automatically evicts least recently used items when cache exceeds max_size.
    Thread-safe for single-threaded operations.

    Type Parameters:
        K: Key type
        V: Value type

    Example:
        >>> cache: LRUCache[str, int] = LRUCache(max_size=100)
        >>> cache.set("key1", 42)
        >>> value = cache.get("key1")  # Returns 42
        >>> stats = cache.stats()  # Get hit/miss statistics
    """

    def __init__(self, max_size: int = 100, enable_stats: bool = True):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
            enable_stats: Whether to track hit/miss statistics
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._max_size = max_size
        self._enable_stats = enable_stats

        # Statistics tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.debug("LRU Cache initialized: max_size=%d, stats=%s", max_size, enable_stats)

    def get(self, key: K) -> V | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            if self._enable_stats:
                self._hits += 1
            return self._cache[key]

        if self._enable_stats:
            self._misses += 1
        return None

    def set(self, key: K, value: V) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            # Update existing key, move to end
            self._cache.move_to_end(key)
        else:
            # New key, check if we need to evict
            while len(self._cache) >= self._max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                if self._enable_stats:
                    self._evictions += 1
                logger.debug("Cache eviction: key=%s", evicted_key)

        self._cache[key] = value

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache (does not affect LRU order)."""
        return key in self._cache

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)

    def __getitem__(self, key: K) -> V:
        """Get item using bracket notation, raises KeyError if not found."""
        value = self.get(key)
        if value is None and key not in self._cache:
            raise KeyError(key)
        return value  # type: ignore

    def __setitem__(self, key: K, value: V) -> None:
        """Set item using bracket notation."""
        self.set(key, value)

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def keys(self):
        """Return cache keys."""
        return self._cache.keys()

    def values(self):
        """Return cache values."""
        return self._cache.values()

    def items(self):
        """Return cache items."""
        return self._cache.items()

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self._max_size if self._max_size > 0 else 0.0
        }

    def get_stats(self) -> dict[str, Any]:
        """Alias for stats() for backward compatibility."""
        return self.stats()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def max_size(self) -> int:
        """Get maximum cache size."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        """Set maximum cache size, evicting items if necessary."""
        self._max_size = value
        while len(self._cache) > self._max_size:
            evicted_key = next(iter(self._cache))
            del self._cache[evicted_key]
            if self._enable_stats:
                self._evictions += 1


class TTLCache(Generic[K, V]):
    """
    Time-To-Live (TTL) cache with automatic expiration.

    Items are automatically expired after the specified TTL.
    Useful for caching data that becomes stale over time.

    Type Parameters:
        K: Key type
        V: Value type
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0):
        """
        Initialize TTL cache.

        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live in seconds for cached items
        """
        import time
        self._cache: OrderedDict[K, tuple] = OrderedDict()  # (value, timestamp)
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._time_fn = time.time

    def get(self, key: K) -> V | None:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if self._time_fn() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                return value
            else:
                # Expired, remove
                del self._cache[key]
        return None

    def set(self, key: K, value: V) -> None:
        """Set value in cache with current timestamp."""
        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            del self._cache[next(iter(self._cache))]

        self._cache[key] = (value, self._time_fn())

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired items. Returns count of removed items."""
        current_time = self._time_fn()
        expired_keys = [
            k for k, (v, ts) in self._cache.items()
            if current_time - ts >= self._ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)


__all__ = ['LRUCache', 'TTLCache']
