"""
TPU Cache Utilities

Provides LRU cache implementation with size limits to prevent unbounded growth.
"""

import logging
from collections import OrderedDict
from typing import Any, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """
    Least Recently Used (LRU) cache with size limits.

    Automatically evicts least recently used items when cache exceeds max_size.
    Thread-safe for single-threaded TPU operations.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.debug("LRU Cache initialized: max_size=%d", max_size)

    def get(self, key: K) -> Optional[V]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

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
            # Update existing key and move to end
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            # Add new key
            self._cache[key] = value

            # Evict oldest item if cache exceeds max size
            if len(self._cache) > self._max_size:
                evicted_key, evicted_value = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(
                    "LRU Cache eviction: key=%s, cache_size=%d/%d",
                    evicted_key,
                    len(self._cache),
                    self._max_size
                )

    def clear(self) -> None:
        """Clear all cache entries."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.debug("LRU Cache cleared: evicted=%d entries", cache_size)

    def __len__(self) -> int:
        """Get number of cached items."""
        return len(self._cache)

    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, evictions, size, hit_rate)
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'size': len(self._cache),
            'max_size': self._max_size,
            'hit_rate': hit_rate,
            'utilization': len(self._cache) / self._max_size if self._max_size > 0 else 0.0
        }

    def reset_stats(self) -> None:
        """Reset cache statistics (hits, misses, evictions)."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __repr__(self) -> str:
        """String representation of cache."""
        stats = self.get_stats()
        return (
            f"LRUCache(size={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.2%}, evictions={stats['evictions']})"
        )
