"""
Quantized KV-Cache with Prefix Caching

Provides KV-cache quantization (dtype casting per backend compatibility matrix)
and RadixAttention-style prefix caching for shared prompt deduplication.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from torchbridge.models.llm.kv_cache import CacheConfig, KVCacheManager

from .cache_compatibility import KVCacheCompatibilityMatrix
from .cache_dtype import KV_DTYPE_SPECS, KVCacheDtype

logger = logging.getLogger(__name__)


# -- Prefix Cache ------------------------------------------------------------


@dataclass
class PrefixCacheEntry:
    """A cached prefix with its KV tensors."""
    token_ids: tuple[int, ...]
    kv_tensors: list[tuple[torch.Tensor, torch.Tensor]]
    num_tokens: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class PrefixCache:
    """RadixAttention-style hash-based prefix cache with LRU eviction.

    Caches KV tensors for common prompt prefixes so that repeated prefixes
    do not require recomputation.

    Args:
        max_entries: Maximum number of cached prefix entries.
        max_tokens: Maximum total tokens across all cached entries.
    """

    def __init__(
        self,
        max_entries: int = 1024,
        max_tokens: int = 65536,
    ) -> None:
        self._max_entries = max_entries
        self._max_tokens = max_tokens
        self._cache: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self._total_tokens = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def _hash_tokens(token_ids: tuple[int, ...]) -> str:
        """SHA-256 hash of a token ID sequence."""
        raw = ",".join(str(t) for t in token_ids).encode()
        return hashlib.sha256(raw).hexdigest()

    def lookup(self, token_ids: tuple[int, ...]) -> PrefixCacheEntry | None:
        """Look up a prefix by token IDs.

        Returns the cached entry if found, or None on miss.
        """
        key = self._hash_tokens(token_ids)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                # Verify token_ids match (guard against hash collisions)
                if entry.token_ids != token_ids:
                    self._misses += 1
                    return None
                self._hits += 1
                entry.last_accessed = time.time()
                # Move to end for LRU
                self._cache.move_to_end(key)
                return entry
            self._misses += 1
            return None

    def insert(
        self,
        token_ids: tuple[int, ...],
        kv_tensors: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        """Insert a prefix and its KV tensors into the cache.

        Returns True if inserted, False if eviction could not make room.
        """
        num_tokens = len(token_ids)
        key = self._hash_tokens(token_ids)

        with self._lock:
            # Already cached — update access time
            if key in self._cache:
                self._cache[key].last_accessed = time.time()
                self._cache.move_to_end(key)
                return True

            # Evict until we have room
            while (
                len(self._cache) >= self._max_entries
                or self._total_tokens + num_tokens > self._max_tokens
            ) and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._total_tokens = max(0, self._total_tokens - evicted.num_tokens)

            # Check if single entry exceeds max
            if num_tokens > self._max_tokens:
                return False

            entry = PrefixCacheEntry(
                token_ids=token_ids,
                kv_tensors=kv_tensors,
                num_tokens=num_tokens,
            )
            self._cache[key] = entry
            self._total_tokens += num_tokens
            return True

    def get_stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "hits": self._hits,
                "misses": self._misses,
                "cached_entries": len(self._cache),
                "cached_tokens": self._total_tokens,
                "max_entries": self._max_entries,
                "max_tokens": self._max_tokens,
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._total_tokens = 0


# -- Quantized KV-Cache Config -----------------------------------------------


@dataclass
class QuantizedCacheConfig:
    """Configuration for quantized KV-cache."""
    cache_config: CacheConfig
    kv_dtype: KVCacheDtype | None = None  # None = auto-select
    enable_prefix_caching: bool = False
    prefix_cache_max_entries: int = 1024
    prefix_cache_max_tokens: int = 65536


# -- Quantized KV-Cache ------------------------------------------------------


class QuantizedKVCache:
    """KV-cache wrapper with backend-aware quantization and prefix caching.

    Wraps an inner ``KVCacheManager`` and applies dtype quantization based on
    the hardware compatibility matrix. Optionally integrates prefix caching
    for shared prompt deduplication.

    Args:
        config: Quantized cache configuration.
        backend_name: Backend name string (e.g. "cuda", "amd", "cpu").
        architecture: Optional architecture enum for fine-grained dtype selection.
    """

    def __init__(
        self,
        config: QuantizedCacheConfig,
        backend_name: str = "cpu",
        architecture=None,
    ) -> None:
        self._config = config
        self._inner = KVCacheManager(config.cache_config)
        self._backend_name = backend_name
        self._architecture = architecture

        # Resolve KV dtype
        self._kv_dtype = self._resolve_dtype(config.kv_dtype)
        self._dtype_spec = KV_DTYPE_SPECS.get(self._kv_dtype)

        # Prefix cache
        self._prefix_cache: PrefixCache | None = None
        if config.enable_prefix_caching:
            self._prefix_cache = PrefixCache(
                max_entries=config.prefix_cache_max_entries,
                max_tokens=config.prefix_cache_max_tokens,
            )

        logger.info(
            "QuantizedKVCache initialized: dtype=%s, prefix_caching=%s",
            self._kv_dtype.value,
            config.enable_prefix_caching,
        )

    @property
    def kv_dtype(self) -> KVCacheDtype:
        """The resolved KV-cache dtype."""
        return self._kv_dtype

    def _resolve_dtype(self, requested: KVCacheDtype | None) -> KVCacheDtype:
        """Auto-select KV dtype from backend compatibility matrix."""
        from torchbridge.core.config import HardwareBackend

        backend_map = {
            "cuda": HardwareBackend.CUDA,
            "amd": HardwareBackend.AMD,
            "trainium": HardwareBackend.TRAINIUM,
            "tpu": HardwareBackend.TPU,
            "cpu": HardwareBackend.CPU,
        }
        hw_backend = backend_map.get(self._backend_name, HardwareBackend.CPU)

        if requested is not None:
            if KVCacheCompatibilityMatrix.is_dtype_supported(
                requested, hw_backend, self._architecture
            ):
                return requested
            # Fallback to optimal
            logger.warning(
                "Requested KV dtype %s not supported on %s, falling back to optimal",
                requested.value,
                self._backend_name,
            )

        return KVCacheCompatibilityMatrix.get_optimal_dtype(
            hw_backend, self._architecture
        )

    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor to the resolved KV dtype."""
        if self._kv_dtype == KVCacheDtype.PASSTHROUGH:
            return tensor
        if self._dtype_spec is None or self._dtype_spec.torch_dtype is None:
            return tensor
        try:
            return tensor.to(self._dtype_spec.torch_dtype)
        except (RuntimeError, TypeError):
            logger.warning(
                "Failed to cast to %s, keeping original dtype",
                self._kv_dtype.value,
            )
            return tensor

    def create_cache(self, batch_size: int):
        """Create a new KV cache via the inner manager."""
        return self._inner.create_cache(batch_size)

    def update_cache(self, cache, new_keys, new_values, layer_idx):
        """Update cache with quantized KV tensors."""
        q_keys = self._quantize_tensor(new_keys)
        q_values = self._quantize_tensor(new_values)
        return self._inner.update_cache(cache, q_keys, q_values, layer_idx)

    def lookup_prefix(
        self, token_ids: tuple[int, ...]
    ) -> PrefixCacheEntry | None:
        """Look up cached KV tensors for a token prefix."""
        if self._prefix_cache is None:
            return None
        return self._prefix_cache.lookup(token_ids)

    def store_prefix(
        self,
        token_ids: tuple[int, ...],
        kv_tensors: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        """Store KV tensors for a token prefix."""
        if self._prefix_cache is None:
            return False
        return self._prefix_cache.insert(token_ids, kv_tensors)

    def get_memory_usage(self, cache=None) -> dict[str, Any]:
        """Get memory usage info, extended with KV dtype metadata."""
        if cache is not None:
            base: dict[str, Any] = self._inner.get_memory_usage(cache)
        else:
            base: dict[str, Any] = {"total_bytes": 0, "num_layers": 0}

        base["kv_dtype"] = self._kv_dtype.value
        if self._dtype_spec:
            base["memory_factor"] = self._dtype_spec.memory_factor
        return base

    def get_prefix_cache_stats(self) -> dict | None:
        """Get prefix cache statistics, or None if prefix caching is disabled."""
        if self._prefix_cache is None:
            return None
        return self._prefix_cache.get_stats()
