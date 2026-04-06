"""
Quantized KV-Cache Configuration

Provides backend-aware KV-cache dtype selection via the hardware compatibility
matrix. ``QuantizedKVCache`` resolves the optimal (or explicitly requested) KV
cache dtype for a given backend and architecture, falling back gracefully when
the requested dtype is unsupported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .cache_compatibility import KVCacheCompatibilityMatrix
from .cache_dtype import KVCacheDtype

logger = logging.getLogger(__name__)


@dataclass
class QuantizedCacheConfig:
    """Configuration for quantized KV-cache."""

    kv_cache_dtype: KVCacheDtype | None = None  # None = auto-select from matrix


class QuantizedKVCache:
    """Backend-aware KV-cache dtype resolver.

    Selects the optimal KV-cache dtype for the given backend and architecture
    using the hardware compatibility matrix. If a specific dtype is requested
    and supported, it is used directly; otherwise the matrix optimal is chosen.

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
        self._backend_name = backend_name
        self._architecture = architecture
        self._kv_dtype = self._resolve_dtype(config.kv_cache_dtype)

    @property
    def kv_dtype(self) -> KVCacheDtype:
        """The resolved KV-cache dtype."""
        return self._kv_dtype

    def _resolve_dtype(self, requested: KVCacheDtype | None) -> KVCacheDtype:
        """Select KV dtype from backend compatibility matrix."""
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
            logger.warning(
                "Requested KV dtype %s not supported on %s, falling back to optimal",
                requested.value,
                self._backend_name,
            )

        return KVCacheCompatibilityMatrix.get_optimal_dtype(
            hw_backend, self._architecture
        )
