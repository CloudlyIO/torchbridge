"""
KV-Cache Optimization Subpackage

Backend-aware KV-cache quantization and prefix caching for LLM serving.
"""

from .cache_compatibility import KVCacheCompatibilityMatrix
from .cache_dtype import KV_DTYPE_SPECS, KVCacheDtype, KVDtypeSpec, get_kv_dtype_spec
from .quantized_cache import PrefixCache, QuantizedCacheConfig, QuantizedKVCache

__all__ = [
    "KVCacheDtype",
    "KVDtypeSpec",
    "KV_DTYPE_SPECS",
    "get_kv_dtype_spec",
    "KVCacheCompatibilityMatrix",
    "QuantizedCacheConfig",
    "QuantizedKVCache",
    "PrefixCache",
]
