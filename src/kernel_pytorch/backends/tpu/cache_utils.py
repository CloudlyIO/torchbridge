"""
TPU Cache Utilities

Provides LRU cache implementation with size limits to prevent unbounded growth.

Note: This module now re-exports from the shared cache utility for consistency.
The LRUCache class is available from kernel_pytorch.utils.cache.
"""

# Re-export from shared cache module for backward compatibility
from kernel_pytorch.utils.cache import LRUCache

__all__ = ['LRUCache']
