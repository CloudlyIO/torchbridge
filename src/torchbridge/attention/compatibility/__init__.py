"""
Backward Compatibility Layer

Provides compatibility for imports from the old attention/ and advanced_attention/
directories to ensure existing code continues to work.
"""

import warnings
from typing import Any

from ..core import AttentionConfig, AttentionPatterns, FP8AttentionConfig

try:
    from ..distributed.context_parallel import ContextParallelAttention
    from ..distributed.ring_attention import RingAttentionLayer
except ImportError:
    ContextParallelAttention = None
    RingAttentionLayer = None

try:
    from ..fusion.neural_operator import create_unified_attention_fusion
except ImportError:
    create_unified_attention_fusion = None

# Import all components for re-export
from ..implementations.flash_attention import FlashAttention2, FlashAttention3
from ..implementations.memory_efficient import MemoryEfficientAttention
from ..implementations.sparse import DynamicSparseAttention


def __getattr__(name: str) -> Any:
    """
    Provide backward compatibility for old import patterns.
    Issues deprecation warnings for imports that should be updated.
    """
    # Map old names to new implementations
    compatibility_map = {
        # From advanced_attention/
        'FlashAttention3': FlashAttention3,
        'MultiHeadFlashAttention3': FlashAttention3,  # Alias
        'RingAttentionLayer': RingAttentionLayer,
        'DynamicSparseAttention': DynamicSparseAttention,
        'ContextParallelAttention': ContextParallelAttention,

        # From attention/
        'MemoryEfficientAttention': MemoryEfficientAttention,
        'FlashAttention2': FlashAttention2,

        # Configuration classes
        'FP8AttentionConfig': FP8AttentionConfig,
        'AttentionConfig': AttentionConfig,
        'AttentionPatterns': AttentionPatterns,
    }

    if name in compatibility_map:
        warnings.warn(
            f"Importing {name} from compatibility layer is deprecated. "
            f"Use: from torchbridge.attention_unified import {name}",
            DeprecationWarning,
            stacklevel=2
        )
        return compatibility_map[name]

    raise AttributeError(f"module 'torchbridge.attention_unified.compatibility' has no attribute '{name}'")


# Export everything for direct access
__all__ = [
    'FlashAttention3',
    'FlashAttention2',
    'MemoryEfficientAttention',
    'DynamicSparseAttention',
    'RingAttentionLayer',
    'ContextParallelAttention',
    'create_unified_attention_fusion',
    'AttentionConfig',
    'AttentionPatterns',
    'FP8AttentionConfig'
]
