"""
Unified Attention Module System

Consolidates all attention implementations into a well-organized hierarchy:
- Core attention utilities and base classes
- FlashAttention variants (v2, v3, FP8 support)
- Memory-efficient attention implementations
- Specialized attention patterns (Ring, Sparse, Differential)

This replaces the scattered attention implementations across multiple modules
with a clean, maintainable architecture.
"""

# Core attention utilities
from .core import (
    BaseAttention,
    AttentionConfig,
    AttentionPatterns,
    create_attention,
    get_attention_registry
)

# FlashAttention implementations
from .flash_attention import (
    FlashAttention2,
    FlashAttention3,
    FP8AttentionConfig,
    MultiHeadFlashAttention
)

# Memory-efficient attention
from .efficient_attention import (
    MemoryEfficientAttention,
    LongSequenceAttention,
    RingAttention,
    ChunkedAttention
)

# Specialized attention patterns
from .specialized import (
    SparseAttentionPattern,
    DifferentialAttention,
    FlexAttentionAPI,
    LocalAttention,
    SlidingWindowAttention
)

# Backward compatibility imports (deprecated)
from .legacy import (
    # Re-exports from old modules for compatibility
    CompilerOptimizedMultiHeadAttention,
    FlashAttentionWrapper
)

__all__ = [
    # Core utilities
    'BaseAttention',
    'AttentionConfig',
    'AttentionPatterns',
    'create_attention',
    'get_attention_registry',

    # FlashAttention family
    'FlashAttention2',
    'FlashAttention3',
    'FP8AttentionConfig',
    'MultiHeadFlashAttention',

    # Efficient attention
    'MemoryEfficientAttention',
    'LongSequenceAttention',
    'RingAttention',
    'ChunkedAttention',

    # Specialized patterns
    'SparseAttentionPattern',
    'DifferentialAttention',
    'FlexAttentionAPI',
    'LocalAttention',
    'SlidingWindowAttention',

    # Legacy (deprecated - use unified variants above)
    'CompilerOptimizedMultiHeadAttention',
    'FlashAttentionWrapper',
]