"""
Unified Attention Implementations

Consolidated implementations from attention/ and advanced_attention/ directories
with enhanced features and consistent interfaces.
"""

from .flash_attention import FlashAttention2, FlashAttention3
from .flex_attention import (
    FlexAttentionCausal,
    FlexAttentionLayer,
    FlexAttentionMaskGenerators,
    FlexAttentionScoreMods,
    FlexAttentionSlidingWindow,
    create_flex_attention,
    get_flex_attention_info,
    is_flex_attention_available,
)
from .memory_efficient import (
    ChunkedAttention,
    LongSequenceAttention,
    MemoryEfficientAttention,
)
from .sparse import DynamicSparseAttention, SparseAttentionPattern

__all__ = [
    # Flash attention implementations
    'FlashAttention3',
    'FlashAttention2',

    # FlexAttention implementations (PyTorch 2.5+)
    'FlexAttentionLayer',
    'FlexAttentionCausal',
    'FlexAttentionSlidingWindow',
    'FlexAttentionScoreMods',
    'FlexAttentionMaskGenerators',
    'create_flex_attention',
    'is_flex_attention_available',
    'get_flex_attention_info',

    # Memory-efficient implementations
    'MemoryEfficientAttention',
    'ChunkedAttention',
    'LongSequenceAttention',

    # Sparse attention implementations
    'DynamicSparseAttention',
    'SparseAttentionPattern'
]
