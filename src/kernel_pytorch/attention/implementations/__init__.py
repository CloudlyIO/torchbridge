"""
Unified Attention Implementations

Consolidated implementations from attention/ and advanced_attention/ directories
with enhanced features and consistent interfaces.
"""

from .flash_attention import FlashAttention3, FlashAttention2
from .memory_efficient import MemoryEfficientAttention, ChunkedAttention, LongSequenceAttention
from .sparse import DynamicSparseAttention, SparseAttentionPattern
from .flex_attention import (
    FlexAttentionLayer,
    FlexAttentionCausal,
    FlexAttentionSlidingWindow,
    FlexAttentionScoreMods,
    FlexAttentionMaskGenerators,
    create_flex_attention,
    is_flex_attention_available,
    get_flex_attention_info,
)

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