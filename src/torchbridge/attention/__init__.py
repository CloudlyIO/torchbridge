"""
Unified Attention Framework

Consolidates all attention implementations from attention/ and advanced_attention/
into a single, unified framework with consistent interfaces and enhanced capabilities.
"""

# Core framework
from .core import (
    AttentionConfig,  # Backward compat alias for AttentionModuleConfig
    AttentionModuleConfig,
    AttentionPatterns,
    BaseAttention,
    DynamicSparseConfig,
    FP8AttentionConfig,
    RingAttentionConfig,
    create_attention,
    register_attention,
)

# Main implementations
from .implementations.flash_attention import FlashAttention2, FlashAttention3
from .implementations.flex_attention import (
    FlexAttentionCausal,
    FlexAttentionLayer,
    FlexAttentionScoreMods,
    FlexAttentionSlidingWindow,
    create_flex_attention,
    get_flex_attention_info,
    is_flex_attention_available,
)

# Build dynamic exports
__all__ = [
    # Core framework
    'BaseAttention',
    'AttentionConfig',  # Backward compat alias
    'AttentionModuleConfig',
    'AttentionPatterns',
    'FP8AttentionConfig',
    'DynamicSparseConfig',
    'RingAttentionConfig',
    'register_attention',
    'create_attention',

    # FlashAttention implementations
    'FlashAttention3',
    'FlashAttention2',

    # FlexAttention implementations (PyTorch 2.5+)
    'FlexAttentionLayer',
    'FlexAttentionCausal',
    'FlexAttentionSlidingWindow',
    'FlexAttentionScoreMods',
    'create_flex_attention',
    'is_flex_attention_available',
    'get_flex_attention_info',
]
