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

# Distributed attention - import what exists
try:
    from .distributed.ring_attention import RingAttentionLayer
except ImportError:
    RingAttentionLayer = None

try:
    from .distributed.context_parallel import ContextParallelAttention
except ImportError:
    ContextParallelAttention = None

# Advanced fusion - import what exists
try:
    from .fusion.neural_operator import create_unified_attention_fusion
except ImportError:
    create_unified_attention_fusion = None

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

# Add available distributed components
if RingAttentionLayer is not None:
    __all__.append('RingAttentionLayer')

if ContextParallelAttention is not None:
    __all__.append('ContextParallelAttention')

# Add available fusion components
if create_unified_attention_fusion is not None:
    __all__.append('create_unified_attention_fusion')
