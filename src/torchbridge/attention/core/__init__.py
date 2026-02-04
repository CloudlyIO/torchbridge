"""
Core attention framework components
"""

from .attention_ops import (
    check_cuda_kernel_available,
    check_flash_attention_available,
    compute_attention_scale,
    flash_attention_forward,
    scaled_dot_product_attention,
    validate_attention_inputs,
)
from .base import AttentionWithCache, BaseAttention
from .config import (
    AttentionConfig,  # Backward compat alias for AttentionModuleConfig
    AttentionModuleConfig,
    AttentionPatterns,
    DynamicSparseConfig,
    FP8AttentionConfig,
    RingAttentionConfig,
)
from .registry import (
    create_attention,
    create_flash_attention,
    create_memory_efficient_attention,
    create_sparse_attention,
    get_attention_registry,
    list_available_attention,
    register_attention,
)

__all__ = [
    'BaseAttention',
    'AttentionWithCache',
    'AttentionConfig',  # Backward compat alias
    'AttentionModuleConfig',
    'AttentionPatterns',
    'FP8AttentionConfig',
    'DynamicSparseConfig',
    'RingAttentionConfig',
    'register_attention',
    'create_attention',
    'get_attention_registry',
    'list_available_attention',
    'create_flash_attention',
    'create_memory_efficient_attention',
    'create_sparse_attention',
    # Core attention operations
    'check_flash_attention_available',
    'check_cuda_kernel_available',
    'validate_attention_inputs',
    'compute_attention_scale',
    'scaled_dot_product_attention',
    'flash_attention_forward',
]
