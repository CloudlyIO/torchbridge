"""
Core attention framework components
"""

from .base import BaseAttention, AttentionWithCache
from .config import (
    AttentionConfig,  # Backward compat alias for AttentionModuleConfig
    AttentionModuleConfig,
    AttentionPatterns,
    FP8AttentionConfig,
    DynamicSparseConfig,
    RingAttentionConfig
)
from .registry import (
    register_attention,
    create_attention,
    get_attention_registry,
    list_available_attention,
    create_flash_attention,
    create_memory_efficient_attention,
    create_sparse_attention
)
from .attention_ops import (
    check_flash_attention_available,
    check_cuda_kernel_available,
    validate_attention_inputs,
    compute_attention_scale,
    scaled_dot_product_attention,
    flash_attention_forward,
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