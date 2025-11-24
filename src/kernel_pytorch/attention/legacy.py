"""
Legacy Attention Implementation Compatibility Layer

Provides backward compatibility imports for the old scattered attention implementations.
These imports redirect to the new unified attention system while maintaining the
original API surface for existing code.

DEPRECATED: These imports are provided for backward compatibility only.
New code should use the unified attention system from the main __init__.py.
"""

import warnings
from typing import Optional

from .core import AttentionConfig, BaseAttention, create_attention
from .flash_attention import FlashAttention2, FlashAttention3, MultiHeadFlashAttention
from .efficient_attention import MemoryEfficientAttention, LongSequenceAttention
from .specialized import SparseAttentionPattern, DifferentialAttention

# Legacy wrapper classes for backward compatibility


class CompilerOptimizedMultiHeadAttention(BaseAttention):
    """
    DEPRECATED: Legacy compiler-optimized multi-head attention.

    This class provides backward compatibility with the old compiler-optimized
    attention implementation. New code should use the unified attention system.
    """

    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        warnings.warn(
            "CompilerOptimizedMultiHeadAttention is deprecated. "
            "Use create_attention() with AttentionConfig instead.",
            DeprecationWarning,
            stacklevel=2
        )

        config = AttentionConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_flash_attention=True,
            enable_compilation=True,
            **kwargs
        )
        super().__init__(config)

    def _compute_attention(self, q, k, v, attention_mask=None):
        # Delegate to FlashAttention3 for best performance
        flash_attn = FlashAttention3(self.config)
        return flash_attn._compute_attention(q, k, v, attention_mask)


class FlashAttentionWrapper(BaseAttention):
    """
    DEPRECATED: Legacy FlashAttention wrapper.

    This class provides backward compatibility with the old FlashAttention
    wrapper implementation. New code should use FlashAttention3 directly.
    """

    def __init__(self, embed_dim: int, num_heads: int, version: str = "v3", **kwargs):
        warnings.warn(
            "FlashAttentionWrapper is deprecated. "
            "Use FlashAttention2, FlashAttention3, or MultiHeadFlashAttention directly.",
            DeprecationWarning,
            stacklevel=2
        )

        config = AttentionConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_flash_attention=True,
            **kwargs
        )
        super().__init__(config)
        self.version = version

    def _compute_attention(self, q, k, v, attention_mask=None):
        if self.version == "v3":
            flash_attn = FlashAttention3(self.config)
        else:
            flash_attn = FlashAttention2(self.config)
        return flash_attn._compute_attention(q, k, v, attention_mask)


# Legacy factory functions for backward compatibility

def create_legacy_attention(attention_type: str, embed_dim: int, num_heads: int, **kwargs) -> BaseAttention:
    """
    DEPRECATED: Legacy attention factory function.

    Creates attention modules using the old naming scheme for backward compatibility.
    New code should use create_attention() with AttentionConfig.
    """
    warnings.warn(
        f"create_legacy_attention is deprecated. Use create_attention() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Map old names to new implementations
    type_mapping = {
        'flash': 'flash_attention3',
        'flash_v2': 'flash_attention2',
        'flash_v3': 'flash_attention3',
        'memory_efficient': 'memory_efficient',
        'sparse': 'sparse',
        'differential': 'differential',
        'standard': 'standard',
        'optimized': 'flash_attention3',  # Map old "optimized" to FlashAttention3
        'compiler_optimized': 'flash_attention3'
    }

    implementation = type_mapping.get(attention_type, 'standard')
    config = AttentionConfig(embed_dim=embed_dim, num_heads=num_heads, **kwargs)

    return create_attention(config, implementation)


# Compatibility aliases for scattered module imports
class OptimizedAttention(FlashAttention3):
    """DEPRECATED: Use FlashAttention3 directly"""
    def __init__(self, *args, **kwargs):
        warnings.warn("OptimizedAttention is deprecated. Use FlashAttention3 instead.",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class EfficientAttention(MemoryEfficientAttention):
    """DEPRECATED: Use MemoryEfficientAttention directly"""
    def __init__(self, *args, **kwargs):
        warnings.warn("EfficientAttention is deprecated. Use MemoryEfficientAttention instead.",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class FlexibleAttention(SparseAttentionPattern):
    """DEPRECATED: Use SparseAttentionPattern or FlexAttentionAPI directly"""
    def __init__(self, *args, **kwargs):
        warnings.warn("FlexibleAttention is deprecated. Use SparseAttentionPattern or FlexAttentionAPI instead.",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


# Export legacy classes for import compatibility
__all__ = [
    'CompilerOptimizedMultiHeadAttention',
    'FlashAttentionWrapper',
    'OptimizedAttention',
    'EfficientAttention',
    'FlexibleAttention',
    'create_legacy_attention'
]