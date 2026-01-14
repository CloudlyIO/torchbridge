"""
Unified Configuration System for Attention Mechanisms

This module provides attention-specific configuration classes.
Common configs (AttentionPatterns, FP8AttentionConfig, etc.) are
now centralized in kernel_pytorch.core.config.

Backward compatibility is maintained through re-exports.
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Import shared configs from core (centralized in Phase 3)
from kernel_pytorch.core.config import (
    AttentionPatterns,
    FP8AttentionConfig,
    DynamicSparseConfig,
    RingAttentionConfig,
)


@dataclass
class AttentionModuleConfig:
    """
    Detailed configuration for attention module instances.

    This config is used when creating attention layers with specific
    model parameters (embed_dim, num_heads, etc.).

    Note: Renamed from AttentionConfig to avoid conflict with the
    high-level AttentionConfig in core/config.py. A backward-compatible
    alias is provided below.
    """

    # Core parameters
    embed_dim: int
    num_heads: int
    head_dim: Optional[int] = None

    # Pattern and behavior
    pattern: AttentionPatterns = AttentionPatterns.FULL
    causal: bool = False

    # Performance optimizations
    use_flash_attention: bool = True
    use_memory_efficient: bool = False
    enable_compilation: bool = True
    gradient_checkpointing: bool = False

    # Precision settings
    precision: str = "float32"  # float32, float16, bfloat16, fp8
    use_fp8: bool = False
    fp8_config: Optional[FP8AttentionConfig] = None

    # Advanced configurations
    sparse_config: Optional[DynamicSparseConfig] = None
    ring_config: Optional[RingAttentionConfig] = None

    # Hardware optimization
    enable_hopper_optimization: bool = True
    warp_specialization: bool = True
    device_mesh: Optional[Tuple[int, ...]] = None

    # Sequence length handling
    max_sequence_length: int = 8192
    sliding_window_size: Optional[int] = None

    # Memory optimization
    memory_efficient_backend: str = "auto"  # "auto", "triton", "flash_attn", "pytorch"
    chunk_size: Optional[int] = None

    # Dropout and regularization
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    # Compatibility settings
    legacy_mode: bool = False
    backward_compatible: bool = True

    def __post_init__(self):
        """Validate and set defaults"""
        if self.head_dim is None:
            if self.embed_dim % self.num_heads != 0:
                raise ValueError(f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}")
            self.head_dim = self.embed_dim // self.num_heads

        # Set FP8 config defaults if using FP8
        if self.use_fp8 and self.fp8_config is None:
            self.fp8_config = FP8AttentionConfig()

        # Set sparse config defaults for sparse patterns
        if self.pattern == AttentionPatterns.DYNAMIC_SPARSE and self.sparse_config is None:
            self.sparse_config = DynamicSparseConfig()

        # Set ring config defaults for ring attention
        if self.pattern == AttentionPatterns.RING and self.ring_config is None:
            self.ring_config = RingAttentionConfig()

        # Validate sliding window
        if self.pattern == AttentionPatterns.SLIDING_WINDOW and self.sliding_window_size is None:
            self.sliding_window_size = min(1024, self.max_sequence_length // 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'pattern': self.pattern.value,
            'causal': self.causal,
            'use_flash_attention': self.use_flash_attention,
            'use_memory_efficient': self.use_memory_efficient,
            'precision': self.precision,
            'use_fp8': self.use_fp8,
            'max_sequence_length': self.max_sequence_length
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AttentionModuleConfig':
        """Create from dictionary"""
        if 'pattern' in config_dict:
            config_dict['pattern'] = AttentionPatterns(config_dict['pattern'])
        return cls(**config_dict)


# Backward compatibility alias
AttentionConfig = AttentionModuleConfig
