"""
Unified Configuration System for Attention Mechanisms

Combines and enhances configuration classes from both attention/ and advanced_attention/
directories to provide a comprehensive, unified configuration system.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class AttentionPatterns(Enum):
    """Supported attention patterns - unified from both implementations"""
    FULL = "full"                           # Standard full attention
    CAUSAL = "causal"                      # Causal/autoregressive attention
    SLIDING_WINDOW = "sliding_window"       # Local sliding window
    SPARSE = "sparse"                      # Sparse attention patterns
    RING = "ring"                          # Ring attention for long sequences
    LOCAL = "local"                        # Local attention (fixed window)
    GLOBAL = "global"                      # Global + local attention
    DIFFERENTIAL = "differential"          # Differential attention
    DYNAMIC_SPARSE = "dynamic_sparse"      # Dynamic sparse attention


@dataclass
class FP8AttentionConfig:
    """Enhanced FP8 configuration - merged from both implementations"""
    use_fp8: bool = False
    fp8_format: str = "e4m3"  # "e4m3" or "e5m2"
    async_compute: bool = True
    warp_specialization: bool = True
    tensor_core_utilization: float = 0.75
    sequence_length_threshold: int = 8192  # Use FP8 for sequences longer than this

    # Additional options from attention/
    gradient_checkpointing: bool = False
    mixed_precision: bool = True


@dataclass
class DynamicSparseConfig:
    """Configuration for dynamic sparse attention - from advanced_attention/"""
    sparsity_threshold: float = 0.1
    adaptive_threshold: bool = True
    content_aware: bool = True
    efficiency_target: float = 0.8
    pattern_learning: bool = False
    min_sparsity: float = 0.05
    max_sparsity: float = 0.9


@dataclass
class RingAttentionConfig:
    """Configuration for ring attention - from advanced_attention/"""
    segment_size: int = 2048
    communication_backend: str = "nccl"  # "nccl", "gloo", "mpi"
    overlap_communication: bool = True
    pipeline_parallel: bool = False
    memory_efficient: bool = True


@dataclass
class AttentionConfig:
    """Unified configuration for all attention implementations"""

    # Core parameters (from both implementations)
    embed_dim: int
    num_heads: int
    head_dim: Optional[int] = None

    # Pattern and behavior
    pattern: AttentionPatterns = AttentionPatterns.FULL
    causal: bool = False

    # Performance optimizations (enhanced)
    use_flash_attention: bool = True
    use_memory_efficient: bool = False
    enable_compilation: bool = True
    gradient_checkpointing: bool = False

    # Precision settings (enhanced)
    precision: str = "float32"  # float32, float16, bfloat16, fp8
    use_fp8: bool = False
    fp8_config: Optional[FP8AttentionConfig] = None

    # Advanced configurations (new)
    sparse_config: Optional[DynamicSparseConfig] = None
    ring_config: Optional[RingAttentionConfig] = None

    # Hardware optimization (from advanced_attention/)
    enable_hopper_optimization: bool = True
    warp_specialization: bool = True
    device_mesh: Optional[Tuple[int, ...]] = None

    # Sequence length handling (merged)
    max_sequence_length: int = 8192
    sliding_window_size: Optional[int] = None

    # Memory optimization (enhanced)
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AttentionConfig':
        """Create from dictionary"""
        if 'pattern' in config_dict:
            config_dict['pattern'] = AttentionPatterns(config_dict['pattern'])
        return cls(**config_dict)