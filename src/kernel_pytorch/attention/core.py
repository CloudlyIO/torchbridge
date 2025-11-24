"""
Core Attention Framework

Provides base classes, configuration, and utilities for all attention implementations.
This module establishes the common interface and patterns used throughout the
unified attention system.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class AttentionPatterns(Enum):
    """Supported attention patterns"""
    FULL = "full"                           # Standard full attention
    CAUSAL = "causal"                      # Causal/autoregressive attention
    SLIDING_WINDOW = "sliding_window"       # Local sliding window
    SPARSE = "sparse"                      # Sparse attention patterns
    RING = "ring"                          # Ring attention for long sequences
    LOCAL = "local"                        # Local attention (fixed window)
    GLOBAL = "global"                      # Global + local attention
    DIFFERENTIAL = "differential"          # Differential attention


@dataclass
class AttentionConfig:
    """Unified configuration for all attention implementations"""
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

    # Precision settings
    precision: str = "float32"  # float32, float16, bfloat16, fp8
    use_fp8: bool = False

    # Memory settings
    chunk_size: Optional[int] = None
    window_size: Optional[int] = None

    # Training settings
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Advanced settings
    scaling_factor: Optional[float] = None
    bias: bool = True

    def __post_init__(self):
        """Validate and set derived parameters"""
        if self.head_dim is None:
            if self.embed_dim % self.num_heads != 0:
                raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")
            self.head_dim = self.embed_dim // self.num_heads

        if self.scaling_factor is None:
            self.scaling_factor = self.head_dim ** -0.5


class BaseAttention(nn.Module, ABC):
    """
    Base class for all attention implementations.

    Provides common interface and functionality for all attention variants.
    Concrete implementations should inherit from this class.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scaling = config.scaling_factor

        # Create projection layers
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters following standard attention initialization"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.config.bias:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def _shape_for_multihead(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention: [B, S, D] -> [B, H, S, D_h]"""
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _unshape_from_multihead(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape tensor from multi-head attention: [B, H, S, D_h] -> [B, S, D]"""
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    @abstractmethod
    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention weights and apply to values.

        Args:
            q: Query tensor [B, H, S, D_h]
            k: Key tensor [B, H, S, D_h]
            v: Value tensor [B, H, S, D_h]
            attention_mask: Optional mask [B, H, S, S] or [B, 1, S, S]

        Returns:
            Attention output [B, H, S, D_h]
        """
        pass

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for attention computation.

        Args:
            query: Query tensor [B, S, D]
            key: Key tensor [B, S, D] (defaults to query for self-attention)
            value: Value tensor [B, S, D] (defaults to key)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights

        Returns:
            Output tensor [B, S, D] and optionally attention weights [B, H, S, S]
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape

        # Apply projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = self._shape_for_multihead(q, batch_size, seq_len)
        k = self._shape_for_multihead(k, batch_size, key.shape[1])
        v = self._shape_for_multihead(v, batch_size, value.shape[1])

        # Compute attention (implemented by subclasses)
        attn_output = self._compute_attention(q, k, v, attention_mask)

        # Reshape back to original format
        attn_output = self._unshape_from_multihead(attn_output, batch_size, seq_len)

        # Apply output projection and dropout
        output = self.out_proj(attn_output)
        output = self.output_dropout(output)

        if return_attention_weights:
            # For now, return dummy weights - subclasses can override if needed
            attn_weights = torch.zeros(batch_size, self.num_heads, seq_len, seq_len,
                                     device=output.device, dtype=output.dtype)
            return output, attn_weights

        return output


class StandardAttention(BaseAttention):
    """
    Standard scaled dot-product attention implementation.

    This serves as the reference implementation and fallback when
    specialized optimizations are not available.
    """

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard scaled dot-product attention"""
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask to [B, H, S, S]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                # Expand mask to [B, H, S, S]
                attention_mask = attention_mask.unsqueeze(1)

            # Apply mask (large negative value for masked positions)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        # Apply causal masking if needed
        if self.config.causal:
            seq_len = q.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        return attn_output


# Global registry for attention implementations
_ATTENTION_REGISTRY: Dict[str, Type[BaseAttention]] = {
    'standard': StandardAttention,
}


def register_attention(name: str, attention_class: Type[BaseAttention]):
    """Register a new attention implementation"""
    if not issubclass(attention_class, BaseAttention):
        raise ValueError(f"Attention class {attention_class} must inherit from BaseAttention")

    _ATTENTION_REGISTRY[name] = attention_class


def get_attention_registry() -> Dict[str, Type[BaseAttention]]:
    """Get the current attention registry"""
    return _ATTENTION_REGISTRY.copy()


def create_attention(config: AttentionConfig,
                    implementation: Optional[str] = None) -> BaseAttention:
    """
    Factory function to create attention modules.

    Args:
        config: AttentionConfig with parameters
        implementation: Specific implementation to use (None for auto-selection)

    Returns:
        Configured attention module
    """
    if implementation is None:
        # Auto-select best available implementation
        if config.use_flash_attention and 'flash_attention3' in _ATTENTION_REGISTRY:
            implementation = 'flash_attention3'
        elif config.use_memory_efficient and 'memory_efficient' in _ATTENTION_REGISTRY:
            implementation = 'memory_efficient'
        else:
            implementation = 'standard'

    if implementation not in _ATTENTION_REGISTRY:
        available = list(_ATTENTION_REGISTRY.keys())
        warnings.warn(f"Implementation '{implementation}' not found. Available: {available}. Using 'standard'.")
        implementation = 'standard'

    attention_class = _ATTENTION_REGISTRY[implementation]
    return attention_class(config)