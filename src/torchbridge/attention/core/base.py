"""
Base Attention Classes

Unified base classes that combine the best aspects from both attention/ and
advanced_attention/ implementations.
"""

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from .config import AttentionConfig, AttentionPatterns


class BaseAttention(nn.Module, ABC):
    """
    Unified base class for all attention implementations.

    Combines the best aspects of both attention/core.py and advanced_attention
    base classes to provide a comprehensive foundation.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or (config.embed_dim // config.num_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Initialize projection layers
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        # Pattern-specific initialization
        self._initialize_pattern_specific()

    def _initialize_pattern_specific(self):
        """Initialize pattern-specific components"""
        if self.config.pattern == AttentionPatterns.SLIDING_WINDOW:
            self.window_size = self.config.sliding_window_size
        elif self.config.pattern == AttentionPatterns.SPARSE:
            self._initialize_sparse_components()
        elif self.config.pattern == AttentionPatterns.RING:
            self._initialize_ring_components()

    def _initialize_sparse_components(self):
        """Initialize sparse attention components"""
        if self.config.sparse_config and self.config.sparse_config.pattern_learning:
            # Initialize learned sparsity predictor
            self.sparsity_predictor = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim // 4),
                nn.ReLU(),
                nn.Linear(self.head_dim // 4, 1),
                nn.Sigmoid()
            )

    def _initialize_ring_components(self):
        """Initialize ring attention components"""
        if self.config.ring_config:
            self.segment_size = self.config.ring_config.segment_size
            # Ring attention specific initialization would go here
            # For now, we'll use simulation mode

    def _shape_for_multihead(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention: [B, S, D] -> [B, H, S, D_h]"""
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _unshape_from_multihead(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape tensor from multi-head attention: [B, H, S, D_h] -> [B, S, D]"""
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask"""
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
        return mask

    def _apply_attention_mask(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Apply attention mask to scores"""
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float('-inf'))
            else:
                scores = scores + mask
        return scores

    @abstractmethod
    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: torch.Tensor | None = None) -> torch.Tensor:
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
                key: torch.Tensor | None = None,
                value: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                return_attention_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward pass for all attention implementations.

        Args:
            query: Query tensor [B, S, D]
            key: Key tensor [B, S, D] (defaults to query for self-attention)
            value: Value tensor [B, S, D] (defaults to key)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights

        Returns:
            Output tensor [B, S, D] and optionally attention weights
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = self._shape_for_multihead(q, batch_size, seq_len)
        k = self._shape_for_multihead(k, batch_size, seq_len)
        v = self._shape_for_multihead(v, batch_size, seq_len)

        # Apply pattern-specific masking
        pattern_mask = self._create_pattern_mask(seq_len, q.device)
        if pattern_mask is not None:
            if attention_mask is not None:
                attention_mask = attention_mask | pattern_mask
            else:
                attention_mask = pattern_mask

        # Compute attention
        attn_output = self._compute_attention(q, k, v, attention_mask)

        # Reshape back to original dimensions
        attn_output = self._unshape_from_multihead(attn_output, batch_size, seq_len)

        # Apply output projection and dropout
        output = self.out_proj(attn_output)
        output = self.residual_dropout(output)

        if return_attention_weights:
            # This would need to be implemented in subclasses
            warnings.warn("Attention weights not implemented for this attention type", stacklevel=2)
            return output, None

        return output

    def _create_pattern_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        """Create mask based on attention pattern"""
        if self.config.causal or self.config.pattern == AttentionPatterns.CAUSAL:
            return self._create_causal_mask(seq_len, device)
        elif self.config.pattern == AttentionPatterns.SLIDING_WINDOW:
            return self._create_sliding_window_mask(seq_len, device)
        # Other patterns handled in specific implementations
        return None

    def get_attention_stats(self) -> dict[str, Any]:
        """Get statistics about the attention layer"""
        return {
            'pattern': self.config.pattern.value,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'scale': self.scale,
            'max_sequence_length': self.config.max_sequence_length,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def reset_parameters(self):
        """Reset all parameters to default initialization"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def extra_repr(self) -> str:
        """Extra representation for printing"""
        return (f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, pattern={self.config.pattern.value}')


class AttentionWithCache(BaseAttention):
    """
    Base class for attention implementations that support KV caching.
    Useful for autoregressive generation and inference optimization.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.cache_enabled = False
        self.kv_cache: dict[str, torch.Tensor] | None = None

    def enable_cache(self, max_batch_size: int = 1, max_seq_len: int | None = None):
        """Enable KV caching for faster autoregressive generation"""
        if max_seq_len is None:
            max_seq_len = self.config.max_sequence_length

        self.cache_enabled = True
        device = next(self.parameters()).device

        self.kv_cache = {
            'keys': torch.zeros(max_batch_size, self.num_heads, max_seq_len, self.head_dim, device=device),
            'values': torch.zeros(max_batch_size, self.num_heads, max_seq_len, self.head_dim, device=device),
            'seq_len': 0
        }

    def disable_cache(self):
        """Disable KV caching"""
        self.cache_enabled = False
        self.kv_cache = None

    def clear_cache(self):
        """Clear the KV cache"""
        if self.kv_cache is not None:
            self.kv_cache['seq_len'] = 0
