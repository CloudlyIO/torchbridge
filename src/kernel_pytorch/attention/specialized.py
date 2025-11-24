"""
Specialized Attention Patterns

Collection of specialized attention mechanisms including sparse attention,
differential attention, and flexible attention patterns for specific use cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable
import math

from .core import BaseAttention, AttentionConfig, AttentionPatterns, register_attention


class SparseAttentionPattern(BaseAttention):
    """
    Sparse attention with configurable sparsity patterns.

    Supports various sparsity patterns like local windows, strided attention,
    and custom sparse patterns to reduce computational complexity.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.sparsity_pattern = getattr(config, 'sparsity_pattern', 'local')
        self.window_size = config.window_size if config.window_size is not None else 64
        self.stride = getattr(config, 'stride', 32)
        self.global_tokens = getattr(config, 'global_tokens', 16)

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sparse attention computation with efficient pattern"""

        batch_size, num_heads, seq_len, head_dim = q.shape

        if self.sparsity_pattern == 'local':
            return self._local_sparse_attention(q, k, v, attention_mask)
        elif self.sparsity_pattern == 'strided':
            return self._strided_sparse_attention(q, k, v, attention_mask)
        elif self.sparsity_pattern == 'global_local':
            return self._global_local_attention(q, k, v, attention_mask)
        else:
            # Fallback to dense attention
            return super()._compute_attention(q, k, v, attention_mask)

    def _local_sparse_attention(self,
                               q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Local windowed sparse attention"""

        batch_size, num_heads, seq_len, head_dim = q.shape
        window_size = min(self.window_size, seq_len)

        # Create local attention mask
        indices = torch.arange(seq_len, device=q.device)
        row_indices = indices.unsqueeze(1)  # [seq_len, 1]
        col_indices = indices.unsqueeze(0)  # [1, seq_len]

        # Local window mask: |i - j| <= window_size // 2
        local_mask = torch.abs(row_indices - col_indices) <= window_size // 2

        # Convert to attention mask format
        sparse_mask = local_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        sparse_mask = sparse_mask.expand(batch_size, num_heads, -1, -1)

        # Combine with existing mask
        if attention_mask is not None:
            sparse_mask = sparse_mask & attention_mask

        # Compute dense attention with sparse mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_scores = attn_scores.masked_fill(~sparse_mask, -1e9)

        if self.config.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _strided_sparse_attention(self,
                                 q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Strided sparse attention pattern"""

        batch_size, num_heads, seq_len, head_dim = q.shape
        stride = self.stride

        # Create strided attention pattern
        indices = torch.arange(seq_len, device=q.device)
        row_indices = indices.unsqueeze(1)
        col_indices = indices.unsqueeze(0)

        # Strided pattern: attend to positions at regular intervals
        strided_mask = (col_indices % stride) == (row_indices % stride)

        # Also include local connections
        local_mask = torch.abs(row_indices - col_indices) <= 2

        # Combine patterns
        sparse_mask = strided_mask | local_mask
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)
        sparse_mask = sparse_mask.expand(batch_size, num_heads, -1, -1)

        if attention_mask is not None:
            sparse_mask = sparse_mask & attention_mask

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_scores = attn_scores.masked_fill(~sparse_mask, -1e9)

        if self.config.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _global_local_attention(self,
                               q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Global + local attention pattern (like Longformer)"""

        batch_size, num_heads, seq_len, head_dim = q.shape

        # First few tokens are global (attend to all), rest are local
        global_tokens = min(self.global_tokens, seq_len // 4)
        window_size = self.window_size

        # Create mask
        mask = torch.zeros(seq_len, seq_len, device=q.device, dtype=torch.bool)

        # Global tokens attend to everything and everything attends to them
        mask[:global_tokens, :] = True
        mask[:, :global_tokens] = True

        # Local attention for the rest
        for i in range(global_tokens, seq_len):
            start = max(global_tokens, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True

        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

        if attention_mask is not None:
            mask = mask & attention_mask

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_scores = attn_scores.masked_fill(~mask, -1e9)

        if self.config.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)


class DifferentialAttention(BaseAttention):
    """
    Differential Attention implementation.

    Uses two separate attention mechanisms and computes their difference
    to capture more nuanced attention patterns.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        # Create separate projection layers for differential attention
        self.q_proj_2 = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.k_proj_2 = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.v_proj_2 = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        # Lambda parameter for combining the two attention mechanisms
        self.lambda_param = nn.Parameter(torch.ones(config.num_heads, 1, 1))

        self.reset_differential_parameters()

    def reset_differential_parameters(self):
        """Initialize the additional parameters for differential attention"""
        nn.init.xavier_uniform_(self.q_proj_2.weight)
        nn.init.xavier_uniform_(self.k_proj_2.weight)
        nn.init.xavier_uniform_(self.v_proj_2.weight)

        if self.config.bias:
            nn.init.constant_(self.q_proj_2.bias, 0.)
            nn.init.constant_(self.k_proj_2.bias, 0.)
            nn.init.constant_(self.v_proj_2.bias, 0.)

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Differential attention forward pass"""

        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape

        # First attention mechanism (standard)
        q1 = self.q_proj(query)
        k1 = self.k_proj(key)
        v1 = self.v_proj(value)

        # Second attention mechanism
        q2 = self.q_proj_2(query)
        k2 = self.k_proj_2(key)
        v2 = self.v_proj_2(value)

        # Reshape for multi-head attention
        q1 = self._shape_for_multihead(q1, batch_size, seq_len)
        k1 = self._shape_for_multihead(k1, batch_size, key.shape[1])
        v1 = self._shape_for_multihead(v1, batch_size, value.shape[1])

        q2 = self._shape_for_multihead(q2, batch_size, seq_len)
        k2 = self._shape_for_multihead(k2, batch_size, key.shape[1])
        v2 = self._shape_for_multihead(v2, batch_size, value.shape[1])

        # Compute both attention mechanisms
        attn1 = self._compute_single_attention(q1, k1, v1, attention_mask)
        attn2 = self._compute_single_attention(q2, k2, v2, attention_mask)

        # Combine with learnable parameter
        lambda_expanded = self.lambda_param.unsqueeze(0)  # [1, H, 1, 1]
        differential_output = lambda_expanded * attn1 - (1 - lambda_expanded) * attn2

        # Reshape back and apply output projection
        differential_output = self._unshape_from_multihead(differential_output, batch_size, seq_len)
        output = self.out_proj(differential_output)
        output = self.output_dropout(output)

        if return_attention_weights:
            # Return dummy weights for now
            attn_weights = torch.zeros(batch_size, self.num_heads, seq_len, seq_len,
                                     device=output.device, dtype=output.dtype)
            return output, attn_weights

        return output

    def _compute_single_attention(self,
                                 q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute single attention mechanism"""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        if self.config.causal:
            seq_len = q.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """This should not be called directly for differential attention"""
        return self._compute_single_attention(q, k, v, attention_mask)


class FlexAttentionAPI(BaseAttention):
    """
    Flexible attention API that supports multiple attention patterns
    and can switch between them dynamically.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.current_pattern = config.pattern

    def set_pattern(self, pattern: AttentionPatterns):
        """Change the attention pattern"""
        self.current_pattern = pattern
        self.config.pattern = pattern

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flexible attention computation based on current pattern"""

        if self.current_pattern == AttentionPatterns.FULL:
            return self._full_attention(q, k, v, attention_mask)
        elif self.current_pattern == AttentionPatterns.CAUSAL:
            return self._causal_attention(q, k, v, attention_mask)
        elif self.current_pattern == AttentionPatterns.SLIDING_WINDOW:
            return self._sliding_window_attention(q, k, v, attention_mask)
        elif self.current_pattern == AttentionPatterns.LOCAL:
            return self._local_attention(q, k, v, attention_mask)
        else:
            # Default to full attention
            return self._full_attention(q, k, v, attention_mask)

    def _full_attention(self, q, k, v, attention_mask):
        """Standard full attention"""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def _causal_attention(self, q, k, v, attention_mask):
        """Causal attention"""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply causal mask
        seq_len = q.size(-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def _sliding_window_attention(self, q, k, v, attention_mask):
        """Sliding window attention"""
        window_size = getattr(self.config, 'window_size', 256)
        return SparseAttentionPattern(self.config)._local_sparse_attention(q, k, v, attention_mask)

    def _local_attention(self, q, k, v, attention_mask):
        """Local attention"""
        return self._sliding_window_attention(q, k, v, attention_mask)


class LocalAttention(SparseAttentionPattern):
    """Local attention with fixed window size"""

    def __init__(self, config: AttentionConfig):
        config.sparsity_pattern = 'local'
        super().__init__(config)


class SlidingWindowAttention(SparseAttentionPattern):
    """Sliding window attention"""

    def __init__(self, config: AttentionConfig):
        config.sparsity_pattern = 'local'
        super().__init__(config)


# Register specialized attention implementations
register_attention('sparse', SparseAttentionPattern)
register_attention('differential', DifferentialAttention)
register_attention('flex', FlexAttentionAPI)
register_attention('local', LocalAttention)
register_attention('sliding_window', SlidingWindowAttention)


# Factory functions
def create_sparse_attention(embed_dim: int,
                           num_heads: int,
                           sparsity_pattern: str = 'local',
                           window_size: int = 64,
                           **kwargs) -> SparseAttentionPattern:
    """Create sparse attention with specified pattern"""
    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparsity_pattern=sparsity_pattern,
        window_size=window_size,
        **kwargs
    )
    return SparseAttentionPattern(config)


def create_differential_attention(embed_dim: int,
                                 num_heads: int,
                                 **kwargs) -> DifferentialAttention:
    """Create differential attention"""
    config = AttentionConfig(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
    return DifferentialAttention(config)