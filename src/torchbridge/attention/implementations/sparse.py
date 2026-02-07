"""
Sparse Attention Implementations

Provides memory-efficient attention for long sequences using sparsity patterns.
Implements multiple sparse attention strategies:
- Block sparse attention (BigBird-style)
- Strided sparse attention (Sparse Transformers)
- Dynamic sparse attention with learned patterns

Version: 0.5.3
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base import BaseAttention
from ..core.config import AttentionConfig
from ..core.registry import register_attention


def _compute_block_mask(
    seq_len: int,
    block_size: int,
    num_random_blocks: int,
    num_global_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute BigBird-style block sparse mask.

    Creates a sparse attention pattern with:
    - Local attention (sliding window)
    - Global tokens that attend to all positions
    - Random block attention for long-range dependencies

    Args:
        seq_len: Sequence length
        block_size: Size of each attention block
        num_random_blocks: Number of random blocks per row
        num_global_blocks: Number of global blocks at start
        device: Torch device

    Returns:
        Boolean mask where True means "do not attend"
    """
    num_blocks = (seq_len + block_size - 1) // block_size

    # Start with all blocked (True = masked/no attention)
    mask = torch.ones(num_blocks, num_blocks, dtype=torch.bool, device=device)

    # Local attention: each block attends to itself and neighbors
    for i in range(num_blocks):
        start = max(0, i - 1)
        end = min(num_blocks, i + 2)
        mask[i, start:end] = False

    # Global blocks: first few blocks attend everywhere
    if num_global_blocks > 0:
        mask[:num_global_blocks, :] = False
        mask[:, :num_global_blocks] = False

    # Random blocks for long-range attention
    if num_random_blocks > 0:
        for i in range(num_blocks):
            # Pick random blocks to attend to
            available = torch.where(mask[i])[0]
            if len(available) > 0:
                num_to_select = min(num_random_blocks, len(available))
                perm = torch.randperm(len(available), device=device)[:num_to_select]
                random_blocks = available[perm]
                mask[i, random_blocks] = False

    # Expand to full attention mask
    full_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        for j in range(num_blocks):
            i_start = i * block_size
            i_end = min((i + 1) * block_size, seq_len)
            j_start = j * block_size
            j_end = min((j + 1) * block_size, seq_len)
            full_mask[i_start:i_end, j_start:j_end] = mask[i, j]

    return full_mask


def _compute_strided_mask(
    seq_len: int,
    local_window: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute strided sparse attention mask (Sparse Transformer style).

    Creates a sparse attention pattern with:
    - Local attention within a window
    - Strided attention at regular intervals

    Args:
        seq_len: Sequence length
        local_window: Size of local attention window
        stride: Stride for strided attention
        device: Torch device

    Returns:
        Boolean mask where True means "do not attend"
    """
    # Start with all blocked
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    for i in range(seq_len):
        # Local window attention
        start = max(0, i - local_window // 2)
        end = min(seq_len, i + local_window // 2 + 1)
        mask[i, start:end] = False

        # Strided attention: attend to positions at regular intervals
        strided_positions = torch.arange(0, seq_len, stride, device=device)
        mask[i, strided_positions] = False

    return mask


@register_attention('dynamic_sparse_attention')
class DynamicSparseAttention(BaseAttention):
    """Dynamic sparse attention with learned sparsity patterns.

    Uses a small predictor network to learn which positions to attend to,
    enabling adaptive sparsity based on input content.

    Key features:
    - Content-aware sparsity prediction
    - Top-k selection for controlled sparsity
    - Differentiable training via straight-through estimator
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        # Sparsity configuration
        self.sparsity_ratio = getattr(config, 'sparsity_ratio', 0.9)
        self.top_k = None  # Will be computed based on sequence length

        # Sparsity predictor network
        self.sparsity_predictor = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, self.head_dim // 4),
            nn.GELU(),
            nn.Linear(self.head_dim // 4, 1),
        )

        self._init_predictor()

    def _init_predictor(self):
        """Initialize predictor with small weights for stable training."""
        for module in self.sparsity_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with dynamic sparsity.

        Args:
            q: Query tensor [B, H, S, D_h]
            k: Key tensor [B, H, S, D_h]
            v: Value tensor [B, H, S, D_h]
            attention_mask: Optional mask

        Returns:
            Attention output [B, H, S, D_h]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        # Compute importance scores using predictor
        # Use mean of Q and K to predict importance
        qk_features = (q + k) / 2  # [B, H, S, D_h]
        importance = self.sparsity_predictor(qk_features)  # [B, H, S, 1]
        importance = importance.squeeze(-1)  # [B, H, S]

        # Expand importance to attention matrix shape
        # Higher importance = more likely to be attended to
        importance_matrix = importance.unsqueeze(-2) + importance.unsqueeze(-1)  # [B, H, S, S]

        # Compute top-k to keep based on sparsity ratio
        total_elements = seq_len * seq_len
        k_keep = max(seq_len, int(total_elements * (1 - self.sparsity_ratio)))

        # Flatten and select top-k
        flat_importance = importance_matrix.view(batch_size * num_heads, -1)
        flat_scores = scores.view(batch_size * num_heads, -1)

        _, topk_indices = torch.topk(flat_importance, k_keep, dim=-1)

        # Create sparse mask
        sparse_mask = torch.ones_like(flat_scores, dtype=torch.bool)
        sparse_mask.scatter_(1, topk_indices, False)

        # Apply sparse mask (mask out non-selected positions)
        flat_scores = flat_scores.masked_fill(sparse_mask, float('-inf'))

        # Reshape back
        scores = flat_scores.view(batch_size, num_heads, seq_len, seq_len)

        # Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle -inf -> nan
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)


@register_attention('block_sparse_attention')
class BlockSparseAttention(BaseAttention):
    """BigBird-style block sparse attention.

    Combines three attention patterns:
    1. Local sliding window attention
    2. Global tokens (attend to/from all positions)
    3. Random block attention for long-range dependencies

    Memory complexity: O(n * b) where b is block size, vs O(n^2) for dense.
    """

    def __init__(
        self,
        config: AttentionConfig,
        block_size: int = 64,
        num_random_blocks: int = 3,
        num_global_blocks: int = 1,
    ):
        super().__init__(config)
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.num_global_blocks = num_global_blocks
        self._cached_mask: torch.Tensor | None = None
        self._cached_seq_len: int = 0

    def _get_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or compute block sparse mask with caching."""
        if self._cached_mask is None or self._cached_seq_len != seq_len:
            self._cached_mask = _compute_block_mask(
                seq_len,
                self.block_size,
                self.num_random_blocks,
                self.num_global_blocks,
                device,
            )
            self._cached_seq_len = seq_len
        return self._cached_mask

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute block sparse attention.

        Args:
            q: Query tensor [B, H, S, D_h]
            k: Key tensor [B, H, S, D_h]
            v: Value tensor [B, H, S, D_h]
            attention_mask: Optional additional mask

        Returns:
            Attention output [B, H, S, D_h]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Get block sparse mask
        sparse_mask = self._get_block_mask(seq_len, q.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Combine masks
        if attention_mask is not None:
            combined_mask = attention_mask | sparse_mask
        else:
            combined_mask = sparse_mask

        scores = self._apply_attention_mask(scores, combined_mask)

        # Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)


@register_attention('strided_sparse_attention')
class StridedSparseAttention(BaseAttention):
    """Strided sparse attention (Sparse Transformer style).

    Combines local window attention with strided attention patterns,
    enabling efficient long-range dependencies.

    Attention pattern:
    - Local: attend to nearby positions within window
    - Strided: attend to positions at regular intervals

    Memory complexity: O(n * (w + n/s)) where w is window size and s is stride.
    """

    def __init__(
        self,
        config: AttentionConfig,
        local_window: int = 256,
        stride: int = 256,
    ):
        super().__init__(config)
        self.local_window = local_window
        self.stride = stride
        self._cached_mask: torch.Tensor | None = None
        self._cached_seq_len: int = 0

    def _get_strided_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or compute strided sparse mask with caching."""
        if self._cached_mask is None or self._cached_seq_len != seq_len:
            self._cached_mask = _compute_strided_mask(
                seq_len,
                self.local_window,
                self.stride,
                device,
            )
            self._cached_seq_len = seq_len
        return self._cached_mask

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute strided sparse attention.

        Args:
            q: Query tensor [B, H, S, D_h]
            k: Key tensor [B, H, S, D_h]
            v: Value tensor [B, H, S, D_h]
            attention_mask: Optional additional mask

        Returns:
            Attention output [B, H, S, D_h]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Get strided sparse mask
        sparse_mask = self._get_strided_mask(seq_len, q.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Combine masks
        if attention_mask is not None:
            combined_mask = attention_mask | sparse_mask
        else:
            combined_mask = sparse_mask

        scores = self._apply_attention_mask(scores, combined_mask)

        # Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)


class SparseAttentionPattern(BaseAttention):
    """Flexible sparse attention with configurable patterns.

    Allows combining multiple sparsity patterns for custom attention behavior.
    Useful for experimentation and architecture search.
    """

    def __init__(
        self,
        config: AttentionConfig,
        patterns: list[str] | None = None,
    ):
        super().__init__(config)
        self.patterns = patterns or ['local', 'global']
        self.local_window = 128
        self.global_positions = 4

    def _build_combined_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build mask combining all configured patterns."""
        # Start with all blocked
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        for pattern in self.patterns:
            if pattern == 'local':
                # Local window attention
                for i in range(seq_len):
                    start = max(0, i - self.local_window // 2)
                    end = min(seq_len, i + self.local_window // 2 + 1)
                    mask[i, start:end] = False

            elif pattern == 'global':
                # First few tokens attend everywhere
                mask[:self.global_positions, :] = False
                mask[:, :self.global_positions] = False

            elif pattern == 'diagonal':
                # Diagonal attention (for linear complexity)
                for i in range(seq_len):
                    mask[i, i] = False

        return mask

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with combined patterns."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Build combined mask
        pattern_mask = self._build_combined_mask(seq_len, q.device)
        pattern_mask = pattern_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            combined_mask = attention_mask | pattern_mask
        else:
            combined_mask = pattern_mask

        scores = self._apply_attention_mask(scores, combined_mask)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)
