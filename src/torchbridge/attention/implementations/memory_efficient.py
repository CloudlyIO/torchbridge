"""
Memory-Efficient Attention Implementations

Provides memory-optimized attention implementations for handling long sequences
and reducing peak memory usage during training and inference.

Implements:
- Chunked attention (processes attention in chunks)
- Gradient checkpointing attention
- FlashAttention-style memory patterns
- Long sequence attention with linear memory

"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..core.base import AttentionWithCache, BaseAttention
from ..core.config import AttentionConfig
from ..core.registry import register_attention


@register_attention('memory_efficient_attention')
class MemoryEfficientAttention(BaseAttention):
    """Memory-efficient attention using chunked computation.

    Processes attention in chunks to reduce peak memory usage.
    Instead of materializing the full NxN attention matrix, we:
    1. Compute attention for chunks of queries
    2. Accumulate weighted values incrementally
    3. Never store the full attention matrix

    Memory reduction:
    - Standard: O(N^2) for attention matrix
    - Chunked: O(N * chunk_size) for attention matrix

    This is inspired by FlashAttention but implemented in pure PyTorch
    for portability across devices.
    """

    def __init__(self, config: AttentionConfig, chunk_size: int = 256):
        super().__init__(config)
        self.chunk_size = chunk_size
        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory savings during training."""
        self.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with chunked memory efficiency.

        Args:
            q: Query tensor [B, H, S, D_h]
            k: Key tensor [B, H, S, D_h]
            v: Value tensor [B, H, S, D_h]
            attention_mask: Optional mask

        Returns:
            Attention output [B, H, S, D_h]
        """
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self._chunked_attention,
                q, k, v, attention_mask,
                use_reentrant=False
            )
        return self._chunked_attention(q, k, v, attention_mask)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention in chunks."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Scale queries
        q = q * self.scale

        # Output accumulator
        output = torch.zeros_like(q)

        # Process queries in chunks
        for q_start in range(0, seq_len, self.chunk_size):
            q_end = min(q_start + self.chunk_size, seq_len)
            q_chunk = q[:, :, q_start:q_end, :]

            # Compute attention scores for this query chunk
            # Shape: [B, H, chunk_size, S]
            scores = torch.matmul(q_chunk, k.transpose(-2, -1))

            # Apply mask for this chunk
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    mask_chunk = attention_mask[:, :, q_start:q_end, :]
                elif attention_mask.dim() == 2:
                    mask_chunk = attention_mask[q_start:q_end, :]
                else:
                    mask_chunk = attention_mask
                scores = self._apply_attention_mask(scores, mask_chunk)

            # Softmax over key dimension
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)

            # Compute output for this chunk
            output[:, :, q_start:q_end, :] = torch.matmul(attn_weights, v)

        return output

@register_attention('chunked_attention')
class ChunkedAttention(BaseAttention):
    """Chunked attention for very long sequences.

    Processes both queries and keys/values in chunks to handle
    sequences that don't fit in memory even with query-only chunking.

    Double chunking strategy:
    1. Outer loop: Process queries in chunks
    2. Inner loop: Process keys/values in chunks
    3. Use online softmax to combine results

    Memory: O(chunk_q * chunk_kv) for attention computation
    """

    def __init__(
        self,
        config: AttentionConfig,
        query_chunk_size: int = 256,
        kv_chunk_size: int = 256,
    ):
        super().__init__(config)
        self.query_chunk_size = query_chunk_size
        self.kv_chunk_size = kv_chunk_size

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with double chunking.

        Uses online softmax for numerically stable combination of chunks.
        """
        batch_size, num_heads, q_len, head_dim = q.shape
        _, _, kv_len, _ = k.shape

        # Scale queries once
        q = q * self.scale

        # Output and normalization accumulators
        output = torch.zeros_like(q)
        torch.zeros(batch_size, num_heads, q_len, 1, device=q.device)
        torch.full(
            (batch_size, num_heads, q_len, 1),
            float('-inf'),
            device=q.device
        )

        # Process queries in chunks
        for q_start in range(0, q_len, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, q_len)
            q_chunk = q[:, :, q_start:q_end, :]

            # Accumulators for this query chunk
            chunk_output = torch.zeros_like(q_chunk)
            chunk_normalizer = torch.zeros(
                batch_size, num_heads, q_end - q_start, 1,
                device=q.device
            )
            chunk_max = torch.full(
                (batch_size, num_heads, q_end - q_start, 1),
                float('-inf'),
                device=q.device
            )

            # Process keys/values in chunks
            for kv_start in range(0, kv_len, self.kv_chunk_size):
                kv_end = min(kv_start + self.kv_chunk_size, kv_len)

                k_chunk = k[:, :, kv_start:kv_end, :]
                v_chunk = v[:, :, kv_start:kv_end, :]

                # Compute scores for this chunk
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))

                # Apply mask if provided
                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        mask_chunk = attention_mask[
                            :, :, q_start:q_end, kv_start:kv_end
                        ]
                    elif attention_mask.dim() == 2:
                        mask_chunk = attention_mask[q_start:q_end, kv_start:kv_end]
                    else:
                        mask_chunk = attention_mask
                    scores = self._apply_attention_mask(scores, mask_chunk)

                # Online softmax: update running max
                new_max = torch.maximum(
                    chunk_max,
                    scores.max(dim=-1, keepdim=True).values
                )

                # Correction factor for previous accumulated values
                correction = torch.exp(chunk_max - new_max)

                # Correct previous accumulation
                chunk_output = chunk_output * correction
                chunk_normalizer = chunk_normalizer * correction

                # Add this chunk's contribution
                exp_scores = torch.exp(scores - new_max)
                chunk_normalizer = chunk_normalizer + exp_scores.sum(dim=-1, keepdim=True)
                chunk_output = chunk_output + torch.matmul(exp_scores, v_chunk)

                chunk_max = new_max

            # Apply dropout to final weights
            # Note: We apply at the end to maintain numerical stability
            chunk_output = self.attention_dropout(chunk_output / chunk_normalizer)

            output[:, :, q_start:q_end, :] = chunk_output

        return output

@register_attention('long_sequence_attention')
class LongSequenceAttention(AttentionWithCache):
    """Attention optimized for very long sequences.

    Combines multiple strategies for handling sequences up to millions of tokens:
    1. Sliding window local attention
    2. Strided global attention
    3. KV cache for autoregressive generation
    4. Chunked computation for memory efficiency

    Memory complexity: O(N * window_size) + O(N / stride)
    """

    def __init__(
        self,
        config: AttentionConfig,
        window_size: int = 512,
        global_stride: int = 256,
        chunk_size: int = 256,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.global_stride = global_stride
        self.chunk_size = chunk_size

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute long sequence attention.

        Combines local window and strided global attention patterns.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # For very long sequences, use combined local + global attention
        if seq_len > self.window_size * 2:
            return self._combined_local_global(q, k, v, attention_mask)
        else:
            # For shorter sequences, use chunked attention
            return self._chunked_attention(q, k, v, attention_mask)

    def _combined_local_global(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combined local window + global strided attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Scale queries
        q = q * self.scale

        output = torch.zeros_like(q)

        # Select global positions (strided)
        global_positions = torch.arange(0, seq_len, self.global_stride, device=q.device)
        k_global = k[:, :, global_positions, :]
        v_global = v[:, :, global_positions, :]

        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]

            # Local attention: window around current position
            local_start = max(0, i - self.window_size // 2)
            local_end = min(seq_len, end_i + self.window_size // 2)

            k_local = k[:, :, local_start:local_end, :]
            v_local = v[:, :, local_start:local_end, :]

            # Concatenate local and global keys/values
            k_combined = torch.cat([k_local, k_global], dim=2)
            v_combined = torch.cat([v_local, v_global], dim=2)

            # Compute attention
            scores = torch.matmul(q_chunk, k_combined.transpose(-2, -1))

            # Apply mask if needed
            if attention_mask is not None:
                # Create combined mask for local + global
                local_len = local_end - local_start
                global_len = len(global_positions)
                local_len + global_len

                if attention_mask.dim() == 4:
                    # Extract local mask
                    local_mask = attention_mask[:, :, i:end_i, local_start:local_end]
                    # Global positions mask
                    global_mask = attention_mask[:, :, i:end_i, :][:, :, :, global_positions]
                    combined_mask = torch.cat([local_mask, global_mask], dim=-1)
                    scores = self._apply_attention_mask(scores, combined_mask)

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)

            output[:, :, i:end_i, :] = torch.matmul(attn_weights, v_combined)

        return output

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard chunked attention for moderate sequence lengths."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        q = q * self.scale
        output = torch.zeros_like(q)

        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]

            scores = torch.matmul(q_chunk, k.transpose(-2, -1))

            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    mask_chunk = attention_mask[:, :, i:end_i, :]
                else:
                    mask_chunk = attention_mask
                scores = self._apply_attention_mask(scores, mask_chunk)

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)

            output[:, :, i:end_i, :] = torch.matmul(attn_weights, v)

        return output

class GradientCheckpointedAttention(BaseAttention):
    """Attention with gradient checkpointing for training memory efficiency.

    Trades compute for memory by not storing intermediate activations during
    forward pass and recomputing them during backward pass.

    Memory reduction: ~3-4x for attention layer during training
    Compute overhead: ~25-33% (recomputation during backward)
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self._checkpointing_enabled = True

    def enable_checkpointing(self):
        """Enable gradient checkpointing."""
        self._checkpointing_enabled = True

    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        self._checkpointing_enabled = False

    def _attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Core attention computation (wrapped by checkpoint)."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with optional gradient checkpointing."""
        if self._checkpointing_enabled and self.training:
            return checkpoint(
                self._attention_core,
                q, k, v, attention_mask,
                use_reentrant=False
            )
        return self._attention_core(q, k, v, attention_mask)

class SlidingWindowAttention(BaseAttention):
    """Sliding window attention for linear memory complexity.

    Each token only attends to tokens within a local window,
    providing O(N * W) memory complexity instead of O(N^2).

    Window size determines the receptive field - larger windows
    capture more context but use more memory.
    """

    def __init__(self, config: AttentionConfig, window_size: int = 256):
        super().__init__(config)
        self.window_size = window_size

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute sliding window attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Scale queries
        q = q * self.scale

        output = torch.zeros_like(q)

        half_window = self.window_size // 2

        for i in range(seq_len):
            # Define window bounds
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)

            # Get window keys and values
            k_window = k[:, :, start:end, :]
            v_window = v[:, :, start:end, :]

            # Query for this position
            q_i = q[:, :, i:i+1, :]

            # Compute attention for this position
            scores = torch.matmul(q_i, k_window.transpose(-2, -1))

            # Apply mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    mask_window = attention_mask[:, :, i:i+1, start:end]
                elif attention_mask.dim() == 2:
                    mask_window = attention_mask[i:i+1, start:end]
                else:
                    mask_window = attention_mask
                scores = self._apply_attention_mask(scores, mask_window)

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)

            output[:, :, i:i+1, :] = torch.matmul(attn_weights, v_window)

        return output
