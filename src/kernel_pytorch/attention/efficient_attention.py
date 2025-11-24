"""
Memory-Efficient Attention Implementations

Collection of attention mechanisms optimized for memory efficiency,
including ring attention, chunked attention, and long sequence support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

from .core import BaseAttention, AttentionConfig, register_attention


class MemoryEfficientAttention(BaseAttention):
    """
    Memory-efficient attention that reduces memory usage through
    gradient checkpointing and efficient computation patterns.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.checkpoint_attention = getattr(config, 'checkpoint_attention', True)

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient attention computation"""

        if self.checkpoint_attention and self.training:
            # Use gradient checkpointing for memory efficiency during training
            return torch.utils.checkpoint.checkpoint(
                self._attention_forward,
                q, k, v, attention_mask,
                use_reentrant=False
            )
        else:
            return self._attention_forward(q, k, v, attention_mask)

    def _attention_forward(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core attention computation"""

        try:
            # Use PyTorch's memory-efficient attention if available
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=self.config.causal
            )
        except AttributeError:
            # Fallback to manual implementation
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

            if self.config.causal:
                seq_len = q.size(-2)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
                attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)

            return torch.matmul(attn_weights, v)


class LongSequenceAttention(BaseAttention):
    """
    Attention optimized for very long sequences using chunking and
    sliding window patterns to reduce computational complexity.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.chunk_size = config.chunk_size if config.chunk_size is not None else 512
        self.window_size = config.window_size if config.window_size is not None else 256
        self.overlap_size = getattr(config, 'overlap_size', 64)

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Long sequence attention with chunking"""

        batch_size, num_heads, seq_len, head_dim = q.shape

        # For short sequences, use standard attention
        if seq_len <= self.chunk_size:
            return MemoryEfficientAttention(self.config)._compute_attention(q, k, v, attention_mask)

        # For long sequences, use chunked processing
        return self._chunked_long_attention(q, k, v, attention_mask)

    def _chunked_long_attention(self,
                               q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process long sequences in chunks with overlapping windows"""

        batch_size, num_heads, seq_len, head_dim = q.shape
        output = torch.zeros_like(q)

        chunk_size = self.chunk_size
        overlap = self.overlap_size

        for start_idx in range(0, seq_len, chunk_size - overlap):
            end_idx = min(start_idx + chunk_size, seq_len)

            # Extract chunks
            q_chunk = q[:, :, start_idx:end_idx]

            # For keys and values, use a larger window for context
            k_start = max(0, start_idx - self.window_size)
            k_end = min(seq_len, end_idx + self.window_size)
            k_chunk = k[:, :, k_start:k_end]
            v_chunk = v[:, :, k_start:k_end]

            # Adjust attention mask for chunk
            mask_chunk = None
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, start_idx:end_idx, k_start:k_end]

            # Compute attention for chunk
            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scaling

            if mask_chunk is not None:
                attn_scores = attn_scores.masked_fill(mask_chunk == 0, -1e9)

            if self.config.causal:
                # Create causal mask for the chunk
                chunk_len = end_idx - start_idx
                key_len = k_end - k_start
                causal_mask = torch.ones(chunk_len, key_len, device=q.device, dtype=torch.bool)

                # Only attend to positions up to the current position
                for i in range(chunk_len):
                    global_pos = start_idx + i
                    local_limit = min(key_len, global_pos - k_start + 1)
                    if local_limit > 0:
                        causal_mask[i, local_limit:] = False

                attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)

            chunk_output = torch.matmul(attn_weights, v_chunk)

            # Handle overlapping regions by averaging
            if start_idx > 0 and start_idx < seq_len - chunk_size + overlap:
                # Overlap region - blend with previous computation
                overlap_start = start_idx
                overlap_end = min(start_idx + overlap, end_idx)

                if overlap_end > overlap_start:
                    # Simple averaging for overlap (could use more sophisticated blending)
                    overlap_weight = 0.5
                    output[:, :, overlap_start:overlap_end] = (
                        overlap_weight * output[:, :, overlap_start:overlap_end] +
                        (1 - overlap_weight) * chunk_output[:, :, :overlap_end - overlap_start]
                    )

                    # Non-overlap region
                    if overlap_end < end_idx:
                        output[:, :, overlap_end:end_idx] = chunk_output[:, :, overlap_end - start_idx:]
                else:
                    output[:, :, start_idx:end_idx] = chunk_output
            else:
                output[:, :, start_idx:end_idx] = chunk_output

        return output


class RingAttention(BaseAttention):
    """
    Ring Attention implementation for distributed long sequence processing.

    Distributes attention computation across multiple devices/ranks
    to handle extremely long sequences efficiently.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.ring_size = getattr(config, 'ring_size', 1)
        self.sequence_parallel = self.ring_size > 1

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ring attention computation"""

        if not self.sequence_parallel or self.ring_size <= 1:
            # Fall back to regular attention if no sequence parallelism
            return MemoryEfficientAttention(self.config)._compute_attention(q, k, v, attention_mask)

        # For actual ring attention, we'd need distributed communication
        # This is a simplified version that simulates the concept
        return self._simulate_ring_attention(q, k, v, attention_mask)

    def _simulate_ring_attention(self,
                                q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simulate ring attention by processing in segments"""

        batch_size, num_heads, seq_len, head_dim = q.shape

        # Split sequence into ring segments
        segment_size = seq_len // self.ring_size
        output = torch.zeros_like(q)

        for segment_idx in range(self.ring_size):
            start_idx = segment_idx * segment_size
            end_idx = (segment_idx + 1) * segment_size if segment_idx < self.ring_size - 1 else seq_len

            # Process this segment's queries against all keys
            q_segment = q[:, :, start_idx:end_idx]

            attn_scores = torch.matmul(q_segment, k.transpose(-2, -1)) * self.scaling

            if attention_mask is not None:
                mask_segment = attention_mask[:, :, start_idx:end_idx]
                attn_scores = attn_scores.masked_fill(mask_segment == 0, -1e9)

            if self.config.causal:
                # Causal mask for this segment
                segment_len = end_idx - start_idx
                causal_mask = torch.ones(segment_len, seq_len, device=q.device, dtype=torch.bool)
                for i in range(segment_len):
                    global_pos = start_idx + i
                    causal_mask[i, global_pos + 1:] = False

                attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)

            output[:, :, start_idx:end_idx] = torch.matmul(attn_weights, v)

        return output


class ChunkedAttention(BaseAttention):
    """
    Simple chunked attention for processing sequences in fixed-size chunks.
    Useful for controlling memory usage with a straightforward implementation.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.chunk_size = config.chunk_size if config.chunk_size is not None else 1024

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Chunked attention computation"""

        batch_size, num_heads, seq_len, head_dim = q.shape

        if seq_len <= self.chunk_size:
            # No chunking needed for short sequences
            return MemoryEfficientAttention(self.config)._compute_attention(q, k, v, attention_mask)

        output = torch.zeros_like(q)

        for start_idx in range(0, seq_len, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)

            q_chunk = q[:, :, start_idx:end_idx]
            k_chunk = k[:, :, start_idx:end_idx] if not self.config.causal else k
            v_chunk = v[:, :, start_idx:end_idx] if not self.config.causal else v

            mask_chunk = None
            if attention_mask is not None:
                if self.config.causal:
                    mask_chunk = attention_mask[:, :, start_idx:end_idx, :]
                else:
                    mask_chunk = attention_mask[:, :, start_idx:end_idx, start_idx:end_idx]

            chunk_output = MemoryEfficientAttention(self.config)._compute_attention(
                q_chunk, k_chunk, v_chunk, mask_chunk
            )

            output[:, :, start_idx:end_idx] = chunk_output

        return output


# Register efficient attention implementations
register_attention('memory_efficient', MemoryEfficientAttention)
register_attention('long_sequence', LongSequenceAttention)
register_attention('ring_attention', RingAttention)
register_attention('chunked', ChunkedAttention)


# Factory functions
def create_memory_efficient_attention(embed_dim: int,
                                     num_heads: int,
                                     checkpoint_attention: bool = True,
                                     **kwargs) -> MemoryEfficientAttention:
    """Create memory-efficient attention with gradient checkpointing"""
    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        checkpoint_attention=checkpoint_attention,
        **kwargs
    )
    return MemoryEfficientAttention(config)


def create_long_sequence_attention(embed_dim: int,
                                  num_heads: int,
                                  chunk_size: int = 512,
                                  window_size: int = 256,
                                  **kwargs) -> LongSequenceAttention:
    """Create attention optimized for long sequences"""
    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        chunk_size=chunk_size,
        window_size=window_size,
        **kwargs
    )
    return LongSequenceAttention(config)


def create_ring_attention(embed_dim: int,
                         num_heads: int,
                         ring_size: int = 1,
                         **kwargs) -> RingAttention:
    """Create ring attention for sequence parallelism"""
    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ring_size=ring_size,
        **kwargs
    )
    return RingAttention(config)