"""
Memory-Efficient Attention Variants

Implementations of memory-efficient attention mechanisms for long sequences
and resource-constrained environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention using chunked computation

    Processes attention in chunks to reduce memory usage for long sequences
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        chunk_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.dropout = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient forward pass"""
        B, T, C = x.shape

        if T <= self.chunk_size:
            # Use standard attention for short sequences
            return self._standard_attention(x)
        else:
            # Use chunked attention for long sequences
            return self._chunked_attention(x)

    def _standard_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard attention computation"""
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)

    def _chunked_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Chunked attention computation"""
        B, T, C = x.shape
        outputs = []

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)

        # Process in chunks
        for i in range(0, T, self.chunk_size):
            end_i = min(i + self.chunk_size, T)

            q_chunk = q[:, :, i:end_i, :]  # [B, H, chunk_size, D]

            # Compute attention with full key/value
            attn_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)

            if self.dropout > 0:
                attn_chunk = F.dropout(attn_chunk, p=self.dropout, training=self.training)

            out_chunk = torch.matmul(attn_chunk, v)
            outputs.append(out_chunk)

        # Concatenate chunks
        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class LongSequenceAttention(nn.Module):
    """
    Attention mechanism optimized for very long sequences

    Uses techniques like sliding window and sparse patterns
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 512,
        sparse_pattern: str = "local",
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.sparse_pattern = sparse_pattern
        self.dropout = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with long sequence optimization"""
        if self.sparse_pattern == "local":
            return self._local_attention(x)
        elif self.sparse_pattern == "sliding":
            return self._sliding_window_attention(x)
        else:
            return self._standard_attention(x)

    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Local attention within windows"""
        B, T, C = x.shape

        # Pad sequence to be divisible by window size
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Reshape into windows
        T_padded = x.size(1)
        num_windows = T_padded // self.window_size
        x = x.view(B, num_windows, self.window_size, C)

        outputs = []
        for i in range(num_windows):
            window = x[:, i, :, :]  # [B, window_size, C]

            qkv = self.qkv_proj(window)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)

            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)

            if self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout, training=self.training)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, self.window_size, C)

            outputs.append(out)

        # Concatenate windows
        out = torch.cat(outputs, dim=1)

        # Remove padding
        if pad_len > 0:
            out = out[:, :T, :]

        return self.out_proj(out)

    def _sliding_window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Sliding window attention"""
        # Simplified sliding window - would implement more sophisticated version
        return self._local_attention(x)

    def _standard_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback to standard attention"""
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)