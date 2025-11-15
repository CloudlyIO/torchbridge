"""
Level 1: Basic PyTorch Optimized Components

These components use PyTorch's native operations that automatically map to
optimized kernels (cuDNN, cuBLAS) while following kernel-friendly patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OptimizedLinear(nn.Module):
    """
    Linear layer optimized for kernel efficiency.

    Key optimizations:
    - Uses tensor operations that map to optimized BLAS kernels
    - Minimizes memory allocations
    - Supports batched operations efficiently
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights for good numerical properties
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use F.linear for optimized GEMM kernel dispatch
        return F.linear(x, self.weight, self.bias)


class FusedLinearActivation(nn.Module):
    """
    Linear layer with fused activation - designed for kernel fusion.

    This pattern allows optimization frameworks to fuse the linear
    operation with the activation in a single kernel.
    """
    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        super().__init__()
        self.linear = OptimizedLinear(in_features, out_features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)

        # Use operations that can be fused with preceding linear layer
        if self.activation == "relu":
            return F.relu(x, inplace=True)  # Can fuse with linear
        elif self.activation == "gelu":
            return F.gelu(x)  # cuDNN has fused GELU implementations
        elif self.activation == "swish":
            return x * torch.sigmoid(x)  # Can be fused
        else:
            return x


class OptimizedLayerNorm(nn.Module):
    """
    Layer normalization optimized for GPU execution.

    Uses operations that map well to GPU's SIMT model and
    can utilize hardware-optimized normalization kernels.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use F.layer_norm for optimized kernel dispatch
        return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)


class OptimizedMultiHeadAttention(nn.Module):
    """
    Multi-head attention optimized for kernel efficiency.

    Key optimizations:
    - Single matrix multiplication for Q, K, V projections
    - Uses scaled_dot_product_attention for Flash Attention
    - Memory-efficient reshape operations
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Single projection for Q, K, V to enable kernel fusion
        self.qkv_proj = OptimizedLinear(dim, 3 * dim, bias=False)
        self.out_proj = OptimizedLinear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        # Single matrix multiplication for Q, K, V
        qkv = self.qkv_proj(x)

        # Reshape and split - these operations are memory layout efficient
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv.unbind(0)

        # Use PyTorch's optimized attention (Flash Attention when available)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            if self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, dim
        )

        return self.out_proj(attn_output)


class OptimizedMLP(nn.Module):
    """
    MLP block optimized for kernel fusion patterns.

    Uses SwiGLU activation which can be implemented efficiently
    and demonstrates gated activation patterns.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim

        # For SwiGLU, we need two projections that can be computed together
        self.gate_proj = OptimizedLinear(dim, hidden_dim, bias=False)
        self.up_proj = OptimizedLinear(dim, hidden_dim, bias=False)
        self.down_proj = OptimizedLinear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: gate * up_proj where gate has SiLU activation
        gate = F.silu(self.gate_proj(x))  # SiLU = x * sigmoid(x)
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


class OptimizedTransformerBlock(nn.Module):
    """
    Complete transformer block with kernel-optimized components.

    Demonstrates how to combine optimized components while
    maintaining the semantic meaning of transformer architecture.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = OptimizedLayerNorm(dim)
        self.attn = OptimizedMultiHeadAttention(dim, num_heads)
        self.norm2 = OptimizedLayerNorm(dim)
        self.mlp = OptimizedMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture for better gradient flow
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    """
    Optimized positional encoding using precomputed sinusoids.

    Demonstrates memory-efficient parameter sharing and
    operations that map well to element-wise kernels.
    """
    def __init__(self, dim: int, max_seq_length: int = 8192):
        super().__init__()
        self.dim = dim

        # Precompute positional encodings
        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, dim, 2).float() *
                            -(torch.log(torch.tensor(10000.0)) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        # Element-wise addition maps to efficient kernels
        return x + self.pe[:, :seq_len]


# Example usage demonstrating semantic ML concepts
class SimpleTransformer(nn.Module):
    """
    Complete transformer model showing how optimized components
    maintain semantic meaning while improving kernel efficiency.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 2048
    ):
        super().__init__()
        self.dim = dim

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = PositionalEncoding(dim, max_seq_length)

        # Transformer layers
        self.layers = nn.ModuleList([
            OptimizedTransformerBlock(dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm = OptimizedLayerNorm(dim)
        self.output_proj = OptimizedLinear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embedding lookup - maps to efficient gather kernels
        x = self.token_embedding(input_ids)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final norm and projection
        x = self.norm(x)
        logits = self.output_proj(x)

        return logits

    def get_num_params(self) -> int:
        """Utility to count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)