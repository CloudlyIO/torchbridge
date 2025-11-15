"""
Level 2: TorchScript JIT Optimized Components

These components use TorchScript's JIT compilation to enable kernel fusion
and graph-level optimizations while maintaining Python semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


@torch.jit.script
def fused_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    JIT-compiled layer normalization that can fuse with surrounding operations.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


@torch.jit.script
def fused_swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU activation that combines gate and up projections.
    This can be optimized into fewer memory accesses by the JIT.
    """
    gate = torch.matmul(x, w_gate.t())
    up = torch.matmul(x, w_up.t())
    return F.silu(gate) * up


@torch.jit.script
def fused_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    causal_mask: bool = False
) -> torch.Tensor:
    """
    Fused computation of attention scores with optional causal masking.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal_mask:
        seq_len = q.size(-2)
        # Create causal mask efficiently
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    return F.softmax(scores, dim=-1)


@torch.jit.script
def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled rotary positional embedding.
    Demonstrates efficient tensor operations that can be fused.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)


class JITOptimizedLinear(nn.Module):
    """
    Linear layer that uses JIT compilation for potential kernel fusion.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class JITOptimizedLayerNorm(nn.Module):
    """
    Layer normalization using JIT-compiled fusion.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_layer_norm(x, self.weight, self.bias, self.eps)


class JITOptimizedMLP(nn.Module):
    """
    MLP with JIT-compiled SwiGLU activation.
    Demonstrates how JIT can optimize complex activation patterns.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.gate_weight = nn.Parameter(torch.randn(hidden_dim, dim) * (2.0 / dim) ** 0.5)
        self.up_weight = nn.Parameter(torch.randn(hidden_dim, dim) * (2.0 / dim) ** 0.5)
        self.down = JITOptimizedLinear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = fused_swiglu(x, self.gate_weight, self.up_weight)
        return self.down(hidden)


class JITRotaryAttention(nn.Module):
    """
    Attention mechanism with JIT-optimized rotary positional embeddings.
    Shows how to optimize positional encoding computations.
    """
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = JITOptimizedLinear(dim, 3 * dim, bias=False)
        self.out_proj = JITOptimizedLinear(dim, dim)

        # Precompute rotary embeddings
        self._init_rotary_embeddings(max_seq_len)

    def _init_rotary_embeddings(self, max_seq_len: int):
        """Initialize rotary embedding parameters"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute for efficiency
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply rotary embeddings using JIT-optimized function
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        q = rotary_embedding(q, cos, sin)
        k = rotary_embedding(k, cos, sin)

        # Attention computation with JIT-optimized scores
        attn_weights = fused_attention_scores(q, k, self.scale)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, dim
        )
        return self.out_proj(attn_output)


@torch.jit.script
def fused_transformer_block_forward(
    x: torch.Tensor,
    # Attention parameters
    attn_qkv_weight: torch.Tensor,
    attn_out_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    # MLP parameters
    mlp_gate_weight: torch.Tensor,
    mlp_up_weight: torch.Tensor,
    mlp_down_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    # Configuration
    num_heads: int,
    head_dim: int,
    scale: float
) -> torch.Tensor:
    """
    Fully JIT-compiled transformer block for maximum fusion opportunities.
    This demonstrates how to compile entire blocks for optimal performance.
    """
    batch_size, seq_len, dim = x.shape

    # Layer 1: Attention with residual
    residual1 = x
    x_norm1 = fused_layer_norm(x, norm1_weight, norm1_bias)

    # QKV projection and attention
    qkv = F.linear(x_norm1, attn_qkv_weight)
    qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Attention computation
    attn_weights = fused_attention_scores(q, k, scale)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
    attn_output = F.linear(attn_output, attn_out_weight)

    x = residual1 + attn_output

    # Layer 2: MLP with residual
    residual2 = x
    x_norm2 = fused_layer_norm(x, norm2_weight, norm2_bias)

    # SwiGLU MLP
    mlp_hidden = fused_swiglu(x_norm2, mlp_gate_weight, mlp_up_weight)
    mlp_output = F.linear(mlp_hidden, mlp_down_weight)

    return residual2 + mlp_output


class FullyJITTransformerBlock(nn.Module):
    """
    Transformer block that compiles the entire forward pass for maximum optimization.
    This shows the extreme end of JIT optimization.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mlp_hidden_dim = dim * mlp_ratio

        # All parameters as direct tensors for JIT efficiency
        self.attn_qkv_weight = nn.Parameter(torch.randn(3 * dim, dim) * (2.0 / dim) ** 0.5)
        self.attn_out_weight = nn.Parameter(torch.randn(dim, dim) * (2.0 / dim) ** 0.5)
        self.norm1_weight = nn.Parameter(torch.ones(dim))
        self.norm1_bias = nn.Parameter(torch.zeros(dim))

        self.mlp_gate_weight = nn.Parameter(torch.randn(self.mlp_hidden_dim, dim) * (2.0 / dim) ** 0.5)
        self.mlp_up_weight = nn.Parameter(torch.randn(self.mlp_hidden_dim, dim) * (2.0 / dim) ** 0.5)
        self.mlp_down_weight = nn.Parameter(torch.randn(dim, self.mlp_hidden_dim) * (2.0 / self.mlp_hidden_dim) ** 0.5)
        self.norm2_weight = nn.Parameter(torch.ones(dim))
        self.norm2_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_transformer_block_forward(
            x,
            self.attn_qkv_weight,
            self.attn_out_weight,
            self.norm1_weight,
            self.norm1_bias,
            self.mlp_gate_weight,
            self.mlp_up_weight,
            self.mlp_down_weight,
            self.norm2_weight,
            self.norm2_bias,
            self.num_heads,
            self.head_dim,
            self.scale
        )


class JITOptimizedTransformer(nn.Module):
    """
    Complete transformer using JIT-optimized components.
    Demonstrates progressive optimization while maintaining semantic clarity.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        use_rotary: bool = True,
        use_fully_jit_blocks: bool = False
    ):
        super().__init__()
        self.use_rotary = use_rotary

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Choose attention type
        if use_rotary:
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    JITOptimizedLayerNorm(dim),
                    JITRotaryAttention(dim, num_heads),
                    JITOptimizedLayerNorm(dim),
                    JITOptimizedMLP(dim, 4 * dim)
                ])
                for _ in range(num_layers)
            ])
        elif use_fully_jit_blocks:
            self.layers = nn.ModuleList([
                FullyJITTransformerBlock(dim, num_heads)
                for _ in range(num_layers)
            ])
        else:
            # Standard JIT-optimized blocks
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    JITOptimizedLayerNorm(dim),
                    JITOptimizedLinear(dim, dim),  # Simplified for demo
                    JITOptimizedLayerNorm(dim),
                    JITOptimizedMLP(dim, 4 * dim)
                ])
                for _ in range(num_layers)
            ])

        self.norm = JITOptimizedLayerNorm(dim)
        self.output_proj = JITOptimizedLinear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)

        if isinstance(self.layers[0], FullyJITTransformerBlock):
            # Fully JIT-optimized path
            for layer in self.layers:
                x = layer(x)
        else:
            # Component-wise JIT optimization
            for layer_components in self.layers:
                if self.use_rotary:
                    norm1, attn, norm2, mlp = layer_components
                    x = x + attn(norm1(x))
                    x = x + mlp(norm2(x))
                else:
                    # Simplified for demonstration
                    norm1, proj, norm2, mlp = layer_components
                    x = x + proj(norm1(x))
                    x = x + mlp(norm2(x))

        x = self.norm(x)
        return self.output_proj(x)

    def compile_model(self) -> 'JITOptimizedTransformer':
        """
        Compile the entire model for production use.
        This creates a fully optimized TorchScript module.
        """
        self.eval()
        example_input = torch.randint(0, 1000, (1, 64))  # Example input
        return torch.jit.trace(self, example_input)