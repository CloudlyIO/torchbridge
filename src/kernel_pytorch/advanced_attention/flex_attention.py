"""
FlexAttention API Integration (2024)

Implementation of PyTorch's FlexAttention API for flexible attention mechanisms.
Based on PyTorch 2.5+ FlexAttention that leverages torch.compile to generate
fused FlashAttention kernels for various attention patterns.

Features:
- Sliding window attention
- Causal attention with arbitrary patterns
- PrefixLM attention
- Block-sparse attention patterns
- Custom attention masks with automatic kernel fusion

References:
    - FlexAttention Blog: https://pytorch.org/blog/flexattention/
    - FlexAttention Tutorial: https://pytorch.org/tutorials/intermediate/flex_attention_tutorial.html
    - FlexAttention API Docs: https://pytorch.org/docs/main/generated/torch.nn.attention.flex_attention.html
    - Attention Patterns Paper: https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any, Union, Tuple
import math
from enum import Enum

try:
    # PyTorch 2.5+ FlexAttention
    from torch.nn.attention.flex_attention import (
        flex_attention,
        BlockMask,
        create_block_mask,
        and_masks,
        or_masks
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    # Fallback implementations


class AttentionPatterns(Enum):
    """Predefined attention patterns for common use cases"""
    CAUSAL = "causal"
    SLIDING_WINDOW = "sliding_window"
    DILATED = "dilated"
    BLOCK_SPARSE = "block_sparse"
    PREFIX_LM = "prefix_lm"
    GLOBAL_LOCAL = "global_local"
    STRIDED = "strided"
    ALIBI = "alibi"


class FlexAttentionAPI(nn.Module):
    """
    Flexible attention implementation using PyTorch's FlexAttention API

    Supports various attention patterns with automatic kernel fusion
    through torch.compile integration.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        pattern: AttentionPatterns = AttentionPatterns.CAUSAL,
        pattern_kwargs: Optional[Dict[str, Any]] = None,
        compile_mode: str = "max-autotune",
        enable_flash: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.pattern = pattern
        self.pattern_kwargs = pattern_kwargs or {}
        self.enable_flash = enable_flash

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Compile attention function for optimal performance
        if FLEX_ATTENTION_AVAILABLE:
            self._compiled_attention = self._create_compiled_attention(compile_mode)
        else:
            self._compiled_attention = None

        # Cache for block masks to avoid recomputation
        self._mask_cache = {}

    def _create_compiled_attention(self, compile_mode: str = "max-autotune"):
        """Create compiled attention function based on pattern"""

        def attention_fn(q, k, v, score_mod=None, block_mask=None):
            if FLEX_ATTENTION_AVAILABLE and score_mod is not None:
                return flex_attention(
                    q, k, v,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    enable_gqa=(self.num_heads != k.size(1))  # Grouped Query Attention
                )
            else:
                return self._manual_attention(q, k, v, score_mod, block_mask)

        # Only compile if CUDA is available and FlexAttention is supported
        if FLEX_ATTENTION_AVAILABLE and torch.cuda.is_available():
            return torch.compile(attention_fn, mode=compile_mode)
        else:
            return attention_fn

    def forward(
        self,
        x: torch.Tensor,
        custom_score_mod: Optional[Callable] = None,
        custom_block_mask: Optional = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with flexible attention patterns

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            custom_score_mod: Custom score modification function
            custom_block_mask: Custom block mask
            seq_lens: Sequence lengths for variable-length sequences
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Get score modification and block mask
        score_mod = custom_score_mod or self._get_score_mod(seq_len)
        block_mask = custom_block_mask or self._get_block_mask(seq_len)

        # Apply attention
        if self._compiled_attention is not None:
            attn_output = self._compiled_attention(q, k, v, score_mod, block_mask)
        else:
            attn_output = self._manual_attention(q, k, v, score_mod, block_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        return self.out_proj(attn_output)

    def _get_score_mod(self, seq_len: int) -> Optional[Callable]:
        """Get score modification function for the specified pattern"""

        if self.pattern == AttentionPatterns.CAUSAL:
            def causal_mask(score, b, h, q_idx, kv_idx):
                return torch.where(q_idx >= kv_idx, score, -float('inf'))
            return causal_mask

        elif self.pattern == AttentionPatterns.SLIDING_WINDOW:
            window_size = self.pattern_kwargs.get('window_size', 256)
            def sliding_window_mask(score, b, h, q_idx, kv_idx):
                return torch.where(
                    (q_idx - kv_idx).abs() <= window_size,
                    score,
                    -float('inf')
                )
            return sliding_window_mask

        elif self.pattern == AttentionPatterns.ALIBI:
            def alibi_bias(score, b, h, q_idx, kv_idx):
                # ALiBi: Attention with Linear Biases
                slope = 1.0 / (2 ** (8 * (h + 1) / self.num_heads))
                bias = -slope * (q_idx - kv_idx).abs()
                return score + bias
            return alibi_bias

        elif self.pattern == AttentionPatterns.PREFIX_LM:
            prefix_length = self.pattern_kwargs.get('prefix_length', seq_len // 4)
            def prefix_lm_mask(score, b, h, q_idx, kv_idx):
                # Bidirectional attention within prefix, causal after
                prefix_mask = (q_idx < prefix_length) & (kv_idx < prefix_length)
                causal_mask = (q_idx >= prefix_length) & (kv_idx >= prefix_length) & (q_idx >= kv_idx)
                return torch.where(prefix_mask | causal_mask, score, -float('inf'))
            return prefix_lm_mask

        elif self.pattern == AttentionPatterns.DILATED:
            dilation = self.pattern_kwargs.get('dilation', 2)
            def dilated_mask(score, b, h, q_idx, kv_idx):
                return torch.where(
                    (q_idx - kv_idx) % dilation == 0,
                    score,
                    -float('inf')
                )
            return dilated_mask

        return None

    def _get_block_mask(self, seq_len: int) -> Optional:
        """Get block mask for the specified pattern"""
        if not FLEX_ATTENTION_AVAILABLE:
            return None

        cache_key = (self.pattern, seq_len, tuple(sorted(self.pattern_kwargs.items())))

        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        block_mask = None

        if self.pattern == AttentionPatterns.BLOCK_SPARSE:
            block_size = self.pattern_kwargs.get('block_size', 64)
            sparsity_pattern = self.pattern_kwargs.get('sparsity_pattern', 'diagonal')

            def sparse_mask(b, h, q_idx, kv_idx):
                if sparsity_pattern == 'diagonal':
                    block_q = q_idx // block_size
                    block_k = kv_idx // block_size
                    return torch.abs(block_q - block_k) <= 1
                elif sparsity_pattern == 'strided':
                    stride = self.pattern_kwargs.get('stride', 2)
                    return (q_idx // block_size) % stride == (kv_idx // block_size) % stride
                return True

            block_mask = create_block_mask(sparse_mask, seq_len, seq_len)

        elif self.pattern == AttentionPatterns.GLOBAL_LOCAL:
            local_window = self.pattern_kwargs.get('local_window', 128)
            global_tokens = self.pattern_kwargs.get('global_tokens', 32)

            def global_local_mask(b, h, q_idx, kv_idx):
                # Global attention for first global_tokens
                global_attn = (q_idx < global_tokens) | (kv_idx < global_tokens)
                # Local attention within window
                local_attn = torch.abs(q_idx - kv_idx) <= local_window
                return global_attn | local_attn

            block_mask = create_block_mask(global_local_mask, seq_len, seq_len)

        self._mask_cache[cache_key] = block_mask
        return block_mask

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        score_mod: Optional[Callable] = None,
        block_mask: Optional = None
    ) -> torch.Tensor:
        """Manual attention computation for fallback"""

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply score modification if provided
        if score_mod is not None:
            batch_size, num_heads, seq_len = scores.shape[:3]

            # Create index tensors
            q_indices = torch.arange(seq_len, device=scores.device)
            kv_indices = torch.arange(seq_len, device=scores.device)
            q_idx, kv_idx = torch.meshgrid(q_indices, kv_indices, indexing='ij')

            # Apply score modification
            for b in range(batch_size):
                for h in range(num_heads):
                    scores[b, h] = score_mod(scores[b, h], b, h, q_idx, kv_idx)

        # Apply attention weights and return
        attn_weights = F.softmax(scores, dim=-1)

        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        return torch.matmul(attn_weights, v)

    def set_pattern(
        self,
        pattern: AttentionPatterns,
        pattern_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Change attention pattern dynamically"""
        self.pattern = pattern
        self.pattern_kwargs = pattern_kwargs or {}

        # Clear mask cache when pattern changes
        self._mask_cache.clear()

        # Recreate compiled attention if needed
        if FLEX_ATTENTION_AVAILABLE:
            self._compiled_attention = self._create_compiled_attention()

    def benchmark_patterns(
        self,
        input_tensor: torch.Tensor,
        patterns_to_test: Optional[list] = None,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark different attention patterns"""
        import time

        if patterns_to_test is None:
            patterns_to_test = [
                AttentionPatterns.CAUSAL,
                AttentionPatterns.SLIDING_WINDOW,
                AttentionPatterns.ALIBI,
                AttentionPatterns.PREFIX_LM
            ]

        results = {}
        original_pattern = self.pattern
        original_kwargs = self.pattern_kwargs.copy()

        for pattern in patterns_to_test:
            self.set_pattern(pattern)

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = self(input_tensor)

            if input_tensor.is_cuda:
                torch.cuda.synchronize()

            # Benchmark
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self(input_tensor)

            if input_tensor.is_cuda:
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            results[pattern.value] = (end_time - start_time) / num_iterations * 1000  # ms

        # Restore original pattern
        self.set_pattern(original_pattern, original_kwargs)

        return results


class MultiPatternAttention(nn.Module):
    """
    Attention layer that can dynamically switch between multiple patterns
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        patterns: Dict[str, Tuple[AttentionPatterns, Dict[str, Any]]],
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patterns = patterns

        # Create attention modules for each pattern
        self.attention_modules = nn.ModuleDict()
        for name, (pattern, kwargs) in patterns.items():
            self.attention_modules[name] = FlexAttentionAPI(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                pattern=pattern,
                pattern_kwargs=kwargs
            )

    def forward(
        self,
        x: torch.Tensor,
        pattern_name: str = "default"
    ) -> torch.Tensor:
        """Forward with specified pattern"""
        if pattern_name not in self.attention_modules:
            pattern_name = next(iter(self.attention_modules.keys()))

        return self.attention_modules[pattern_name](x)


# Utility functions for creating common patterns
def create_sliding_window_attention(
    embed_dim: int,
    num_heads: int,
    window_size: int = 256,
    **kwargs
) -> FlexAttentionAPI:
    """Create sliding window attention"""
    return FlexAttentionAPI(
        embed_dim=embed_dim,
        num_heads=num_heads,
        pattern=AttentionPatterns.SLIDING_WINDOW,
        pattern_kwargs={'window_size': window_size},
        **kwargs
    )


def create_block_sparse_attention(
    embed_dim: int,
    num_heads: int,
    block_size: int = 64,
    sparsity_pattern: str = 'diagonal',
    **kwargs
) -> FlexAttentionAPI:
    """Create block sparse attention"""
    return FlexAttentionAPI(
        embed_dim=embed_dim,
        num_heads=num_heads,
        pattern=AttentionPatterns.BLOCK_SPARSE,
        pattern_kwargs={
            'block_size': block_size,
            'sparsity_pattern': sparsity_pattern
        },
        **kwargs
    )


def create_prefix_lm_attention(
    embed_dim: int,
    num_heads: int,
    prefix_length: int,
    **kwargs
) -> FlexAttentionAPI:
    """Create PrefixLM attention (bidirectional prefix + causal suffix)"""
    return FlexAttentionAPI(
        embed_dim=embed_dim,
        num_heads=num_heads,
        pattern=AttentionPatterns.PREFIX_LM,
        pattern_kwargs={'prefix_length': prefix_length},
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    embed_dim = 768
    num_heads = 12
    seq_len = 1024
    batch_size = 2

    # Test different attention patterns
    patterns_to_test = {
        'causal': (AttentionPatterns.CAUSAL, {}),
        'sliding_window': (AttentionPatterns.SLIDING_WINDOW, {'window_size': 256}),
        'alibi': (AttentionPatterns.ALIBI, {}),
        'prefix_lm': (AttentionPatterns.PREFIX_LM, {'prefix_length': 256}),
        'block_sparse': (AttentionPatterns.BLOCK_SPARSE, {
            'block_size': 64,
            'sparsity_pattern': 'diagonal'
        })
    }

    # Create multi-pattern attention
    multi_attn = MultiPatternAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        patterns=patterns_to_test,
        dropout=0.1
    )

    # Test input
    x = torch.randn(batch_size, seq_len, embed_dim)
    if torch.cuda.is_available():
        x = x.cuda()
        multi_attn = multi_attn.cuda()

    print(f"FlexAttention available: {FLEX_ATTENTION_AVAILABLE}")
    print(f"Input shape: {x.shape}")

    # Test each pattern
    for pattern_name in patterns_to_test.keys():
        output = multi_attn(x, pattern_name)
        print(f"{pattern_name} output shape: {output.shape}")

    # Benchmark patterns
    single_attn = FlexAttentionAPI(embed_dim, num_heads)
    if torch.cuda.is_available():
        single_attn = single_attn.cuda()

    benchmark_results = single_attn.benchmark_patterns(x, num_iterations=50)
    print("\nBenchmark Results (ms per forward pass):")
    for pattern, time_ms in benchmark_results.items():
        print(f"  {pattern}: {time_ms:.2f}ms")