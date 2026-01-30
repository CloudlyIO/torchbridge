"""
FlexAttention Implementation (PyTorch 2.5+)

Native integration with PyTorch's FlexAttention API for flexible attention patterns
with FlashAttention-like performance for arbitrary attention masks and score modifications.

Key Features:
- Native PyTorch 2.5+ FlexAttention integration
- Common score_mod patterns (causal, sliding window, document masking, ALiBi)
- Block mask generation for efficient computation
- Automatic fallback to standard attention when FlexAttention unavailable
- Full integration with KernelPyTorch attention registry
"""

import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from ..core.base import AttentionWithCache
from ..core.config import AttentionConfig, AttentionPatterns
from ..core.registry import register_attention

# Check for FlexAttention availability (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        create_mask,
        flex_attention,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None
    create_mask = None

# Check for torch.compile availability
try:
    torch_compile = torch.compile
    TORCH_COMPILE_AVAILABLE = True
except AttributeError:
    TORCH_COMPILE_AVAILABLE = False
    torch_compile = lambda f, **kwargs: f  # noqa: E731


class FlexAttentionScoreMods:
    """
    Collection of common score_mod functions for FlexAttention.

    Each score_mod function takes (score, b, h, q_idx, kv_idx) and returns modified score.
    These can be composed and combined for complex attention patterns.
    """

    @staticmethod
    def causal(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        """Causal/autoregressive masking - only attend to previous positions"""
        # Handle both tensor and scalar indices
        if isinstance(q_idx, int) and isinstance(kv_idx, int):
            return score if q_idx >= kv_idx else torch.tensor(float('-inf'))
        return torch.where(q_idx >= kv_idx, score, torch.tensor(float('-inf'), device=score.device))

    @staticmethod
    def sliding_window(window_size: int):
        """Sliding window attention - attend to positions within window"""
        def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
            # Handle both tensor and scalar indices
            if isinstance(q_idx, int) and isinstance(kv_idx, int):
                distance = q_idx - kv_idx
                in_window = (distance >= 0) and (distance < window_size)
                return score if in_window else torch.tensor(float('-inf'))
            distance = q_idx - kv_idx
            in_window = (distance >= 0) & (distance < window_size)
            return torch.where(in_window, score, torch.tensor(float('-inf'), device=score.device))
        return score_mod

    @staticmethod
    def causal_sliding_window(window_size: int):
        """Causal + sliding window - attend to previous positions within window"""
        def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
            # Handle both tensor and scalar indices
            if isinstance(q_idx, int) and isinstance(kv_idx, int):
                distance = q_idx - kv_idx
                in_window = (distance >= 0) and (distance < window_size)
                return score if in_window else torch.tensor(float('-inf'))
            distance = q_idx - kv_idx
            in_window = (distance >= 0) & (distance < window_size)
            return torch.where(in_window, score, torch.tensor(float('-inf'), device=score.device))
        return score_mod

    @staticmethod
    def alibi(num_heads: int, alibi_slopes: torch.Tensor | None = None):
        """ALiBi positional bias"""
        if alibi_slopes is None:
            # Default ALiBi slopes: geometric sequence
            ratio = 2 ** (-8 / num_heads)
            alibi_slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])

        def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
            bias = alibi_slopes[h] * (kv_idx - q_idx)
            return score + bias
        return score_mod

    @staticmethod
    def relative_position_bias(max_distance: int = 128):
        """Learned relative position bias (T5-style)"""
        def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
            distance = q_idx - kv_idx
            # Bucket the distance
            distance = torch.clamp(distance, -max_distance, max_distance)
            return score  # Bias would be added from learned embeddings
        return score_mod

    @staticmethod
    def document_masking(document_ids: torch.Tensor):
        """Document masking - only attend within same document"""
        def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
            same_doc = document_ids[b, q_idx] == document_ids[b, kv_idx]
            return torch.where(same_doc, score, float('-inf'))
        return score_mod

    @staticmethod
    def prefix_lm(prefix_length: int):
        """Prefix LM masking - bidirectional on prefix, causal after"""
        def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
            # Full attention within prefix
            in_prefix = (q_idx < prefix_length) & (kv_idx < prefix_length)
            # Causal attention after prefix, can attend to all prefix
            causal_ok = (q_idx >= prefix_length) & (kv_idx <= q_idx)
            valid = in_prefix | causal_ok
            return torch.where(valid, score, float('-inf'))
        return score_mod

    @staticmethod
    def soft_cap(cap_value: float = 50.0):
        """Soft capping of attention logits (Gemma 2 style)"""
        def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
            return cap_value * torch.tanh(score / cap_value)
        return score_mod


class FlexAttentionMaskGenerators:
    """
    Block mask generators for efficient FlexAttention computation.

    Block masks enable FlexAttention to skip computation for blocks that
    are entirely masked, improving performance for sparse patterns.
    """

    @staticmethod
    def causal_mask(batch_size: int, num_heads: int, seq_len: int, device: torch.device):
        """Generate causal block mask"""
        if not FLEX_ATTENTION_AVAILABLE:
            return None
        # Block masks require CUDA
        if device.type != 'cuda':
            return None

        try:
            def mask_fn(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            return create_block_mask(mask_fn, B=batch_size, H=num_heads, Q_LEN=seq_len, KV_LEN=seq_len)
        except Exception:
            return None

    @staticmethod
    def sliding_window_mask(batch_size: int, num_heads: int, seq_len: int,
                           window_size: int, device: torch.device):
        """Generate sliding window block mask"""
        if not FLEX_ATTENTION_AVAILABLE:
            return None
        # Block masks require CUDA
        if device.type != 'cuda':
            return None

        try:
            def mask_fn(b, h, q_idx, kv_idx):
                distance = q_idx - kv_idx
                return (distance >= 0) & (distance < window_size)

            return create_block_mask(mask_fn, B=batch_size, H=num_heads, Q_LEN=seq_len, KV_LEN=seq_len)
        except Exception:
            return None

    @staticmethod
    def full_mask(batch_size: int, num_heads: int, seq_len: int, device: torch.device):
        """Generate full attention block mask (no masking)"""
        if not FLEX_ATTENTION_AVAILABLE:
            return None
        # Block masks require CUDA
        if device.type != 'cuda':
            return None

        try:
            def mask_fn(b, h, q_idx, kv_idx):
                return True

            return create_block_mask(mask_fn, B=batch_size, H=num_heads, Q_LEN=seq_len, KV_LEN=seq_len)
        except Exception:
            return None


@register_attention('flex_attention')
class FlexAttentionLayer(AttentionWithCache):
    """
    FlexAttention layer with PyTorch 2.5+ native integration.

    Provides FlashAttention-like performance for arbitrary attention patterns
    through user-defined score_mod functions and efficient block masking.

    Args:
        config: AttentionConfig with attention parameters
        score_mod: Optional score modification function or name of built-in pattern
        use_block_mask: Whether to use block masks for efficiency
        compile_score_mod: Whether to compile score_mod with torch.compile

    Example:
        config = AttentionConfig(embed_dim=512, num_heads=8, pattern=AttentionPatterns.CAUSAL)
        attention = FlexAttentionLayer(config)

        # With custom score_mod
        def my_score_mod(score, b, h, q_idx, kv_idx):
            return score + custom_bias[q_idx, kv_idx]
        attention = FlexAttentionLayer(config, score_mod=my_score_mod)
    """

    def __init__(
        self,
        config: AttentionConfig,
        score_mod: str | Callable | None = None,
        use_block_mask: bool = True,
        compile_score_mod: bool = True,
    ):
        super().__init__(config)

        self.use_block_mask = use_block_mask and FLEX_ATTENTION_AVAILABLE
        self.compile_score_mod = compile_score_mod and TORCH_COMPILE_AVAILABLE

        # Determine score_mod based on config pattern or explicit argument
        self._score_mod_fn = self._resolve_score_mod(score_mod, config)

        # Compile score_mod for better performance
        if self.compile_score_mod and self._score_mod_fn is not None:
            self._compiled_score_mod = torch_compile(self._score_mod_fn, fullgraph=True)
        else:
            self._compiled_score_mod = self._score_mod_fn

        # Cache for block masks
        self._block_mask_cache: dict[tuple, Any] = {}

        # Track FlexAttention availability
        self.flex_attention_available = FLEX_ATTENTION_AVAILABLE

        if not FLEX_ATTENTION_AVAILABLE:
            warnings.warn(
                "FlexAttention not available (requires PyTorch 2.5+). "
                "Falling back to standard attention implementation.",
            stacklevel=2,
            )

    def _resolve_score_mod(
        self,
        score_mod: str | Callable | None,
        config: AttentionConfig
    ) -> Callable | None:
        """Resolve score_mod from string name or callable"""

        # If explicit score_mod provided
        if callable(score_mod):
            return score_mod

        if isinstance(score_mod, str):
            if score_mod == 'causal':
                return FlexAttentionScoreMods.causal
            elif score_mod == 'sliding_window':
                window_size = config.sliding_window_size or 256
                return FlexAttentionScoreMods.sliding_window(window_size)
            elif score_mod == 'causal_sliding_window':
                window_size = config.sliding_window_size or 256
                return FlexAttentionScoreMods.causal_sliding_window(window_size)
            elif score_mod == 'alibi':
                return FlexAttentionScoreMods.alibi(config.num_heads)
            elif score_mod == 'soft_cap':
                return FlexAttentionScoreMods.soft_cap()
            else:
                raise ValueError(f"Unknown score_mod: {score_mod}")

        # Infer from config pattern
        if config.pattern == AttentionPatterns.CAUSAL:
            return FlexAttentionScoreMods.causal
        elif config.pattern == AttentionPatterns.SLIDING_WINDOW:
            window_size = config.sliding_window_size or 256
            if config.causal:
                return FlexAttentionScoreMods.causal_sliding_window(window_size)
            else:
                return FlexAttentionScoreMods.sliding_window(window_size)

        # Full attention - no score_mod needed
        return None

    def _get_block_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> Any | None:
        """Get or create cached block mask"""
        if not self.use_block_mask:
            return None

        cache_key = (batch_size, self.num_heads, seq_len, str(device), self.config.pattern)

        if cache_key not in self._block_mask_cache:
            if self.config.pattern == AttentionPatterns.CAUSAL:
                mask = FlexAttentionMaskGenerators.causal_mask(
                    batch_size, self.num_heads, seq_len, device
                )
            elif self.config.pattern == AttentionPatterns.SLIDING_WINDOW:
                mask = FlexAttentionMaskGenerators.sliding_window_mask(
                    batch_size, self.num_heads, seq_len,
                    self.config.sliding_window_size or 256, device
                )
            else:
                mask = FlexAttentionMaskGenerators.full_mask(
                    batch_size, self.num_heads, seq_len, device
                )

            self._block_mask_cache[cache_key] = mask

        return self._block_mask_cache[cache_key]

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute attention using FlexAttention or fallback.

        Args:
            q: Query tensor [B, H, S, D]
            k: Key tensor [B, H, S, D]
            v: Value tensor [B, H, S, D]
            attention_mask: Optional attention mask

        Returns:
            Attention output [B, H, S, D]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # FlexAttention requires CUDA and the native API
        can_use_flex = (
            FLEX_ATTENTION_AVAILABLE and
            self._compiled_score_mod is not None and
            q.device.type == 'cuda'
        )

        if can_use_flex:
            return self._flex_attention_forward(q, k, v, attention_mask)
        else:
            return self._standard_attention_forward(q, k, v, attention_mask)

    def _flex_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """FlexAttention forward pass using PyTorch native API"""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Get block mask for efficiency
        block_mask = self._get_block_mask(batch_size, seq_len, q.device)

        try:
            # Use native FlexAttention
            # Note: flex_attention expects [B, H, S, D] format
            output = flex_attention(
                q, k, v,
                score_mod=self._compiled_score_mod,
                block_mask=block_mask,
                scale=self.scale,
            )
            return output

        except Exception as e:
            warnings.warn(f"FlexAttention failed: {e}. Falling back to standard attention.", stacklevel=2)
            return self._standard_attention_forward(q, k, v, attention_mask)

    def _standard_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Standard attention fallback implementation"""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply pattern-specific masking
        if self.config.pattern == AttentionPatterns.CAUSAL or self.config.causal:
            causal_mask = self._create_causal_mask(seq_len, q.device)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        elif self.config.pattern == AttentionPatterns.SLIDING_WINDOW:
            window_mask = self._create_sliding_window_mask(seq_len, q.device)
            scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply explicit attention mask
        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply to values
        output = torch.matmul(attn_weights, v)

        return output

    def clear_block_mask_cache(self):
        """Clear the block mask cache"""
        self._block_mask_cache.clear()

    def get_attention_stats(self) -> dict[str, Any]:
        """Get attention statistics including FlexAttention info"""
        stats = super().get_attention_stats()
        stats.update({
            'flex_attention_available': self.flex_attention_available,
            'using_flex_attention': FLEX_ATTENTION_AVAILABLE and self._compiled_score_mod is not None,
            'score_mod_compiled': self.compile_score_mod,
            'using_block_mask': self.use_block_mask,
            'block_mask_cache_size': len(self._block_mask_cache),
            'pattern': self.config.pattern.value if hasattr(self.config.pattern, 'value') else str(self.config.pattern),
        })
        return stats


@register_attention('flex_attention_causal')
class FlexAttentionCausal(FlexAttentionLayer):
    """FlexAttention with causal masking"""

    def __init__(self, config: AttentionConfig, **kwargs):
        config.pattern = AttentionPatterns.CAUSAL
        config.causal = True
        super().__init__(config, score_mod='causal', **kwargs)


@register_attention('flex_attention_sliding_window')
class FlexAttentionSlidingWindow(FlexAttentionLayer):
    """FlexAttention with sliding window"""

    def __init__(self, config: AttentionConfig, window_size: int = 256, **kwargs):
        config.pattern = AttentionPatterns.SLIDING_WINDOW
        config.sliding_window_size = window_size
        super().__init__(config, score_mod='causal_sliding_window', **kwargs)


# Convenience factory functions
def create_flex_attention(
    embed_dim: int,
    num_heads: int,
    pattern: str = 'full',
    score_mod: str | Callable | None = None,
    **kwargs
) -> FlexAttentionLayer:
    """
    Factory function for creating FlexAttention layers.

    Args:
        embed_dim: Model embedding dimension
        num_heads: Number of attention heads
        pattern: Attention pattern ('full', 'causal', 'sliding_window')
        score_mod: Score modification function or name
        **kwargs: Additional config parameters

    Returns:
        Configured FlexAttentionLayer

    Example:
        # Causal attention
        attn = create_flex_attention(512, 8, pattern='causal')

        # Sliding window
        attn = create_flex_attention(512, 8, pattern='sliding_window', sliding_window_size=128)

        # Custom score_mod
        def my_mod(score, b, h, q, kv):
            return score + bias[q, kv]
        attn = create_flex_attention(512, 8, score_mod=my_mod)
    """
    pattern_map = {
        'full': AttentionPatterns.FULL,
        'causal': AttentionPatterns.CAUSAL,
        'sliding_window': AttentionPatterns.SLIDING_WINDOW,
        'sparse': AttentionPatterns.SPARSE,
    }

    attention_pattern = pattern_map.get(pattern, AttentionPatterns.FULL)

    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        pattern=attention_pattern,
        causal=(pattern == 'causal'),
        **kwargs
    )

    return FlexAttentionLayer(config, score_mod=score_mod)


# Module-level check for FlexAttention availability
def is_flex_attention_available() -> bool:
    """Check if FlexAttention is available (PyTorch 2.5+)"""
    return FLEX_ATTENTION_AVAILABLE


def get_flex_attention_info() -> dict[str, Any]:
    """Get information about FlexAttention availability and features"""
    return {
        'available': FLEX_ATTENTION_AVAILABLE,
        'torch_version': torch.__version__,
        'torch_compile_available': TORCH_COMPILE_AVAILABLE,
        'supported_patterns': [
            'causal',
            'sliding_window',
            'causal_sliding_window',
            'alibi',
            'document_masking',
            'prefix_lm',
            'soft_cap',
            'custom',
        ],
    }
