"""
Unified FlashAttention Implementation

Combines FlashAttention v2/v3 implementations from both attention/ and advanced_attention/
directories into a single, comprehensive implementation with the best features from both.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from ..core.base import BaseAttention, AttentionWithCache
from ..core.config import AttentionConfig, FP8AttentionConfig
from ..core.registry import register_attention

try:
    import flash_attn
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True

    # Check for FlashAttention-3 specific features
    FLASH_ATTN_3_AVAILABLE = hasattr(flash_attn, 'flash_attn_v3') or hasattr(flash_attn, 'flash_attn_kvpacked_func')
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    FLASH_ATTN_3_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@register_attention('flash_attention3')
class FlashAttention3(AttentionWithCache):
    """
    FlashAttention-3 with unified best practices from both implementations.

    Features:
    - FP8 precision support with error reduction (from advanced_attention/)
    - Unified base class architecture (from attention/)
    - Automatic backend selection (enhanced)
    - Comprehensive fallback chain (unified)
    - KV caching support (from base class)
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        # Validate configuration
        if config.use_fp8 and config.fp8_config is None:
            config.fp8_config = FP8AttentionConfig()

        self.fp8_config = config.fp8_config

        # Initialize FP8 scaling factors if needed
        if config.use_fp8 and self.fp8_config:
            self.register_buffer('q_scale', torch.tensor(1.0))
            self.register_buffer('k_scale', torch.tensor(1.0))
            self.register_buffer('v_scale', torch.tensor(1.0))
            self.register_buffer('out_scale', torch.tensor(1.0))

        # Determine optimal backend
        self._select_backend()

    def _select_backend(self):
        """Select the best available attention backend"""
        device_available = torch.cuda.is_available()
        device_capability = torch.cuda.get_device_capability() if device_available else (0, 0)

        # FlashAttention-3 (Hopper H100+ architecture)
        self.use_flash_attn3 = (
            FLASH_ATTN_3_AVAILABLE and
            device_available and
            device_capability[0] >= 9 and
            self.config.use_fp8
        )

        # FlashAttention-2 (Ampere A100+ architecture)
        self.use_flash_attn2 = (
            FLASH_ATTN_AVAILABLE and
            device_available and
            device_capability[0] >= 8
        )

        # Triton fallback
        self.use_triton = TRITON_AVAILABLE and device_available

        # Selected backend info
        if self.use_flash_attn3:
            self.backend = "flash_attention_3"
        elif self.use_flash_attn2:
            self.backend = "flash_attention_2"
        elif self.use_triton:
            self.backend = "triton"
        else:
            self.backend = "pytorch_native"

    def _apply_fp8_optimization(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply FP8 optimizations from advanced_attention implementation"""
        if not (self.config.use_fp8 and self.fp8_config):
            return q, k, v

        if self.fp8_config.fp8_format == "e4m3":
            # E4M3 format optimization
            q = q * self.q_scale
            k = k * self.k_scale
            v = v * self.v_scale

            # Simulate FP8 quantization (actual implementation would use native ops)
            q = q.to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else q.half()
            k = k.to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else k.half()
            v = v.to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else v.half()

        return q, k, v

    def _compute_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unified attention computation with best available backend"""

        batch_size, num_heads, seq_len, head_dim = q.shape

        # Apply FP8 optimization if enabled
        if self.config.use_fp8 and self.training:
            q, k, v = self._apply_fp8_optimization(q, k, v)

        # Rearrange for flash attention: [B, H, S, D] -> [B, S, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Choose backend
        if self.use_flash_attn3 and seq_len >= (self.fp8_config.sequence_length_threshold if self.fp8_config else 8192):
            attn_output = self._flash_attention3_forward(q, k, v, attention_mask)
        elif self.use_flash_attn2:
            attn_output = self._flash_attention2_forward(q, k, v, attention_mask)
        elif self.use_triton:
            attn_output = self._triton_attention_forward(q, k, v, attention_mask)
        else:
            attn_output = self._pytorch_attention_forward(q, k, v, attention_mask)

        # Rearrange back: [B, S, H, D] -> [B, H, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output

    def _flash_attention3_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """FlashAttention-3 forward pass with FP8 optimizations"""
        try:
            if hasattr(flash_attn, 'flash_attn_v3'):
                return flash_attn.flash_attn_v3(
                    q, k, v,
                    dropout_p=self.config.attention_dropout if self.training else 0.0,
                    causal=self.config.causal,
                    softmax_scale=self.scale,
                    return_attn_probs=False
                )[0]
        except Exception as e:
            warnings.warn(f"FlashAttention-3 failed: {e}, falling back to FlashAttention-2")
            return self._flash_attention2_forward(q, k, v, attn_mask)

    def _flash_attention2_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """FlashAttention-2 forward pass"""
        try:
            return flash_attn_func(
                q, k, v,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                causal=self.config.causal,
                softmax_scale=self.scale
            )
        except Exception as e:
            warnings.warn(f"FlashAttention-2 failed: {e}, falling back to Triton")
            return self._triton_attention_forward(q, k, v, attn_mask)

    def _triton_attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Triton-based attention implementation"""
        # For now, fallback to PyTorch implementation
        # In a full implementation, this would use Triton kernels
        return self._pytorch_attention_forward(q, k, v, attn_mask)

    def _pytorch_attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """PyTorch native attention implementation"""
        # Rearrange to [B, S, H, D] -> [B*H, S, D] for efficient computation
        batch_size, seq_len, num_heads, head_dim = q.shape

        q = q.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        v = v.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        # Apply mask if provided
        if attn_mask is not None:
            scores = self._apply_attention_mask(scores, attn_mask)

        # Apply causal mask if needed
        if self.config.causal:
            causal_mask = self._create_causal_mask(seq_len, q.device)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.bmm(attn_weights, v)

        # Reshape back to [B, S, H, D]
        attn_output = attn_output.reshape(batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2)

        return attn_output

    def get_attention_stats(self):
        """Get enhanced attention statistics"""
        stats = super().get_attention_stats()
        stats.update({
            'backend': self.backend,
            'flash_attn_3_available': self.use_flash_attn3,
            'flash_attn_2_available': self.use_flash_attn2,
            'triton_available': self.use_triton,
            'fp8_enabled': self.config.use_fp8,
            'fp8_format': self.fp8_config.fp8_format if self.fp8_config else None
        })
        return stats


@register_attention('flash_attention2')
class FlashAttention2(FlashAttention3):
    """
    FlashAttention-2 implementation (legacy compatibility).
    Inherits from FlashAttention3 but forces v2 backend.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        # Force FlashAttention-2 backend
        self.use_flash_attn3 = False
        if self.use_flash_attn2:
            self.backend = "flash_attention_2"

    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Force FlashAttention-2 only"""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Rearrange for flash attention
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if self.use_flash_attn2:
            attn_output = self._flash_attention2_forward(q, k, v, attention_mask)
        else:
            attn_output = self._pytorch_attention_forward(q, k, v, attention_mask)

        # Rearrange back
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output


# Register aliases for backward compatibility
register_attention('flash_attention')(FlashAttention3)
register_attention('flashattention3')(FlashAttention3)
register_attention('flashattention2')(FlashAttention2)