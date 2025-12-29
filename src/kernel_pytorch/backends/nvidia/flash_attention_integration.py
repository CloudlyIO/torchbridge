"""
FlashAttention-3 Integration for NVIDIA GPUs

Optimized attention implementation for H100, Blackwell, and newer NVIDIA GPUs.
"""

import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAArchitecture

logger = logging.getLogger(__name__)


class FlashAttention3(nn.Module):
    """
    FlashAttention-3 implementation for NVIDIA H100/Blackwell.

    Provides memory-efficient attention with 3x memory reduction
    and faster execution compared to standard attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        config: Optional[KernelPyTorchConfig] = None
    ):
        """
        Initialize FlashAttention-3.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            causal: Whether to use causal (autoregressive) masking
            config: KernelPyTorch configuration
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.config = config or KernelPyTorchConfig()
        self.nvidia_config = self.config.hardware.nvidia

        # Check for optimal head dimensions
        optimal_div = 16 if self.nvidia_config.tensor_core_version >= 4 else 8
        if self.head_dim % optimal_div != 0:
            warnings.warn(
                f"Head dimension {self.head_dim} not divisible by {optimal_div}. "
                f"Consider using head_dim divisible by {optimal_div} for optimal Tensor Core performance."
            )

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # FlashAttention availability
        self._flash_available = self._check_flash_availability()

        # Use FlashAttention-3 if available
        self.use_flash_attention = (
            self._flash_available and
            self.nvidia_config.flash_attention_enabled and
            self.nvidia_config.flash_attention_version == "3"
        )

    def _check_flash_availability(self) -> bool:
        """Check if FlashAttention is available."""
        try:
            # Check for flash_attn package
            import flash_attn
            return True
        except ImportError:
            # FlashAttention not available, will use PyTorch implementation
            return False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with FlashAttention-3.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply FlashAttention if available
        if self.use_flash_attention and not return_attention_weights:
            attn_output = self._flash_attention_forward(q, k, v, attention_mask)
            attn_weights = None
        else:
            # Fall back to standard attention
            attn_output, attn_weights = self._standard_attention_forward(
                q, k, v, attention_mask, return_attention_weights
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)

        if self.resid_dropout:
            output = self.resid_dropout(output)

        return output, attn_weights

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FlashAttention-3 forward pass.

        Args:
            q, k, v: Query, key, value tensors
            attention_mask: Optional attention mask

        Returns:
            Attention output
        """
        try:
            from flash_attn import flash_attn_func

            # Prepare inputs for FlashAttention
            # FlashAttention expects (batch, seq_len, num_heads, head_dim)
            q = q.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply FlashAttention
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=None,  # Use default 1/sqrt(head_dim)
                causal=self.causal,
            )

            # Transpose back
            output = output.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

            return output

        except Exception as e:
            warnings.warn(
                f"FlashAttention failed ({e}), falling back to standard attention"
            )
            # Fall back to standard attention
            return self._standard_attention_forward(q, k, v, attention_mask, False)[0]

    def _standard_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard scaled dot-product attention.

        Args:
            q, k, v: Query, key, value tensors
            attention_mask: Optional attention mask
            return_weights: Whether to return attention weights

        Returns:
            Tuple of (output, weights)
        """
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if self.attn_dropout and self.training:
            attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        if return_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None


def create_flash_attention_3(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    bias: bool = True,
    config: Optional[KernelPyTorchConfig] = None
) -> FlashAttention3:
    """
    Factory function to create FlashAttention-3 module.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias
        config: KernelPyTorch configuration

    Returns:
        FlashAttention3 module
    """
    return FlashAttention3(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        config=config
    )


class FlashAttention3Optimizer:
    """
    Optimizer for FlashAttention-3 integration.

    Helps convert existing attention modules to FlashAttention-3.
    """

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize FlashAttention-3 optimizer.

        Args:
            config: KernelPyTorch configuration
        """
        self.config = config or KernelPyTorchConfig()

    def convert_attention_layer(
        self,
        module: nn.Module
    ) -> nn.Module:
        """
        Convert attention module to FlashAttention-3.

        Args:
            module: Attention module to convert

        Returns:
            FlashAttention-3 module or original if conversion not possible
        """
        # Check if module is MultiheadAttention
        if isinstance(module, nn.MultiheadAttention):
            return FlashAttention3(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                config=self.config
            )

        # Return original module if not convertible
        return module

    def optimize_model(self, model: nn.Module) -> Tuple[nn.Module, int]:
        """
        Convert all attention layers in model to FlashAttention-3.

        Args:
            model: PyTorch model

        Returns:
            Tuple of (optimized_model, num_conversions)
        """
        num_conversions = 0

        for name, module in model.named_children():
            if isinstance(module, nn.MultiheadAttention):
                # Replace with FlashAttention-3
                flash_attn = self.convert_attention_layer(module)
                setattr(model, name, flash_attn)
                num_conversions += 1
            elif len(list(module.children())) > 0:
                # Recursively optimize child modules
                _, child_conversions = self.optimize_model(module)
                num_conversions += child_conversions

        return model, num_conversions
