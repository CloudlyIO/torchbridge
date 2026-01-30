"""
Core Attention Operations

Shared attention computation functions used across all FlashAttention implementations.
This module consolidates the core attention logic to eliminate code duplication.
"""

import math
import warnings

import torch
import torch.nn.functional as F


def check_flash_attention_available() -> bool:
    """Check if FlashAttention library is available."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def check_cuda_kernel_available() -> bool:
    """Check if custom CUDA kernel is available."""
    try:
        import kernel_pytorch_cuda
        return hasattr(kernel_pytorch_cuda, 'flash_attention_v3')
    except (ImportError, AttributeError):
        return False


def validate_attention_inputs(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    expected_dims: int = 4
) -> None:
    """
    Validate attention input tensor shapes and types.

    Args:
        Q, K, V: Query, key, value tensors
        expected_dims: Expected number of dimensions (3 or 4)

    Raises:
        AssertionError: If validation fails
    """
    assert Q.dim() == expected_dims, f"Q must be {expected_dims}D, got {Q.dim()}D"
    assert K.dim() == expected_dims, f"K must be {expected_dims}D, got {K.dim()}D"
    assert V.dim() == expected_dims, f"V must be {expected_dims}D, got {V.dim()}D"

    # Check shape compatibility
    if expected_dims == 4:
        # [batch, heads, seq, head_dim]
        assert Q.shape == K.shape, f"Q and K shapes must match: {Q.shape} vs {K.shape}"
        assert Q.shape[0] == V.shape[0], "Batch sizes must match"
        assert Q.shape[1] == V.shape[1], "Number of heads must match"
        assert K.shape[2] == V.shape[2], "K and V sequence lengths must match"
        assert K.shape[3] == V.shape[3], "K and V head dimensions must match"


def compute_attention_scale(head_dim: int, custom_scale: float | None = None) -> float:
    """Compute attention scale factor."""
    if custom_scale is not None:
        return custom_scale
    return 1.0 / math.sqrt(head_dim)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    dropout: float = 0.0,
    training: bool = False,
    attention_mask: torch.Tensor | None = None,
    return_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute scaled dot-product attention.

    This is the canonical implementation used by all FlashAttention variants.

    Args:
        Q: Query tensor [batch, num_heads, seq_len, head_dim]
        K: Key tensor [batch, num_heads, seq_len, head_dim]
        V: Value tensor [batch, num_heads, seq_len, head_dim]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        dropout: Dropout probability
        training: Whether in training mode
        attention_mask: Optional attention mask to add to scores
        return_weights: Whether to return attention weights

    Returns:
        Tuple of (output, attention_weights or None)
    """
    # Compute scale
    head_dim = Q.size(-1)
    scale = compute_attention_scale(head_dim, scale)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask
    if causal:
        seq_len_q = Q.size(-2)
        seq_len_k = K.size(-2)
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Apply custom attention mask
    if attention_mask is not None:
        scores = scores + attention_mask

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    # Compute output
    output = torch.matmul(attn_weights, V)

    if return_weights:
        return output, attn_weights
    return output, None


def flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    dropout: float = 0.0,
    training: bool = False,
    attention_mask: torch.Tensor | None = None,
    return_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    FlashAttention forward pass with automatic fallback.

    Attempts to use FlashAttention library if available, otherwise falls back
    to PyTorch implementation.

    Args:
        Q: Query tensor [batch, num_heads, seq_len, head_dim]
        K: Key tensor [batch, num_heads, seq_len, head_dim]
        V: Value tensor [batch, num_heads, seq_len, head_dim]
        scale: Attention scale factor
        causal: Whether to apply causal masking
        dropout: Dropout probability
        training: Whether in training mode
        attention_mask: Optional attention mask
        return_weights: Whether to return attention weights

    Returns:
        Tuple of (output, attention_weights or None)
    """
    # If returning weights, must use standard implementation
    if return_weights:
        return scaled_dot_product_attention(
            Q, K, V, scale, causal, dropout, training, attention_mask, return_weights
        )

    # Try FlashAttention library
    if check_flash_attention_available() and Q.is_cuda:
        try:
            from flash_attn import flash_attn_func

            # FlashAttention expects (batch, seq_len, num_heads, head_dim)
            Q_fa = Q.transpose(1, 2)
            K_fa = K.transpose(1, 2)
            V_fa = V.transpose(1, 2)

            output = flash_attn_func(
                Q_fa, K_fa, V_fa,
                dropout_p=dropout if training else 0.0,
                softmax_scale=scale,
                causal=causal,
            )

            # Transpose back to (batch, num_heads, seq_len, head_dim)
            output = output.transpose(1, 2)
            return output, None

        except Exception as e:
            warnings.warn(f"FlashAttention failed ({e}), using fallback", stacklevel=2)

    # Try custom CUDA kernel
    if check_cuda_kernel_available() and Q.is_cuda:
        try:
            import kernel_pytorch_cuda
            head_dim = Q.size(-1)
            scale_val = compute_attention_scale(head_dim, scale)
            output = kernel_pytorch_cuda.flash_attention_v3(Q, K, V, scale_val, causal)
            return output, None
        except Exception as e:
            warnings.warn(f"CUDA kernel failed ({e}), using fallback", stacklevel=2)

    # Fallback to PyTorch implementation
    return scaled_dot_product_attention(
        Q, K, V, scale, causal, dropout, training, attention_mask, return_weights
    )


__all__ = [
    'check_flash_attention_available',
    'check_cuda_kernel_available',
    'validate_attention_inputs',
    'compute_attention_scale',
    'scaled_dot_product_attention',
    'flash_attention_forward',
]
