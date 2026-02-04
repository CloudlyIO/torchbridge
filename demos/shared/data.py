"""
Shared data generation utilities for TorchBridge demos.

This module provides common data generation functions used across demo scripts.

Version: 0.3.6
"""

import torch
from typing import Dict, Tuple, Optional


# ============================================================================
# Input Data Generators
# ============================================================================

def create_linear_input(
    batch_size: int = 32,
    features: int = 128,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create sample input for linear/MLP models.

    Args:
        batch_size: Batch size
        features: Number of features
        device: Device to create tensor on
        dtype: Data type

    Returns:
        Random tensor of shape (batch_size, features)
    """
    x = torch.randn(batch_size, features, dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def create_transformer_input(
    batch_size: int = 8,
    seq_length: int = 64,
    vocab_size: int = 1000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create sample input for transformer models (token IDs).

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        vocab_size: Vocabulary size for random tokens
        device: Device to create tensor on

    Returns:
        Random token tensor of shape (batch_size, seq_length)
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    if device is not None:
        x = x.to(device)
    return x


def create_transformer_embedding_input(
    batch_size: int = 8,
    seq_length: int = 64,
    d_model: int = 256,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create sample embedding input for transformer models.

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        d_model: Model dimension
        device: Device to create tensor on
        dtype: Data type

    Returns:
        Random tensor of shape (batch_size, seq_length, d_model)
    """
    x = torch.randn(batch_size, seq_length, d_model, dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def create_vision_input(
    batch_size: int = 16,
    channels: int = 3,
    height: int = 224,
    width: int = 224,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create sample input for vision models.

    Args:
        batch_size: Batch size
        channels: Number of image channels
        height: Image height
        width: Image width
        device: Device to create tensor on
        dtype: Data type

    Returns:
        Random tensor of shape (batch_size, channels, height, width)
    """
    x = torch.randn(batch_size, channels, height, width, dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def create_attention_input(
    batch_size: int = 8,
    seq_length: int = 64,
    num_heads: int = 8,
    head_dim: int = 64,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample Q, K, V tensors for attention testing.

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to create tensors on
        dtype: Data type

    Returns:
        Tuple of (query, key, value) tensors
    """
    shape = (batch_size, num_heads, seq_length, head_dim)
    q = torch.randn(*shape, dtype=dtype)
    k = torch.randn(*shape, dtype=dtype)
    v = torch.randn(*shape, dtype=dtype)

    if device is not None:
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)

    return q, k, v


# ============================================================================
# Training Data Generators
# ============================================================================

def create_sample_batch(
    model_type: str = "mlp",
    batch_size: int = 32,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Create a sample training batch for different model types.

    Args:
        model_type: Type of model ("mlp", "transformer", "cnn")
        batch_size: Batch size
        num_classes: Number of output classes
        device: Device to create tensors on
        **kwargs: Additional arguments for input creation

    Returns:
        Dictionary with "inputs" and "labels" tensors
    """
    if model_type == "mlp" or model_type == "linear":
        features = kwargs.get("features", 128)
        inputs = create_linear_input(batch_size, features, device)
        labels = torch.randint(0, num_classes, (batch_size,))

    elif model_type == "transformer":
        seq_length = kwargs.get("seq_length", 64)
        vocab_size = kwargs.get("vocab_size", 1000)
        inputs = create_transformer_input(batch_size, seq_length, vocab_size, device)
        # Token-level labels for language modeling
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))

    elif model_type == "cnn" or model_type == "vision":
        channels = kwargs.get("channels", 3)
        height = kwargs.get("height", 224)
        width = kwargs.get("width", 224)
        inputs = create_vision_input(batch_size, channels, height, width, device)
        labels = torch.randint(0, num_classes, (batch_size,))

    elif model_type == "attention":
        seq_length = kwargs.get("seq_length", 64)
        d_model = kwargs.get("d_model", 256)
        inputs = create_transformer_embedding_input(batch_size, seq_length, d_model, device)
        # No specific labels for attention testing
        labels = torch.zeros(batch_size, dtype=torch.long)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if device is not None:
        labels = labels.to(device)

    return {"inputs": inputs, "labels": labels}
