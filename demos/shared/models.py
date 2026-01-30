"""
Shared model architectures for TorchBridge demos.

This module provides common model architectures used across demo scripts.

Version: 0.3.6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


# ============================================================================
# Simple Models
# ============================================================================

class SimpleLinear(nn.Module):
    """Simple linear layer for basic testing."""

    def __init__(self, in_features: int = 128, out_features: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleMLP(nn.Module):
    """Simple MLP for basic testing."""

    def __init__(
        self,
        in_features: int = 128,
        hidden_features: int = 256,
        out_features: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ============================================================================
# Transformer Models
# ============================================================================

class SimpleTransformer(nn.Module):
    """
    Simple transformer model for demonstration.

    Args:
        vocab_size: Vocabulary size for embedding
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward dimension (default: 4*d_model)
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.embedding(x) * (self.d_model ** 0.5) + pos
        x = self.transformer(x)
        return self.output_projection(x)


class SimpleAttention(nn.Module):
    """
    Simple attention module for testing attention-specific optimizations.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key is None:
            key = x
        if value is None:
            value = x

        attn_out, _ = self.attention(x, key, value)
        return self.norm(x + attn_out)


# ============================================================================
# Vision Models
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN for vision testing.

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of output classes
        base_channels: Base number of channels (scaled up in deeper layers)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ============================================================================
# Model Factory
# ============================================================================

ModelType = Literal["linear", "mlp", "transformer", "attention", "cnn"]


def create_model(
    model_type: ModelType,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create demo models.

    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model

    Example:
        >>> model = create_model("transformer", d_model=256, num_layers=4)
        >>> model = create_model("cnn", num_classes=100)
    """
    models = {
        "linear": SimpleLinear,
        "mlp": SimpleMLP,
        "transformer": SimpleTransformer,
        "attention": SimpleAttention,
        "cnn": SimpleCNN,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)
