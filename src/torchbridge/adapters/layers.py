"""
LoRA and DoRA Linear Layers

Lightweight adapter layer implementations for parameter-efficient
fine-tuning. Pure PyTorch — no external dependencies (PEFT, torchtune).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import InitMethod


def _init_adapter_weights(
    lora_A: nn.Linear, lora_B: nn.Linear, rank: int, method: InitMethod
) -> None:
    """Initialize adapter matrices (shared by LoRA and DoRA)."""
    if method == InitMethod.KAIMING:
        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_B.weight)
    elif method == InitMethod.GAUSSIAN:
        nn.init.normal_(lora_A.weight, std=1.0 / math.sqrt(rank))
        nn.init.zeros_(lora_B.weight)
    elif method == InitMethod.ZEROS:
        nn.init.zeros_(lora_A.weight)
        nn.init.zeros_(lora_B.weight)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear.

    Computes: x -> base(x) + (alpha/rank) * B(A(dropout(x)))

    The base linear is frozen. Only lora_A and lora_B are trainable.

    Args:
        base_linear: The original nn.Linear to wrap.
        rank: Low-rank dimension (r).
        alpha: Scaling factor. Output is scaled by alpha/rank.
        dropout: Dropout probability on the adapter input path.
        init_method: Initialization for adapter matrices.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_method: InitMethod = InitMethod.KAIMING,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self._scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Adapter matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout (only during training)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._init_weights(init_method)

        # Freeze base
        for param in self.base_linear.parameters():
            param.requires_grad = False

    def _init_weights(self, method: InitMethod) -> None:
        """Initialize adapter matrices."""
        _init_adapter_weights(self.lora_A, self.lora_B, self.rank, method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + scaled adapter output."""
        base_out = self.base_linear(x)
        adapter_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + self._scaling * adapter_out

    def merge(self) -> nn.Linear:
        """Merge adapter weights into base and return a plain nn.Linear.

        The merged linear is mathematically equivalent to this module
        but has no adapter overhead at inference time.

        Returns:
            A new nn.Linear with merged weights.
        """
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
        )
        with torch.no_grad():
            # W_merged = W_base + (alpha/rank) * B @ A
            delta = self.lora_B.weight @ self.lora_A.weight
            merged.weight.copy_(self.base_linear.weight + self._scaling * delta)
            if self.base_linear.bias is not None:
                merged.bias.copy_(self.base_linear.bias)
        return merged

    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters (adapter only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        """Count of all parameters (base + adapter)."""
        return sum(p.numel() for p in self.parameters())


class DoRALinear(nn.Module):
    """Weight-decomposed low-rank adapter (DoRA).

    Decomposes the weight update into magnitude and direction:
    - m: learnable magnitude vector (one scalar per output feature)
    - Direction: base weight + low-rank update, column-normalized

    Output: x -> m * column_norm(W_base + (alpha/rank) * B @ A) @ x + bias

    DoRA outperforms LoRA at low ranks (r=8-16) by decoupling magnitude
    and directional changes, matching full fine-tuning update patterns.

    Args:
        base_linear: The original nn.Linear to wrap.
        rank: Low-rank dimension (r).
        alpha: Scaling factor. Low-rank update scaled by alpha/rank.
        dropout: Dropout probability on the adapter input path.
        init_method: Initialization for adapter matrices.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_method: InitMethod = InitMethod.KAIMING,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self._scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Low-rank matrices (same as LoRA)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Magnitude vector: initialized from column norms of base weight
        with torch.no_grad():
            weight_norms = base_linear.weight.norm(dim=1)
        self.magnitude = nn.Parameter(weight_norms.clone())

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize adapter matrices
        self._init_weights(init_method)

        # Freeze base
        for param in self.base_linear.parameters():
            param.requires_grad = False

    def _init_weights(self, method: InitMethod) -> None:
        """Initialize adapter matrices."""
        _init_adapter_weights(self.lora_A, self.lora_B, self.rank, method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight decomposition.

        Computes: m * normalize_columns(W + scaling * B @ A) @ x + bias
        """
        # Adapted weight: W' = W_base + (alpha/rank) * B @ A
        delta = self.lora_B.weight @ self.lora_A.weight
        adapted_weight = self.base_linear.weight + self._scaling * delta

        # Column-normalize (per output feature)
        col_norms = adapted_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        direction = adapted_weight / col_norms

        # Scale by learnable magnitude
        weight = self.magnitude.unsqueeze(1) * direction

        # Apply dropout to input
        x_dropped = self.lora_dropout(x)

        # Linear operation
        out = torch.nn.functional.linear(x_dropped, weight, self.base_linear.bias)
        return out

    def merge(self) -> nn.Linear:
        """Merge adapter + magnitude into base weights.

        Returns:
            A new nn.Linear with merged weights (zero overhead at inference).
        """
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
        )
        with torch.no_grad():
            delta = self.lora_B.weight @ self.lora_A.weight
            adapted = self.base_linear.weight + self._scaling * delta
            col_norms = adapted.norm(dim=1, keepdim=True).clamp(min=1e-8)
            direction = adapted / col_norms
            merged.weight.copy_(self.magnitude.unsqueeze(1) * direction)
            if self.base_linear.bias is not None:
                merged.bias.copy_(self.base_linear.bias)
        return merged

    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters (adapter + magnitude)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        """Count of all parameters (base + adapter + magnitude)."""
        return sum(p.numel() for p in self.parameters())
