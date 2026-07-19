# SPDX-License-Identifier: Apache-2.0
"""
Adapter Layer Implementations for TorchBridge

Provides LoRA, DoRA, QLoRA, and QDoRA layer classes for parameter-efficient
fine-tuning. QLoRA/QDoRA use torchao for base weight quantization; they fall
back gracefully when torchao is not installed.

Key properties per class:
  LoRALinear   — standard LoRA; base FP32/FP16; merge() returns nn.Linear
  DoRALinear   — weight-decomposed LoRA; base FP32/FP16; merge() raises
  QLoRALinear  — INT4/INT8 quantized base + LoRA adapters; requires torchao
  QDoRALinear  — INT4/INT8 quantized base + DoRA adapters; requires torchao
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbridge.adapters.config import InitMethod

if TYPE_CHECKING:
    from torchbridge.precision.formats import QuantizationFormat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional torchao import
# ---------------------------------------------------------------------------

try:
    from torchao.quantization import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        quantize_,
    )

    _TORCHAO_AVAILABLE = True
except ImportError:
    _TORCHAO_AVAILABLE = False


def _get_quant_callable(quant_format: QuantizationFormat):  # type: ignore[name-defined]
    """Return the torchao quantization callable for the given format."""
    from torchbridge.precision.formats import QuantizationFormat

    if quant_format == QuantizationFormat.INT4_WEIGHT_ONLY:
        return int4_weight_only()
    if quant_format == QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS:
        return int8_dynamic_activation_int8_weight()
    raise ValueError(f"Unsupported quantization format for QLoRA: {quant_format}")


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around nn.Linear.

    Adds two small matrices (lora_A, lora_B) alongside the frozen base
    weight. Only the adapter matrices are trainable.

    forward(x) = base_linear(x) + scaling * lora_B(lora_A(dropout(x)))
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_method: InitMethod = InitMethod.KAIMING,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self._scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base weights
        for p in self.base_linear.parameters():
            p.requires_grad_(False)

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights(init_method, rank)

    def _init_weights(self, init_method: InitMethod, rank: int) -> None:
        if init_method == InitMethod.KAIMING:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_method == InitMethod.GAUSSIAN:
            nn.init.normal_(self.lora_A.weight, std=1.0 / math.sqrt(rank))
        else:
            nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + self._scaling * self.lora_B(
            self.lora_A(self.lora_dropout(x))
        )

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear with adapter weights merged into base."""
        delta_w = self.lora_B.weight @ self.lora_A.weight  # [out, in]
        merged_weight = self.base_linear.weight.data + self._scaling * delta_w
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
        )
        merged.weight.data = merged_weight
        if self.base_linear.bias is not None:
            merged.bias.data = self.base_linear.bias.data.clone()
        return merged

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# DoRALinear
# ---------------------------------------------------------------------------


class DoRALinear(nn.Module):
    """Weight-decomposed LoRA (DoRA) around nn.Linear.

    Decomposes the adapted weight into magnitude and direction, allowing
    independent learning of each component.

    forward(x) = F.linear(x, magnitude * direction(W_0 + scaling * BA))
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_method: InitMethod = InitMethod.KAIMING,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self._scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base weights
        for p in self.base_linear.parameters():
            p.requires_grad_(False)

        # Magnitude: per-row norm of the base weight (trainable)
        with torch.no_grad():
            row_norms = base_linear.weight.norm(p=2, dim=1)
        self.magnitude = nn.Parameter(row_norms.clone())

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if init_method == InitMethod.KAIMING:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_method == InitMethod.GAUSSIAN:
            nn.init.normal_(self.lora_A.weight, std=1.0 / math.sqrt(rank))
        else:
            nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self.lora_B.weight @ self.lora_A.weight  # [out, in]
        adapted = self.base_linear.weight + self._scaling * delta_w  # [out, in]
        row_norms = adapted.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        direction = adapted / row_norms  # [out, in]
        final_weight = self.magnitude.unsqueeze(1) * direction  # [out, in]
        bias = self.base_linear.bias if self.base_linear.bias is not None else None
        return F.linear(x, final_weight, bias)

    def merge(self) -> None:
        raise NotImplementedError(
            "DoRALinear.merge() is not supported — direction-magnitude decomposition "
            "cannot be collapsed back to a plain weight matrix."
        )

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# QLoRALinear
# ---------------------------------------------------------------------------


class QLoRALinear(nn.Module):
    """QLoRA: INT4/INT8 quantized base weight + LoRA adapters.

    The base nn.Linear is quantized in-place using torchao before storing.
    Only lora_A and lora_B remain trainable; base weight is frozen + quantized.

    Requires torchao. Raises RuntimeError at construction if unavailable.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_method: InitMethod = InitMethod.KAIMING,
        quant_format: QuantizationFormat | None = None,
    ) -> None:
        super().__init__()

        if not _TORCHAO_AVAILABLE:
            raise RuntimeError(
                "torchao is required for QLoRALinear. "
                "Install it with: pip install torchao"
            )

        from torchbridge.precision.formats import QuantizationFormat

        if quant_format is None:
            quant_format = QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS

        # Quantize base in-place before freezing
        quant_callable = _get_quant_callable(quant_format)
        quantize_(base_linear, quant_callable)

        # Freeze all base parameters (weight is now an AffineQuantizedTensor)
        for p in base_linear.parameters():
            p.requires_grad_(False)

        self.base_linear = base_linear
        self._scaling = alpha / rank
        self._quant_format = quant_format

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if init_method == InitMethod.KAIMING:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_method == InitMethod.GAUSSIAN:
            nn.init.normal_(self.lora_A.weight, std=1.0 / math.sqrt(rank))
        else:
            nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchao dequantizes internally inside base_linear(x)
        return self.base_linear(x) + self._scaling * self.lora_B(
            self.lora_A(self.lora_dropout(x))
        )

    def merge(self) -> None:
        raise NotImplementedError(
            "Cannot merge adapter weights into a quantized base. "
            "Save adapter weights separately and load at inference time."
        )

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# QDoRALinear
# ---------------------------------------------------------------------------


class QDoRALinear(nn.Module):
    """QDoRA: INT4/INT8 quantized base weight + DoRA adapters.

    Magnitude is computed from the pre-quantization float weight norms.
    Base weight is then quantized in-place.

    Requires torchao. Raises RuntimeError at construction if unavailable.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_method: InitMethod = InitMethod.KAIMING,
        quant_format: QuantizationFormat | None = None,
    ) -> None:
        super().__init__()

        if not _TORCHAO_AVAILABLE:
            raise RuntimeError(
                "torchao is required for QDoRALinear. "
                "Install it with: pip install torchao"
            )

        from torchbridge.precision.formats import QuantizationFormat

        if quant_format is None:
            quant_format = QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS

        # Compute magnitude BEFORE quantizing (float weights have exact norms)
        with torch.no_grad():
            row_norms = base_linear.weight.norm(p=2, dim=1)

        # Quantize base in-place
        quant_callable = _get_quant_callable(quant_format)
        quantize_(base_linear, quant_callable)

        for p in base_linear.parameters():
            p.requires_grad_(False)

        self.base_linear = base_linear
        self._scaling = alpha / rank
        self._quant_format = quant_format
        self.magnitude = nn.Parameter(row_norms.clone())

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if init_method == InitMethod.KAIMING:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_method == InitMethod.GAUSSIAN:
            nn.init.normal_(self.lora_A.weight, std=1.0 / math.sqrt(rank))
        else:
            nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize base weight for direction computation
        base_w = self.base_linear.weight
        if hasattr(base_w, "dequantize"):
            base_w = base_w.dequantize()
        else:
            base_w = base_w.float()

        delta_w = self.lora_B.weight @ self.lora_A.weight  # [out, in]
        adapted = base_w + self._scaling * delta_w  # [out, in]
        row_norms = adapted.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        direction = adapted / row_norms
        final_weight = self.magnitude.unsqueeze(1) * direction
        bias = self.base_linear.bias if self.base_linear.bias is not None else None
        return F.linear(x, final_weight, bias)

    def merge(self) -> None:
        raise NotImplementedError(
            "Cannot merge adapter weights into a quantized base. "
            "Save adapter weights separately and load at inference time."
        )

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
