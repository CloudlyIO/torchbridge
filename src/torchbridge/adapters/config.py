# SPDX-License-Identifier: Apache-2.0
"""
Adapter Configuration

Defines adapter methods, configuration, and initialization options for
backend-aware parameter-efficient fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from torchbridge.precision.formats import QuantizationFormat


class AdapterMethod(Enum):
    """Adapter training method."""

    LORA = "lora"  # Standard low-rank adaptation
    QLORA = "qlora"  # 4-bit quantized base + LoRA adapters
    DORA = "dora"  # Weight-decomposed LoRA (magnitude + direction)
    QDORA = "qdora"  # 4-bit quantized base + DoRA adapters


class InitMethod(Enum):
    """Weight initialization method for adapter layers."""

    KAIMING = "kaiming"  # Kaiming uniform (default)
    GAUSSIAN = "gaussian"  # Normal distribution with std = 1/sqrt(rank)
    ZEROS = "zeros"  # Both A and B initialized to zeros


@dataclass
class AdapterConfig:
    """Configuration for adapter injection and training.

    Attributes:
        method: Adapter method to apply.
        rank: Low-rank dimension (r). Higher = more capacity, more params.
        alpha: Scaling factor (alpha). Output scaled by alpha/rank.
        dropout: Dropout probability on the adapter path (0 = disabled).
        target_modules: Module name patterns to adapt. Only nn.Linear
            modules whose name ends with one of these strings are adapted.
        quantize_base: Quantize base model weights (auto-set for QLORA/QDORA).
        base_quant_format: Quantization format for base model. Auto-selected
            per backend if None.
        auto_detect_targets: When True, engine auto-detects target modules
            based on model family at inject time, overriding defaults.
        init_method: Weight initialization strategy for adapter matrices.
        merge_on_save: Merge adapter weights into base before saving.
    """

    method: AdapterMethod = AdapterMethod.LORA
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    quantize_base: bool = False
    base_quant_format: QuantizationFormat | None = None
    auto_detect_targets: bool = True
    init_method: InitMethod = InitMethod.KAIMING
    merge_on_save: bool = False

    _DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        # Track whether target_modules was explicitly overridden
        self._targets_explicitly_set = (
            self.target_modules != self._DEFAULT_TARGET_MODULES
        )
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if any(not t for t in self.target_modules):
            raise ValueError("target_modules must not contain empty strings")

        # Auto-set quantize_base for QLoRA/QDoRA
        if self.method in (AdapterMethod.QLORA, AdapterMethod.QDORA):
            self.quantize_base = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "method": self.method.value,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "quantize_base": self.quantize_base,
            "base_quant_format": (
                self.base_quant_format.value if self.base_quant_format else None
            ),
            "auto_detect_targets": self.auto_detect_targets,
            "init_method": self.init_method.value,
            "merge_on_save": self.merge_on_save,
        }
