# SPDX-License-Identifier: Apache-2.0
"""
Quantization Format Definitions

Defines the internal quantization formats and their metadata for backend-aware
quantization dispatch. QuantizationFormat is the detailed internal enum used by
the compatibility matrix and engine; the user-facing QuantizationMode in
llm_optimizer.py maps to these formats.
"""

from dataclasses import dataclass
from enum import Enum


class QuantizationFormat(Enum):
    """Internal quantization formats for dispatch."""

    NONE = "none"
    INT8_DYNAMIC = "int8_dynamic"
    INT8_DYNAMIC_ACTIVATIONS = "int8_dynamic_activations"
    INT4_WEIGHT_ONLY = "int4_weight_only"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    NVFP4 = "nvfp4"
    BF16 = "bf16"

    @classmethod
    def from_string(cls, value: str) -> "QuantizationFormat":
        """Convert string to QuantizationFormat, case-insensitive."""
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "int8": cls.INT8_DYNAMIC,
            "int4": cls.INT4_WEIGHT_ONLY,
            "fp8": cls.FP8_E4M3,
            "fp4": cls.NVFP4,
            "auto": cls.NONE,  # AUTO is resolved by the engine, not here
        }
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown quantization format: '{value}'. "
            f"Valid formats: {[m.value for m in cls]}"
        )


@dataclass(frozen=True)
class FormatSpec:
    """Metadata for a quantization format."""

    bits: int
    display_name: str
    perplexity_tolerance_pct: float
    memory_reduction_pct: float
    requires_calibration: bool
    requires_torchao: bool


FORMAT_SPECS: dict[QuantizationFormat, FormatSpec] = {
    QuantizationFormat.NONE: FormatSpec(
        bits=32,
        display_name="Full Precision",
        perplexity_tolerance_pct=0.0,
        memory_reduction_pct=0.0,
        requires_calibration=False,
        requires_torchao=False,
    ),
    QuantizationFormat.INT8_DYNAMIC: FormatSpec(
        bits=8,
        display_name="INT8 Dynamic",
        perplexity_tolerance_pct=2.0,
        memory_reduction_pct=50.0,
        requires_calibration=False,
        requires_torchao=False,
    ),
    QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS: FormatSpec(
        bits=8,
        display_name="INT8 Dynamic Activations",
        perplexity_tolerance_pct=1.0,
        memory_reduction_pct=50.0,
        requires_calibration=False,
        requires_torchao=True,
    ),
    QuantizationFormat.INT4_WEIGHT_ONLY: FormatSpec(
        bits=4,
        display_name="INT4 Weight-Only",
        perplexity_tolerance_pct=5.0,
        memory_reduction_pct=75.0,
        requires_calibration=False,
        requires_torchao=True,
    ),
    QuantizationFormat.FP8_E4M3: FormatSpec(
        bits=8,
        display_name="FP8 E4M3",
        perplexity_tolerance_pct=1.0,
        memory_reduction_pct=50.0,
        requires_calibration=False,
        requires_torchao=False,
    ),
    QuantizationFormat.FP8_E5M2: FormatSpec(
        bits=8,
        display_name="FP8 E5M2",
        perplexity_tolerance_pct=1.5,
        memory_reduction_pct=50.0,
        requires_calibration=False,
        requires_torchao=False,
    ),
    QuantizationFormat.NVFP4: FormatSpec(
        bits=4,
        display_name="NVFP4 (Blackwell DC)",
        perplexity_tolerance_pct=2.0,
        memory_reduction_pct=87.5,
        requires_calibration=False,
        requires_torchao=False,
    ),
    QuantizationFormat.BF16: FormatSpec(
        bits=16,
        display_name="BFloat16",
        perplexity_tolerance_pct=0.1,
        memory_reduction_pct=50.0,
        requires_calibration=False,
        requires_torchao=False,
    ),
}


def get_format_spec(fmt: QuantizationFormat) -> FormatSpec:
    """Get the metadata spec for a quantization format."""
    return FORMAT_SPECS[fmt]
