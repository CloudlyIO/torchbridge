"""
Precision Module

Backend-aware quantization format selection and torchao dispatch.
"""

from .compatibility import QuantizationCompatibilityMatrix
from .engine import QuantizationEngine, QuantizationResult
from .formats import (
    FORMAT_SPECS,
    FormatSpec,
    QuantizationFormat,
    get_format_spec,
)
from .torchao_integration import TORCHAO_AVAILABLE, TorchAOBackend

__all__ = [
    "QuantizationFormat",
    "FormatSpec",
    "FORMAT_SPECS",
    "get_format_spec",
    "QuantizationCompatibilityMatrix",
    "QuantizationEngine",
    "QuantizationResult",
    "TorchAOBackend",
    "TORCHAO_AVAILABLE",
]
