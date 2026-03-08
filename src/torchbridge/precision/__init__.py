"""
Precision Module

Backend-aware quantization format selection and torchao dispatch.
"""

from .quantization import (
    FORMAT_SPECS,
    TORCHAO_AVAILABLE,
    FormatSpec,
    QuantizationCompatibilityMatrix,
    QuantizationEngine,
    QuantizationFormat,
    QuantizationResult,
    TorchAOBackend,
    get_format_spec,
)

__all__ = [
    # Quantization subpackage
    'QuantizationFormat',
    'FormatSpec',
    'FORMAT_SPECS',
    'get_format_spec',
    'QuantizationCompatibilityMatrix',
    'QuantizationEngine',
    'QuantizationResult',
    'TorchAOBackend',
    'TORCHAO_AVAILABLE',
]
