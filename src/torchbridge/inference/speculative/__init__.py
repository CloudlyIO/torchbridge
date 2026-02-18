"""
Speculative Decoding Subpackage

Backend-aware speculative decoding with compatibility matrix, method
selection, and HuggingFace generate() integration.
"""

from .compatibility import SpeculationCompatibilityMatrix
from .engine import SpeculationConfig, SpeculationEngine
from .methods import (
    SPECULATIVE_METHOD_SPECS,
    SpeculativeMethod,
    SpeculativeMethodSpec,
    get_method_spec,
)

__all__ = [
    "SpeculativeMethod",
    "SpeculativeMethodSpec",
    "SPECULATIVE_METHOD_SPECS",
    "get_method_spec",
    "SpeculationCompatibilityMatrix",
    "SpeculationConfig",
    "SpeculationEngine",
]
