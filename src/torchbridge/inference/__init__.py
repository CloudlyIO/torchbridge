"""
Inference Subpackage

Speculative decoding, structured output, and disaggregated serving
phase detection for TorchBridge.
"""

from .phase_detection import PhaseDetector, PhaseProfile, PhaseType
from .speculative import (
    SpeculationCompatibilityMatrix,
    SpeculationConfig,
    SpeculationEngine,
    SpeculativeMethod,
)
from .structured import OutputFormat, StructuredOutputProcessor

__all__ = [
    # Speculative decoding
    "SpeculativeMethod",
    "SpeculationCompatibilityMatrix",
    "SpeculationConfig",
    "SpeculationEngine",
    # Structured output
    "OutputFormat",
    "StructuredOutputProcessor",
    # Phase detection
    "PhaseType",
    "PhaseProfile",
    "PhaseDetector",
]
