"""
Inference Subpackage

Backend-aware speculative method selection, structured output formats,
and disaggregated serving phase detection for TorchBridge.
"""

from .disaggregated import (
    DisaggregatedFleetAdvisor,
    DisaggregatedFleetConfig,
    DisaggregatedRoleConfig,
)
from .kv_handoff import KVHandoffNegotiator, KVHandoffSpec
from .phase_detection import PhaseDetector, PhaseProfile, PhaseType
from .speculative import (
    SPECULATIVE_METHOD_SPECS,
    SpeculationCompatibilityMatrix,
    SpeculativeMethod,
    SpeculativeMethodSpec,
    get_method_spec,
)
from .structured import OutputFormat, OutputFormatSpec, get_format_spec

__all__ = [
    # Disaggregated serving fleet advisor
    "DisaggregatedFleetAdvisor",
    "DisaggregatedFleetConfig",
    "DisaggregatedRoleConfig",
    # KV cache handoff physical spec
    "KVHandoffNegotiator",
    "KVHandoffSpec",
    # Speculative decoding compatibility
    "SpeculativeMethod",
    "SpeculativeMethodSpec",
    "SPECULATIVE_METHOD_SPECS",
    "get_method_spec",
    "SpeculationCompatibilityMatrix",
    # Structured output formats
    "OutputFormat",
    "OutputFormatSpec",
    "get_format_spec",
    # Phase detection
    "PhaseType",
    "PhaseProfile",
    "PhaseDetector",
]
