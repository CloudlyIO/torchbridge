"""
Inference Subpackage

Backend-aware speculative method selection, structured output formats,
and disaggregated serving configuration for TorchBridge.
"""

from .disaggregated import (
    DisaggregatedFleetAdvisor,
    DisaggregatedFleetConfig,
    DisaggregatedRoleConfig,
)
from .kv_handoff import KVHandoffNegotiator, KVHandoffSpec
from .output_format import OutputFormat, OutputFormatSpec
from .speculative import (
    SPECULATIVE_METHOD_SPECS,
    SpeculationCompatibilityMatrix,
    SpeculativeMethod,
    SpeculativeMethodSpec,
)

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
    "SpeculationCompatibilityMatrix",
    # Structured output formats
    "OutputFormat",
    "OutputFormatSpec",
]
