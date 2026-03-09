"""
Adapter Compatibility for TorchBridge

Backend-aware adapter method selection: which method (LoRA/DoRA/QLoRA/QDoRA)
is optimal for the given hardware backend, with fallback chains and base
quantization format recommendations.

For adapter implementation use PEFT, torchao, or Unsloth directly.
For method selection use AdapterCompatibilityMatrix.
"""

from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
from torchbridge.adapters.config import AdapterConfig, AdapterMethod

__all__ = [
    "AdapterCompatibilityMatrix",
    "AdapterConfig",
    "AdapterMethod",
]
