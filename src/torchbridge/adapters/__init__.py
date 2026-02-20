"""
Adapter Training for TorchBridge

Unified LoRA/QLoRA/DoRA/QDoRA API with backend-optimized PEFT,
multi-adapter serving, and cross-backend compatibility.
"""

from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
from torchbridge.adapters.config import AdapterConfig, AdapterMethod
from torchbridge.adapters.engine import AdapterEngine, AdapterResult
from torchbridge.adapters.layers import DoRALinear, LoRALinear
from torchbridge.adapters.serving import MultiAdapterManager

__all__ = [
    "AdapterCompatibilityMatrix",
    "AdapterConfig",
    "AdapterEngine",
    "AdapterMethod",
    "AdapterResult",
    "DoRALinear",
    "LoRALinear",
    "MultiAdapterManager",
]
