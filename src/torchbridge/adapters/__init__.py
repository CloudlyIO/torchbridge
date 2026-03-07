"""
Adapter Training for TorchBridge

Unified LoRA/QLoRA/DoRA/QDoRA API with backend-optimized PEFT,
multi-adapter serving, and cross-backend compatibility.
"""

from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
from torchbridge.adapters.config import AdapterConfig, AdapterMethod
from torchbridge.adapters.engine import AdapterEngine, AdapterResult
from torchbridge.adapters.layers import DoRALinear, LoRALinear, QDoRALinear, QLoRALinear
from torchbridge.adapters.model_families import (
    ModelFamily,
    ModelFamilySpec,
    detect_model_family,
    get_model_family_spec,
    get_target_modules,
)
from torchbridge.adapters.serving import MultiAdapterManager

__all__ = [
    "AdapterCompatibilityMatrix",
    "AdapterConfig",
    "AdapterEngine",
    "AdapterMethod",
    "AdapterResult",
    "DoRALinear",
    "LoRALinear",
    "ModelFamily",
    "ModelFamilySpec",
    "MultiAdapterManager",
    "QDoRALinear",
    "QLoRALinear",
    "detect_model_family",
    "get_model_family_spec",
    "get_target_modules",
]
