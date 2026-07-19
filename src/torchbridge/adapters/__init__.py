# SPDX-License-Identifier: Apache-2.0
"""
Adapter System for TorchBridge

Backend-aware LoRA/DoRA/QLoRA/QDoRA injection with hardware-appropriate
quantization format selection.

Quick start::

    from torchbridge.adapters import AdapterEngine, AdapterConfig, AdapterMethod
    from torchbridge.core.config import HardwareBackend

    config = AdapterConfig(method=AdapterMethod.QLORA, rank=16, alpha=32.0)
    engine = AdapterEngine(config=config, backend=HardwareBackend.CUDA)
    result = engine.inject(model)
"""

from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
from torchbridge.adapters.config import AdapterConfig, AdapterMethod
from torchbridge.adapters.engine import AdapterEngine, AdapterResult
from torchbridge.adapters.layers import (
    DoRALinear,
    LoRALinear,
    QDoRALinear,
    QLoRALinear,
)

__all__ = [
    "AdapterCompatibilityMatrix",
    "AdapterConfig",
    "AdapterEngine",
    "AdapterMethod",
    "AdapterResult",
    "DoRALinear",
    "LoRALinear",
    "QDoRALinear",
    "QLoRALinear",
]
