# SPDX-License-Identifier: Apache-2.0
"""
TPU Backend Module for TorchBridge

This module provides TPU-specific backend implementations for cross-backend
validation and configuration intelligence using PyTorch/XLA integration.

Key Components:
- TPU Device Management
- PyTorch/XLA Integration
- TPU-specific Memory Management
- XLA Compilation and Backend Integration
- TPU Model Deployment Support

Example:
    ```python
    from torchbridge.backends.tpu import TPUBackend, TPUAdapter

    # Initialize TPU backend
    backend = TPUBackend()
    model = backend.prepare_model(your_model)

    # Optimize for TPU
    adapter = TPUAdapter(backend.config)
    optimized_model = adapter.optimize(model)
    ```
"""

from .tpu_adapter import TPUAdapter
from .tpu_backend import TPUBackend
from .xla_compiler import XLACompiler
from .xla_integration import (
    XLADeviceManager,
    XLADistributedTraining,
    XLAOptimizations,
    XLAUtilities,
    create_xla_integration,
)

__all__ = [
    "TPUBackend",
    "TPUAdapter",
    "XLACompiler",
    "XLADeviceManager",
    "XLADistributedTraining",
    "XLAOptimizations",
    "XLAUtilities",
    "create_xla_integration",
]

# Version compatibility
__tpu_support__ = True

try:
    import torch_xla

    __xla_available__ = True
    __torch_xla_version__ = torch_xla.__version__
except ImportError:
    __xla_available__ = False
    __torch_xla_version__ = None
