"""
TPU Backend Module for KernelPyTorch

This module provides TPU-specific implementations and optimizations
for PyTorch models using PyTorch/XLA integration.

Key Components:
- TPU Device Management
- PyTorch/XLA Integration
- TPU-specific Memory Optimization
- XLA Compilation and Optimization
- TPU Model Deployment Support

Example:
    ```python
    from kernel_pytorch.backends.tpu import TPUBackend, TPUOptimizer

    # Initialize TPU backend
    backend = TPUBackend()
    model = backend.prepare_model(your_model)

    # Optimize for TPU
    optimizer = TPUOptimizer(backend.config)
    optimized_model = optimizer.optimize(model)
    ```
"""

from .tpu_backend import TPUBackend
from .tpu_optimizer import TPUOptimizer
from .xla_compiler import XLACompiler
from .memory_manager import TPUMemoryManager
from .xla_integration import (
    XLADeviceManager,
    XLADistributedTraining,
    XLAOptimizations,
    XLAUtilities,
    create_xla_integration
)

__all__ = [
    'TPUBackend',
    'TPUOptimizer',
    'XLACompiler',
    'TPUMemoryManager',
    'XLADeviceManager',
    'XLADistributedTraining',
    'XLAOptimizations',
    'XLAUtilities',
    'create_xla_integration'
]

# Version compatibility
__version__ = "0.3.8"
__tpu_support__ = True
__phase__ = "Phase 4C-Pre Complete"

try:
    import torch_xla
    __xla_available__ = True
    __torch_xla_version__ = torch_xla.__version__
except ImportError:
    __xla_available__ = False
    __torch_xla_version__ = None