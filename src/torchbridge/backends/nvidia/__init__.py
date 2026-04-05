"""
NVIDIA Backend Module for TorchBridge

This module provides NVIDIA GPU-specific backend implementations for
PyTorch models targeting H100, Blackwell, Ampere, and other NVIDIA architectures.

Example:
    ```python
    from torchbridge.backends.nvidia import NVIDIABackend, NVIDIAAdapter

    # Initialize NVIDIA backend
    backend = NVIDIABackend()
    model = backend.prepare_model(your_model)

    # Optimize for NVIDIA GPU
    adapter = NVIDIAAdapter(backend.config)
    optimized_model = adapter.optimize(model)
    ```
"""

from .cuda_utilities import (
    CUDADeviceManager,
    CUDAOptimizations,
    CUDAUtilities,
    create_cuda_integration,
)
from .nvidia_adapter import NVIDIAAdapter
from .nvidia_backend import NVIDIABackend

__all__ = [
    "NVIDIABackend",
    "NVIDIAAdapter",
    "CUDADeviceManager",
    "CUDAOptimizations",
    "CUDAUtilities",
    "create_cuda_integration",
]

# Version compatibility
__nvidia_support__ = True

try:
    import torch

    __cuda_available__ = torch.cuda.is_available()
    if __cuda_available__:
        __cuda_version__ = torch.version.cuda
        __cudnn_version__ = torch.backends.cudnn.version()
    else:
        __cuda_version__ = None
        __cudnn_version__ = None
except ImportError:
    __cuda_available__ = False
    __cuda_version__ = None
    __cudnn_version__ = None
