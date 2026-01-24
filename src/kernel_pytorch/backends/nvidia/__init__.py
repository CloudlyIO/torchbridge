"""
NVIDIA Backend Module for KernelPyTorch

This module provides NVIDIA GPU-specific implementations and optimizations
for PyTorch models targeting H100, Blackwell, and other NVIDIA architectures.

Key Components:
- NVIDIA Device Management
- FP8 Training Pipeline
- FlashAttention-3 Integration
- CUDA-specific Memory Optimization
- Tensor Core Optimization
- NVIDIA Model Deployment Support

Example:
    ```python
    from kernel_pytorch.backends.nvidia import NVIDIABackend, NVIDIAOptimizer

    # Initialize NVIDIA backend
    backend = NVIDIABackend()
    model = backend.prepare_model(your_model)

    # Optimize for NVIDIA GPU
    optimizer = NVIDIAOptimizer(backend.config)
    optimized_model = optimizer.optimize(model)
    ```
"""

from .nvidia_backend import NVIDIABackend
from .nvidia_optimizer import NVIDIAOptimizer
from .fp8_compiler import FP8Compiler
from .memory_manager import NVIDIAMemoryManager
from .flash_attention_integration import (
    FlashAttention3,
    create_flash_attention_3,
)
from .cuda_utilities import (
    CUDADeviceManager,
    CUDAOptimizations,
    CUDAUtilities,
    create_cuda_integration
)

__all__ = [
    'NVIDIABackend',
    'NVIDIAOptimizer',
    'FP8Compiler',
    'NVIDIAMemoryManager',
    'FlashAttention3',
    'create_flash_attention_3',
    'CUDADeviceManager',
    'CUDAOptimizations',
    'CUDAUtilities',
    'create_cuda_integration'
]

# Version compatibility
__version__ = "0.4.19"
__nvidia_support__ = True
__phase__ = "Phase 4C-Pre Complete"

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
