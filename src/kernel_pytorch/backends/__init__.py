"""
Backend implementations for KernelPyTorch.

This module provides hardware-specific optimizations for:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm/HIP)
- Google TPUs (PyTorch/XLA)

All backends inherit from shared base classes for consistent interfaces.

Version: 0.3.7
"""

from .base_exceptions import (
    BackendError,
    DeviceNotAvailableError,
    DeviceError,
    MemoryError,
    OutOfMemoryError,
    MemoryAllocationError,
    MemoryPoolError,
    CompilationError,
    KernelCompilationError,
    OptimizationError,
    ModelOptimizationError,
    ConfigurationError,
    InvalidArchitectureError,
    KernelError,
    KernelLaunchError,
    raise_or_warn,
)

from .base_memory_manager import (
    BaseMemoryManager,
    BaseMemoryStats,
    MemoryAllocationInfo,
)

__all__ = [
    # Base exceptions
    'BackendError',
    'DeviceNotAvailableError',
    'DeviceError',
    'MemoryError',
    'OutOfMemoryError',
    'MemoryAllocationError',
    'MemoryPoolError',
    'CompilationError',
    'KernelCompilationError',
    'OptimizationError',
    'ModelOptimizationError',
    'ConfigurationError',
    'InvalidArchitectureError',
    'KernelError',
    'KernelLaunchError',
    'raise_or_warn',
    # Base memory manager
    'BaseMemoryManager',
    'BaseMemoryStats',
    'MemoryAllocationInfo',
]
