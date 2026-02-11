"""
Backend implementations for TorchBridge.

This module provides hardware-specific backend implementations for:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm/HIP)
- Google TPUs (PyTorch/XLA)
- AWS Trainium/Inferentia (NeuronX)

All backends inherit from shared base classes for consistent interfaces.

"""

# Base exceptions
# Backend factory
from .backend_factory import (
    BackendFactory,
    BackendType,
    detect_best_backend,
    get_backend,
    get_optimizer,
    list_available_backends,
)

# Base adapter
from .base_adapter import (
    BaseAdapter,
    BaseKernelAdapter,
    CPUAdapter,
    KernelConfig,
    OptimizationStrategy,
)

# Base backend
from .base_backend import (
    BaseBackend,
    CPUBackend,
    DeviceInfo,
    OptimizationLevel,
    OptimizationResult,
)
from .base_exceptions import (
    BackendError,
    CompilationError,
    ConfigurationError,
    DeviceError,
    DeviceNotAvailableError,
    InvalidArchitectureError,
    KernelCompilationError,
    KernelError,
    KernelLaunchError,
    MemoryAllocationError,
    MemoryError,
    MemoryPoolError,
    ModelOptimizationError,
    OptimizationError,
    OutOfMemoryError,
    raise_or_warn,
)

# Base memory manager
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
    # Base backend
    'BaseBackend',
    'CPUBackend',
    'OptimizationLevel',
    'DeviceInfo',
    'OptimizationResult',
    # Base adapter
    'BaseAdapter',
    'BaseKernelAdapter',
    'CPUAdapter',
    'KernelConfig',
    'OptimizationStrategy',
    # Backend factory
    'BackendFactory',
    'BackendType',
    'get_backend',
    'get_optimizer',
    'detect_best_backend',
    'list_available_backends',
]
