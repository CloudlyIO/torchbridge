"""
Intel XPU Backend for KernelPyTorch

Provides support for Intel XPU devices including:
- Intel Data Center Max Series (Ponte Vecchio)
- Intel Arc GPUs (DG2)
- Intel integrated graphics

Features:
- IPEX (Intel Extension for PyTorch) integration
- oneDNN operator fusion
- SYCL kernel support
- Memory management for XPU devices
- Multi-device coordination

Usage:
    from kernel_pytorch.backends.intel import IntelBackend

    # Initialize backend
    backend = IntelBackend(config)

    # Prepare model for Intel XPU
    model = backend.prepare_model(model)

    # Optimize for inference
    model = backend.optimize_for_inference(model)

Requirements:
    - Intel Extension for PyTorch (IPEX)
    - Intel oneAPI Base Toolkit
    - PyTorch with XPU support
"""

from .intel_exceptions import (
    IntelBackendError,
    IntelXPUError,
    XPUNotAvailableError,
    IPEXNotInstalledError,
    XPUDeviceError,
    XPUOutOfMemoryError,
    XPUMemoryAllocationError,
    OneDNNError,
    SYCLCompilationError,
    DPCPPError,
    XPUOptimizationError,
    AMXNotSupportedError,
    InvalidXPUArchitectureError,
    XPUConfigurationError,
)

from .xpu_utilities import (
    XPUDeviceManager,
    XPUDeviceInfo,
    XPUOptimizations,
    get_xpu_device_count,
    is_xpu_available,
    is_ipex_available,
    get_ipex_version,
    xpu_synchronize,
    xpu_empty_cache,
    IPEX_AVAILABLE,
    XPU_AVAILABLE,
)

from .memory_manager import IntelMemoryManager

from .intel_backend import IntelBackend

from .intel_optimizer import (
    IntelOptimizer,
    IntelKernelOptimizer,
    IntelOptimizationLevel,
    OptimizationResult,
)

__version__ = "0.4.7"

__all__ = [
    # Main backend
    'IntelBackend',

    # Memory management
    'IntelMemoryManager',

    # Device utilities
    'XPUDeviceManager',
    'XPUDeviceInfo',
    'XPUOptimizations',
    'get_xpu_device_count',
    'is_xpu_available',
    'is_ipex_available',
    'get_ipex_version',
    'xpu_synchronize',
    'xpu_empty_cache',

    # Optimization
    'IntelOptimizer',
    'IntelKernelOptimizer',
    'IntelOptimizationLevel',
    'OptimizationResult',

    # Exceptions
    'IntelBackendError',
    'IntelXPUError',
    'XPUNotAvailableError',
    'IPEXNotInstalledError',
    'XPUDeviceError',
    'XPUOutOfMemoryError',
    'XPUMemoryAllocationError',
    'OneDNNError',
    'SYCLCompilationError',
    'DPCPPError',
    'XPUOptimizationError',
    'AMXNotSupportedError',
    'InvalidXPUArchitectureError',
    'XPUConfigurationError',

    # Constants
    'IPEX_AVAILABLE',
    'XPU_AVAILABLE',
]
