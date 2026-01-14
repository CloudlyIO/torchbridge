"""
NVIDIA Backend Custom Exceptions

Custom exception hierarchy for NVIDIA backend error handling,
inheriting from the shared base_exceptions module.

Version: 0.3.7
"""

from typing import Optional, Any, List

from kernel_pytorch.backends.base_exceptions import (
    BackendError,
    DeviceNotAvailableError,
    DeviceError,
    MemoryError,
    MemoryAllocationError as BaseMemoryAllocationError,
    OutOfMemoryError as BaseOutOfMemoryError,
    CompilationError,
    OptimizationError,
    ModelOptimizationError as BaseModelOptimizationError,
    ConfigurationError as BaseConfigurationError,
    InvalidArchitectureError as BaseInvalidArchitectureError,
    KernelLaunchError as BaseKernelLaunchError,
    raise_or_warn,
)


class NVIDIABackendError(BackendError):
    """Base exception for all NVIDIA backend errors."""
    pass


class CUDANotAvailableError(DeviceNotAvailableError):
    """CUDA is not available on this system."""

    def __init__(self, message: str = "CUDA not available"):
        super().__init__("CUDA", message)


class CUDADeviceError(DeviceError):
    """Error related to CUDA device operations."""

    def __init__(self, device_id: int = 0, operation: str = "", error_message: str = ""):
        super().__init__(device_id, operation, error_message)


class FP8CompilationError(CompilationError):
    """Error during FP8 model compilation."""

    def __init__(self, error_message: str):
        super().__init__("FP8", error_message)


class FlashAttentionError(NVIDIABackendError):
    """Error in FlashAttention operations."""

    def __init__(self, operation: str, error_message: str):
        message = f"FlashAttention {operation} failed: {error_message}"
        super().__init__(message, {"operation": operation, "error": error_message})


class MemoryAllocationError(BaseMemoryAllocationError):
    """Error allocating GPU memory."""
    pass


class OutOfMemoryError(BaseOutOfMemoryError):
    """GPU out of memory error."""

    def __init__(self, message: str = "", required_bytes: Optional[int] = None, available_bytes: Optional[int] = None):
        if message:
            # Support legacy string-based initialization
            BackendError.__init__(self, message, {
                "required_bytes": required_bytes,
                "available_bytes": available_bytes,
                "device": "cuda"
            })
        else:
            super().__init__(required_bytes, available_bytes, "cuda")


class InvalidComputeCapabilityError(NVIDIABackendError):
    """Invalid or unsupported compute capability."""

    def __init__(self, compute_capability: str, required: str):
        message = f"Invalid compute capability {compute_capability}, required {required}"
        super().__init__(message, {
            "compute_capability": compute_capability,
            "required": required
        })


class KernelLaunchError(BaseKernelLaunchError):
    """Error launching CUDA kernel."""
    pass


class ModelOptimizationError(BaseModelOptimizationError):
    """Error during model optimization."""

    def __init__(self, model_name: str = "", optimization_type: str = "", error_message: str = ""):
        super().__init__(model_name, optimization_type, error_message)


class InvalidArchitectureError(BaseInvalidArchitectureError):
    """Invalid or unsupported NVIDIA GPU architecture."""

    def __init__(self, architecture: str, supported: Optional[List[str]] = None):
        super().__init__(architecture, supported or [])


class ConfigurationError(BaseConfigurationError):
    """Error in NVIDIA backend configuration."""

    def __init__(self, parameter: str = "", value: Any = None, reason: str = ""):
        super().__init__(parameter, value, reason)


__all__ = [
    "NVIDIABackendError",
    "CUDANotAvailableError",
    "CUDADeviceError",
    "FP8CompilationError",
    "FlashAttentionError",
    "MemoryAllocationError",
    "OutOfMemoryError",
    "InvalidComputeCapabilityError",
    "KernelLaunchError",
    "ModelOptimizationError",
    "InvalidArchitectureError",
    "ConfigurationError",
    "raise_or_warn",  # Re-export from base
]
