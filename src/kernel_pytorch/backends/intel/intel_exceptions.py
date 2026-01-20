"""
Intel XPU Backend Exceptions

Provides structured exception handling for Intel XPU operations,
following the same patterns as NVIDIA and AMD backends.
"""

from ..base_exceptions import (
    BackendError,
    DeviceNotAvailableError,
    DeviceError,
    MemoryError,
    OutOfMemoryError,
    MemoryAllocationError,
    CompilationError,
    KernelCompilationError,
    OptimizationError,
    ConfigurationError,
    InvalidArchitectureError,
)


class IntelBackendError(BackendError):
    """Base exception for all Intel XPU backend errors."""
    pass


class XPUNotAvailableError(DeviceNotAvailableError):
    """Intel XPU runtime is not available."""

    def __init__(self, message: str = "Intel XPU (IPEX) is not available"):
        super().__init__("INTEL", message)


class IPEXNotInstalledError(DeviceNotAvailableError):
    """Intel Extension for PyTorch (IPEX) is not installed."""

    def __init__(self, message: str = "Intel Extension for PyTorch (IPEX) is not installed. Install with: pip install intel-extension-for-pytorch"):
        super().__init__("INTEL", message)


class XPUDeviceError(DeviceError):
    """Error during XPU device operations."""

    def __init__(self, message: str, device_id: int = 0, details: dict = None):
        super().__init__(
            device_id=device_id,
            operation="xpu_operation",
            error_message=message
        )
        self.device_id = device_id
        if details:
            self.details.update(details)


class XPUOutOfMemoryError(OutOfMemoryError):
    """Out of memory error on Intel XPU device."""

    def __init__(
        self,
        requested_bytes: int,
        available_bytes: int,
        device_id: int = 0,
        details: dict = None
    ):
        super().__init__(
            required_bytes=requested_bytes,
            available_bytes=available_bytes,
            device=f"xpu:{device_id}"
        )
        self.device_id = device_id
        self.requested_bytes = requested_bytes
        if details:
            self.details.update(details)


class XPUMemoryAllocationError(MemoryAllocationError):
    """Failed to allocate memory on Intel XPU."""

    def __init__(self, message: str, size_bytes: int = 0, details: dict = None):
        super().__init__(operation="xpu_allocation", error_message=message)
        self.size_bytes = size_bytes
        if details:
            self.details.update(details)


class OneDNNError(IntelBackendError):
    """Error in oneDNN operations."""

    def __init__(self, message: str, operation: str = None, details: dict = None):
        full_message = f"oneDNN error: {message}"
        if operation:
            full_message = f"oneDNN error in {operation}: {message}"
        super().__init__(full_message, details)
        self.operation = operation


class SYCLCompilationError(KernelCompilationError):
    """Error during SYCL kernel compilation."""

    def __init__(self, message: str, kernel_name: str = None, details: dict = None):
        super().__init__(
            kernel_name=kernel_name or "unknown",
            compiler="SYCL/DPC++",
            error_message=message
        )
        if details:
            self.details.update(details)


class DPCPPError(CompilationError):
    """Error in DPC++ (Data Parallel C++) operations."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(compiler="DPC++", error_message=message)
        if details:
            self.details.update(details)


class XPUOptimizationError(OptimizationError):
    """Error during Intel XPU optimization."""

    def __init__(self, message: str, optimization_type: str = None, details: dict = None):
        super().__init__(
            optimization_type=optimization_type or "xpu_optimization",
            error_message=message
        )
        if details:
            self.details.update(details)


class AMXNotSupportedError(IntelBackendError):
    """Intel Advanced Matrix Extensions (AMX) is not supported on this hardware."""

    def __init__(self, message: str = "Intel AMX is not supported on this CPU/XPU"):
        super().__init__(message)


class InvalidXPUArchitectureError(InvalidArchitectureError):
    """Invalid or unsupported Intel XPU architecture."""

    def __init__(self, architecture: str, supported: list = None):
        supported = supported or ["pvc", "ats", "dg2", "flex"]
        super().__init__(architecture=architecture, supported=supported)


class XPUConfigurationError(ConfigurationError):
    """Invalid Intel XPU configuration."""

    def __init__(self, message: str, config_key: str = None, details: dict = None):
        super().__init__(
            parameter=config_key or "unknown",
            value=details.get('value') if details else None,
            reason=message
        )
        if details:
            self.details.update(details)


# Alias for convenience
IntelXPUError = IntelBackendError


__all__ = [
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
]
