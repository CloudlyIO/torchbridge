"""
Custom exceptions for AMD ROCm backend.

This module provides a comprehensive exception hierarchy for AMD GPU
operations, inheriting from the shared base_exceptions module.

Exception Hierarchy:
- AMDBackendError (BackendError): Base exception for all AMD backend errors
  - ROCmNotAvailableError: ROCm runtime not available
  - HIPCompilationError: HIP kernel compilation failed
  - ROCmMemoryError: Memory allocation/management errors
  - MIOpenError: MIOpen (cuDNN equivalent) errors
  - ROCBLASError: rocBLAS (cuBLAS equivalent) errors
  - AMDDeviceError: Device management errors
  - AMDConfigurationError: Configuration validation errors
  - MatrixCoreError: Matrix core operation errors

Version: 0.3.7
"""

import logging
from typing import Any

from kernel_pytorch.backends.base_exceptions import (
    BackendError,
    ConfigurationError,
    DeviceError,
    DeviceNotAvailableError,
    KernelCompilationError,
    KernelError,
    MemoryError,
    OptimizationError,
    raise_or_warn,
)

logger = logging.getLogger(__name__)


class AMDBackendError(BackendError):
    """Base exception for all AMD backend errors."""
    pass


class ROCmNotAvailableError(DeviceNotAvailableError):
    """Raised when ROCm runtime is not available."""

    def __init__(self, message: str = "ROCm runtime not available"):
        super().__init__("ROCm", message)


class HIPCompilationError(KernelCompilationError):
    """Raised when HIP kernel compilation fails."""

    def __init__(self, kernel_name: str, error_message: str):
        super().__init__(kernel_name, "HIP", error_message)


class ROCmMemoryError(MemoryError):
    """Raised when GPU memory operations fail."""

    def __init__(self, operation: str, required_mb: float = 0.0, available_mb: float = 0.0):
        message = f"Memory {operation} failed: required {required_mb:.1f}MB, available {available_mb:.1f}MB"
        super().__init__(message, {
            "operation": operation,
            "required_mb": required_mb,
            "available_mb": available_mb
        })


class MIOpenError(AMDBackendError):
    """Raised when MIOpen operations fail."""

    def __init__(self, operation: str, error_message: str):
        message = f"MIOpen {operation} failed: {error_message}"
        super().__init__(message, {"operation": operation, "error": error_message})


class ROCBLASError(AMDBackendError):
    """Raised when rocBLAS operations fail."""

    def __init__(self, operation: str, error_message: str):
        message = f"rocBLAS {operation} failed: {error_message}"
        super().__init__(message, {"operation": operation, "error": error_message})


class AMDDeviceError(DeviceError):
    """Raised when device management operations fail."""

    def __init__(self, device_id: int, operation: str, error_message: str):
        super().__init__(device_id, operation, error_message)


class AMDConfigurationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def __init__(self, parameter: str, value: Any, reason: str):
        super().__init__(parameter, value, reason)


class MatrixCoreError(AMDBackendError):
    """Raised when Matrix Core operations fail."""

    def __init__(self, operation: str, architecture: str, error_message: str):
        message = f"Matrix Core {operation} failed on {architecture}: {error_message}"
        super().__init__(message, {
            "operation": operation,
            "architecture": architecture,
            "error": error_message
        })


class AMDOptimizationError(OptimizationError):
    """Raised when optimization operations fail."""

    def __init__(self, optimization_level: str, error_message: str):
        super().__init__(optimization_level, error_message)


class HIPKernelError(KernelError):
    """Raised when HIP kernel execution fails."""

    def __init__(self, kernel_name: str, error_code: int, error_message: str):
        super().__init__(kernel_name, error_code, error_message)


__all__ = [
    "AMDBackendError",
    "ROCmNotAvailableError",
    "HIPCompilationError",
    "ROCmMemoryError",
    "MIOpenError",
    "ROCBLASError",
    "AMDDeviceError",
    "AMDConfigurationError",
    "MatrixCoreError",
    "AMDOptimizationError",
    "HIPKernelError",
    "raise_or_warn",  # Re-export from base
]
