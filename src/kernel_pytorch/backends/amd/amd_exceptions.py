"""
Custom exceptions for AMD ROCm backend.

This module provides a comprehensive exception hierarchy for AMD GPU
operations, following the hardened pattern from NVIDIA and TPU backends.

Exception Hierarchy:
- AMDBackendError: Base exception for all AMD backend errors
  - ROCmNotAvailableError: ROCm runtime not available
  - HIPCompilationError: HIP kernel compilation failed
  - ROCmMemoryError: Memory allocation/management errors
  - MIOpenError: MIOpen (cuDNN equivalent) errors
  - ROCBLASError: rocBLAS (cuBLAS equivalent) errors
  - AMDDeviceError: Device management errors
  - AMDConfigurationError: Configuration validation errors
  - MatrixCoreError: Matrix core operation errors

Version: 0.3.4
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AMDBackendError(Exception):
    """Base exception for all AMD backend errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ROCmNotAvailableError(AMDBackendError):
    """Raised when ROCm runtime is not available."""

    def __init__(self, message: str = "ROCm runtime not available"):
        super().__init__(message)


class HIPCompilationError(AMDBackendError):
    """Raised when HIP kernel compilation fails."""

    def __init__(self, kernel_name: str, error_message: str):
        message = f"HIP compilation failed for kernel '{kernel_name}': {error_message}"
        details = {"kernel": kernel_name, "error": error_message}
        super().__init__(message, details)


class ROCmMemoryError(AMDBackendError):
    """Raised when GPU memory operations fail."""

    def __init__(self, operation: str, required_mb: float, available_mb: float):
        message = f"Memory {operation} failed: required {required_mb}MB, available {available_mb}MB"
        details = {
            "operation": operation,
            "required_mb": required_mb,
            "available_mb": available_mb
        }
        super().__init__(message, details)


class MIOpenError(AMDBackendError):
    """Raised when MIOpen operations fail."""

    def __init__(self, operation: str, error_message: str):
        message = f"MIOpen {operation} failed: {error_message}"
        details = {"operation": operation, "error": error_message}
        super().__init__(message, details)


class ROCBLASError(AMDBackendError):
    """Raised when rocBLAS operations fail."""

    def __init__(self, operation: str, error_message: str):
        message = f"rocBLAS {operation} failed: {error_message}"
        details = {"operation": operation, "error": error_message}
        super().__init__(message, details)


class AMDDeviceError(AMDBackendError):
    """Raised when device management operations fail."""

    def __init__(self, device_id: int, operation: str, error_message: str):
        message = f"Device {device_id} {operation} failed: {error_message}"
        details = {
            "device_id": device_id,
            "operation": operation,
            "error": error_message
        }
        super().__init__(message, details)


class AMDConfigurationError(AMDBackendError):
    """Raised when configuration validation fails."""

    def __init__(self, parameter: str, value: any, reason: str):
        message = f"Invalid configuration for '{parameter}': {value} - {reason}"
        details = {"parameter": parameter, "value": value, "reason": reason}
        super().__init__(message, details)


class MatrixCoreError(AMDBackendError):
    """Raised when Matrix Core operations fail."""

    def __init__(self, operation: str, architecture: str, error_message: str):
        message = f"Matrix Core {operation} failed on {architecture}: {error_message}"
        details = {
            "operation": operation,
            "architecture": architecture,
            "error": error_message
        }
        super().__init__(message, details)


class AMDOptimizationError(AMDBackendError):
    """Raised when optimization operations fail."""

    def __init__(self, optimization_level: str, error_message: str):
        message = f"Optimization ({optimization_level}) failed: {error_message}"
        details = {"level": optimization_level, "error": error_message}
        super().__init__(message, details)


class HIPKernelError(AMDBackendError):
    """Raised when HIP kernel execution fails."""

    def __init__(self, kernel_name: str, error_code: int, error_message: str):
        message = f"HIP kernel '{kernel_name}' failed with code {error_code}: {error_message}"
        details = {
            "kernel": kernel_name,
            "error_code": error_code,
            "error": error_message
        }
        super().__init__(message, details)


# Utility function for error handling pattern
def raise_or_warn(
    exception_class: type,
    message: str,
    strict_mode: bool = False,
    **kwargs
):
    """
    Raise exception in strict mode, otherwise log warning.

    This pattern is used throughout the AMD backend for flexible error handling.

    Args:
        exception_class: Exception class to raise/warn
        message: Error message
        strict_mode: If True, raise exception; if False, log warning
        **kwargs: Additional arguments for exception
    """
    if strict_mode:
        raise exception_class(message, **kwargs)
    else:
        logger.warning(f"{exception_class.__name__}: {message}")


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
    "raise_or_warn",
]
