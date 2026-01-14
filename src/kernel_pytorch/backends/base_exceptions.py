"""
Base Exception Hierarchy for Backend Implementations

This module provides the shared exception hierarchy that all backend-specific
exceptions should inherit from. This ensures consistent error handling
across NVIDIA, AMD, and TPU backends.

Exception Hierarchy:
- BackendError: Base for all backend errors
  - DeviceNotAvailableError: Device/runtime not available
  - DeviceError: Device operations failure
  - MemoryError: Memory-related errors
    - OutOfMemoryError: Out of memory
    - MemoryAllocationError: General allocation failure
    - MemoryPoolError: Pool operations failure
  - CompilationError: Compilation failures
  - OptimizationError: Optimization failures
  - ConfigurationError: Configuration validation errors
  - KernelError: Kernel execution errors

Version: 0.3.7
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """
    Base exception for all backend errors.

    All backend-specific exceptions should inherit from this class.
    Supports optional details dictionary for structured error information.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize backend error.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


# =============================================================================
# Device Availability Errors
# =============================================================================

class DeviceNotAvailableError(BackendError):
    """Raised when device or runtime is not available."""

    def __init__(self, backend: str, reason: str = ""):
        message = f"{backend} not available"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, {"backend": backend, "reason": reason})


class DeviceError(BackendError):
    """Raised when device operations fail."""

    def __init__(self, device_id: int, operation: str, error_message: str):
        message = f"Device {device_id} {operation} failed: {error_message}"
        super().__init__(message, {
            "device_id": device_id,
            "operation": operation,
            "error": error_message
        })


# =============================================================================
# Memory Errors
# =============================================================================

class MemoryError(BackendError):
    """Base exception for memory-related errors."""
    pass


class OutOfMemoryError(MemoryError):
    """Raised when device runs out of memory."""

    def __init__(
        self,
        required_bytes: Optional[int] = None,
        available_bytes: Optional[int] = None,
        device: str = "unknown"
    ):
        if required_bytes is not None and available_bytes is not None:
            required_mb = required_bytes / (1024 ** 2)
            available_mb = available_bytes / (1024 ** 2)
            message = f"Out of memory on {device}: required {required_mb:.1f}MB, available {available_mb:.1f}MB"
        else:
            message = f"Out of memory on {device}"

        super().__init__(message, {
            "required_bytes": required_bytes,
            "available_bytes": available_bytes,
            "device": device
        })


class MemoryAllocationError(MemoryError):
    """Raised when memory allocation fails."""

    def __init__(self, operation: str, error_message: str):
        message = f"Memory allocation failed during {operation}: {error_message}"
        super().__init__(message, {"operation": operation, "error": error_message})


class MemoryPoolError(MemoryError):
    """Raised when memory pool operations fail."""

    def __init__(self, pool_id: str, operation: str, error_message: str):
        message = f"Memory pool '{pool_id}' {operation} failed: {error_message}"
        super().__init__(message, {
            "pool_id": pool_id,
            "operation": operation,
            "error": error_message
        })


# =============================================================================
# Compilation Errors
# =============================================================================

class CompilationError(BackendError):
    """Base exception for compilation failures."""

    def __init__(self, compiler: str, error_message: str):
        message = f"{compiler} compilation failed: {error_message}"
        super().__init__(message, {"compiler": compiler, "error": error_message})


class KernelCompilationError(CompilationError):
    """Raised when kernel compilation fails."""

    def __init__(self, kernel_name: str, compiler: str, error_message: str):
        self.kernel_name = kernel_name
        message = f"{compiler} compilation failed for kernel '{kernel_name}': {error_message}"
        # Call BackendError.__init__ directly to avoid double formatting
        BackendError.__init__(self, message, {
            "kernel": kernel_name,
            "compiler": compiler,
            "error": error_message
        })


# =============================================================================
# Optimization Errors
# =============================================================================

class OptimizationError(BackendError):
    """Raised when optimization operations fail."""

    def __init__(self, optimization_type: str, error_message: str):
        message = f"Optimization ({optimization_type}) failed: {error_message}"
        super().__init__(message, {"type": optimization_type, "error": error_message})


class ModelOptimizationError(OptimizationError):
    """Raised when model optimization fails."""

    def __init__(self, model_name: str, optimization_type: str, error_message: str):
        self.model_name = model_name
        message = f"Model '{model_name}' optimization ({optimization_type}) failed: {error_message}"
        BackendError.__init__(self, message, {
            "model": model_name,
            "type": optimization_type,
            "error": error_message
        })


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(BackendError):
    """Raised when configuration validation fails."""

    def __init__(self, parameter: str, value: Any, reason: str):
        message = f"Invalid configuration for '{parameter}': {value} - {reason}"
        super().__init__(message, {"parameter": parameter, "value": value, "reason": reason})


class InvalidArchitectureError(ConfigurationError):
    """Raised when an unsupported architecture is specified."""

    def __init__(self, architecture: str, supported: list):
        message = f"Invalid architecture '{architecture}'. Supported: {supported}"
        BackendError.__init__(self, message, {
            "architecture": architecture,
            "supported": supported
        })


# =============================================================================
# Kernel Errors
# =============================================================================

class KernelError(BackendError):
    """Raised when kernel execution fails."""

    def __init__(self, kernel_name: str, error_code: Optional[int], error_message: str):
        if error_code is not None:
            message = f"Kernel '{kernel_name}' failed with code {error_code}: {error_message}"
        else:
            message = f"Kernel '{kernel_name}' failed: {error_message}"
        super().__init__(message, {
            "kernel": kernel_name,
            "error_code": error_code,
            "error": error_message
        })


class KernelLaunchError(KernelError):
    """Raised when kernel launch fails."""
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def raise_or_warn(
    message: str,
    exception_class: type = BackendError,
    strict_mode: bool = False,
    log: Optional[logging.Logger] = None,
    **kwargs
) -> None:
    """
    Raise exception in strict mode, otherwise log warning.

    This pattern is used throughout backends for flexible error handling.

    Args:
        message: Error message
        exception_class: Exception class to raise/warn
        strict_mode: If True, raise exception; if False, log warning
        log: Optional logger instance
        **kwargs: Additional arguments for exception
    """
    if strict_mode:
        if kwargs:
            raise exception_class(message, **kwargs)
        else:
            raise exception_class(message)
    else:
        log_instance = log or logger
        log_instance.warning(f"{exception_class.__name__}: {message}")


__all__ = [
    # Base
    'BackendError',
    # Device
    'DeviceNotAvailableError',
    'DeviceError',
    # Memory
    'MemoryError',
    'OutOfMemoryError',
    'MemoryAllocationError',
    'MemoryPoolError',
    # Compilation
    'CompilationError',
    'KernelCompilationError',
    # Optimization
    'OptimizationError',
    'ModelOptimizationError',
    # Configuration
    'ConfigurationError',
    'InvalidArchitectureError',
    # Kernel
    'KernelError',
    'KernelLaunchError',
    # Utility
    'raise_or_warn',
]
