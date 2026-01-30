"""
TPU Backend Custom Exceptions

Defines a hierarchy of custom exceptions for TPU backend operations,
inheriting from the shared base_exceptions module.

Version: 0.3.7
"""

import logging
from typing import Any

from kernel_pytorch.backends.base_exceptions import (
    BackendError,
    CompilationError,
    ConfigurationError,
    DeviceError,
    DeviceNotAvailableError,
    MemoryError,
    MemoryPoolError,
    OptimizationError,
    raise_or_warn,
)

logger = logging.getLogger(__name__)


class TPUBackendError(BackendError):
    """Base exception for all TPU backend errors."""
    pass


class TPUNotAvailableError(DeviceNotAvailableError, TPUBackendError):
    """Raised when TPU hardware or PyTorch/XLA is not available."""

    def __init__(self, message: str = "TPU/PyTorch-XLA not available"):
        DeviceNotAvailableError.__init__(self, "TPU", message)


class XLACompilationError(CompilationError, TPUBackendError):
    """Raised when XLA compilation fails."""

    def __init__(self, error_message: str):
        CompilationError.__init__(self, "XLA", error_message)


class XLACompilationTimeoutError(XLACompilationError):
    """Raised when XLA compilation exceeds timeout."""

    def __init__(self, timeout_seconds: float, error_message: str = ""):
        message = f"timeout after {timeout_seconds}s"
        if error_message:
            message = f"{message}: {error_message}"
        super().__init__(message)


class TPUMemoryError(MemoryError, TPUBackendError):
    """Base exception for TPU memory-related errors."""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None):
        MemoryError.__init__(self, message, details)


class TPUOutOfMemoryError(TPUMemoryError):
    """Raised when TPU runs out of memory during allocation."""

    def __init__(self, required_bytes: int | None = None, available_bytes: int | None = None):
        if required_bytes is not None and available_bytes is not None:
            message = f"Out of TPU memory: required {required_bytes/1e6:.1f}MB, available {available_bytes/1e6:.1f}MB"
        else:
            message = "Out of TPU memory"
        super().__init__(message, {
            "required_bytes": required_bytes,
            "available_bytes": available_bytes
        })


class TPUMemoryPoolError(MemoryPoolError):
    """Raised when memory pool operations fail."""

    def __init__(self, pool_id: str, operation: str, error_message: str):
        super().__init__(pool_id, operation, error_message)


class TPUCacheError(TPUBackendError):
    """Raised when cache operations fail."""

    def __init__(self, operation: str, error_message: str):
        message = f"TPU cache {operation} failed: {error_message}"
        super().__init__(message, {"operation": operation, "error": error_message})


class TPUModelPreparationError(TPUBackendError):
    """Raised when model preparation for TPU fails."""

    def __init__(self, model_name: str, error_message: str):
        message = f"TPU model preparation for '{model_name}' failed: {error_message}"
        super().__init__(message, {"model": model_name, "error": error_message})


class TPUOptimizationError(OptimizationError):
    """Raised when TPU-specific optimization fails."""

    def __init__(self, optimization_type: str, error_message: str):
        super().__init__(optimization_type, error_message)


class TPUValidationError(TPUBackendError):
    """Raised when validation checks fail."""

    def __init__(self, validation_type: str, error_message: str):
        message = f"TPU validation ({validation_type}) failed: {error_message}"
        super().__init__(message, {"type": validation_type, "error": error_message})


class TPUDistributedError(TPUBackendError):
    """Raised when distributed TPU operations fail."""

    def __init__(self, operation: str, error_message: str):
        message = f"TPU distributed {operation} failed: {error_message}"
        super().__init__(message, {"operation": operation, "error": error_message})


class TPUCheckpointError(TPUBackendError):
    """Raised when model checkpoint save/load operations fail."""

    def __init__(self, operation: str, checkpoint_path: str, error_message: str):
        message = f"TPU checkpoint {operation} failed for '{checkpoint_path}': {error_message}"
        super().__init__(message, {
            "operation": operation,
            "checkpoint_path": checkpoint_path,
            "error": error_message
        })


class TPUConfigurationError(ConfigurationError):
    """Raised when TPU configuration is invalid."""

    def __init__(self, parameter: str, value: Any, reason: str):
        super().__init__(parameter, value, reason)


class TPUDeviceError(DeviceError):
    """Raised when TPU device operations fail."""

    def __init__(self, device_id: int, operation: str, error_message: str):
        super().__init__(device_id, operation, error_message)


__all__ = [
    "TPUBackendError",
    "TPUNotAvailableError",
    "XLACompilationError",
    "XLACompilationTimeoutError",
    "TPUMemoryError",
    "TPUOutOfMemoryError",
    "TPUMemoryPoolError",
    "TPUCacheError",
    "TPUModelPreparationError",
    "TPUOptimizationError",
    "TPUValidationError",
    "TPUDistributedError",
    "TPUCheckpointError",
    "TPUConfigurationError",
    "TPUDeviceError",
    "raise_or_warn",  # Re-export from base
]
