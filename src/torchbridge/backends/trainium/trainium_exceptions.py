# SPDX-License-Identifier: Apache-2.0
"""
Trainium Backend Custom Exceptions

Defines a hierarchy of custom exceptions for Trainium backend operations,
inheriting from the shared base_exceptions module.

"""

import logging
from typing import Any

from torchbridge.backends.base_exceptions import (
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


class TrainiumBackendError(BackendError):
    """Base exception for all Trainium backend errors."""

    pass


class TrainiumNotAvailableError(DeviceNotAvailableError, TrainiumBackendError):
    """Raised when Trainium hardware or Neuron SDK is not available."""

    def __init__(self, message: str = "Trainium/Neuron SDK not available"):
        DeviceNotAvailableError.__init__(self, "Trainium", message)


class NeuronCompilationError(CompilationError, TrainiumBackendError):
    """Raised when Neuron compilation fails."""

    def __init__(self, error_message: str):
        CompilationError.__init__(self, "Neuron", error_message)


class NeuronCompilationTimeoutError(NeuronCompilationError):
    """Raised when Neuron compilation exceeds timeout."""

    def __init__(self, timeout_seconds: float, error_message: str = ""):
        message = f"timeout after {timeout_seconds}s"
        if error_message:
            message = f"{message}: {error_message}"
        super().__init__(message)


class TrainiumMemoryError(MemoryError, TrainiumBackendError):
    """Base exception for Trainium memory-related errors."""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None):
        MemoryError.__init__(self, message, details)


class TrainiumOutOfMemoryError(TrainiumMemoryError):
    """Raised when Trainium runs out of HBM during allocation."""

    def __init__(
        self, required_bytes: int | None = None, available_bytes: int | None = None
    ):
        if required_bytes is not None and available_bytes is not None:
            message = f"Out of Trainium memory: required {required_bytes / 1e6:.1f}MB, available {available_bytes / 1e6:.1f}MB"
        else:
            message = "Out of Trainium memory"
        super().__init__(
            message,
            {"required_bytes": required_bytes, "available_bytes": available_bytes},
        )


class TrainiumMemoryPoolError(MemoryPoolError):
    """Raised when memory pool operations fail."""

    def __init__(self, pool_id: str, operation: str, error_message: str):
        super().__init__(pool_id, operation, error_message)


class TrainiumCacheError(TrainiumBackendError):
    """Raised when cache operations fail."""

    def __init__(self, operation: str, error_message: str):
        message = f"Trainium cache {operation} failed: {error_message}"
        super().__init__(message, {"operation": operation, "error": error_message})


class TrainiumModelPreparationError(TrainiumBackendError):
    """Raised when model preparation for Trainium fails."""

    def __init__(self, model_name: str, error_message: str):
        message = (
            f"Trainium model preparation for '{model_name}' failed: {error_message}"
        )
        super().__init__(message, {"model": model_name, "error": error_message})


class TrainiumOptimizationError(OptimizationError):
    """Raised when Trainium-specific optimization fails."""

    def __init__(self, optimization_type: str, error_message: str):
        super().__init__(optimization_type, error_message)


class TrainiumValidationError(TrainiumBackendError):
    """Raised when validation checks fail."""

    def __init__(self, validation_type: str, error_message: str):
        message = f"Trainium validation ({validation_type}) failed: {error_message}"
        super().__init__(message, {"type": validation_type, "error": error_message})


class TrainiumConfigurationError(ConfigurationError):
    """Raised when Trainium configuration is invalid."""

    def __init__(self, parameter: str, value: Any, reason: str):
        super().__init__(parameter, value, reason)


class TrainiumDeviceError(DeviceError):
    """Raised when Trainium device operations fail."""

    def __init__(self, device_id: int, operation: str, error_message: str):
        super().__init__(device_id, operation, error_message)


__all__ = [
    "TrainiumBackendError",
    "TrainiumNotAvailableError",
    "NeuronCompilationError",
    "NeuronCompilationTimeoutError",
    "TrainiumMemoryError",
    "TrainiumOutOfMemoryError",
    "TrainiumMemoryPoolError",
    "TrainiumCacheError",
    "TrainiumModelPreparationError",
    "TrainiumOptimizationError",
    "TrainiumValidationError",
    "TrainiumConfigurationError",
    "TrainiumDeviceError",
    "raise_or_warn",  # Re-export from base
]
