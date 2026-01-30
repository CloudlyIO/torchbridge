"""
Unified Error Handling Framework for TorchBridge

This module provides the complete exception hierarchy for the framework:
- TorchBridgeError: Base exception for all framework errors
- ValidationError: Configuration and input validation errors
- HardwareError: Hardware detection and operation errors
- OptimizationError: Model optimization failures
- DeploymentError: Export and serving errors
- MonitoringError: Metrics and health monitoring errors

All backend-specific exceptions (NVIDIA, AMD, TPU) inherit from
the base_exceptions module which inherits from TorchBridgeError.

Version: 0.3.11
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Base Exception
# =============================================================================

class TorchBridgeError(Exception):
    """
    Base exception for all TorchBridge errors.

    All framework exceptions should inherit from this class.
    Supports structured error details for debugging and logging.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        """
        Initialize TorchBridge error.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
            cause: Optional underlying exception
        """
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"({details_str})")
        if self.cause:
            parts.append(f"[caused by: {type(self.cause).__name__}: {self.cause}]")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(TorchBridgeError):
    """Base exception for validation failures."""
    pass


class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        config_name: str,
        parameter: str,
        value: Any,
        reason: str
    ):
        message = f"Invalid {config_name} configuration: '{parameter}' = {value!r} - {reason}"
        super().__init__(message, {
            "config": config_name,
            "parameter": parameter,
            "value": value,
            "reason": reason
        })


class InputValidationError(ValidationError):
    """Raised when input validation fails."""

    def __init__(
        self,
        input_name: str,
        expected: str,
        actual: str,
        reason: str = ""
    ):
        message = f"Invalid input '{input_name}': expected {expected}, got {actual}"
        if reason:
            message = f"{message}. {reason}"
        super().__init__(message, {
            "input": input_name,
            "expected": expected,
            "actual": actual,
            "reason": reason
        })


class ModelValidationError(ValidationError):
    """Raised when model validation fails."""

    def __init__(
        self,
        model_name: str,
        issues: list[str]
    ):
        issues_str = "; ".join(issues)
        message = f"Model '{model_name}' validation failed: {issues_str}"
        super().__init__(message, {
            "model": model_name,
            "issues": issues
        })


# =============================================================================
# Hardware Errors
# =============================================================================

class HardwareError(TorchBridgeError):
    """Base exception for hardware-related errors."""
    pass


class HardwareDetectionError(HardwareError):
    """Raised when hardware detection fails."""

    def __init__(self, hardware_type: str, reason: str):
        message = f"Failed to detect {hardware_type}: {reason}"
        super().__init__(message, {
            "hardware_type": hardware_type,
            "reason": reason
        })


class HardwareNotFoundError(HardwareError):
    """Raised when required hardware is not available."""

    def __init__(self, hardware_type: str, requirements: list[str] | None = None):
        message = f"Required hardware not found: {hardware_type}"
        if requirements:
            message = f"{message}. Requirements: {', '.join(requirements)}"
        super().__init__(message, {
            "hardware_type": hardware_type,
            "requirements": requirements
        })


class HardwareCapabilityError(HardwareError):
    """Raised when hardware lacks required capabilities."""

    def __init__(
        self,
        hardware_name: str,
        required_capability: str,
        available_capabilities: list[str]
    ):
        message = f"Hardware '{hardware_name}' lacks capability: {required_capability}"
        super().__init__(message, {
            "hardware": hardware_name,
            "required": required_capability,
            "available": available_capabilities
        })


# =============================================================================
# Optimization Errors
# =============================================================================

class OptimizationError(TorchBridgeError):
    """Base exception for optimization failures."""
    pass


class CompilationError(OptimizationError):
    """Raised when model compilation fails."""

    def __init__(self, compiler: str, model_name: str, error_message: str):
        message = f"{compiler} compilation failed for '{model_name}': {error_message}"
        super().__init__(message, {
            "compiler": compiler,
            "model": model_name,
            "error": error_message
        })


class FusionError(OptimizationError):
    """Raised when operator fusion fails."""

    def __init__(self, pattern: str, reason: str):
        message = f"Fusion pattern '{pattern}' failed: {reason}"
        super().__init__(message, {
            "pattern": pattern,
            "reason": reason
        })


class PrecisionError(OptimizationError):
    """Raised when precision conversion fails."""

    def __init__(
        self,
        source_precision: str,
        target_precision: str,
        reason: str
    ):
        message = f"Precision conversion {source_precision} -> {target_precision} failed: {reason}"
        super().__init__(message, {
            "source": source_precision,
            "target": target_precision,
            "reason": reason
        })


# =============================================================================
# Deployment Errors
# =============================================================================

class DeploymentError(TorchBridgeError):
    """Base exception for deployment failures."""
    pass


class ExportError(DeploymentError):
    """Raised when model export fails."""

    def __init__(self, format_name: str, model_name: str, reason: str):
        message = f"Export to {format_name} failed for '{model_name}': {reason}"
        super().__init__(message, {
            "format": format_name,
            "model": model_name,
            "reason": reason
        })


class ServingError(DeploymentError):
    """Raised when inference serving fails."""

    def __init__(self, server_type: str, operation: str, reason: str):
        message = f"{server_type} serving {operation} failed: {reason}"
        super().__init__(message, {
            "server": server_type,
            "operation": operation,
            "reason": reason
        })


class ContainerError(DeploymentError):
    """Raised when container operations fail."""

    def __init__(self, container_type: str, operation: str, reason: str):
        message = f"{container_type} container {operation} failed: {reason}"
        super().__init__(message, {
            "container": container_type,
            "operation": operation,
            "reason": reason
        })


# =============================================================================
# Monitoring Errors
# =============================================================================

class MonitoringError(TorchBridgeError):
    """Base exception for monitoring failures."""
    pass


class MetricsError(MonitoringError):
    """Raised when metrics collection/export fails."""

    def __init__(self, metric_name: str, operation: str, reason: str):
        message = f"Metrics '{metric_name}' {operation} failed: {reason}"
        super().__init__(message, {
            "metric": metric_name,
            "operation": operation,
            "reason": reason
        })


class HealthCheckError(MonitoringError):
    """Raised when health check fails."""

    def __init__(self, component: str, status: str, reason: str):
        message = f"Health check failed for '{component}': {status} - {reason}"
        super().__init__(message, {
            "component": component,
            "status": status,
            "reason": reason
        })


# =============================================================================
# Utility Functions
# =============================================================================

def raise_or_warn(
    exception_class: type,
    message: str,
    strict_mode: bool = False,
    log: logging.Logger | None = None,
    **kwargs
) -> None:
    """
    Raise exception in strict mode, otherwise log warning.

    This pattern allows flexible error handling based on configuration.

    Args:
        exception_class: Exception class to raise/warn
        message: Error message
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


def format_error_chain(exc: Exception, max_depth: int = 5) -> str:
    """
    Format an exception chain for logging.

    Args:
        exc: The exception to format
        max_depth: Maximum depth of cause chain to include

    Returns:
        Formatted error string with cause chain
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    current = exc
    depth = 0

    while hasattr(current, 'cause') and current.cause and depth < max_depth:
        current = current.cause
        parts.append(f"  Caused by: {type(current).__name__}: {current}")
        depth += 1

    if hasattr(exc, '__cause__') and exc.__cause__ and depth < max_depth:
        parts.append(f"  Python cause: {type(exc.__cause__).__name__}: {exc.__cause__}")

    return "\n".join(parts)


__all__ = [
    # Base
    'TorchBridgeError',
    # Validation
    'ValidationError',
    'ConfigValidationError',
    'InputValidationError',
    'ModelValidationError',
    # Hardware
    'HardwareError',
    'HardwareDetectionError',
    'HardwareNotFoundError',
    'HardwareCapabilityError',
    # Optimization
    'OptimizationError',
    'CompilationError',
    'FusionError',
    'PrecisionError',
    # Deployment
    'DeploymentError',
    'ExportError',
    'ServingError',
    'ContainerError',
    # Monitoring
    'MonitoringError',
    'MetricsError',
    'HealthCheckError',
    # Utilities
    'raise_or_warn',
    'format_error_chain',
]
