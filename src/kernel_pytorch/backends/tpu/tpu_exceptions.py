"""
TPU Backend Custom Exceptions

Defines a hierarchy of custom exceptions for TPU backend operations,
providing structured error handling and better debugging information.
"""


class TPUBackendError(Exception):
    """Base exception for all TPU backend errors."""
    pass


class TPUNotAvailableError(TPUBackendError):
    """Raised when TPU hardware or PyTorch/XLA is not available."""
    pass


class XLACompilationError(TPUBackendError):
    """Raised when XLA compilation fails."""
    pass


class XLACompilationTimeoutError(XLACompilationError):
    """Raised when XLA compilation exceeds timeout."""
    pass


class TPUMemoryError(TPUBackendError):
    """Base exception for TPU memory-related errors."""
    pass


class TPUOutOfMemoryError(TPUMemoryError):
    """Raised when TPU runs out of memory during allocation."""
    pass


class TPUMemoryPoolError(TPUMemoryError):
    """Raised when memory pool operations fail."""
    pass


class TPUCacheError(TPUBackendError):
    """Raised when cache operations fail."""
    pass


class TPUModelPreparationError(TPUBackendError):
    """Raised when model preparation for TPU fails."""
    pass


class TPUOptimizationError(TPUBackendError):
    """Raised when TPU-specific optimization fails."""
    pass


class TPUValidationError(TPUBackendError):
    """Raised when validation checks fail."""
    pass


class TPUDistributedError(TPUBackendError):
    """Raised when distributed TPU operations fail."""
    pass


class TPUCheckpointError(TPUBackendError):
    """Raised when model checkpoint save/load operations fail."""
    pass


class TPUConfigurationError(TPUBackendError):
    """Raised when TPU configuration is invalid."""
    pass


class TPUDeviceError(TPUBackendError):
    """Raised when TPU device operations fail."""
    pass


# Utility function for strict validation mode
def raise_or_warn(
    message: str,
    exception_class: type,
    strict_mode: bool = False,
    logger=None
):
    """
    Raise an exception or log a warning based on strict mode.

    Args:
        message: Error/warning message
        exception_class: Exception class to raise if in strict mode
        strict_mode: Whether to raise exceptions (True) or warnings (False)
        logger: Optional logger instance for warnings

    Raises:
        exception_class: If strict_mode is True
    """
    if strict_mode:
        raise exception_class(message)
    else:
        if logger:
            logger.warning(message)
        else:
            import warnings
            warnings.warn(message, RuntimeWarning)
