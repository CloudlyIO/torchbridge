"""
NVIDIA Backend Custom Exceptions

Custom exception hierarchy for NVIDIA backend error handling.
"""


class NVIDIABackendError(Exception):
    """Base exception for all NVIDIA backend errors."""
    pass


class CUDANotAvailableError(NVIDIABackendError):
    """CUDA is not available on this system."""
    pass


class CUDADeviceError(NVIDIABackendError):
    """Error related to CUDA device operations."""
    pass


class FP8CompilationError(NVIDIABackendError):
    """Error during FP8 model compilation."""
    pass


class FlashAttentionError(NVIDIABackendError):
    """Error in FlashAttention operations."""
    pass


class MemoryAllocationError(NVIDIABackendError):
    """Error allocating GPU memory."""
    pass


class OutOfMemoryError(MemoryAllocationError):
    """GPU out of memory error."""
    pass


class InvalidComputeCapabilityError(NVIDIABackendError):
    """Invalid or unsupported compute capability."""
    pass


class KernelLaunchError(NVIDIABackendError):
    """Error launching CUDA kernel."""
    pass


class ModelOptimizationError(NVIDIABackendError):
    """Error during model optimization."""
    pass


class InvalidArchitectureError(NVIDIABackendError):
    """Invalid or unsupported NVIDIA GPU architecture."""
    pass


class ConfigurationError(NVIDIABackendError):
    """Error in NVIDIA backend configuration."""
    pass
