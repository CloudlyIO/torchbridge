"""
Base Backend for All Hardware Implementations

This module provides the abstract base class for all backend implementations,
defining the common interface and shared functionality.

Backends (NVIDIA, AMD, TPU, Intel) inherit from this base and implement
device-specific optimizations while maintaining a consistent API.

Version: 0.4.8
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import torch
import torch.nn as nn

from .base_exceptions import BackendError
from .base_memory_manager import BaseMemoryManager, BaseMemoryStats

logger = logging.getLogger(__name__)

# Type variable for config types
ConfigT = TypeVar('ConfigT')


class OptimizationLevel(Enum):
    """
    Standardized optimization levels across all backends.

    O0: No optimizations (debug mode)
    O1: Conservative optimizations (safe, minimal impact)
    O2: Balanced optimizations (performance + stability)
    O3: Aggressive optimizations (maximum performance)
    """
    O0 = "O0"
    O1 = "O1"
    O2 = "O2"
    O3 = "O3"

    # Aliases for compatibility
    DEBUG = "O0"
    CONSERVATIVE = "O1"
    BALANCED = "O2"
    AGGRESSIVE = "O3"

    @classmethod
    def from_string(cls, level: str) -> "OptimizationLevel":
        """
        Convert string to OptimizationLevel.

        Args:
            level: String like "O0", "O1", "conservative", "balanced", etc.

        Returns:
            OptimizationLevel enum value
        """
        level_upper = level.upper()

        # Direct match
        if level_upper in ("O0", "DEBUG"):
            return cls.O0
        elif level_upper in ("O1", "CONSERVATIVE"):
            return cls.O1
        elif level_upper in ("O2", "BALANCED"):
            return cls.O2
        elif level_upper in ("O3", "AGGRESSIVE"):
            return cls.O3
        else:
            # Default to balanced
            logger.warning(f"Unknown optimization level '{level}', defaulting to O2 (balanced)")
            return cls.O2


@dataclass
class DeviceInfo:
    """
    Standardized device information structure.

    All backends return this structure from get_device_info().
    """
    backend: str  # "nvidia", "amd", "tpu", "intel", "cpu"
    device_type: str  # Device string (e.g., "cuda:0", "xpu:0", "xla:0")
    device_id: int
    device_name: str
    compute_capability: str | None = None  # Architecture-specific version
    total_memory_bytes: int = 0
    driver_version: str | None = None
    is_available: bool = True
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory_bytes / (1024 ** 3)

    @property
    def total_memory_mb(self) -> float:
        """Total memory in MB."""
        return self.total_memory_bytes / (1024 ** 2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backend': self.backend,
            'device_type': self.device_type,
            'device_id': self.device_id,
            'device_name': self.device_name,
            'compute_capability': self.compute_capability,
            'total_memory_gb': self.total_memory_gb,
            'driver_version': self.driver_version,
            'is_available': self.is_available,
            'properties': self.properties
        }


@dataclass
class OptimizationResult:
    """
    Standardized result from optimization operations.

    All backends return this structure from optimize() methods.
    """
    success: bool
    model: nn.Module
    level: OptimizationLevel
    optimizations_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure level is OptimizationLevel enum
        if isinstance(self.level, str):
            self.level = OptimizationLevel.from_string(self.level)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'level': self.level.value,
            'optimizations_applied': self.optimizations_applied,
            'warnings': self.warnings,
            'errors': self.errors,
            'metrics': self.metrics
        }


class BaseBackend(ABC):
    """
    Abstract base class for all hardware backends.

    This class defines the common interface that all backend implementations
    (NVIDIA, AMD, TPU, Intel) must implement, while providing shared functionality.

    Subclasses must implement:
    - _setup_environment() -> None
    - _check_availability() -> bool
    - _get_device() -> torch.device
    - _get_device_info(device_id: int) -> DeviceInfo
    - prepare_model(model: nn.Module) -> nn.Module
    - optimize_for_inference(model: nn.Module, ...) -> nn.Module
    - optimize_for_training(model: nn.Module, ...) -> nn.Module

    Optional overrides for device-specific behavior:
    - synchronize()
    - empty_cache()
    - get_memory_stats()
    """

    # Backend name - should be overridden by subclasses
    BACKEND_NAME: str = "base"

    def __init__(self, config: Any = None):
        """
        Initialize the backend.

        Args:
            config: Backend-specific configuration object
        """
        self.config = config
        self._initialized = False
        self._device: torch.device | None = None
        self._memory_manager: BaseMemoryManager | None = None
        self._device_cache: dict[int, DeviceInfo] = {}

        # Setup environment
        try:
            self._setup_environment()
            self._initialized = True
        except Exception as e:
            logger.warning(
                f"{self.__class__.__name__} initialization failed: {e}. "
                "Falling back to CPU."
            )
            self._device = torch.device('cpu')
            self._initialized = False

        logger.debug(
            "%s initialized: device=%s, available=%s",
            self.__class__.__name__,
            self._device,
            self._initialized
        )

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _setup_environment(self) -> None:
        """
        Set up the backend environment.

        This method should:
        - Initialize device-specific runtime (CUDA, ROCm, XLA, etc.)
        - Set environment variables if needed
        - Initialize the device
        - Set self._device to the appropriate device
        """
        pass

    @abstractmethod
    def _check_availability(self) -> bool:
        """
        Check if the backend is available.

        Returns:
            True if backend is available, False otherwise
        """
        pass

    @abstractmethod
    def _get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """
        Get information about a specific device.

        Args:
            device_id: Device index

        Returns:
            DeviceInfo with device details
        """
        pass

    @abstractmethod
    def prepare_model(
        self,
        model: nn.Module,
        optimization_level: str | OptimizationLevel | None = None
    ) -> nn.Module:
        """
        Prepare a model for this backend.

        This method should:
        - Move model to the appropriate device
        - Apply backend-specific optimizations
        - Configure memory layout

        Args:
            model: PyTorch model
            optimization_level: Optional optimization level

        Returns:
            Prepared model on this backend's device
        """
        pass

    @abstractmethod
    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module:
        """
        Optimize a model for inference.

        Args:
            model: PyTorch model
            sample_input: Optional sample input for tracing
            dtype: Optional dtype for precision

        Returns:
            Inference-optimized model
        """
        pass

    @abstractmethod
    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module | tuple[nn.Module, torch.optim.Optimizer]:
        """
        Optimize a model for training.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer to optimize along with model
            dtype: Optional dtype for precision

        Returns:
            Training-optimized model, or tuple of (model, optimizer)
        """
        pass

    # =========================================================================
    # Common implementations
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Get the primary device for this backend."""
        if self._device is None:
            return torch.device('cpu')
        return self._device

    @property
    def is_available(self) -> bool:
        """Check if the backend is available and initialized."""
        return self._initialized and self._check_availability()

    @property
    def memory_manager(self) -> BaseMemoryManager | None:
        """Get the memory manager for this backend."""
        return self._memory_manager

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """
        Get device information with caching.

        Args:
            device_id: Device index

        Returns:
            DeviceInfo with device details
        """
        if device_id not in self._device_cache:
            self._device_cache[device_id] = self._get_device_info(device_id)
        return self._device_cache[device_id]

    def get_all_devices(self) -> list[DeviceInfo]:
        """
        Get information about all available devices.

        Returns:
            List of DeviceInfo for all devices
        """
        devices = []
        device_count = self.device_count
        for i in range(device_count):
            devices.append(self.get_device_info(i))
        return devices

    @property
    def device_count(self) -> int:
        """
        Get the number of available devices.

        Override in subclasses for actual device count.
        """
        return 1 if self.is_available else 0

    def set_device(self, device_id: int) -> None:
        """
        Set the active device.

        Args:
            device_id: Device index to set as active
        """
        if device_id >= self.device_count:
            raise BackendError(
                f"Device {device_id} not available. "
                f"Available devices: 0-{self.device_count - 1}"
            )
        # Subclasses should override to set actual device
        logger.info(f"Setting device to {device_id}")

    def synchronize(self) -> None:  # noqa: B027
        """
        Synchronize all pending operations.

        Default implementation does nothing (CPU doesn't need sync).
        Subclasses should override for actual synchronization.
        """
        pass

    def empty_cache(self) -> None:
        """
        Empty the memory cache.

        Default implementation does nothing.
        Subclasses should override for actual cache clearing.
        """
        if self._memory_manager:
            self._memory_manager.empty_cache()

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        if self._memory_manager:
            stats = self._memory_manager.get_memory_stats()
            if isinstance(stats, BaseMemoryStats):
                return stats.to_dict()
            return stats

        # Default CPU-based stats
        return {
            'allocated_mb': 0,
            'reserved_mb': 0,
            'total_mb': 0,
            'free_mb': 0,
            'device': str(self.device),
            'backend': self.BACKEND_NAME
        }

    def get_memory_summary(self) -> str:
        """
        Get a human-readable memory summary.

        Returns:
            Formatted string with memory information
        """
        stats = self.get_memory_stats()
        lines = [
            f"=== {self.BACKEND_NAME.upper()} Memory Summary ===",
            f"Device: {self.device}",
        ]

        if 'allocated_mb' in stats:
            lines.append(f"Allocated: {stats.get('allocated_mb', 0):.2f} MB")
        if 'reserved_mb' in stats:
            lines.append(f"Reserved: {stats.get('reserved_mb', 0):.2f} MB")
        if 'total_mb' in stats:
            lines.append(f"Total: {stats.get('total_mb', 0):.2f} MB")
        if 'free_mb' in stats:
            lines.append(f"Free: {stats.get('free_mb', 0):.2f} MB")

        return "\n".join(lines)

    def to_device(self, tensor_or_model: torch.Tensor | nn.Module) -> torch.Tensor | nn.Module:
        """
        Move tensor or model to this backend's device.

        Args:
            tensor_or_model: Tensor or model to move

        Returns:
            Tensor or model on this backend's device
        """
        return tensor_or_model.to(self.device)

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        pool_id: str | None = None
    ) -> torch.Tensor:
        """
        Allocate a tensor on this backend's device.

        Args:
            shape: Tensor shape
            dtype: Data type
            requires_grad: Whether to track gradients
            pool_id: Optional pool ID for memory pooling

        Returns:
            Allocated tensor
        """
        if self._memory_manager:
            return self._memory_manager.allocate_tensor(
                shape=shape,
                dtype=dtype,
                requires_grad=requires_grad,
                pool_id=pool_id
            )

        return torch.zeros(
            shape,
            dtype=dtype,
            device=self.device,
            requires_grad=requires_grad
        )

    def cleanup(self) -> None:
        """
        Clean up backend resources.
        """
        logger.info(f"Cleaning up {self.__class__.__name__}...")

        if self._memory_manager:
            self._memory_manager.cleanup()

        self.empty_cache()
        self._device_cache.clear()

        logger.info("Cleanup complete")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"available={self.is_available}, "
            f"initialized={self._initialized})"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


class CPUBackend(BaseBackend):
    """
    CPU backend implementation.

    This serves as the fallback backend when no accelerator is available.
    """

    BACKEND_NAME = "cpu"

    def _setup_environment(self) -> None:
        """Set up CPU environment."""
        self._device = torch.device('cpu')

    def _check_availability(self) -> bool:
        """CPU is always available."""
        return True

    def _get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Get CPU device info."""
        import platform

        return DeviceInfo(
            backend="cpu",
            device_type="cpu",
            device_id=0,
            device_name=platform.processor() or "CPU",
            compute_capability=None,
            total_memory_bytes=0,  # Could use psutil if available
            driver_version=None,
            is_available=True,
            properties={
                'platform': platform.platform(),
                'python_version': platform.python_version()
            }
        )

    def prepare_model(
        self,
        model: nn.Module,
        optimization_level: str | OptimizationLevel | None = None
    ) -> nn.Module:
        """Prepare model for CPU."""
        return model.to('cpu')

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module:
        """Optimize for CPU inference."""
        model = model.eval()

        # Apply torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        return model

    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module | tuple[nn.Module, torch.optim.Optimizer]:
        """Optimize for CPU training."""
        model = model.train()

        if optimizer:
            return model, optimizer
        return model


__all__ = [
    'BaseBackend',
    'CPUBackend',
    'OptimizationLevel',
    'DeviceInfo',
    'OptimizationResult',
]
