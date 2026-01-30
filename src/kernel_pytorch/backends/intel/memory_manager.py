"""
Intel XPU Memory Manager

Provides memory management utilities for Intel XPU devices,
following the BaseMemoryManager pattern.
"""

import logging
import time

import torch

from ..base_memory_manager import (
    BaseMemoryManager,
    BaseMemoryStats,
    MemoryAllocationInfo,
)
from .intel_exceptions import (
    XPUMemoryAllocationError,
    XPUOutOfMemoryError,
)
from .xpu_utilities import XPU_AVAILABLE

logger = logging.getLogger(__name__)


class IntelMemoryManager(BaseMemoryManager):
    """
    Memory manager for Intel XPU devices.

    Provides:
    - Memory allocation and deallocation tracking
    - Memory pool management for efficient tensor reuse
    - Memory statistics and monitoring
    - OOM protection and handling
    """

    def __init__(self, config=None, device_id: int = 0):
        """
        Initialize Intel XPU memory manager.

        Args:
            config: IntelConfig instance (optional)
            device_id: XPU device ID to manage
        """
        self._config = config
        self._device_id = device_id
        self._peak_allocated = 0

        # Initialize base class
        super().__init__(config)

        logger.debug(f"IntelMemoryManager initialized for XPU:{device_id}")

    def _get_device(self) -> torch.device:
        """Get the torch device for this memory manager."""
        if not XPU_AVAILABLE:
            logger.warning("XPU not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("xpu", self._device_id)

    def _get_optimal_alignment(self) -> int:
        """
        Get optimal memory alignment for Intel XPU.

        Intel XPUs work best with 64-byte alignment for vector operations,
        but 128-byte alignment for matrix operations.
        """
        return 64  # 64-byte alignment for general use

    def _get_total_memory_bytes(self) -> int:
        """Get total device memory in bytes."""
        if not XPU_AVAILABLE:
            return 0

        try:
            props = torch.xpu.get_device_properties(self._device_id)
            return props.total_memory
        except Exception as e:
            logger.warning(f"Failed to get total memory: {e}")
            return 0

    def _get_allocated_memory_bytes(self) -> int:
        """Get currently allocated memory in bytes."""
        if not XPU_AVAILABLE:
            return 0

        try:
            return torch.xpu.memory_allocated(self._device_id)
        except Exception as e:
            logger.warning(f"Failed to get allocated memory: {e}")
            return 0

    def _get_reserved_memory_bytes(self) -> int:
        """Get reserved (cached) memory in bytes."""
        if not XPU_AVAILABLE:
            return 0

        try:
            return torch.xpu.memory_reserved(self._device_id)
        except Exception as e:
            logger.warning(f"Failed to get reserved memory: {e}")
            return 0

    def _device_synchronize(self) -> None:
        """Synchronize the XPU device."""
        if not XPU_AVAILABLE:
            return

        try:
            torch.xpu.synchronize(self._device_id)
        except Exception as e:
            logger.warning(f"XPU synchronization failed: {e}")

    def _empty_device_cache(self) -> None:
        """Empty the XPU memory cache."""
        if not XPU_AVAILABLE:
            return

        try:
            torch.xpu.empty_cache()
            logger.debug("XPU memory cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear XPU cache: {e}")

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        purpose: str = "unknown",
        use_pool: bool = True
    ) -> torch.Tensor:
        """
        Allocate a tensor on the XPU device.

        Args:
            shape: Tensor shape
            dtype: Data type
            purpose: Description of tensor usage
            use_pool: Whether to try memory pool first

        Returns:
            Allocated tensor

        Raises:
            XPUOutOfMemoryError: If allocation fails due to OOM
            XPUMemoryAllocationError: If allocation fails for other reasons
        """
        # Check if we should use the memory pool
        if use_pool and self.pool_enabled:
            pooled = self._try_get_from_pool(shape, dtype)
            if pooled is not None:
                return pooled

        # Calculate required size
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        size_bytes = num_elements * element_size

        # Check memory availability
        if XPU_AVAILABLE:
            available = self._get_total_memory_bytes() - self._get_allocated_memory_bytes()
            if size_bytes > available * 0.95:  # Leave 5% buffer
                raise XPUOutOfMemoryError(
                    requested_bytes=size_bytes,
                    available_bytes=available,
                    device_id=self._device_id,
                    details={"shape": shape, "dtype": str(dtype), "purpose": purpose}
                )

        try:
            device = self._get_device()
            tensor = torch.empty(shape, dtype=dtype, device=device)

            # Track allocation
            self._track_allocation(shape, dtype, size_bytes, purpose)

            return tensor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise XPUOutOfMemoryError(
                    requested_bytes=size_bytes,
                    available_bytes=self._get_total_memory_bytes() - self._get_allocated_memory_bytes(),
                    device_id=self._device_id,
                    details={"shape": shape, "dtype": str(dtype), "error": str(e)}
                ) from e
            raise XPUMemoryAllocationError(
                f"Failed to allocate tensor: {e}",
                size_bytes=size_bytes,
                details={"shape": shape, "dtype": str(dtype)}
            ) from e

    def _track_allocation(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        size_bytes: int,
        purpose: str
    ):
        """Track a memory allocation."""
        info = MemoryAllocationInfo(
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            timestamp=time.time(),
            purpose=purpose
        )
        self._allocation_history.append(info)

        # Update peak tracking
        current_allocated = self._get_allocated_memory_bytes()
        if current_allocated > self._peak_allocated:
            self._peak_allocated = current_allocated

    def get_memory_stats(self) -> BaseMemoryStats:
        """Get current memory statistics."""
        allocated = self._get_allocated_memory_bytes()
        reserved = self._get_reserved_memory_bytes()
        total = self._get_total_memory_bytes()

        return BaseMemoryStats(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            total_bytes=total,
            free_bytes=total - allocated,
            peak_allocated_bytes=max(self._peak_allocated, allocated),
            num_allocations=len(self._allocation_history),
            pool_count=len(self._memory_pools)
        )

    def get_memory_summary(self) -> str:
        """Get a human-readable memory summary."""
        stats = self.get_memory_stats()
        return (
            f"Intel XPU:{self._device_id} Memory:\n"
            f"  Allocated: {stats.allocated_mb:.1f} MB\n"
            f"  Reserved:  {stats.reserved_mb:.1f} MB\n"
            f"  Total:     {stats.total_mb:.1f} MB\n"
            f"  Free:      {stats.free_mb:.1f} MB\n"
            f"  Peak:      {stats.peak_allocated_bytes / (1024**2):.1f} MB\n"
            f"  Utilization: {stats.utilization:.1%}"
        )

    def optimize_for_inference(self):
        """Optimize memory layout for inference workloads."""
        if not XPU_AVAILABLE:
            return

        # Clear caches to free up memory
        self._empty_device_cache()

        # Set memory fraction for inference (can use more memory)
        if self._config and hasattr(self._config, 'max_memory_fraction'):
            logger.debug(f"Memory configured for inference with {self._config.max_memory_fraction:.0%} utilization")

    def optimize_for_training(self):
        """Optimize memory layout for training workloads."""
        if not XPU_AVAILABLE:
            return

        # Clear caches
        self._empty_device_cache()

        # Training needs memory headroom for gradients
        logger.debug("Memory configured for training with gradient headroom")


__all__ = [
    'IntelMemoryManager',
]
