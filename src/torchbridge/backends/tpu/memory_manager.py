"""
TPU Memory Manager

Manages TPU memory allocation, optimization, and monitoring
for efficient TPU model execution.

Inherits from BaseMemoryManager for shared functionality.

Version: 0.3.7
"""

import gc
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from torchbridge.backends.base_memory_manager import (
    BaseMemoryManager,
)
from torchbridge.core.config import TPUConfig, TPUVersion

from . import xla_compat

logger = logging.getLogger(__name__)


@dataclass
class TPUMemoryStats:
    """TPU-specific memory statistics."""
    allocated_memory: int
    cached_memory: int
    reserved_memory: int
    available_memory: int
    memory_fraction: float
    active_tensors: int
    peak_memory: int


class TPUMemoryManager(BaseMemoryManager):
    """
    TPU memory manager for optimal memory usage.

    Provides memory allocation, monitoring, and optimization
    specifically designed for TPU hardware characteristics.

    Inherits from BaseMemoryManager for shared functionality.
    """

    def __init__(self, config: TPUConfig):
        """
        Initialize TPU memory manager.

        Args:
            config: TPU configuration
        """
        self._tpu_config = config

        # TPU-specific memory pool tracking
        self._tpu_memory_pool: dict[str, dict] = {}
        self._tpu_peak_memory = 0

        # Initialize base class
        super().__init__(config)

        # Initialize TPU-specific memory management
        self._setup_memory_management()

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _get_device(self) -> torch.device:
        """Get the XLA/TPU device."""
        try:
            return xla_compat.get_xla_device()
        except Exception:
            warnings.warn("PyTorch/XLA not available. Using CPU fallback.", stacklevel=2)
            return torch.device("cpu")

    def _get_optimal_alignment(self) -> int:
        """
        Get optimal tensor dimension alignment for TPU.

        TPU matrix units work best with dimensions divisible by 8.
        """
        return 8

    def _get_total_memory_bytes(self) -> int:
        """Get total TPU memory in bytes."""
        tpu_memory_gb = self._get_tpu_memory_gb()
        return int(tpu_memory_gb * 1e9)

    def _get_allocated_memory_bytes(self) -> int:
        """
        Get currently allocated memory in bytes.

        Note: XLA doesn't expose memory stats the same way as CUDA,
        so we estimate from allocation history.
        """
        retention_seconds = self._tpu_config.allocation_history_retention_seconds
        return sum(
            alloc.size_bytes for alloc in self._allocation_history
            if time.time() - alloc.timestamp < retention_seconds
        )

    def _get_reserved_memory_bytes(self) -> int:
        """Get reserved (cached) memory in bytes."""
        allocated = self._get_allocated_memory_bytes()
        cached = sum(
            len(pool.get('tensors', [])) * pool['tensors'][0].numel() * pool['tensors'][0].element_size()
            for pool in self._tpu_memory_pool.values()
            if pool.get('tensors')
        )
        return allocated + cached

    def _device_synchronize(self) -> None:
        """Synchronize XLA operations."""
        try:
            xla_compat.sync()
        except Exception:
            pass

    def _empty_device_cache(self) -> None:
        """Empty device cache."""
        try:
            xla_compat.sync()
            gc.collect()
        except Exception:
            pass

    # =========================================================================
    # TPU-specific methods
    # =========================================================================

    def _setup_memory_management(self) -> None:
        """Set up TPU memory management."""
        try:
            import torch_xla  # noqa: F401

            # Set memory fraction
            self._apply_memory_fraction()

            logger.info(
                "TPU Memory Manager initialized: memory_fraction=%.2f, device=%s",
                self._tpu_config.memory_fraction,
                self._device
            )

        except ImportError:
            warnings.warn("PyTorch/XLA not available. Memory manager will use CPU fallback.", stacklevel=2)

    def _apply_memory_fraction(self) -> None:
        """Apply memory fraction limits for TPU."""
        import os

        # Set TPU memory fraction
        if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self._tpu_config.memory_fraction)

        # Enable memory growth for TPUs
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    def _get_tpu_memory_gb(self) -> float:
        """Get TPU memory capacity in GB."""
        memory_map = {
            TPUVersion.V4: 32.0,    # 32GB HBM (verified)
            TPUVersion.V5E: 16.0,   # 16GB HBM (verified)
            TPUVersion.V5P: 95.0,   # 95GB HBM (verified)
            TPUVersion.V6E: self._tpu_config.v6e_memory_gb or 32.0,
            TPUVersion.V7: self._tpu_config.v7_memory_gb or 128.0,
        }
        return memory_map.get(self._tpu_config.version, 32.0)

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        pool_id: str | None = None,
        purpose: str = "unknown"
    ) -> torch.Tensor:
        """
        Allocate tensor with optimal TPU memory placement.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            requires_grad: Whether tensor requires gradients
            pool_id: Optional pool ID for memory pooling
            purpose: Purpose description for tracking

        Returns:
            Allocated tensor on TPU
        """
        # Use base class allocation
        return super().allocate_tensor(shape, dtype, requires_grad, pool_id, purpose)

    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor memory layout for TPU.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with optimized layout
        """
        original_shape = tensor.shape
        optimal_div = self._get_optimal_alignment()

        # For 2D tensors, ensure dimensions are multiples of 8
        if len(original_shape) == 2:
            height, width = original_shape
            if height % optimal_div != 0 or width % optimal_div != 0:
                new_height = ((height + optimal_div - 1) // optimal_div) * optimal_div
                new_width = ((width + optimal_div - 1) // optimal_div) * optimal_div

                if new_height != height or new_width != width:
                    padded_tensor = torch.zeros(
                        (new_height, new_width),
                        dtype=tensor.dtype,
                        device=tensor.device
                    )
                    padded_tensor[:height, :width] = tensor
                    return padded_tensor

        # For 4D tensors (NCHW), optimize channel dimensions
        elif len(original_shape) == 4:
            n, c, h, w = original_shape
            if c % optimal_div != 0:
                new_c = ((c + optimal_div - 1) // optimal_div) * optimal_div
                padded_tensor = torch.zeros(
                    (n, new_c, h, w),
                    dtype=tensor.dtype,
                    device=tensor.device
                )
                padded_tensor[:, :c, :, :] = tensor
                return padded_tensor

        return tensor

    def create_memory_pool(
        self,
        pool_size: int,
        tensor_size: tuple[int, ...],
        dtype: torch.dtype = torch.float32
    ) -> str:
        """
        Create a memory pool for efficient tensor reuse.

        Args:
            pool_size: Number of tensors in pool
            tensor_size: Size of each tensor
            dtype: Tensor data type

        Returns:
            Pool identifier
        """
        pool_id = f"tpu_pool_{len(self._tpu_memory_pool)}_{int(time.time())}"

        # Pre-allocate tensors
        pool_tensors = []
        for _ in range(pool_size):
            tensor = self.allocate_tensor(tensor_size, dtype)
            pool_tensors.append(tensor)

        self._tpu_memory_pool[pool_id] = {
            'tensors': pool_tensors,
            'available': list(range(pool_size)),
            'in_use': [],
            'created_at': time.time()
        }

        logger.debug("Created TPU memory pool '%s': size=%d, tensor_shape=%s",
                     pool_id, pool_size, tensor_size)
        return pool_id

    def get_tensor_from_pool(self, pool_id: str) -> torch.Tensor | None:
        """
        Get tensor from memory pool.

        Args:
            pool_id: Pool identifier

        Returns:
            Tensor from pool or None if pool is empty
        """
        if pool_id not in self._tpu_memory_pool:
            return None

        pool = self._tpu_memory_pool[pool_id]
        if not pool['available']:
            return None

        tensor_idx = pool['available'].pop(0)
        pool['in_use'].append(tensor_idx)

        tensor = pool['tensors'][tensor_idx]
        tensor.zero_()
        return tensor

    def return_tensor_to_pool(self, pool_id: str, tensor: torch.Tensor) -> bool:
        """
        Return tensor to memory pool.

        Args:
            pool_id: Pool identifier
            tensor: Tensor to return

        Returns:
            True if successful, False otherwise
        """
        if pool_id not in self._tpu_memory_pool:
            return False

        pool = self._tpu_memory_pool[pool_id]

        tensor_idx = None
        for idx, pool_tensor in enumerate(pool['tensors']):
            if torch.equal(tensor, pool_tensor):
                tensor_idx = idx
                break

        if tensor_idx is None or tensor_idx not in pool['in_use']:
            return False

        pool['in_use'].remove(tensor_idx)
        pool['available'].append(tensor_idx)
        return True

    def get_memory_stats(self) -> TPUMemoryStats:
        """
        Get current memory statistics.

        Returns TPUMemoryStats dataclass for TPU.
        """
        base_stats = super().get_memory_stats()

        # Calculate cached memory from TPU pools
        cached_memory = sum(
            len(pool['tensors']) * pool['tensors'][0].numel() * pool['tensors'][0].element_size()
            for pool in self._tpu_memory_pool.values()
            if pool.get('tensors')
        )

        return TPUMemoryStats(
            allocated_memory=base_stats.allocated_bytes,
            cached_memory=cached_memory,
            reserved_memory=base_stats.reserved_bytes,
            available_memory=base_stats.free_bytes,
            memory_fraction=self._tpu_config.memory_fraction,
            active_tensors=base_stats.num_allocations,
            peak_memory=base_stats.peak_allocated_bytes,
        )

    def get_tpu_memory_stats(self) -> TPUMemoryStats:
        """
        Get TPU-specific memory statistics as dataclass.

        Alias for get_memory_stats for backward compatibility.

        Returns:
            TPUMemoryStats object with current memory state
        """
        return self.get_memory_stats()

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by clearing caches and consolidating allocations."""
        try:
            # Clear XLA compilation cache
            self._device_synchronize()
            self._empty_device_cache()

            # Clean up old allocations from history
            current_time = time.time()
            retention_seconds = self._tpu_config.allocation_history_retention_seconds
            self._allocation_history = [
                alloc for alloc in self._allocation_history
                if current_time - alloc.timestamp < retention_seconds
            ]

            logger.info("TPU memory optimization completed")

        except Exception:
            pass

    def clear_memory_pools(self) -> None:
        """Clear all TPU memory pools."""
        self._tpu_memory_pool.clear()
        logger.debug("TPU memory pools cleared")

    def get_pool_stats(self, pool_id: str | None = None) -> dict[str, Any]:
        """
        Get TPU memory pool statistics.

        Args:
            pool_id: Specific pool ID, or None for all pools

        Returns:
            Dictionary with pool statistics
        """
        if pool_id and pool_id in self._tpu_memory_pool:
            pool = self._tpu_memory_pool[pool_id]
            return {
                'pool_id': pool_id,
                'total_tensors': len(pool['tensors']),
                'available_tensors': len(pool['available']),
                'in_use_tensors': len(pool['in_use']),
                'utilization': len(pool['in_use']) / len(pool['tensors']) if pool['tensors'] else 0,
                'created_at': pool['created_at']
            }

        pool_stats = {}
        for pid, pool in self._tpu_memory_pool.items():
            pool_stats[pid] = {
                'total_tensors': len(pool['tensors']),
                'available_tensors': len(pool['available']),
                'in_use_tensors': len(pool['in_use']),
                'utilization': len(pool['in_use']) / len(pool['tensors']) if pool['tensors'] else 0,
                'created_at': pool['created_at']
            }

        return {
            'total_pools': len(self._tpu_memory_pool),
            'pool_details': pool_stats
        }

    def monitor_memory(
        self,
        interval: float | None = None,
        duration: float | None = None
    ) -> list[TPUMemoryStats]:
        """
        Monitor memory usage over time.

        Args:
            interval: Monitoring interval in seconds
            duration: Total monitoring duration in seconds

        Returns:
            List of memory statistics over time
        """
        interval = interval or self._tpu_config.monitoring_interval_seconds
        duration = duration or self._tpu_config.monitoring_duration_seconds
        stats_history = []
        start_time = time.time()

        while time.time() - start_time < duration:
            stats = self.get_tpu_memory_stats()
            stats_history.append(stats)

            # Update peak memory
            self._tpu_peak_memory = max(self._tpu_peak_memory, stats.allocated_memory)

            time.sleep(interval)

        return stats_history

    def cleanup(self) -> None:
        """Clean up all tracked allocations and pools."""
        logger.info("Cleaning up TPU memory manager...")
        self.clear_memory_pools()
        super().cleanup()
        logger.info("TPU memory manager cleanup complete")

    def __repr__(self) -> str:
        """String representation of memory manager."""
        stats = self.get_memory_stats()
        return (
            f"TPUMemoryManager(allocated={stats.allocated_memory/1e6:.1f}MB, "
            f"pools={len(self._tpu_memory_pool)}, "
            f"fraction={self._tpu_config.memory_fraction})"
        )


__all__ = ["TPUMemoryManager", "TPUMemoryStats"]
