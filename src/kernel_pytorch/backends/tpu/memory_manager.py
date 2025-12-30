"""
TPU Memory Manager

Manages TPU memory allocation, optimization, and monitoring
for efficient TPU model execution.
"""

import logging
import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import time

from kernel_pytorch.core.config import TPUConfig, TPUVersion
from .tpu_exceptions import TPUMemoryError, raise_or_warn

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """TPU memory statistics."""
    allocated_memory: int
    cached_memory: int
    reserved_memory: int
    available_memory: int
    memory_fraction: float
    active_tensors: int
    peak_memory: int


class TPUMemoryManager:
    """
    TPU memory manager for optimal memory usage.

    Provides memory allocation, monitoring, and optimization
    specifically designed for TPU hardware characteristics.
    """

    def __init__(self, config: TPUConfig):
        """
        Initialize TPU memory manager.

        Args:
            config: TPU configuration
        """
        self.config = config
        self._memory_pool = {}
        self._allocation_history = []
        self._peak_memory = 0

        # Initialize memory management
        self._setup_memory_management()

    def _setup_memory_management(self) -> None:
        """Set up TPU memory management."""
        try:
            import torch_xla.core.xla_model as xm

            # Set memory fraction
            self._apply_memory_fraction()

            # Initialize memory tracking
            self._device = xm.xla_device()
            logger.info(
                "TPU Memory Manager initialized: memory_fraction=%.2f, device=%s",
                self.config.memory_fraction,
                self._device
            )

        except ImportError:
            warnings.warn("PyTorch/XLA not available. Memory manager will use CPU fallback.")
            self._device = torch.device("cpu")

    def _apply_memory_fraction(self) -> None:
        """Apply memory fraction limits for TPU."""
        # TPU memory management is handled through XLA environment variables
        import os

        # Set TPU memory fraction
        if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.config.memory_fraction)

        # Enable memory growth for TPUs
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    def allocate_tensor(self, shape: Tuple[int, ...],
                       dtype: torch.dtype = torch.float32,
                       requires_grad: bool = False) -> torch.Tensor:
        """
        Allocate tensor with optimal TPU memory placement.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            requires_grad: Whether tensor requires gradients

        Returns:
            Allocated tensor on TPU
        """
        # Create tensor on TPU device
        tensor = torch.zeros(shape, dtype=dtype, device=self._device, requires_grad=requires_grad)

        # Track allocation
        allocation_info = {
            'shape': shape,
            'dtype': dtype,
            'size_bytes': tensor.numel() * tensor.element_size(),
            'timestamp': time.time(),
            'requires_grad': requires_grad
        }
        self._allocation_history.append(allocation_info)

        return tensor

    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor memory layout for TPU.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with optimized layout
        """
        # TPU prefers certain memory layouts for performance
        original_shape = tensor.shape

        # For 2D tensors, ensure dimensions are multiples of 8 (TPU matrix units)
        if len(original_shape) == 2:
            height, width = original_shape
            if height % 8 != 0 or width % 8 != 0:
                # Pad to nearest multiple of 8
                new_height = ((height + 7) // 8) * 8
                new_width = ((width + 7) // 8) * 8

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
            if c % 8 != 0:
                # Pad channels to multiple of 8
                new_c = ((c + 7) // 8) * 8
                padded_tensor = torch.zeros(
                    (n, new_c, h, w),
                    dtype=tensor.dtype,
                    device=tensor.device
                )
                padded_tensor[:, :c, :, :] = tensor
                return padded_tensor

        return tensor

    def create_memory_pool(self, pool_size: int, tensor_size: Tuple[int, ...],
                          dtype: torch.dtype = torch.float32) -> str:
        """
        Create a memory pool for efficient tensor reuse.

        Args:
            pool_size: Number of tensors in pool
            tensor_size: Size of each tensor
            dtype: Tensor data type

        Returns:
            Pool identifier
        """
        pool_id = f"pool_{len(self._memory_pool)}_{int(time.time())}"

        # Pre-allocate tensors
        pool_tensors = []
        for _ in range(pool_size):
            tensor = self.allocate_tensor(tensor_size, dtype)
            pool_tensors.append(tensor)

        self._memory_pool[pool_id] = {
            'tensors': pool_tensors,
            'available': list(range(pool_size)),
            'in_use': [],
            'created_at': time.time()
        }

        logger.debug("Created memory pool '%s': size=%d, tensor_shape=%s", pool_id, pool_size, tensor_size)
        return pool_id

    def get_tensor_from_pool(self, pool_id: str) -> Optional[torch.Tensor]:
        """
        Get tensor from memory pool.

        Args:
            pool_id: Pool identifier

        Returns:
            Tensor from pool or None if pool is empty
        """
        if pool_id not in self._memory_pool:
            return None

        pool = self._memory_pool[pool_id]
        if not pool['available']:
            return None

        # Get tensor from available list
        tensor_idx = pool['available'].pop(0)
        pool['in_use'].append(tensor_idx)

        tensor = pool['tensors'][tensor_idx]
        tensor.zero_()  # Clear tensor data
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
        if pool_id not in self._memory_pool:
            return False

        pool = self._memory_pool[pool_id]

        # Find tensor index
        tensor_idx = None
        for idx, pool_tensor in enumerate(pool['tensors']):
            if torch.equal(tensor, pool_tensor):
                tensor_idx = idx
                break

        if tensor_idx is None or tensor_idx not in pool['in_use']:
            return False

        # Move tensor back to available list
        pool['in_use'].remove(tensor_idx)
        pool['available'].append(tensor_idx)

        return True

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            Memory statistics
        """
        try:
            # Calculate memory usage from allocation history
            retention_seconds = self.config.allocation_history_retention_seconds
            total_allocated = sum(
                alloc['size_bytes'] for alloc in self._allocation_history
                if time.time() - alloc['timestamp'] < retention_seconds
            )

            # Estimate cached memory from pools
            cached_memory = sum(
                len(pool['tensors']) * pool['tensors'][0].numel() * pool['tensors'][0].element_size()
                for pool in self._memory_pool.values()
                if pool['tensors']
            )

            # TPU-specific memory info (if available)
            try:
                import torch_xla.core.xla_model as xm
                device_count = xm.xla_device_count()
                reserved_memory = total_allocated + cached_memory
            except ImportError:
                device_count = 1
                reserved_memory = total_allocated

            # Estimate available memory based on TPU type
            tpu_memory_gb = self._get_tpu_memory_gb()
            total_memory = int(tpu_memory_gb * 1e9)  # Convert to bytes
            available_memory = total_memory - reserved_memory

            return MemoryStats(
                allocated_memory=total_allocated,
                cached_memory=cached_memory,
                reserved_memory=reserved_memory,
                available_memory=max(0, available_memory),
                memory_fraction=self.config.memory_fraction,
                active_tensors=len(self._allocation_history),
                peak_memory=self._peak_memory
            )

        except Exception as e:
            error_msg = f"Failed to get memory stats: {e}"
            raise_or_warn(error_msg, TPUMemoryError, strict_mode=self.config.enable_strict_validation, logger=logger)
            return MemoryStats(0, 0, 0, 0, 0.0, 0, 0)

    def _get_tpu_memory_gb(self) -> float:
        """Get TPU memory capacity in GB."""
        # TPU memory capacities by version
        memory_map = {
            TPUVersion.V4: 32.0,    # 32GB HBM (verified)
            TPUVersion.V5E: 16.0,   # 16GB HBM (verified)
            TPUVersion.V5P: 95.0,   # 95GB HBM (verified)
            TPUVersion.V6E: self.config.v6e_memory_gb or 32.0,   # Configurable (default: 32GB)
            TPUVersion.V7: self.config.v7_memory_gb or 128.0,    # Configurable (default: 128GB)
        }

        return memory_map.get(self.config.version, 32.0)  # Default to 32GB

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by clearing caches and consolidating allocations."""
        try:
            import torch_xla.core.xla_model as xm

            # Clear XLA compilation cache
            xm.mark_step()

            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clean up old allocations from history
            current_time = time.time()
            retention_seconds = self.config.allocation_history_retention_seconds
            self._allocation_history = [
                alloc for alloc in self._allocation_history
                if current_time - alloc['timestamp'] < retention_seconds
            ]

            logger.info("Memory optimization completed")

        except ImportError:
            pass

    def clear_memory_pools(self) -> None:
        """Clear all memory pools."""
        self._memory_pool.clear()
        logger.debug("Memory pools cleared")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        pool_stats = {}
        for pool_id, pool in self._memory_pool.items():
            pool_stats[pool_id] = {
                'total_tensors': len(pool['tensors']),
                'available_tensors': len(pool['available']),
                'in_use_tensors': len(pool['in_use']),
                'utilization': len(pool['in_use']) / len(pool['tensors']) if pool['tensors'] else 0,
                'created_at': pool['created_at']
            }

        return {
            'total_pools': len(self._memory_pool),
            'pool_details': pool_stats
        }

    def monitor_memory(
        self,
        interval: Optional[float] = None,
        duration: Optional[float] = None
    ) -> List[MemoryStats]:
        """
        Monitor memory usage over time.

        Args:
            interval: Monitoring interval in seconds (default: from config)
            duration: Total monitoring duration in seconds (default: from config)

        Returns:
            List of memory statistics over time
        """
        interval = interval or self.config.monitoring_interval_seconds
        duration = duration or self.config.monitoring_duration_seconds
        stats_history = []
        start_time = time.time()

        while time.time() - start_time < duration:
            stats = self.get_memory_stats()
            stats_history.append(stats)

            # Update peak memory
            self._peak_memory = max(self._peak_memory, stats.allocated_memory)

            time.sleep(interval)

        return stats_history

    def __repr__(self) -> str:
        """String representation of memory manager."""
        stats = self.get_memory_stats()
        return (
            f"TPUMemoryManager(allocated={stats.allocated_memory/1e6:.1f}MB, "
            f"pools={len(self._memory_pool)}, "
            f"fraction={self.config.memory_fraction})"
        )