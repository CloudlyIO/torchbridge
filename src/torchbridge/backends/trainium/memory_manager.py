"""
Trainium Memory Manager

Manages Trainium HBM memory allocation, optimization, and monitoring
for efficient NeuronCore model execution.

Inherits from BaseMemoryManager for shared functionality.
"""

import gc
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from torchbridge.backends.base_memory_manager import BaseMemoryManager
from torchbridge.core.config import TrainiumArchitecture, TrainiumConfig

from . import neuron_utilities

logger = logging.getLogger(__name__)


@dataclass
class TrainiumMemoryStats:
    """Trainium-specific memory statistics."""
    allocated_memory: int
    cached_memory: int
    reserved_memory: int
    available_memory: int
    memory_fraction: float
    active_tensors: int
    peak_memory: int


class TrainiumMemoryManager(BaseMemoryManager):
    """
    Trainium memory manager for optimal HBM usage.

    Provides memory allocation, monitoring, and optimization
    specifically designed for Trainium hardware characteristics.

    Inherits from BaseMemoryManager for shared functionality.
    """

    def __init__(self, config: TrainiumConfig):
        """
        Initialize Trainium memory manager.

        Args:
            config: Trainium configuration
        """
        self._trainium_config = config

        # Trainium-specific memory pool tracking
        self._trainium_memory_pool: dict[str, dict] = {}
        self._trainium_peak_memory = 0

        # Initialize base class
        super().__init__(config)

        # Initialize Trainium-specific memory management
        self._setup_memory_management()

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _get_device(self) -> torch.device:
        """Get the XLA/Trainium device."""
        try:
            return neuron_utilities.get_xla_device()
        except Exception:
            warnings.warn("Neuron SDK not available. Using CPU fallback.", stacklevel=2)
            return torch.device("cpu")

    def _get_optimal_alignment(self) -> int:
        """
        Get optimal tensor dimension alignment for Trainium.

        NeuronCores work best with dimensions divisible by 8.
        """
        return 8

    def _get_total_memory_bytes(self) -> int:
        """Get total Trainium HBM in bytes."""
        memory_gb = self._get_trainium_memory_gb()
        return int(memory_gb * 1e9)

    def _get_allocated_memory_bytes(self) -> int:
        """
        Get currently allocated memory in bytes.

        XLA doesn't expose memory stats the same way as CUDA,
        so we estimate from allocation history.
        """
        retention_seconds = self._trainium_config.allocation_history_retention_seconds
        return sum(
            alloc.size_bytes for alloc in self._allocation_history
            if time.time() - alloc.timestamp < retention_seconds
        )

    def _get_reserved_memory_bytes(self) -> int:
        """Get reserved (cached) memory in bytes."""
        allocated = self._get_allocated_memory_bytes()
        cached = sum(
            len(pool.get('tensors', [])) * pool['tensors'][0].numel() * pool['tensors'][0].element_size()
            for pool in self._trainium_memory_pool.values()
            if pool.get('tensors')
        )
        return allocated + cached

    def _device_synchronize(self) -> None:
        """Synchronize XLA operations."""
        try:
            neuron_utilities.sync()
        except Exception:
            pass

    def _empty_device_cache(self) -> None:
        """Empty device cache."""
        try:
            neuron_utilities.sync()
            gc.collect()
        except Exception:
            pass

    # =========================================================================
    # Trainium-specific methods
    # =========================================================================

    def _setup_memory_management(self) -> None:
        """Set up Trainium memory management."""
        try:
            import torch_neuronx  # noqa: F401

            self._apply_memory_fraction()

            logger.info(
                "Trainium Memory Manager initialized: memory_fraction=%.2f, device=%s",
                self._trainium_config.memory_fraction,
                self._device
            )

        except ImportError:
            warnings.warn("Neuron SDK not available. Memory manager will use CPU fallback.", stacklevel=2)

    def _apply_memory_fraction(self) -> None:
        """Apply memory fraction limits for Trainium."""
        import os

        if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self._trainium_config.memory_fraction)

        os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

    def _get_trainium_memory_gb(self) -> float:
        """Get Trainium HBM capacity in GB."""
        memory_map = {
            TrainiumArchitecture.TRN1: 32.0,    # 32GB HBM per chip
            TrainiumArchitecture.TRN2: 96.0,    # 96GB HBM per chip
            TrainiumArchitecture.TRN3: 144.0,   # 144GB HBM3e per chip
            TrainiumArchitecture.INF2: 32.0,    # 32GB HBM per chip
        }
        return memory_map.get(self._trainium_config.architecture, 32.0)

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        pool_id: str | None = None,
        purpose: str = "unknown"
    ) -> torch.Tensor:
        """
        Allocate tensor with optimal Trainium memory placement.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            requires_grad: Whether tensor requires gradients
            pool_id: Optional pool ID for memory pooling
            purpose: Purpose description for tracking

        Returns:
            Allocated tensor on Trainium
        """
        return super().allocate_tensor(shape, dtype, requires_grad, pool_id, purpose)

    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor memory layout for Trainium NeuronCores.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with optimized layout
        """
        original_shape = tensor.shape
        optimal_div = self._get_optimal_alignment()

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
        pool_id = f"trn_pool_{len(self._trainium_memory_pool)}_{int(time.time())}"

        pool_tensors = []
        for _ in range(pool_size):
            tensor = self.allocate_tensor(tensor_size, dtype)
            pool_tensors.append(tensor)

        self._trainium_memory_pool[pool_id] = {
            'tensors': pool_tensors,
            'available': list(range(pool_size)),
            'in_use': [],
            'created_at': time.time()
        }

        logger.debug("Created Trainium memory pool '%s': size=%d, tensor_shape=%s",
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
        if pool_id not in self._trainium_memory_pool:
            return None

        pool = self._trainium_memory_pool[pool_id]
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
        if pool_id not in self._trainium_memory_pool:
            return False

        pool = self._trainium_memory_pool[pool_id]

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

    def get_memory_stats(self) -> TrainiumMemoryStats:
        """Get current memory statistics."""
        base_stats = super().get_memory_stats()

        cached_memory = sum(
            len(pool['tensors']) * pool['tensors'][0].numel() * pool['tensors'][0].element_size()
            for pool in self._trainium_memory_pool.values()
            if pool.get('tensors')
        )

        return TrainiumMemoryStats(
            allocated_memory=base_stats.allocated_bytes,
            cached_memory=cached_memory,
            reserved_memory=base_stats.reserved_bytes,
            available_memory=base_stats.free_bytes,
            memory_fraction=self._trainium_config.memory_fraction,
            active_tensors=base_stats.num_allocations,
            peak_memory=base_stats.peak_allocated_bytes,
        )

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by clearing caches and consolidating allocations."""
        try:
            self._device_synchronize()
            self._empty_device_cache()

            current_time = time.time()
            retention_seconds = self._trainium_config.allocation_history_retention_seconds
            self._allocation_history = [
                alloc for alloc in self._allocation_history
                if current_time - alloc.timestamp < retention_seconds
            ]

            logger.info("Trainium memory optimization completed")

        except Exception:
            pass

    def clear_memory_pools(self) -> None:
        """Clear all Trainium memory pools."""
        self._trainium_memory_pool.clear()
        logger.debug("Trainium memory pools cleared")

    def get_pool_stats(self, pool_id: str | None = None) -> dict[str, Any]:
        """
        Get Trainium memory pool statistics.

        Args:
            pool_id: Specific pool ID, or None for all pools

        Returns:
            Dictionary with pool statistics
        """
        if pool_id and pool_id in self._trainium_memory_pool:
            pool = self._trainium_memory_pool[pool_id]
            return {
                'pool_id': pool_id,
                'total_tensors': len(pool['tensors']),
                'available_tensors': len(pool['available']),
                'in_use_tensors': len(pool['in_use']),
                'utilization': len(pool['in_use']) / len(pool['tensors']) if pool['tensors'] else 0,
                'created_at': pool['created_at']
            }

        pool_stats = {}
        for pid, pool in self._trainium_memory_pool.items():
            pool_stats[pid] = {
                'total_tensors': len(pool['tensors']),
                'available_tensors': len(pool['available']),
                'in_use_tensors': len(pool['in_use']),
                'utilization': len(pool['in_use']) / len(pool['tensors']) if pool['tensors'] else 0,
                'created_at': pool['created_at']
            }

        return {
            'total_pools': len(self._trainium_memory_pool),
            'pool_details': pool_stats
        }

    def cleanup(self) -> None:
        """Clean up all tracked allocations and pools."""
        logger.info("Cleaning up Trainium memory manager...")
        self.clear_memory_pools()
        super().cleanup()
        logger.info("Trainium memory manager cleanup complete")

    def __repr__(self) -> str:
        """String representation of memory manager."""
        stats = self.get_memory_stats()
        return (
            f"TrainiumMemoryManager(allocated={stats.allocated_memory / 1e6:.1f}MB, "
            f"pools={len(self._trainium_memory_pool)}, "
            f"fraction={self._trainium_config.memory_fraction})"
        )


__all__ = ["TrainiumMemoryManager", "TrainiumMemoryStats"]
