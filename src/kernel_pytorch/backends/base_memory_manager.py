"""
Base Memory Manager for Backend Implementations

This module provides the abstract base class for all backend memory managers,
defining the common interface and shared functionality.

Backends (NVIDIA, AMD, TPU) inherit from this base and implement
device-specific optimizations while maintaining a consistent API.
"""

import logging
import gc
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, TypeVar
from dataclasses import dataclass, field
from collections import defaultdict
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Type variable for config types
ConfigT = TypeVar('ConfigT')


@dataclass
class MemoryAllocationInfo:
    """Information about a memory allocation."""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int
    timestamp: float = field(default_factory=time.time)
    pool_id: Optional[str] = None
    purpose: str = "unknown"


@dataclass
class BaseMemoryStats:
    """Base memory statistics structure."""
    allocated_bytes: int
    reserved_bytes: int
    total_bytes: int
    free_bytes: int
    peak_allocated_bytes: int
    num_allocations: int
    pool_count: int

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 ** 2)

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / (1024 ** 2)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def free_mb(self) -> float:
        return self.free_bytes / (1024 ** 2)

    @property
    def peak_allocated_mb(self) -> float:
        return self.peak_allocated_bytes / (1024 ** 2)

    @property
    def utilization(self) -> float:
        """Memory utilization as a fraction (0-1)."""
        if self.total_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'allocated_mb': self.allocated_mb,
            'reserved_mb': self.reserved_mb,
            'total_mb': self.total_mb,
            'free_mb': self.free_mb,
            'peak_allocated_mb': self.peak_allocated_mb,
            'num_allocations': self.num_allocations,
            'pool_count': self.pool_count,
            'utilization': self.utilization
        }


class BaseMemoryManager(ABC):
    """
    Abstract base class for backend memory managers.

    This class defines the common interface that all backend memory managers
    (NVIDIA, AMD, TPU) must implement, while providing shared functionality
    for memory pooling, allocation tracking, and statistics.

    Subclasses must implement:
    - _get_device() -> torch.device
    - _get_optimal_alignment() -> int
    - _get_total_memory_bytes() -> int
    - _get_allocated_memory_bytes() -> int
    - _get_reserved_memory_bytes() -> int
    - _device_synchronize() -> None
    - _empty_device_cache() -> None

    Optional overrides for device-specific behavior:
    - allocate_tensor()
    - optimize_tensor_layout()
    - get_memory_stats()
    """

    def __init__(self, config: Any):
        """
        Initialize base memory manager.

        Args:
            config: Backend-specific configuration object
        """
        self.config = config
        self._device = self._get_device()

        # Memory pool management
        self._memory_pools: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # Allocation tracking
        self._allocation_history: List[MemoryAllocationInfo] = []
        self._peak_memory_bytes: int = 0

        # Statistics
        self._stats = {
            'total_allocations': 0,
            'total_frees': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }

        logger.debug(
            "%s initialized: device=%s",
            self.__class__.__name__,
            self._device
        )

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _get_device(self) -> torch.device:
        """Get the device for this memory manager."""
        pass

    @abstractmethod
    def _get_optimal_alignment(self) -> int:
        """
        Get optimal tensor dimension alignment for this backend.

        Returns:
            Optimal divisor for tensor dimensions (e.g., 8, 16, 128)
        """
        pass

    @abstractmethod
    def _get_total_memory_bytes(self) -> int:
        """Get total device memory in bytes."""
        pass

    @abstractmethod
    def _get_allocated_memory_bytes(self) -> int:
        """Get currently allocated memory in bytes."""
        pass

    @abstractmethod
    def _get_reserved_memory_bytes(self) -> int:
        """Get reserved (cached) memory in bytes."""
        pass

    @abstractmethod
    def _device_synchronize(self) -> None:
        """Synchronize device operations."""
        pass

    @abstractmethod
    def _empty_device_cache(self) -> None:
        """Empty device memory cache."""
        pass

    # =========================================================================
    # Common implementations
    # =========================================================================

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        pool_id: Optional[str] = None,
        purpose: str = "unknown"
    ) -> torch.Tensor:
        """
        Allocate tensor with optional memory pooling.

        Args:
            shape: Tensor shape
            dtype: Data type
            requires_grad: Whether tensor requires gradients
            pool_id: Optional pool ID for memory pooling
            purpose: Purpose description for tracking

        Returns:
            Allocated tensor on device
        """
        # Try to reuse from pool if available
        if pool_id and self._memory_pools[pool_id]:
            for i, tensor in enumerate(self._memory_pools[pool_id]):
                if tensor.shape == shape and tensor.dtype == dtype:
                    # Reuse tensor from pool
                    reused = self._memory_pools[pool_id].pop(i)
                    reused.zero_()
                    reused.requires_grad = requires_grad
                    self._stats['pool_hits'] += 1
                    return reused
            self._stats['pool_misses'] += 1

        # Allocate new tensor
        tensor = torch.zeros(
            shape,
            dtype=dtype,
            device=self._device,
            requires_grad=requires_grad
        )

        # Track allocation
        size_bytes = tensor.element_size() * tensor.numel()
        self._allocation_history.append(MemoryAllocationInfo(
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            pool_id=pool_id,
            purpose=purpose
        ))
        self._stats['total_allocations'] += 1

        # Update peak memory
        current_allocated = self._get_allocated_memory_bytes()
        if current_allocated > self._peak_memory_bytes:
            self._peak_memory_bytes = current_allocated

        return tensor

    def return_to_pool(self, tensor: torch.Tensor, pool_id: str) -> None:
        """
        Return tensor to memory pool for reuse.

        Args:
            tensor: Tensor to return
            pool_id: Pool identifier
        """
        if tensor.device == self._device:
            # Detach and zero out
            tensor = tensor.detach()
            tensor.zero_()
            self._memory_pools[pool_id].append(tensor)
            self._stats['total_frees'] += 1

    def clear_pool(self, pool_id: Optional[str] = None) -> None:
        """
        Clear memory pool(s).

        Args:
            pool_id: Pool to clear, or None to clear all pools
        """
        if pool_id:
            self._memory_pools[pool_id].clear()
        else:
            self._memory_pools.clear()

        # Force garbage collection and empty cache
        gc.collect()
        self._empty_device_cache()

    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor layout for this backend's optimal alignment.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with optimized layout (padded if necessary)
        """
        optimal_divisor = self._get_optimal_alignment()
        shape = list(tensor.shape)
        needs_padding = False

        for i, dim in enumerate(shape):
            if dim % optimal_divisor != 0:
                needs_padding = True
                shape[i] = ((dim + optimal_divisor - 1) // optimal_divisor) * optimal_divisor

        if needs_padding:
            # Pad tensor to optimal dimensions
            padded = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

            # Copy original data
            slices = tuple(slice(0, s) for s in tensor.shape)
            padded[slices] = tensor

            return padded

        return tensor

    def get_memory_stats(self) -> BaseMemoryStats:
        """
        Get current memory statistics.

        Returns:
            BaseMemoryStats with current memory state
        """
        return BaseMemoryStats(
            allocated_bytes=self._get_allocated_memory_bytes(),
            reserved_bytes=self._get_reserved_memory_bytes(),
            total_bytes=self._get_total_memory_bytes(),
            free_bytes=self._get_total_memory_bytes() - self._get_allocated_memory_bytes(),
            peak_allocated_bytes=self._peak_memory_bytes,
            num_allocations=self._stats['total_allocations'],
            pool_count=len(self._memory_pools)
        )

    def check_memory_available(self, required_bytes: int) -> bool:
        """
        Check if required memory is available.

        Args:
            required_bytes: Required memory in bytes

        Returns:
            True if memory is available, False otherwise
        """
        free_bytes = self._get_total_memory_bytes() - self._get_reserved_memory_bytes()
        return free_bytes >= required_bytes

    def estimate_tensor_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """
        Estimate tensor size in bytes.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Estimated size in bytes
        """
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        dtype_size = torch.tensor([], dtype=dtype).element_size()
        return num_elements * dtype_size

    def get_pool_stats(self, pool_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for memory pool(s).

        Args:
            pool_id: Specific pool ID, or None for all pools

        Returns:
            Dictionary with pool statistics
        """
        if pool_id:
            pool = self._memory_pools.get(pool_id, [])
            total_bytes = sum(t.element_size() * t.numel() for t in pool)
            return {
                'pool_id': pool_id,
                'tensor_count': len(pool),
                'total_bytes': total_bytes,
                'total_mb': total_bytes / (1024 ** 2),
                'shapes': [tuple(t.shape) for t in pool]
            }
        else:
            return {
                'total_pools': len(self._memory_pools),
                'pools': {
                    pid: {
                        'tensor_count': len(tensors),
                        'total_bytes': sum(t.element_size() * t.numel() for t in tensors)
                    }
                    for pid, tensors in self._memory_pools.items()
                },
                'stats': self._stats.copy()
            }

    def optimize_model_memory(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze model memory usage.

        Args:
            model: PyTorch model to analyze

        Returns:
            Dictionary with memory analysis and recommendations
        """
        results = {
            'parameter_bytes': 0,
            'buffer_bytes': 0,
            'total_bytes': 0,
            'parameter_count': 0,
            'recommendations': []
        }

        # Calculate parameter memory
        for p in model.parameters():
            results['parameter_bytes'] += p.element_size() * p.numel()
            results['parameter_count'] += p.numel()

        # Calculate buffer memory
        for b in model.buffers():
            results['buffer_bytes'] += b.element_size() * b.numel()

        results['total_bytes'] = results['parameter_bytes'] + results['buffer_bytes']
        results['total_mb'] = results['total_bytes'] / (1024 ** 2)

        # Check for suboptimal layer dimensions
        optimal_div = self._get_optimal_alignment()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.in_features % optimal_div != 0 or module.out_features % optimal_div != 0:
                    results['recommendations'].append({
                        'type': 'dimension_padding',
                        'layer': name,
                        'current': f"{module.in_features}x{module.out_features}",
                        'reason': f'Not divisible by {optimal_div}',
                        'expected_benefit': 'Better hardware utilization'
                    })

        return results

    def synchronize(self) -> None:
        """Synchronize device operations."""
        self._device_synchronize()

    def empty_cache(self) -> None:
        """Empty device memory cache."""
        self._empty_device_cache()
        gc.collect()

    def reset_peak_memory(self) -> None:
        """Reset peak memory statistics."""
        self._peak_memory_bytes = self._get_allocated_memory_bytes()

    def cleanup(self) -> None:
        """Clean up all tracked allocations and pools."""
        logger.info("Cleaning up %s...", self.__class__.__name__)
        self.clear_pool()
        self._allocation_history.clear()
        self.empty_cache()
        logger.info("Cleanup complete")

    def __repr__(self) -> str:
        """String representation of memory manager."""
        stats = self.get_memory_stats()
        return (
            f"{self.__class__.__name__}("
            f"device={self._device}, "
            f"allocated={stats.allocated_mb:.2f}MB, "
            f"free={stats.free_mb:.2f}MB, "
            f"pools={stats.pool_count})"
        )


__all__ = ['BaseMemoryManager', 'BaseMemoryStats', 'MemoryAllocationInfo']
