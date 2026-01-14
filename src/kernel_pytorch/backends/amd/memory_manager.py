"""
AMD GPU Memory Management

This module provides specialized memory management for AMD GPUs with
HBM2e/HBM3 memory, implementing efficient allocation, pooling, and
monitoring strategies.

HBM (High Bandwidth Memory) Characteristics:
- HBM2e (MI200): ~1.6 TB/s bandwidth, 32-64 GB capacity
- HBM3 (MI300): ~5.3 TB/s bandwidth, up to 192 GB capacity

Key Features:
- Memory pooling for reduced allocation overhead
- HBM-optimized allocation strategies
- Memory usage monitoring and profiling
- Out-of-memory (OOM) protection
- Automatic cleanup and defragmentation

Inherits from BaseMemoryManager for shared functionality.

Version: 0.3.7
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import time
import gc

from kernel_pytorch.core.config import AMDConfig, AMDArchitecture
from kernel_pytorch.backends.base_memory_manager import BaseMemoryManager, BaseMemoryStats
from .amd_exceptions import ROCmMemoryError, AMDDeviceError

logger = logging.getLogger(__name__)


@dataclass
class AMDMemoryStats:
    """Extended memory statistics for AMD GPU with fragmentation tracking."""

    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    peak_allocated_mb: float
    num_allocations: int
    num_frees: int
    fragmentation_percent: float = 0.0


class AMDMemoryManager(BaseMemoryManager):
    """
    Memory manager for AMD GPUs with HBM optimization.

    This class implements efficient memory management strategies tailored
    for AMD CDNA architectures with high-bandwidth memory.

    Features:
    - Memory pooling to reduce allocation overhead
    - Automatic defragmentation
    - Memory usage tracking and profiling
    - OOM prevention strategies
    - HBM-specific optimizations

    Example:
        >>> config = AMDConfig(memory_pool_size_gb=8.0)
        >>> mem_manager = AMDMemoryManager(config, device_id=0)
        >>> tensor = mem_manager.allocate_tensor((1024, 1024), torch.float32)
    """

    def __init__(self, config: AMDConfig, device_id: int = 0):
        """
        Initialize AMD memory manager.

        Args:
            config: AMD configuration
            device_id: GPU device ID to manage

        Raises:
            AMDDeviceError: If device initialization fails
        """
        self._amd_config = config
        self.device_id = device_id

        # Initialize base class
        super().__init__(config)

        # AMD-specific statistics
        self._amd_stats = {
            "oom_count": 0,
            "defrag_count": 0,
        }

        # Memory pool setup
        self._pool_enabled = config.enable_memory_pooling
        if self._pool_enabled:
            self._initialize_memory_pool()

        logger.info(
            "AMDMemoryManager initialized: device=%d, pool_size=%.2fGB, pooling=%s",
            device_id,
            config.memory_pool_size_gb,
            self._pool_enabled,
        )

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _get_device(self) -> torch.device:
        """Get the AMD GPU device."""
        return torch.device(f"cuda:{self.device_id}")

    def _get_optimal_alignment(self) -> int:
        """
        Get optimal tensor dimension alignment for AMD Matrix Cores.

        CDNA architecture works best with dimensions divisible by 16 (or 32 for MI300).
        """
        # MI300 series benefits from 32-alignment
        if hasattr(self._amd_config, 'architecture'):
            if self._amd_config.architecture in [AMDArchitecture.MI300X, AMDArchitecture.MI300A]:
                return 32
        return 16

    def _get_total_memory_bytes(self) -> int:
        """Get total GPU memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(self.device_id).total_memory

    def _get_allocated_memory_bytes(self) -> int:
        """Get currently allocated memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated(self.device_id)

    def _get_reserved_memory_bytes(self) -> int:
        """Get reserved (cached) memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_reserved(self.device_id)

    def _device_synchronize(self) -> None:
        """Synchronize device operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device_id)

    def _empty_device_cache(self) -> None:
        """Empty CUDA/ROCm cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # AMD-specific methods
    # =========================================================================

    def _initialize_memory_pool(self) -> None:
        """Initialize memory pool for efficient allocations."""
        try:
            # Pre-allocate memory pool
            pool_size_bytes = int(self._amd_config.memory_pool_size_gb * 1024**3)

            # Check if enough memory is available
            free_memory = self.get_free_memory_mb() * 1024**2
            if pool_size_bytes > free_memory * 0.8:  # Leave 20% buffer
                logger.warning(
                    "Requested pool size (%.2fGB) exceeds available memory (%.2fGB), "
                    "reducing pool size",
                    pool_size_bytes / 1024**3,
                    free_memory / 1024**3,
                )
                pool_size_bytes = int(free_memory * 0.5)  # Use 50% of available

            logger.info(
                "Initializing memory pool: %.2fGB", pool_size_bytes / 1024**3
            )

            # Enable PyTorch's caching allocator (works with ROCm too)
            torch.cuda.empty_cache()

            logger.info("Memory pool initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize memory pool: %s", e)
            self._pool_enabled = False

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        pool_id: Optional[str] = None,
        purpose: str = "unknown",
        check_oom: bool = True,
    ) -> torch.Tensor:
        """
        Allocate a tensor on AMD GPU with OOM protection.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            requires_grad: Whether tensor requires gradients
            pool_id: Optional pool ID for memory pooling
            purpose: Purpose description for tracking
            check_oom: Whether to check for OOM before allocation

        Returns:
            Allocated tensor on GPU

        Raises:
            ROCmMemoryError: If allocation fails or OOM detected
        """
        # Estimate memory requirement
        size_bytes = self.estimate_tensor_size(shape, dtype)
        size_mb = size_bytes / (1024**2)

        logger.debug(
            "Allocating tensor: shape=%s, dtype=%s, size=%.2fMB, purpose=%s",
            shape,
            dtype,
            size_mb,
            purpose,
        )

        # OOM check
        if check_oom:
            free_mb = self.get_free_memory_mb()
            if size_mb > free_mb * 0.9:  # Leave 10% buffer
                self._amd_stats["oom_count"] += 1
                raise ROCmMemoryError(
                    "allocation",
                    required_mb=size_mb,
                    available_mb=free_mb,
                )

        try:
            # Use base class allocation for pooling support
            return super().allocate_tensor(shape, dtype, requires_grad, pool_id, purpose)

        except torch.cuda.OutOfMemoryError as e:
            self._amd_stats["oom_count"] += 1
            raise ROCmMemoryError(
                "allocation",
                required_mb=size_mb,
                available_mb=self.get_free_memory_mb(),
            )

    def free_tensor(self, tensor: torch.Tensor) -> None:
        """
        Free a tensor and update tracking.

        Args:
            tensor: Tensor to free
        """
        alloc_id = str(id(tensor))

        logger.debug(
            "Freeing tensor: size=%.2fMB",
            tensor.element_size() * tensor.numel() / (1024**2),
        )

        self._stats['total_frees'] += 1

        # Delete tensor reference
        del tensor

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns extended stats including fragmentation for AMD.
        """
        base_stats = super().get_memory_stats()

        # Calculate fragmentation
        total_mb = base_stats.total_bytes / (1024**2)
        allocated_mb = base_stats.allocated_bytes / (1024**2)
        reserved_mb = base_stats.reserved_bytes / (1024**2)
        fragmentation = (
            ((reserved_mb - allocated_mb) / total_mb * 100)
            if total_mb > 0
            else 0.0
        )

        # Return dict format for AMD-specific stats
        return {
            'total_mb': total_mb,
            'allocated_mb': allocated_mb,
            'reserved_mb': reserved_mb,
            'free_mb': base_stats.free_bytes / (1024**2),
            'peak_allocated_mb': base_stats.peak_allocated_bytes / (1024**2),
            'num_allocations': base_stats.num_allocations,
            'num_frees': self._stats['total_frees'],
            'fragmentation_percent': fragmentation,
            'device': str(self._device),
            'pool_count': base_stats.pool_count,
            'oom_count': self._amd_stats['oom_count'],
            'defrag_count': self._amd_stats['defrag_count'],
        }

    def get_amd_memory_stats(self) -> AMDMemoryStats:
        """
        Get AMD-specific memory statistics as dataclass.

        Returns:
            AMDMemoryStats object with current memory state
        """
        stats = self.get_memory_stats()
        return AMDMemoryStats(
            total_mb=stats['total_mb'],
            allocated_mb=stats['allocated_mb'],
            reserved_mb=stats['reserved_mb'],
            free_mb=stats['free_mb'],
            peak_allocated_mb=stats['peak_allocated_mb'],
            num_allocations=stats['num_allocations'],
            num_frees=stats['num_frees'],
            fragmentation_percent=stats['fragmentation_percent'],
        )

    def get_allocated_memory_mb(self) -> float:
        """
        Get currently allocated memory in MB.

        Returns:
            Allocated memory in MB
        """
        return self._get_allocated_memory_bytes() / (1024**2)

    def get_free_memory_mb(self) -> float:
        """
        Get free memory in MB.

        Returns:
            Free memory in MB
        """
        total = self._get_total_memory_bytes()
        allocated = self._get_allocated_memory_bytes()
        return (total - allocated) / (1024**2)

    def get_total_memory_mb(self) -> float:
        """
        Get total GPU memory in MB.

        Returns:
            Total memory in MB
        """
        return self._get_total_memory_bytes() / (1024**2)

    def defragment(self) -> None:
        """
        Defragment GPU memory to reduce fragmentation.

        This method triggers PyTorch's cache cleanup to consolidate
        free memory blocks.
        """
        logger.info("Defragmenting GPU memory...")

        try:
            # Empty cache to consolidate free blocks
            gc.collect()
            self._empty_device_cache()

            self._amd_stats["defrag_count"] += 1
            logger.info("Memory defragmentation complete")

        except Exception as e:
            logger.warning("Defragmentation failed: %s", e)

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        super().reset_peak_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
        logger.debug("Peak memory stats reset")

    def get_allocation_summary(self) -> Dict[str, int]:
        """
        Get summary of allocations by purpose.

        Returns:
            Dictionary mapping purpose to count
        """
        summary = defaultdict(int)
        for allocation in self._allocation_history:
            summary[allocation.purpose] += 1
        return dict(summary)

    def cleanup(self) -> None:
        """Clean up all tracked allocations."""
        logger.info("Cleaning up AMD memory manager...")
        super().cleanup()
        logger.info("AMD memory manager cleanup complete")

    def __repr__(self) -> str:
        """String representation of memory manager."""
        stats = self.get_memory_stats()
        return (
            f"AMDMemoryManager("
            f"device={self.device_id}, "
            f"allocated={stats['allocated_mb']:.2f}MB, "
            f"free={stats['free_mb']:.2f}MB, "
            f"fragmentation={stats['fragmentation_percent']:.1f}%)"
        )


__all__ = ["AMDMemoryManager", "AMDMemoryStats"]
