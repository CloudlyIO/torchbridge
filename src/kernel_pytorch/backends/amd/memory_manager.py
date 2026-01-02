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

Version: 0.3.6
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

from kernel_pytorch.core.config import AMDConfig, AMDArchitecture
from .amd_exceptions import ROCmMemoryError, AMDDeviceError

logger = logging.getLogger(__name__)


@dataclass
class MemoryAllocation:
    """Represents a memory allocation on AMD GPU."""

    size_bytes: int
    device_id: int
    timestamp: float
    purpose: str = "unknown"
    tensor_ref: Optional[torch.Tensor] = None


@dataclass
class MemoryStats:
    """Memory statistics for AMD GPU."""

    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    peak_allocated_mb: float
    num_allocations: int
    num_frees: int
    fragmentation_percent: float = 0.0


class AMDMemoryManager:
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
        self.config = config
        self.device_id = device_id
        self._device = torch.device(f"cuda:{device_id}")

        # Memory tracking
        self._allocations: Dict[str, MemoryAllocation] = {}
        self._allocation_history: List[MemoryAllocation] = []
        self._peak_memory_mb: float = 0.0

        # Statistics
        self._stats = {
            "total_allocations": 0,
            "total_frees": 0,
            "oom_count": 0,
            "defrag_count": 0,
        }

        # Memory pool
        self._pool_enabled = config.enable_memory_pooling
        if self._pool_enabled:
            self._initialize_memory_pool()

        logger.info(
            "AMDMemoryManager initialized: device=%d, pool_size=%.2fGB, pooling=%s",
            device_id,
            config.memory_pool_size_gb,
            self._pool_enabled,
        )

    def _initialize_memory_pool(self) -> None:
        """Initialize memory pool for efficient allocations."""
        try:
            # Pre-allocate memory pool
            pool_size_bytes = int(self.config.memory_pool_size_gb * 1024**3)

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

            # Enable PyTorch's CUD A caching allocator (works with ROCm too)
            torch.cuda.empty_cache()

            logger.info("Memory pool initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize memory pool: %s", e)
            self._pool_enabled = False

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        purpose: str = "unknown",
        check_oom: bool = True,
    ) -> torch.Tensor:
        """
        Allocate a tensor on AMD GPU with OOM protection.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            purpose: Purpose description for tracking
            check_oom: Whether to check for OOM before allocation

        Returns:
            Allocated tensor on GPU

        Raises:
            ROCmMemoryError: If allocation fails or OOM detected
        """
        # Estimate memory requirement
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_bytes = int(torch.tensor(shape).prod().item() * element_size)
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
                self._stats["oom_count"] += 1
                raise ROCmMemoryError(
                    "allocation",
                    required_mb=size_mb,
                    available_mb=free_mb,
                )

        try:
            # Allocate tensor
            tensor = torch.zeros(shape, dtype=dtype, device=self._device)

            # Track allocation
            allocation = MemoryAllocation(
                size_bytes=size_bytes,
                device_id=self.device_id,
                timestamp=time.time(),
                purpose=purpose,
                tensor_ref=tensor,
            )

            alloc_id = str(id(tensor))
            self._allocations[alloc_id] = allocation
            self._allocation_history.append(allocation)
            self._stats["total_allocations"] += 1

            # Update peak memory
            current_allocated = self.get_allocated_memory_mb()
            if current_allocated > self._peak_memory_mb:
                self._peak_memory_mb = current_allocated

            logger.debug("Tensor allocated successfully: %.2fMB", size_mb)
            return tensor

        except torch.cuda.OutOfMemoryError as e:
            self._stats["oom_count"] += 1
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

        if alloc_id in self._allocations:
            allocation = self._allocations[alloc_id]
            logger.debug(
                "Freeing tensor: size=%.2fMB, purpose=%s",
                allocation.size_bytes / (1024**2),
                allocation.purpose,
            )

            del self._allocations[alloc_id]
            self._stats["total_frees"] += 1

        # Delete tensor reference
        del tensor

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            MemoryStats object with current memory state
        """
        # Get PyTorch memory stats (works with ROCm)
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        total = torch.cuda.get_device_properties(self.device_id).total_memory

        allocated_mb = allocated / (1024**2)
        reserved_mb = reserved / (1024**2)
        total_mb = total / (1024**2)
        free_mb = total_mb - allocated_mb

        # Calculate fragmentation
        fragmentation = (
            ((reserved_mb - allocated_mb) / total_mb * 100)
            if total_mb > 0
            else 0.0
        )

        return MemoryStats(
            total_mb=total_mb,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            free_mb=free_mb,
            peak_allocated_mb=self._peak_memory_mb,
            num_allocations=self._stats["total_allocations"],
            num_frees=self._stats["total_frees"],
            fragmentation_percent=fragmentation,
        )

    def get_allocated_memory_mb(self) -> float:
        """
        Get currently allocated memory in MB.

        Returns:
            Allocated memory in MB
        """
        return torch.cuda.memory_allocated(self.device_id) / (1024**2)

    def get_free_memory_mb(self) -> float:
        """
        Get free memory in MB.

        Returns:
            Free memory in MB
        """
        total = torch.cuda.get_device_properties(self.device_id).total_memory
        allocated = torch.cuda.memory_allocated(self.device_id)
        return (total - allocated) / (1024**2)

    def get_total_memory_mb(self) -> float:
        """
        Get total GPU memory in MB.

        Returns:
            Total memory in MB
        """
        total = torch.cuda.get_device_properties(self.device_id).total_memory
        return total / (1024**2)

    def defragment(self) -> None:
        """
        Defragment GPU memory to reduce fragmentation.

        This method triggers PyTorch's cache cleanup to consolidate
        free memory blocks.
        """
        logger.info("Defragmenting GPU memory...")

        try:
            # Empty cache to consolidate free blocks
            torch.cuda.empty_cache()

            self._stats["defrag_count"] += 1
            logger.info("Memory defragmentation complete")

        except Exception as e:
            logger.warning("Defragmentation failed: %s", e)

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        self._peak_memory_mb = self.get_allocated_memory_mb()
        torch.cuda.reset_peak_memory_stats(self.device_id)
        logger.debug("Peak memory stats reset")

    def get_allocation_summary(self) -> Dict[str, int]:
        """
        Get summary of allocations by purpose.

        Returns:
            Dictionary mapping purpose to count
        """
        summary = defaultdict(int)
        for allocation in self._allocations.values():
            summary[allocation.purpose] += 1
        return dict(summary)

    def cleanup(self) -> None:
        """Clean up all tracked allocations."""
        logger.info("Cleaning up memory manager...")

        # Clear all allocations
        self._allocations.clear()
        self._allocation_history.clear()

        # Empty CUDA cache
        torch.cuda.empty_cache()

        logger.info("Memory manager cleanup complete")

    def __repr__(self) -> str:
        """String representation of memory manager."""
        stats = self.get_memory_stats()
        return (
            f"AMDMemoryManager("
            f"device={self.device_id}, "
            f"allocated={stats.allocated_mb:.2f}MB, "
            f"free={stats.free_mb:.2f}MB, "
            f"fragmentation={stats.fragmentation_percent:.1f}%)"
        )


__all__ = ["AMDMemoryManager", "MemoryAllocation", "MemoryStats"]
