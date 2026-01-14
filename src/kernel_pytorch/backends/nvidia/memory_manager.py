"""
NVIDIA Memory Manager

GPU memory allocation and optimization for NVIDIA devices.
Inherits from BaseMemoryManager for shared functionality.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import gc

from kernel_pytorch.core.config import KernelPyTorchConfig
from kernel_pytorch.backends.base_memory_manager import BaseMemoryManager, BaseMemoryStats
from .nvidia_exceptions import MemoryAllocationError, OutOfMemoryError

logger = logging.getLogger(__name__)


class NVIDIAMemoryManager(BaseMemoryManager):
    """
    NVIDIA GPU memory manager for efficient allocation and optimization.

    Provides memory pooling, allocation tracking, and optimization
    strategies for NVIDIA GPUs with Tensor Core awareness.
    """

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize NVIDIA memory manager.

        Args:
            config: KernelPyTorch configuration
        """
        self._config = config or KernelPyTorchConfig()
        self.nvidia_config = self._config.hardware.nvidia
        super().__init__(self._config)

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _get_device(self) -> torch.device:
        """Get CUDA device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_optimal_alignment(self) -> int:
        """
        Get optimal tensor dimension alignment for Tensor Cores.

        Tensor Cores work best with dimensions divisible by 8 (or 16 for newer GPUs).
        """
        return 16 if self.nvidia_config.tensor_core_version >= 4 else 8

    def _get_total_memory_bytes(self) -> int:
        """Get total GPU memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(0).total_memory

    def _get_allocated_memory_bytes(self) -> int:
        """Get currently allocated memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated()

    def _get_reserved_memory_bytes(self) -> int:
        """Get reserved (cached) memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_reserved()

    def _device_synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _empty_device_cache(self) -> None:
        """Empty CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # NVIDIA-specific methods
    # =========================================================================

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns extended stats including pool details for NVIDIA.
        """
        base_stats = super().get_memory_stats()

        # Convert to dict format expected by existing code
        stats = {
            'allocated_gb': base_stats.allocated_bytes / 1024**3,
            'reserved_gb': base_stats.reserved_bytes / 1024**3,
            'max_allocated_gb': base_stats.peak_allocated_bytes / 1024**3,
            'device': str(self._device),
            'pool_count': base_stats.pool_count,
            'total_pooled_tensors': sum(len(pool) for pool in self._memory_pools.values()),
            'allocation_count': base_stats.num_allocations
        }

        # Add pool details
        if self._memory_pools:
            stats['pools'] = {
                pool_id: len(tensors)
                for pool_id, tensors in self._memory_pools.items()
            }

        return stats

    def check_memory_available(self, required_mb: float) -> bool:
        """
        Check if required memory is available on GPU.

        Args:
            required_mb: Required memory in megabytes

        Returns:
            True if memory is available, False otherwise
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot check GPU memory")
            return False

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
            available_memory = total_memory - reserved_memory

            logger.debug(
                "Memory check: required=%.1f MB, available=%.1f MB",
                required_mb, available_memory
            )

            return available_memory >= required_mb

        except Exception as e:
            logger.error("Failed to check memory availability: %s", e)
            return False

    def allocate_with_oom_protection(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        safety_margin: float = 1.2
    ) -> torch.Tensor:
        """
        Allocate tensor with out-of-memory protection.

        Args:
            shape: Tensor shape
            dtype: Data type
            requires_grad: Whether tensor requires gradients
            safety_margin: Safety margin multiplier (default 1.2 for 20% buffer)

        Returns:
            Allocated tensor

        Raises:
            OutOfMemoryError: If insufficient memory available
            MemoryAllocationError: If allocation fails for other reasons
        """
        required_mb = self._estimate_tensor_size(shape, dtype)
        required_with_margin = required_mb * safety_margin

        logger.debug(
            "Allocating tensor: shape=%s, dtype=%s, size=%.1f MB",
            shape, dtype, required_mb
        )

        # Check if memory is available
        if not self.check_memory_available(required_with_margin):
            # Try to free up memory
            self.clear_pool()
            gc.collect()
            self._empty_device_cache()

            # Check again after cleanup
            if not self.check_memory_available(required_with_margin):
                stats = self.get_memory_stats()
                raise OutOfMemoryError(
                    f"Insufficient GPU memory for tensor of size {required_mb:.1f} MB. "
                    f"Allocated: {stats.get('allocated_gb', 0):.2f} GB, "
                    f"Reserved: {stats.get('reserved_gb', 0):.2f} GB."
                )

        # Attempt allocation
        try:
            return self.allocate_tensor(shape, dtype, requires_grad)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(f"GPU out of memory while allocating tensor: {e}") from e
            else:
                raise MemoryAllocationError("allocation", str(e)) from e

    def optimize_model_memory(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze and optimize model memory usage for NVIDIA GPUs.

        Args:
            model: PyTorch model to optimize

        Returns:
            Dictionary with optimization results and recommendations
        """
        # Get base analysis
        results = super().optimize_model_memory(model)

        # Add backward compatibility alias
        results['total_memory_mb'] = results['total_mb']

        # Add NVIDIA-specific recommendations
        if results['total_mb'] > 1000:  # > 1GB
            results['recommendations'].append({
                'type': 'gradient_checkpointing',
                'reason': 'Large model (>1GB)',
                'expected_saving': '30-50% memory reduction'
            })

        if self.nvidia_config.fp8_enabled:
            results['recommendations'].append({
                'type': 'fp8_quantization',
                'reason': 'FP8 supported on current hardware',
                'expected_saving': '2x memory reduction'
            })

        return results

    def enable_memory_efficient_mode(self) -> None:
        """Enable memory-efficient execution mode."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True

            if self.nvidia_config.memory_fraction < 1.0:
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    def _estimate_tensor_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> float:
        """
        Estimate tensor size in megabytes (backward compatibility alias).

        Note: Returns MB for backward compatibility with existing tests.
        Use estimate_tensor_size() for bytes.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Estimated size in megabytes
        """
        return self.estimate_tensor_size(shape, dtype) / (1024 ** 2)

    def reset_peak_memory(self) -> None:
        """Reset peak memory statistics."""
        super().reset_peak_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def __repr__(self) -> str:
        """String representation of memory manager."""
        stats = self.get_memory_stats()
        return (
            f"NVIDIAMemoryManager("
            f"device={stats['device']}, "
            f"allocated={stats['allocated_gb']:.2f}GB, "
            f"reserved={stats['reserved_gb']:.2f}GB, "
            f"pools={stats['pool_count']})"
        )
