"""
NVIDIA Memory Manager

GPU memory allocation and optimization for NVIDIA devices.
"""

import logging
import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import gc

from kernel_pytorch.core.config import KernelPyTorchConfig
from .nvidia_exceptions import MemoryAllocationError, OutOfMemoryError

logger = logging.getLogger(__name__)


class NVIDIAMemoryManager:
    """
    NVIDIA GPU memory manager for efficient allocation and optimization.

    Provides memory pooling, allocation tracking, and optimization
    strategies for NVIDIA GPUs.
    """

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize NVIDIA memory manager.

        Args:
            config: KernelPyTorch configuration
        """
        self.config = config or KernelPyTorchConfig()
        self.nvidia_config = self.config.hardware.nvidia

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._memory_pools = defaultdict(list)
        self._allocation_history = []

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        pool_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Allocate tensor with optional memory pooling.

        Args:
            shape: Tensor shape
            dtype: Data type
            requires_grad: Whether tensor requires gradients
            pool_id: Optional pool ID for memory pooling

        Returns:
            Allocated tensor
        """
        # Try to reuse from pool if available
        if pool_id and self._memory_pools[pool_id]:
            for i, tensor in enumerate(self._memory_pools[pool_id]):
                if tensor.shape == shape and tensor.dtype == dtype:
                    # Reuse tensor from pool
                    reused = self._memory_pools[pool_id].pop(i)
                    reused.zero_()
                    reused.requires_grad = requires_grad
                    return reused

        # Allocate new tensor
        tensor = torch.zeros(
            shape,
            dtype=dtype,
            device=self._device,
            requires_grad=requires_grad
        )

        # Track allocation
        self._allocation_history.append({
            'shape': shape,
            'dtype': dtype,
            'size_bytes': tensor.element_size() * tensor.numel(),
            'pool_id': pool_id
        })

        return tensor

    def return_to_pool(self, tensor: torch.Tensor, pool_id: str) -> None:
        """
        Return tensor to memory pool for reuse.

        Args:
            tensor: Tensor to return
            pool_id: Pool identifier
        """
        if tensor.device.type == 'cuda':
            # Detach and zero out
            tensor = tensor.detach()
            tensor.zero_()
            self._memory_pools[pool_id].append(tensor)

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

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor layout for NVIDIA GPUs.

        Args:
            tensor: Input tensor

        Returns:
            Optimized tensor
        """
        # Ensure tensor dimensions are optimal for Tensor Cores
        # Tensor Cores work best with dimensions divisible by 8 (or 16 for newer GPUs)
        optimal_divisor = 16 if self.nvidia_config.tensor_core_version >= 4 else 8

        # Check if padding is beneficial
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

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0,
                'device': 'cpu'
            }

        stats = {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'device': str(self._device),
            'pool_count': len(self._memory_pools),
            'total_pooled_tensors': sum(len(pool) for pool in self._memory_pools.values()),
            'allocation_count': len(self._allocation_history)
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
            # Get current memory stats
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # MB

            # Available memory is total - reserved (not allocated, as reserved includes fragmentation)
            available_memory = total_memory - reserved_memory

            logger.debug("Memory check: required=%.1f MB, available=%.1f MB (total=%.1f MB, reserved=%.1f MB)",
                        required_mb, available_memory, total_memory, reserved_memory)

            return available_memory >= required_mb

        except Exception as e:
            logger.error("Failed to check memory availability: %s", e)
            return False

    def _estimate_tensor_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> float:
        """
        Estimate tensor size in megabytes.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Estimated size in MB
        """
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        # Get dtype size in bytes
        dtype_size = torch.tensor([], dtype=dtype).element_size()

        size_bytes = num_elements * dtype_size
        size_mb = size_bytes / (1024 ** 2)

        return size_mb

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
        # Estimate required memory
        required_mb = self._estimate_tensor_size(shape, dtype)
        required_with_margin = required_mb * safety_margin

        logger.debug("Allocating tensor: shape=%s, dtype=%s, size=%.1f MB (with margin: %.1f MB)",
                    shape, dtype, required_mb, required_with_margin)

        # Check if memory is available
        if not self.check_memory_available(required_with_margin):
            # Try to free up memory
            self.clear_pool()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check again after cleanup
            if not self.check_memory_available(required_with_margin):
                stats = self.get_memory_stats()
                raise OutOfMemoryError(
                    f"Insufficient GPU memory for tensor of size {required_mb:.1f} MB. "
                    f"Allocated: {stats.get('allocated_gb', 0):.2f} GB, "
                    f"Reserved: {stats.get('reserved_gb', 0):.2f} GB. "
                    f"Required: {required_with_margin:.1f} MB"
                )

        # Attempt allocation
        try:
            tensor = torch.zeros(
                shape,
                dtype=dtype,
                device=self._device,
                requires_grad=requires_grad
            )

            # Track allocation
            self._allocation_history.append({
                'shape': shape,
                'dtype': dtype,
                'size_bytes': tensor.element_size() * tensor.numel(),
                'pool_id': None
            })

            logger.debug("Successfully allocated tensor: %s", shape)
            return tensor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(f"GPU out of memory while allocating tensor: {e}") from e
            else:
                raise MemoryAllocationError(f"Failed to allocate tensor: {e}") from e

    def get_pool_stats(self, pool_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific memory pool.

        Args:
            pool_id: Pool identifier

        Returns:
            Dictionary with pool statistics
        """
        if pool_id not in self._memory_pools:
            return {
                'pool_id': pool_id,
                'tensor_count': 0,
                'total_bytes': 0,
                'shapes': []
            }

        pool = self._memory_pools[pool_id]
        total_bytes = sum(
            tensor.element_size() * tensor.numel()
            for tensor in pool
        )

        shapes = [tuple(tensor.shape) for tensor in pool]

        return {
            'pool_id': pool_id,
            'tensor_count': len(pool),
            'total_bytes': total_bytes,
            'total_mb': total_bytes / 1024**2,
            'shapes': shapes
        }

    def optimize_model_memory(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze and optimize model memory usage.

        Args:
            model: PyTorch model to optimize

        Returns:
            Dictionary with optimization results and recommendations
        """
        results = {
            'parameter_memory_mb': 0,
            'buffer_memory_mb': 0,
            'total_memory_mb': 0,
            'recommendations': []
        }

        # Calculate parameter memory
        param_bytes = sum(
            p.element_size() * p.numel()
            for p in model.parameters()
        )
        results['parameter_memory_mb'] = param_bytes / 1024**2

        # Calculate buffer memory
        buffer_bytes = sum(
            b.element_size() * b.numel()
            for b in model.buffers()
        )
        results['buffer_memory_mb'] = buffer_bytes / 1024**2

        results['total_memory_mb'] = results['parameter_memory_mb'] + results['buffer_memory_mb']

        # Provide recommendations
        if results['total_memory_mb'] > 1000:  # > 1GB
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

        # Check for inefficient layer sizes
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                optimal_div = 16 if self.nvidia_config.tensor_core_version >= 4 else 8

                if in_features % optimal_div != 0 or out_features % optimal_div != 0:
                    results['recommendations'].append({
                        'type': 'dimension_padding',
                        'layer': name,
                        'current': f"{in_features}x{out_features}",
                        'reason': f'Not divisible by {optimal_div} (optimal for Tensor Cores)',
                        'expected_benefit': 'Better Tensor Core utilization'
                    })

        return results

    def enable_memory_efficient_mode(self) -> None:
        """Enable memory-efficient execution mode."""
        if torch.cuda.is_available():
            # Enable memory-efficient settings
            torch.backends.cudnn.benchmark = False  # Disable for consistent memory
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for memory efficiency

            # Set memory fraction if configured
            if self.nvidia_config.memory_fraction < 1.0:
                # PyTorch doesn't have direct memory fraction setting like TensorFlow
                # but we can use environment variable
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:512'

    def reset_peak_memory(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        """Empty CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
