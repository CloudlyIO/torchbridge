"""
TPU Backend Implementation

Core TPU backend that provides device management, model preparation,
and integration with PyTorch/XLA for Google Cloud TPUs.

Inherits from BaseBackend to provide a consistent interface across all
hardware backends.

Version: 0.4.8
"""

import logging
import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from kernel_pytorch.core.config import KernelPyTorchConfig, TPUConfig, TPUVersion, TPUTopology
from kernel_pytorch.backends.base_backend import BaseBackend, DeviceInfo, OptimizationLevel
from .cache_utils import LRUCache
from . import xla_compat

logger = logging.getLogger(__name__)


class TPUBackend(BaseBackend):
    """
    Core TPU backend for PyTorch/XLA integration.

    Provides TPU device management, model preparation, and optimization
    specifically designed for Google Cloud TPU deployments.

    Inherits from BaseBackend to provide a unified interface while maintaining
    backward compatibility with existing TPU-specific APIs.
    """

    # Backend identifier
    BACKEND_NAME: str = "tpu"

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize TPU backend.

        Args:
            config: Optional configuration. If None, creates default TPU config.
        """
        self._full_config = config or KernelPyTorchConfig()
        self.tpu_config = self._full_config.hardware.tpu

        # Initialize XLA environment
        self._xla_device = None
        self._world_size = 1
        self._rank = 0

        # Call parent init (which calls _setup_environment)
        super().__init__(config=self._full_config)

        # Alias for backward compatibility
        self.config = self._full_config

        # Performance tracking with LRU caches
        self._model_cache = LRUCache(max_size=self.tpu_config.cache_max_size)
        self._compilation_cache = LRUCache(max_size=self.tpu_config.cache_max_size)

    def _setup_environment(self) -> None:
        """Set up PyTorch/XLA environment (implements BaseBackend abstract method)."""
        self._setup_xla_environment()

    def _check_availability(self) -> bool:
        """Check if XLA/TPU is available (implements BaseBackend abstract method)."""
        try:
            import torch_xla
            return self._xla_device is not None and str(self._xla_device) != 'cpu'
        except ImportError:
            return False

    def _get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Get TPU device info (implements BaseBackend abstract method)."""
        try:
            device_count = xla_compat.get_device_count()
            if device_id >= device_count:
                return DeviceInfo(
                    backend="tpu",
                    device_type="cpu",
                    device_id=0,
                    device_name="CPU (XLA fallback)",
                    is_available=False
                )

            return DeviceInfo(
                backend="tpu",
                device_type=f"xla:{device_id}",
                device_id=device_id,
                device_name=f"TPU {self.tpu_config.version.value}",
                compute_capability=self.tpu_config.version.value,
                total_memory_bytes=self._estimate_tpu_memory(),
                is_available=True,
                properties={
                    'version': self.tpu_config.version.value,
                    'topology': self.tpu_config.topology.value,
                    'world_size': self._world_size,
                    'rank': self._rank,
                }
            )
        except Exception:
            return DeviceInfo(
                backend="tpu",
                device_type="cpu",
                device_id=0,
                device_name="CPU (XLA fallback)",
                is_available=False
            )

    def _estimate_tpu_memory(self) -> int:
        """Estimate TPU memory based on version."""
        memory_map = {
            TPUVersion.V4: 32 * (1024**3),  # 32GB HBM
            TPUVersion.V5E: 16 * (1024**3),  # 16GB HBM
            TPUVersion.V5P: 95 * (1024**3),  # 95GB HBM
            TPUVersion.V6E: 32 * (1024**3),  # 32GB HBM
            TPUVersion.V7: 256 * (1024**3),  # 256GB HBM (estimated)
        }
        return memory_map.get(self.tpu_config.version, 16 * (1024**3))

    def _setup_xla_environment(self) -> None:
        """Set up PyTorch/XLA environment for TPU."""
        try:
            import torch_xla
            import torch_xla.distributed.xla_backend

            # Get TPU device using compatibility layer
            self._xla_device = xla_compat.get_xla_device()
            self._world_size = xla_compat.get_world_size()
            self._rank = xla_compat.get_ordinal()

            # Set up XLA environment variables
            self._configure_xla_flags()

            logger.info(
                "TPU Backend initialized: device=%s, world_size=%d, rank=%d, version=%s, topology=%s",
                self._xla_device,
                self._world_size,
                self._rank,
                self.tpu_config.version.value,
                self.tpu_config.topology.value
            )

        except ImportError:
            warnings.warn(
                "PyTorch/XLA not available. TPU backend will use CPU fallback.",
                RuntimeWarning
            )
            self._xla_device = torch.device("cpu")

    def _configure_xla_flags(self) -> None:
        """Configure XLA flags for optimal TPU performance."""
        import os

        # Set basic XLA flags
        base_flags = [
            f"--xla_optimization_level={self.tpu_config.xla_optimization_level}",
            "--xla_force_host_platform_device_count=1"
        ]

        # Add dynamic shapes support
        if self.tpu_config.enable_xla_dynamic_shapes:
            base_flags.append("--xla_dynamic_shapes=true")

        # Version-specific optimizations
        if self.tpu_config.version in [TPUVersion.V5P, TPUVersion.V6E, TPUVersion.V7]:
            # High-performance TPUs
            base_flags.extend([
                "--xla_enable_async_collectives=true",
                "--xla_tpu_enable_async_collective_fusion=true",
                "--xla_tpu_enable_async_collective_fusion_multiple_steps=true"
            ])

        # Combine with user-provided flags
        all_flags = base_flags
        if self.tpu_config.xla_flags:
            all_flags.append(self.tpu_config.xla_flags)

        # Set environment variable
        os.environ["XLA_FLAGS"] = " ".join(all_flags)

    @property
    def device(self) -> torch.device:
        """Get the TPU device."""
        return self._xla_device

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed TPU setup."""
        return self._world_size > 1

    @property
    def rank(self) -> int:
        """Get the current process rank."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Get the total number of processes."""
        return self._world_size

    def prepare_model(
        self,
        model: nn.Module,
        optimization_level: Optional[Union[str, OptimizationLevel]] = None
    ) -> nn.Module:
        """
        Prepare a PyTorch model for TPU execution (implements BaseBackend abstract method).

        Args:
            model: PyTorch model to prepare
            optimization_level: Optional optimization level

        Returns:
            Model prepared for TPU execution
        """
        # Check cache
        model_id = id(model)
        cached_model = self._model_cache.get(model_id)
        if cached_model is not None:
            return cached_model

        # Move model to TPU device
        model = model.to(self.device)

        # Apply TPU-specific optimizations if level is not O0
        if optimization_level != OptimizationLevel.O0 and optimization_level != "O0":
            model = self._apply_tpu_optimizations(model)

        # Cache the prepared model
        self._model_cache.set(model_id, model)

        return model

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Module:
        """
        Optimize a model for inference (implements BaseBackend abstract method).

        Args:
            model: PyTorch model
            sample_input: Optional sample input for tracing
            dtype: Optional dtype for precision

        Returns:
            Inference-optimized model
        """
        model = self.prepare_model(model, optimization_level=OptimizationLevel.O2)
        model.eval()

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # Synchronize
        self.synchronize()

        return model

    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Union[nn.Module, Tuple[nn.Module, torch.optim.Optimizer]]:
        """
        Optimize a model for training (implements BaseBackend abstract method).

        Args:
            model: PyTorch model
            optimizer: Optional optimizer to optimize along with model
            dtype: Optional dtype for precision

        Returns:
            Training-optimized model, or tuple of (model, optimizer)
        """
        model = self.prepare_model(model, optimization_level=OptimizationLevel.O1)
        model.train()

        if optimizer:
            return model, optimizer
        return model

    @property
    def device_count(self) -> int:
        """Get the number of available TPU devices (overrides BaseBackend)."""
        try:
            return xla_compat.get_device_count()
        except Exception:
            return 0

    def _apply_tpu_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply TPU-specific model optimizations."""

        # Enable gradient checkpointing if configured
        if self.tpu_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        # Apply mixed precision if enabled
        if self.tpu_config.mixed_precision:
            model = self._enable_mixed_precision(model)

        # Apply model-specific optimizations based on TPU version
        if self.tpu_config.version in [TPUVersion.V5P, TPUVersion.V6E, TPUVersion.V7]:
            # High-performance TPU optimizations
            model = self._apply_high_performance_optimizations(model)

        return model

    def _enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision for TPU.

        TPU requires consistent precision across computation graphs.
        Unlike CUDA's autocast which handles mixed precision automatically,
        TPU/XLA requires explicit dtype management.
        """
        if self.tpu_config.precision == "bfloat16":
            # Convert entire model to bfloat16 for TPU consistency
            # This avoids mixed precision errors in XLA compilation
            model = model.to(dtype=torch.bfloat16)

        return model

    def prepare_data(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prepare data for TPU execution.

        Args:
            data: Input data (tensor or dict of tensors)

        Returns:
            Data prepared for TPU (moved to device and dtype-converted if needed)
        """
        target_dtype = torch.bfloat16 if self.tpu_config.precision == "bfloat16" and self.tpu_config.mixed_precision else None

        if isinstance(data, torch.Tensor):
            result = data.to(self.device)
            # Only convert float32 tensors to target dtype (not int tensors)
            if target_dtype and result.is_floating_point() and result.dtype == torch.float32:
                result = result.to(dtype=target_dtype)
            return result
        elif hasattr(data, 'items'):  # Dict-like objects (including BatchEncoding)
            result = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    moved = value.to(self.device)
                    # Only convert float32 tensors (not int tensors like input_ids)
                    if target_dtype and moved.is_floating_point() and moved.dtype == torch.float32:
                        moved = moved.to(dtype=target_dtype)
                    result[key] = moved
                else:
                    result[key] = value  # Keep non-tensor values as-is
            return result
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _apply_high_performance_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply optimizations for high-performance TPUs (v5p+)."""

        # Enable optimizations specific to newer TPU generations
        try:
            # Sync for efficient compilation using compatibility layer
            xla_compat.sync()

            # Enable faster collective operations for distributed training
            if self.is_distributed:
                self._setup_distributed_optimizations()

        except ImportError:
            pass

        return model

    def _setup_distributed_optimizations(self) -> None:
        """Set up optimizations for distributed TPU training."""
        try:
            import torch_xla.distributed.xla_backend as xla_backend

            # Initialize process group for distributed training
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend='xla',
                    rank=self.rank,
                    world_size=self.world_size
                )

        except ImportError:
            warnings.warn("Distributed training setup failed - XLA backend not available")

    def synchronize(self) -> None:
        """Synchronize TPU operations."""
        try:
            xla_compat.sync()
            if self.is_distributed:
                xla_compat.rendezvous('sync')
        except Exception:
            pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get TPU memory statistics."""
        stats = {
            'device': str(self.device),
            'world_size': self.world_size,
            'rank': self.rank,
            'memory_fraction': self.tpu_config.memory_fraction,
            'model_cache_stats': self._model_cache.get_stats(),
            'compilation_cache_stats': self._compilation_cache.get_stats()
        }

        try:
            stats['xla_device_count'] = xla_compat.get_device_count()
        except Exception:
            pass

        return stats

    def clear_cache(self) -> None:
        """Clear model and compilation caches."""
        self._model_cache.clear()
        self._compilation_cache.clear()

        try:
            # Clear XLA compilation cache
            xla_compat.sync()
        except Exception:
            pass

    def save_model(self, model: nn.Module, path: Union[str, Path],
                   save_optimizer: bool = False, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Save TPU model with proper state synchronization.

        Args:
            model: Model to save
            path: Save path
            save_optimizer: Whether to save optimizer state
            optimizer: Optimizer to save (if save_optimizer=True)
        """
        # Synchronize before saving
        self.synchronize()

        # Only save from rank 0 in distributed setup
        if self.rank == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'tpu_config': self.tpu_config.__dict__,
                'world_size': self.world_size
            }

            if save_optimizer and optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(checkpoint, path)

    def load_model(self, model: nn.Module, path: Union[str, Path],
                   load_optimizer: bool = False, optimizer: Optional[torch.optim.Optimizer] = None) -> nn.Module:
        """
        Load TPU model with proper device placement.

        Args:
            model: Model to load into
            path: Load path
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer to load into (if load_optimizer=True)

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        # Load optimizer state if requested
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model

    def __repr__(self) -> str:
        """String representation of TPU backend."""
        return (
            f"TPUBackend(device={self.device}, "
            f"version={self.tpu_config.version.value}, "
            f"topology={self.tpu_config.topology.value}, "
            f"world_size={self.world_size})"
        )