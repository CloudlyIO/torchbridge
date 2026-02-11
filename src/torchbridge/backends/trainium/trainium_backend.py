"""
Trainium Backend Implementation

Core Trainium backend that provides device management, model preparation,
and integration with PyTorch/XLA via the Neuron SDK for AWS Trainium chips.

Inherits from BaseBackend to provide a consistent interface across all
hardware backends.

The current integration (Neuron SDK 2.27, PyTorch 2.9) uses XLA under the
hood (same device mechanism as TPU, with torch.device('xla')). Trainium is
differentiated from TPU via environment variables and torch_neuronx imports.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torchbridge.backends.base_backend import (
    BaseBackend,
    DeviceInfo,
    OptimizationLevel,
)
from torchbridge.core.config import (
    TorchBridgeConfig,
    TrainiumArchitecture,
)
from torchbridge.utils.cache import LRUCache

from . import neuron_utilities

logger = logging.getLogger(__name__)


class TrainiumBackend(BaseBackend):
    """
    Core Trainium backend for PyTorch/Neuron SDK integration.

    Provides Trainium device management, model preparation, and optimization
    specifically designed for AWS Trainium (Trn1/Trn2/Trn3) and Inferentia2
    deployments.

    Inherits from BaseBackend to provide a unified interface while maintaining
    backward compatibility with existing Trainium-specific APIs.
    """

    # Backend identifier
    BACKEND_NAME: str = "trainium"

    def __init__(self, config: TorchBridgeConfig | None = None):
        """
        Initialize Trainium backend.

        Args:
            config: Optional configuration. If None, creates default config.
        """
        self._full_config = config or TorchBridgeConfig()
        self.trainium_config = self._full_config.hardware.trainium

        # Initialize XLA/Neuron environment
        self._xla_device: torch.device | None = None
        self._world_size = 1
        self._rank = 0

        # Call parent init (which calls _setup_environment)
        super().__init__(config=self._full_config)

        # Alias for backward compatibility
        self.config = self._full_config

        # Performance tracking with LRU caches
        self._model_cache = LRUCache(max_size=self.trainium_config.cache_max_size)
        self._compilation_cache = LRUCache(max_size=self.trainium_config.cache_max_size)

    def _setup_environment(self) -> None:
        """Set up Neuron SDK/XLA environment (implements BaseBackend abstract method)."""
        self._setup_neuron_environment()

    def _check_availability(self) -> bool:
        """Check if Neuron/Trainium is available (implements BaseBackend abstract method)."""
        try:
            import torch_neuronx  # noqa: F401
            return self._xla_device is not None and str(self._xla_device) != 'cpu'
        except ImportError:
            return False

    def _get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Get Trainium device info (implements BaseBackend abstract method)."""
        try:
            device_count = neuron_utilities.get_device_count()
            if device_id >= device_count:
                return DeviceInfo(
                    backend="trainium",
                    device_type="cpu",
                    device_id=0,
                    device_name="CPU (Neuron fallback)",
                    is_available=False
                )

            arch = self.trainium_config.architecture
            return DeviceInfo(
                backend="trainium",
                device_type=f"xla:{device_id}",
                device_id=device_id,
                device_name=f"Trainium {arch.value}",
                compute_capability=arch.value,
                total_memory_bytes=self._estimate_trainium_memory(),
                is_available=True,
                properties={
                    'architecture': arch.value,
                    'world_size': self._world_size,
                    'rank': self._rank,
                    'neuron_sdk_version': neuron_utilities.get_neuron_sdk_version(),
                    'instance_type': neuron_utilities.detect_instance_type(),
                }
            )
        except Exception:
            return DeviceInfo(
                backend="trainium",
                device_type="cpu",
                device_id=0,
                device_name="CPU (Neuron fallback)",
                is_available=False
            )

    def _estimate_trainium_memory(self) -> int:
        """Estimate Trainium HBM based on architecture."""
        memory_map = {
            TrainiumArchitecture.TRN1: 32 * (1024**3),    # 32GB HBM
            TrainiumArchitecture.TRN2: 96 * (1024**3),    # 96GB HBM
            TrainiumArchitecture.TRN3: 144 * (1024**3),   # 144GB HBM3e
            TrainiumArchitecture.INF2: 32 * (1024**3),    # 32GB HBM
        }
        return memory_map.get(self.trainium_config.architecture, 32 * (1024**3))

    def _setup_neuron_environment(self) -> None:
        """Set up Neuron SDK/XLA environment for Trainium."""
        try:
            import torch_neuronx  # noqa: F401
            import torch_xla  # noqa: F401

            # Get XLA device via Neuron utilities
            self._xla_device = neuron_utilities.get_xla_device()
            self._world_size = neuron_utilities.get_world_size()
            self._rank = neuron_utilities.get_ordinal()

            # Set up Neuron-specific environment variables
            self._configure_neuron_flags()

            logger.info(
                "Trainium Backend initialized: device=%s, world_size=%d, rank=%d, architecture=%s",
                self._xla_device,
                self._world_size,
                self._rank,
                self.trainium_config.architecture.value
            )

        except ImportError:
            warnings.warn(
                "Neuron SDK not available. Trainium backend will use CPU fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._xla_device = torch.device("cpu")

    def _configure_neuron_flags(self) -> None:
        """Configure Neuron-specific environment flags."""
        import os

        # Set PJRT_DEVICE if not already set
        os.environ.setdefault('PJRT_DEVICE', 'NEURON')

        # Set Neuron compiler flags
        if self.trainium_config.neuron_cc_flags:
            existing = os.environ.get('NEURON_CC_FLAGS', '')
            if existing:
                os.environ['NEURON_CC_FLAGS'] = f"{existing} {self.trainium_config.neuron_cc_flags}"
            else:
                os.environ['NEURON_CC_FLAGS'] = self.trainium_config.neuron_cc_flags

        # Enable graph caching
        if self.trainium_config.enable_graph_caching:
            os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

    @property
    def device(self) -> torch.device:
        """Get the Trainium XLA device."""
        return self._xla_device  # type: ignore[return-value]

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed Trainium setup."""
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
        optimization_level: str | OptimizationLevel | None = None
    ) -> nn.Module:
        """
        Prepare a PyTorch model for Trainium execution.

        Args:
            model: PyTorch model to prepare
            optimization_level: Optional optimization level

        Returns:
            Model prepared for Trainium execution
        """
        model_id = id(model)
        cached_model = self._model_cache.get(model_id)
        if cached_model is not None:
            return cached_model

        # Move model to Trainium device
        model = model.to(self.device)

        # Apply Trainium-specific optimizations
        if optimization_level != OptimizationLevel.O0 and optimization_level != "O0":
            model = self._apply_trainium_optimizations(model)

        # Cache the prepared model
        self._model_cache.set(model_id, model)

        return model

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module:
        """
        Optimize a model for inference on Trainium.

        Args:
            model: PyTorch model
            sample_input: Optional sample input for tracing
            dtype: Optional dtype for precision

        Returns:
            Inference-optimized model
        """
        model = self.prepare_model(model, optimization_level=OptimizationLevel.O2)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        self.synchronize()
        return model

    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module | tuple[nn.Module, torch.optim.Optimizer]:
        """
        Optimize a model for training on Trainium.

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
        """Get the number of available NeuronCores."""
        try:
            return neuron_utilities.get_device_count()
        except Exception:
            return 0

    def _apply_trainium_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply Trainium-specific model optimizations."""

        # Enable gradient checkpointing if configured
        if self.trainium_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        # Apply mixed precision if enabled
        if self.trainium_config.mixed_precision:
            model = self._enable_mixed_precision(model)

        return model

    def _enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision for Trainium.

        Trainium natively supports BF16. The Neuron SDK handles
        mixed precision through XLA, requiring consistent dtype.
        """
        if self.trainium_config.precision == "bfloat16":
            model = model.to(dtype=torch.bfloat16)
        return model

    def prepare_data(self, data: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Prepare data for Trainium execution.

        Args:
            data: Input data (tensor or dict of tensors)

        Returns:
            Data prepared for Trainium (moved to device and dtype-converted if needed)
        """
        target_dtype = torch.bfloat16 if self.trainium_config.precision == "bfloat16" and self.trainium_config.mixed_precision else None

        if isinstance(data, torch.Tensor):
            result = data.to(self.device)
            if target_dtype and result.is_floating_point() and result.dtype == torch.float32:
                result = result.to(dtype=target_dtype)
            return result
        elif hasattr(data, 'items'):
            result: dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    moved = value.to(self.device)
                    if target_dtype and moved.is_floating_point() and moved.dtype == torch.float32:
                        moved = moved.to(dtype=target_dtype)
                    result[key] = moved
                else:
                    result[key] = value
            return result
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def synchronize(self) -> None:
        """Synchronize Trainium/XLA operations."""
        try:
            neuron_utilities.sync()
            if self.is_distributed:
                neuron_utilities.rendezvous('sync')
        except Exception:
            pass

    def get_memory_stats(self) -> dict[str, Any]:
        """Get Trainium memory statistics."""
        stats = {
            'device': str(self.device),
            'world_size': self.world_size,
            'rank': self.rank,
            'memory_fraction': self.trainium_config.memory_fraction,
            'model_cache_stats': self._model_cache.get_stats(),
            'compilation_cache_stats': self._compilation_cache.get_stats()
        }

        try:
            stats['xla_device_count'] = neuron_utilities.get_device_count()
        except Exception:
            pass

        return stats

    def clear_cache(self) -> None:
        """Clear model and compilation caches."""
        self._model_cache.clear()
        self._compilation_cache.clear()

        try:
            neuron_utilities.sync()
        except Exception:
            pass

    def save_model(self, model: nn.Module, path: str | Path,
                   save_optimizer: bool = False, optimizer: torch.optim.Optimizer | None = None) -> None:
        """
        Save Trainium model with proper state synchronization.

        Args:
            model: Model to save
            path: Save path
            save_optimizer: Whether to save optimizer state
            optimizer: Optimizer to save (if save_optimizer=True)
        """
        self.synchronize()

        if self.rank == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'trainium_config': self.trainium_config.__dict__,
                'world_size': self.world_size
            }

            if save_optimizer and optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(checkpoint, path)

    def load_model(self, model: nn.Module, path: str | Path,
                   load_optimizer: bool = False, optimizer: torch.optim.Optimizer | None = None) -> nn.Module:
        """
        Load Trainium model with proper device placement.

        Args:
            model: Model to load into
            path: Load path
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer to load into (if load_optimizer=True)

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model

    def __repr__(self) -> str:
        """String representation of Trainium backend."""
        return (
            f"TrainiumBackend(device={self.device}, "
            f"architecture={self.trainium_config.architecture.value}, "
            f"world_size={self.world_size})"
        )
