"""
Intel XPU Backend Implementation

Core backend for Intel XPU device management and model preparation,
using Intel Extension for PyTorch (IPEX) for optimizations.
"""

import logging
import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .intel_exceptions import (
    XPUNotAvailableError,
    IPEXNotInstalledError,
    XPUDeviceError,
)
from .xpu_utilities import (
    XPUDeviceManager,
    XPUOptimizations,
    XPU_AVAILABLE,
    IPEX_AVAILABLE,
    is_xpu_available,
    is_ipex_available,
    get_ipex_version,
)
from .memory_manager import IntelMemoryManager

logger = logging.getLogger(__name__)


class IntelBackend:
    """
    Core Intel XPU backend for device management and model preparation.

    Provides device coordination, memory management, and model optimization
    for Intel XPU devices (Ponte Vecchio, Arc, Data Center Max, etc.).
    """

    def __init__(self, config=None):
        """
        Initialize Intel XPU backend.

        Args:
            config: KernelPyTorchConfig configuration with Intel settings
        """
        self.config = config
        self._intel_config = None
        if config is not None:
            self._intel_config = getattr(config.hardware, 'intel', None)

        self._device = None
        self._devices = []
        self._device_name = None
        self._device_type = None

        # Initialize device manager
        self._device_manager = XPUDeviceManager()

        # Initialize optimizations helper
        self._optimizations = XPUOptimizations(self._device_manager)

        # Initialize memory manager
        self._memory_manager = None

        self._setup_xpu_environment()

    def _setup_xpu_environment(self) -> None:
        """Set up XPU environment for Intel devices."""
        if not XPU_AVAILABLE:
            warnings.warn("Intel XPU not available. Using CPU fallback.")
            self._device = torch.device("cpu")
            return

        if not IPEX_AVAILABLE:
            warnings.warn(
                "Intel Extension for PyTorch (IPEX) not installed. "
                "Install with: pip install intel-extension-for-pytorch"
            )

        # Get device information
        device_count = self._device_manager.device_count
        if device_count > 0:
            self._devices = [torch.device(f"xpu:{i}") for i in range(device_count)]
            self._device = self._devices[0]

            # Get device info
            device_info = self._device_manager.get_device_info(0)
            self._device_name = device_info.name
            self._device_type = device_info.device_type

            # Initialize memory manager
            self._memory_manager = IntelMemoryManager(
                config=self._intel_config,
                device_id=0
            )

            # Apply XPU-specific optimizations
            self._apply_xpu_environment_settings()

            logger.info(
                "Intel Backend initialized: device=%s, type=%s, "
                "num_devices=%d, ipex_version=%s",
                self._device_name,
                self._device_type,
                len(self._devices),
                get_ipex_version() or "not installed"
            )
        else:
            self._device = torch.device("cpu")
            logger.warning("No Intel XPU devices found, falling back to CPU")

    def _apply_xpu_environment_settings(self) -> None:
        """Apply XPU-specific environment settings."""
        if not IPEX_AVAILABLE:
            return

        try:
            import intel_extension_for_pytorch as ipex

            # Enable oneDNN fusion for better performance
            if hasattr(ipex, 'enable_onednn_fusion'):
                ipex.enable_onednn_fusion(True)
                logger.debug("oneDNN fusion enabled")

            # Set optimal environment variables
            import os
            if 'IPEX_FP32_MATH_MODE' not in os.environ:
                # Use TF32 for better performance (similar to CUDA TF32)
                os.environ['IPEX_FP32_MATH_MODE'] = 'TF32'

        except Exception as e:
            logger.warning(f"Failed to apply XPU environment settings: {e}")

    @property
    def device(self) -> torch.device:
        """Get current XPU device."""
        return self._device

    @property
    def devices(self) -> List[torch.device]:
        """Get all available XPU devices."""
        return self._devices

    @property
    def device_name(self) -> Optional[str]:
        """Get current device name."""
        return self._device_name

    @property
    def device_type(self) -> Optional[str]:
        """Get current device type (data_center, consumer, integrated)."""
        return self._device_type

    @property
    def is_xpu_available(self) -> bool:
        """Check if XPU is available."""
        return self._device.type == "xpu"

    @property
    def is_data_center(self) -> bool:
        """Check if running on data center GPU (Ponte Vecchio/Max)."""
        return self._device_type == "data_center"

    @property
    def is_consumer(self) -> bool:
        """Check if running on consumer GPU (Arc)."""
        return self._device_type == "consumer"

    @property
    def supports_bf16(self) -> bool:
        """Check if device supports BF16."""
        if not self.is_xpu_available:
            return False
        device_info = self._device_manager.get_device_info()
        return device_info.supports_bf16

    @property
    def supports_amx(self) -> bool:
        """Check if device supports Intel AMX (Advanced Matrix Extensions)."""
        if not self.is_xpu_available:
            return False
        device_info = self._device_manager.get_device_info()
        return device_info.supports_amx

    @property
    def memory_manager(self) -> Optional[IntelMemoryManager]:
        """Get memory manager instance."""
        return self._memory_manager

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for Intel XPU execution.

        Args:
            model: PyTorch model to prepare

        Returns:
            Prepared model ready for Intel XPU execution
        """
        if model is None:
            warnings.warn("Model is None, returning unchanged")
            return model

        if not self.is_xpu_available:
            warnings.warn("XPU not available, returning model unchanged")
            return model

        # Move model to device
        model = model.to(self.device)

        # Apply Intel-specific optimizations
        model = self._apply_intel_optimizations(model)

        return model

    def _apply_intel_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply Intel-specific optimizations to model."""
        # Optimize for Vector Engine (similar to Tensor Cores)
        model = self._optimize_for_vector_engine(model)

        # Apply memory layout optimizations
        model = self._optimize_memory_layout(model)

        return model

    def _optimize_for_vector_engine(self, model: nn.Module) -> nn.Module:
        """Optimize model for Intel Vector Engine execution."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if dimensions are optimal for Vector Engine
                in_features, out_features = module.in_features, module.out_features

                # Intel XPU works best with dimensions divisible by 16
                optimal_divisor = 16

                if in_features % optimal_divisor != 0 or out_features % optimal_divisor != 0:
                    logger.debug(
                        f"Linear layer {name} dimensions ({in_features}x{out_features}) "
                        f"not optimal for XPU Vector Engine. Consider padding to multiples of {optimal_divisor}."
                    )

        return model

    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimize memory layout for XPU execution."""
        # Use channels_last format for convolutions when beneficial
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                try:
                    module.to(memory_format=torch.channels_last)
                except (RuntimeError, TypeError) as e:
                    logger.debug(f"Could not convert to channels_last: {e}")

        return model

    def optimize_for_inference(
        self,
        model: nn.Module,
        dtype: Optional[torch.dtype] = None,
        sample_input: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Optimize model for inference using IPEX.

        Args:
            model: Model to optimize
            dtype: Data type to use (auto-detected if None)
            sample_input: Optional sample input for tracing

        Returns:
            Optimized model
        """
        if dtype is None:
            dtype = self._optimizations.get_optimal_dtype()

        return self._optimizations.optimize_model_for_inference(
            model, dtype=dtype, sample_input=sample_input
        )

    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dtype: Optional[torch.dtype] = None
    ) -> tuple:
        """
        Optimize model and optimizer for training using IPEX.

        Args:
            model: Model to optimize
            optimizer: Optimizer to optimize
            dtype: Data type to use

        Returns:
            Tuple of (optimized_model, optimized_optimizer)
        """
        if dtype is None:
            dtype = self._optimizations.get_optimal_dtype()

        return self._optimizations.optimize_model_for_training(
            model, optimizer, dtype=dtype
        )

    def synchronize(self) -> None:
        """Synchronize XPU device."""
        self._device_manager.synchronize()

    def empty_cache(self) -> None:
        """Empty XPU memory cache."""
        self._device_manager.empty_cache()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get XPU memory statistics."""
        if not self.is_xpu_available:
            return {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0,
                'device': 'cpu'
            }

        if self._memory_manager:
            stats = self._memory_manager.get_memory_stats()
            return {
                'allocated': stats.allocated_mb / 1024,  # GB
                'reserved': stats.reserved_mb / 1024,    # GB
                'total': stats.total_mb / 1024,          # GB
                'free': stats.free_mb / 1024,            # GB
                'peak_allocated': stats.peak_allocated_bytes / (1024**3),  # GB
                'utilization': stats.utilization,
                'device': str(self.device),
                'device_name': self._device_name,
            }

        return {
            'allocated': 0,
            'reserved': 0,
            'device': str(self.device),
            'device_name': self._device_name,
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'backend': 'intel',
            'xpu_available': self.is_xpu_available,
            'ipex_available': IPEX_AVAILABLE,
            'ipex_version': get_ipex_version(),
            'device': str(self.device),
            'device_count': len(self._devices),
        }

        if self.is_xpu_available:
            device_info = self._device_manager.get_device_info()
            info.update({
                'device_name': device_info.name,
                'device_type': device_info.device_type,
                'total_memory_gb': device_info.total_memory / (1024**3),
                'driver_version': device_info.driver_version,
                'supports_amx': device_info.supports_amx,
                'supports_fp16': device_info.supports_fp16,
                'supports_bf16': device_info.supports_bf16,
                'max_compute_units': device_info.max_compute_units,
            })

        return info

    def set_device(self, device_id: int) -> None:
        """Set current XPU device."""
        if device_id < len(self._devices):
            self._device_manager.set_device(device_id)
            self._device = self._devices[device_id]

            # Update device info
            device_info = self._device_manager.get_device_info(device_id)
            self._device_name = device_info.name
            self._device_type = device_info.device_type

            # Reinitialize memory manager for new device
            self._memory_manager = IntelMemoryManager(
                config=self._intel_config,
                device_id=device_id
            )
        else:
            warnings.warn(f"Device {device_id} not available, using current device")

    def get_memory_summary(self) -> str:
        """Get a human-readable memory summary."""
        if self._memory_manager:
            return self._memory_manager.get_memory_summary()
        return "Memory manager not initialized"


__all__ = [
    'IntelBackend',
]
