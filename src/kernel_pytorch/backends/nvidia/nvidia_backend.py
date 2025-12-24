"""
NVIDIA Backend Implementation

Core backend for NVIDIA GPU device management and model preparation.
"""

import warnings
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import os

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAConfig, NVIDIAArchitecture


class NVIDIABackend:
    """
    Core NVIDIA GPU backend for device management and model preparation.

    Provides device coordination, memory management, and model optimization
    for NVIDIA GPUs (H100, Blackwell, Ampere, etc.).
    """

    def __init__(self, config: Optional[KernelPyTorchConfig] = None):
        """
        Initialize NVIDIA backend.

        Args:
            config: KernelPyTorch configuration with NVIDIA settings
        """
        self.config = config or KernelPyTorchConfig()
        self.nvidia_config = self.config.hardware.nvidia

        self._device = None
        self._devices = []
        self._compute_capability = None
        self._device_name = None

        self._setup_cuda_environment()

    def _setup_cuda_environment(self) -> None:
        """Set up CUDA environment for NVIDIA GPUs."""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available. Using CPU fallback.")
            self._device = torch.device("cpu")
            return

        # Get device information
        device_count = torch.cuda.device_count()
        self._devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
        self._device = self._devices[0] if self._devices else torch.device("cpu")

        if self._device.type == "cuda":
            # Get compute capability
            props = torch.cuda.get_device_properties(0)
            self._compute_capability = (props.major, props.minor)
            self._device_name = props.name

            # Set up CUDA optimization flags
            if self.nvidia_config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

            # Enable TF32 for Ampere and newer
            if self._compute_capability[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            print(f"🔧 NVIDIA Backend initialized:")
            print(f"   Device: {self._device_name}")
            print(f"   Compute Capability: {self._compute_capability}")
            print(f"   Available devices: {len(self._devices)}")
            print(f"   Architecture: {self.nvidia_config.architecture.value}")
            print(f"   FP8 enabled: {self.nvidia_config.fp8_enabled}")

    @property
    def device(self) -> torch.device:
        """Get current CUDA device."""
        return self._device

    @property
    def devices(self) -> List[torch.device]:
        """Get all available CUDA devices."""
        return self._devices

    @property
    def compute_capability(self) -> Optional[tuple]:
        """Get CUDA compute capability."""
        return self._compute_capability

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._device.type == "cuda"

    @property
    def is_h100(self) -> bool:
        """Check if running on H100 GPU."""
        return self.nvidia_config.architecture == NVIDIAArchitecture.HOPPER

    @property
    def is_blackwell(self) -> bool:
        """Check if running on Blackwell GPU."""
        return self.nvidia_config.architecture == NVIDIAArchitecture.BLACKWELL

    @property
    def supports_fp8(self) -> bool:
        """Check if GPU supports FP8."""
        return self.nvidia_config.architecture in [
            NVIDIAArchitecture.HOPPER,
            NVIDIAArchitecture.BLACKWELL
        ]

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for NVIDIA GPU execution.

        Args:
            model: PyTorch model to prepare

        Returns:
            Prepared model ready for NVIDIA GPU execution
        """
        if not self.is_cuda_available:
            warnings.warn("CUDA not available, returning model unchanged")
            return model

        # Move model to device
        model = model.to(self.device)

        # Apply NVIDIA-specific optimizations
        model = self._apply_nvidia_optimizations(model)

        # Enable gradient checkpointing if configured
        if self.config.memory.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model)

        return model

    def _apply_nvidia_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply NVIDIA-specific optimizations to model."""
        # Apply Tensor Core optimizations
        model = self._optimize_for_tensor_cores(model)

        # Apply memory optimizations
        model = self._optimize_memory_layout(model)

        # Apply kernel fusion hints
        if self.nvidia_config.kernel_fusion_enabled:
            model = self._add_fusion_hints(model)

        return model

    def _optimize_for_tensor_cores(self, model: nn.Module) -> nn.Module:
        """Optimize model for Tensor Core execution."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if dimensions are optimal for Tensor Cores
                in_features, out_features = module.in_features, module.out_features

                # Tensor Cores work best with dimensions divisible by 8 (or 16 for newer)
                optimal_divisor = 16 if self._compute_capability and self._compute_capability[0] >= 8 else 8

                if in_features % optimal_divisor != 0 or out_features % optimal_divisor != 0:
                    warnings.warn(
                        f"Linear layer {name} dimensions ({in_features}x{out_features}) "
                        f"not optimal for Tensor Cores. Consider padding to multiples of {optimal_divisor}."
                    )

        return model

    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimize memory layout for CUDA execution."""
        # Ensure model uses channels_last memory format for CNNs if beneficial
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                # channels_last can be beneficial for convolutions
                if hasattr(module, 'to_memory_format'):
                    try:
                        module.to(memory_format=torch.channels_last)
                    except Exception:
                        pass  # Not all modules support channels_last

        return model

    def _add_fusion_hints(self, model: nn.Module) -> nn.Module:
        """Add kernel fusion hints for CUDA compiler."""
        # Mark modules for potential fusion
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Add metadata for torch.compile
                setattr(module, '_cuda_fusible', True)

        return model

    def _enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        return model

    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        if self.is_cuda_available:
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        """Empty CUDA cache."""
        if self.is_cuda_available:
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get CUDA memory statistics."""
        if not self.is_cuda_available:
            return {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0,
                'device': 'cpu'
            }

        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            'device': str(self.device),
            'device_name': self._device_name,
            'compute_capability': self._compute_capability
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'backend': 'nvidia',
            'cuda_available': self.is_cuda_available,
            'device': str(self.device),
            'device_count': len(self._devices),
        }

        if self.is_cuda_available:
            info.update({
                'device_name': self._device_name,
                'compute_capability': self._compute_capability,
                'architecture': self.nvidia_config.architecture.value,
                'fp8_supported': self.supports_fp8,
                'fp8_enabled': self.nvidia_config.fp8_enabled,
                'tensor_core_version': self.nvidia_config.tensor_core_version,
                'flash_attention_enabled': self.nvidia_config.flash_attention_enabled,
                'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            })

        return info

    def set_device(self, device_id: int) -> None:
        """Set current CUDA device."""
        if device_id < len(self._devices):
            self._device = self._devices[device_id]
            if self.is_cuda_available:
                torch.cuda.set_device(device_id)
        else:
            warnings.warn(f"Device {device_id} not available, using default device")

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.is_cuda_available:
            torch.cuda.reset_peak_memory_stats()
