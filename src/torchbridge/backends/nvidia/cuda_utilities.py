"""
CUDA Utilities and Optimizations

General CUDA utilities and helper functions for NVIDIA GPU optimization.
"""

import logging
import os
import subprocess
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from torchbridge.core.config import NVIDIAArchitecture, TorchBridgeConfig

logger = logging.getLogger(__name__)


class CUDADeviceManager:
    """
    CUDA device management and coordination.

    Provides device information, multi-GPU coordination, and
    CUDA environment setup.
    """

    def __init__(self, config: TorchBridgeConfig | None = None):
        """
        Initialize CUDA device manager.

        Args:
            config: TorchBridge configuration
        """
        self.config = config or TorchBridgeConfig()
        self.nvidia_config = self.config.hardware.nvidia

        self._devices = []
        self._current_device = None
        self._device_properties = {}

        self._setup_cuda_devices()

    def _setup_cuda_devices(self) -> None:
        """Set up CUDA devices."""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available", stacklevel=2)
            self._current_device = torch.device("cpu")
            return

        device_count = torch.cuda.device_count()
        self._devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
        self._current_device = self._devices[0] if self._devices else torch.device("cpu")

        # Collect device properties
        for i, _device in enumerate(self._devices):
            props = torch.cuda.get_device_properties(i)
            self._device_properties[i] = {
                'name': props.name,
                'compute_capability': (props.major, props.minor),
                'total_memory_gb': props.total_memory / 1024**3,
                'multi_processor_count': props.multi_processor_count,
                # These attributes may not exist in all PyTorch versions
                'max_threads_per_block': getattr(props, 'max_threads_per_block', 1024),
                'max_shared_memory_per_block': getattr(props, 'max_shared_memory_per_block', 49152),
            }

        logger.info("CUDA Device Manager initialized: num_devices=%d", device_count)
        for i, props in self._device_properties.items():
            logger.debug("  Device %d: %s (CC %s, %.1f GB)",
                        i, props['name'], props['compute_capability'],
                        props['total_memory_gb'])

    @property
    def device(self) -> torch.device:
        """Get current CUDA device."""
        return self._current_device

    @property
    def devices(self) -> list[torch.device]:
        """Get all available CUDA devices."""
        return self._devices

    @property
    def device_count(self) -> int:
        """Get number of available CUDA devices."""
        return len(self._devices)

    def get_device_properties(self, device_id: int = 0) -> dict[str, Any]:
        """Get properties for specific device."""
        return self._device_properties.get(device_id, {})

    def get_all_device_properties(self) -> dict[int, dict[str, Any]]:
        """Get properties for all devices."""
        return self._device_properties

    def set_device(self, device_id: int) -> None:
        """Set current CUDA device."""
        if device_id < len(self._devices):
            self._current_device = self._devices[device_id]
            torch.cuda.set_device(device_id)
        else:
            warnings.warn(f"Device {device_id} not available", stacklevel=2)

    def synchronize_all(self) -> None:
        """Synchronize all CUDA devices."""
        for i in range(len(self._devices)):
            with torch.cuda.device(i):
                torch.cuda.synchronize()


class CUDAOptimizations:
    """
    CUDA-specific optimization utilities.

    Provides optimization hints, kernel configuration,
    and performance tuning for CUDA execution.
    """

    def __init__(self, config: TorchBridgeConfig | None = None):
        """
        Initialize CUDA optimizations.

        Args:
            config: TorchBridge configuration
        """
        self.config = config or TorchBridgeConfig()
        self.nvidia_config = self.config.hardware.nvidia

    def optimize_model_for_cuda(self, model: nn.Module) -> nn.Module:
        """
        Apply CUDA-specific optimizations to model.

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        # Optimize layer dimensions for Tensor Cores
        model = self._optimize_layer_dimensions(model)

        # Add CUDA kernel fusion hints
        model = self._add_cuda_fusion_hints(model)

        # Optimize memory layout
        model = self._optimize_memory_layout(model)

        return model

    def _optimize_layer_dimensions(self, model: nn.Module) -> nn.Module:
        """Optimize layer dimensions for Tensor Cores."""
        optimal_div = 16 if self.nvidia_config.tensor_core_version >= 4 else 8

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_f, out_f = module.in_features, module.out_features

                if in_f % optimal_div != 0 or out_f % optimal_div != 0:
                    warnings.warn(
                        f"Layer {name} dimensions ({in_f}x{out_f}) not optimal for "
                        f"Tensor Cores. Consider padding to multiples of {optimal_div}.",
                    stacklevel=2,
                    )

        return model

    def _add_cuda_fusion_hints(self, model: nn.Module) -> nn.Module:
        """Add hints for CUDA kernel fusion."""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                module._cuda_fusible = True
                module._cuda_fusion_priority = 1

        return model

    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimize memory layout for CUDA."""
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                # Use channels_last for better memory access patterns
                try:
                    if hasattr(module, 'to_memory_format'):
                        module.to(memory_format=torch.channels_last)
                except (RuntimeError, TypeError) as e:
                    logger.debug("Could not convert to channels_last: %s", e)

        return model

    def get_cuda_optimization_config(self) -> dict[str, Any]:
        """Get recommended CUDA optimization configuration."""
        config = {
            'cudnn_benchmark': self.nvidia_config.cudnn_benchmark,
            'tf32_enabled': self.nvidia_config.tensor_core_version >= 3,
            'allow_fp16_reduction': True,
            'optimal_dimension_divisor': 16 if self.nvidia_config.tensor_core_version >= 4 else 8,
        }

        if self.nvidia_config.architecture in [NVIDIAArchitecture.HOPPER, NVIDIAArchitecture.BLACKWELL]:
            config.update({
                'fp8_enabled': True,
                'flash_attention_3': True,
                'tensor_core_version': 4,
            })

        return config


class CUDAUtilities:
    """
    General CUDA utility functions.

    Provides profiling, debugging, and utility functions
    for CUDA development.
    """

    @staticmethod
    def get_cuda_env_info() -> dict[str, Any]:
        """Get CUDA environment information."""
        env_info = {
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            env_info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
            })

            # Get compute capability
            props = torch.cuda.get_device_properties(0)
            env_info['compute_capability'] = (props.major, props.minor)

            # Environment variables
            env_info['env_vars'] = {
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
                'CUDA_LAUNCH_BLOCKING': os.environ.get('CUDA_LAUNCH_BLOCKING', 'not set'),
            }

        return env_info

    @staticmethod
    def profile_cuda_kernel(
        func: Callable,  # type: ignore[type-arg]
        *args,
        num_warmup: int = 5,
        num_iterations: int = 100,
        **kwargs
    ) -> dict[str, Any]:
        """
        Profile CUDA kernel performance.

        Args:
            func: Function to profile
            args: Function arguments
            num_warmup: Number of warmup iterations
            num_iterations: Number of profiling iterations
            kwargs: Function keyword arguments

        Returns:
            Profiling results
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}


        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = func(*args, **kwargs)
        torch.cuda.synchronize()

        # Profile
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_iterations

        return {
            'total_time_ms': elapsed_time_ms,
            'avg_time_ms': avg_time_ms,
            'iterations': num_iterations,
            'throughput_ops_per_sec': 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
        }

    @staticmethod
    def get_gpu_utilization() -> dict[str, Any]:
        """Get current GPU utilization."""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        try:
            # Try to use nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization_percent': float(values[0]),
                    'memory_utilization_percent': float(values[1]),
                    'memory_used_mb': float(values[2]),
                    'memory_total_mb': float(values[3]),
                }

        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            # nvidia-smi not available or failed, fall back to PyTorch metrics
            logger.debug("nvidia-smi failed, using PyTorch metrics: %s", e)

        # Fallback to PyTorch metrics
        return {
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }

    @staticmethod
    def optimize_cuda_flags(architecture: NVIDIAArchitecture) -> dict[str, str]:
        """Get optimized CUDA flags for specific architecture."""
        flags = {
            'CUDA_LAUNCH_BLOCKING': '0',  # Async kernel launches
        }

        if architecture in [NVIDIAArchitecture.HOPPER, NVIDIAArchitecture.BLACKWELL]:
            # H100/Blackwell optimizations
            flags.update({
                'CUDA_DEVICE_MAX_CONNECTIONS': '32',  # More concurrent streams
            })

        elif architecture == NVIDIAArchitecture.AMPERE:
            # A100 optimizations
            flags.update({
                'CUDA_DEVICE_MAX_CONNECTIONS': '16',
            })

        return flags


# Integration factory function
def create_cuda_integration(
    config: TorchBridgeConfig | None = None
) -> tuple[CUDADeviceManager, CUDAOptimizations]:
    """
    Create complete CUDA integration setup.

    Args:
        config: TorchBridge configuration

    Returns:
        Tuple of (device_manager, optimizations)
    """
    device_manager = CUDADeviceManager(config)
    optimizations = CUDAOptimizations(config)

    return device_manager, optimizations
