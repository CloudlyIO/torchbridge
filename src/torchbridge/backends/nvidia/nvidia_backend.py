"""
NVIDIA Backend Implementation

Core backend for NVIDIA GPU device management and model preparation.

Inherits from BaseBackend to provide a consistent interface across all
hardware backends while implementing NVIDIA-specific optimizations.

Version: 0.5.3
"""

import logging
import warnings
from typing import Any

import torch
import torch.nn as nn

from torchbridge.backends.base_backend import (
    BaseBackend,
    DeviceInfo,
    OptimizationLevel,
)
from torchbridge.core.config import (
    NVIDIAArchitecture,
    PrecisionFormat,
    TorchBridgeConfig,
)
from torchbridge.core.kernel_registry import (
    KernelBackend,
    KernelMetadata,
    KernelRegistry,
    KernelType,
)

logger = logging.getLogger(__name__)


class NVIDIABackend(BaseBackend):
    """
    Core NVIDIA GPU backend for device management and model preparation.

    Provides device coordination, memory management, and model optimization
    for NVIDIA GPUs (H100, Blackwell, Ampere, etc.).

    Inherits from BaseBackend to provide a unified interface while maintaining
    backward compatibility with existing NVIDIA-specific APIs.
    """

    # Backend identifier
    BACKEND_NAME: str = "nvidia"

    def __init__(self, config: TorchBridgeConfig | None = None):
        """
        Initialize NVIDIA backend.

        Args:
            config: TorchBridge configuration with NVIDIA settings
        """
        # Store config before calling super().__init__
        self._full_config = config or TorchBridgeConfig()
        self.nvidia_config = self._full_config.hardware.nvidia

        self._devices = []
        self._compute_capability = None
        self._device_name = None

        # Initialize kernel registry
        self._kernel_registry = KernelRegistry()

        # Call parent init (which calls _setup_environment)
        super().__init__(config=self._full_config)

        # Alias for backward compatibility
        self.config = self._full_config

        # Register default kernels after CUDA environment is set up
        if self._full_config.kernel.enabled and self.is_cuda_available:
            self._register_default_kernels()

    def _setup_environment(self) -> None:
        """Set up CUDA environment for NVIDIA GPUs (implements BaseBackend abstract method)."""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available. Using CPU fallback.", stacklevel=2)
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

            logger.info("NVIDIA Backend initialized: device=%s, compute_capability=%s, "
                       "num_devices=%d, architecture=%s, fp8_enabled=%s",
                       self._device_name, self._compute_capability, len(self._devices),
                       self.nvidia_config.architecture.value, self.nvidia_config.fp8_enabled)

    def _check_availability(self) -> bool:
        """Check if CUDA is available (implements BaseBackend abstract method)."""
        return torch.cuda.is_available() and self._device is not None and self._device.type == "cuda"

    def _get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Get information about a specific CUDA device (implements BaseBackend abstract method)."""
        if not self._check_availability() or device_id >= len(self._devices):
            return DeviceInfo(
                backend="nvidia",
                device_type="cpu",
                device_id=0,
                device_name="CPU (CUDA not available)",
                is_available=False
            )

        props = torch.cuda.get_device_properties(device_id)
        return DeviceInfo(
            backend="nvidia",
            device_type=f"cuda:{device_id}",
            device_id=device_id,
            device_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            total_memory_bytes=props.total_memory,
            driver_version=torch.version.cuda,
            is_available=True,
            properties={
                'multi_processor_count': props.multi_processor_count,
                'major': props.major,
                'minor': props.minor,
                'architecture': self.nvidia_config.architecture.value,
                'fp8_supported': self.supports_fp8,
                'tensor_core_version': self.nvidia_config.tensor_core_version,
            }
        )

    @property
    def device(self) -> torch.device:
        """Get current CUDA device."""
        return self._device

    @property
    def devices(self) -> list[torch.device]:
        """Get all available CUDA devices."""
        return self._devices

    @property
    def device_count(self) -> int:
        """Get the number of available CUDA devices (overrides BaseBackend)."""
        return len(self._devices)

    @property
    def compute_capability(self) -> tuple | None:
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
        return self.nvidia_config.architecture in [
            NVIDIAArchitecture.BLACKWELL_DC,
            NVIDIAArchitecture.BLACKWELL_CONSUMER,
        ]

    @property
    def supports_fp8(self) -> bool:
        """Check if GPU supports FP8."""
        return self.nvidia_config.architecture in [
            NVIDIAArchitecture.HOPPER,
            NVIDIAArchitecture.BLACKWELL_DC,
            NVIDIAArchitecture.BLACKWELL_CONSUMER
        ]

    def prepare_model(
        self,
        model: nn.Module,
        optimization_level: str | OptimizationLevel | None = None
    ) -> nn.Module:
        """
        Prepare model for NVIDIA GPU execution (implements BaseBackend abstract method).

        Args:
            model: PyTorch model to prepare
            optimization_level: Optional optimization level (O0, O1, O2, O3)

        Returns:
            Prepared model ready for NVIDIA GPU execution
        """
        if model is None:
            warnings.warn("Model is None, returning unchanged", stacklevel=2)
            return model

        if not self.is_cuda_available:
            warnings.warn("CUDA not available, returning model unchanged", stacklevel=2)
            return model

        # Move model to device
        model = model.to(self.device)

        # Apply NVIDIA-specific optimizations based on level
        if optimization_level is None or optimization_level == OptimizationLevel.O0:
            # O0: Minimal optimizations (just device placement)
            pass
        else:
            # O1+: Apply NVIDIA-specific optimizations
            model = self._apply_nvidia_optimizations(model)

            # O2+: Enable gradient checkpointing if configured
            if optimization_level in (OptimizationLevel.O2, OptimizationLevel.O3, "O2", "O3", "balanced", "aggressive"):
                if self.config.memory.gradient_checkpointing:
                    model = self._enable_gradient_checkpointing(model)

        return model

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
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
        # Prepare model with aggressive optimizations
        model = self.prepare_model(model, optimization_level=OptimizationLevel.O2)

        # Set to eval mode
        model.eval()

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # Apply torch.compile if available and sample input provided
        if sample_input is not None and hasattr(torch, 'compile'):
            try:
                mode = 'max-autotune' if (self.is_h100 or self.is_blackwell) else 'reduce-overhead'
                model = torch.compile(model, mode=mode)
                # Warm up
                with torch.no_grad():
                    _ = model(sample_input.to(self.device))
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        return model

    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module | tuple[nn.Module, torch.optim.Optimizer]:
        """
        Optimize a model for training (implements BaseBackend abstract method).

        Args:
            model: PyTorch model
            optimizer: Optional optimizer to optimize along with model
            dtype: Optional dtype for precision

        Returns:
            Training-optimized model, or tuple of (model, optimizer)
        """
        # Prepare model with balanced optimizations
        model = self.prepare_model(model, optimization_level=OptimizationLevel.O1)

        # Ensure train mode
        model.train()

        # Apply mixed precision settings if configured
        if self.nvidia_config.mixed_precision_enabled:
            model._mixed_precision_enabled = True

        if optimizer:
            return model, optimizer
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
                        f"not optimal for Tensor Cores. Consider padding to multiples of {optimal_divisor}.",
                    stacklevel=2,
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
                    except (RuntimeError, TypeError) as e:
                        # Not all modules support channels_last
                        logger.debug("Could not convert to channels_last: %s", e)

        return model

    def _add_fusion_hints(self, model: nn.Module) -> nn.Module:
        """Add kernel fusion hints for CUDA compiler."""
        # Mark modules for potential fusion
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Add metadata for torch.compile
                module._cuda_fusible = True

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

    def get_memory_stats(self) -> dict[str, Any]:
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

    def get_device_info_dict(self) -> dict[str, Any]:
        """Get comprehensive device information (legacy format)."""
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
            warnings.warn(f"Device {device_id} not available, using default device", stacklevel=2)

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.is_cuda_available:
            torch.cuda.reset_peak_memory_stats()

    # ===== Custom Kernel Management =====

    def _register_default_kernels(self) -> None:
        """Register default CUDA kernels based on hardware capabilities."""
        try:
            # Import custom kernels
            from torchbridge.hardware.gpu.custom_kernels import (
                FlashAttentionV3,
                FusedLinearGELU,
                FusedLinearSiLU,
            )

            # Determine supported precisions based on architecture
            precisions = [PrecisionFormat.FP32, PrecisionFormat.FP16, PrecisionFormat.BF16]
            if self.supports_fp8:
                precisions.append(PrecisionFormat.FP8_E4M3)

            # Register FlashAttention-3 if enabled
            if self.config.kernel.flash_attention_enabled:
                fa3_metadata = KernelMetadata(
                    kernel_id="flash_attention_v3",
                    kernel_type=KernelType.ATTENTION,
                    version="3.0",
                    backend=KernelBackend.CUDA,
                    description="FlashAttention-3 with FP8 support and Split-K optimization",
                    min_compute_capability=(7, 0),  # Volta and newer
                    precision_support=precisions,
                    kernel_fn=FlashAttentionV3
                )
                self._kernel_registry.register_kernel(fa3_metadata)

            # Register Fused Linear+GELU if enabled
            if self.config.kernel.fused_gelu_enabled:
                gelu_metadata = KernelMetadata(
                    kernel_id="fused_linear_gelu",
                    kernel_type=KernelType.FUSION,
                    version="1.0",
                    backend=KernelBackend.CUDA,
                    description="Fused Linear + GELU activation",
                    min_compute_capability=(7, 0),
                    precision_support=precisions,
                    kernel_fn=FusedLinearGELU
                )
                self._kernel_registry.register_kernel(gelu_metadata)

            # Register Fused Linear+SiLU if enabled
            if self.config.kernel.fused_silu_enabled:
                silu_metadata = KernelMetadata(
                    kernel_id="fused_linear_silu",
                    kernel_type=KernelType.FUSION,
                    version="1.0",
                    backend=KernelBackend.CUDA,
                    description="Fused Linear + SiLU activation",
                    min_compute_capability=(7, 0),
                    precision_support=precisions,
                    kernel_fn=FusedLinearSiLU
                )
                self._kernel_registry.register_kernel(silu_metadata)

        except ImportError as e:
            warnings.warn(f"Could not import custom kernels: {e}", stacklevel=2)

    def get_optimal_attention_kernel(self,
                                     head_dim: int,
                                     precision: PrecisionFormat | None = None) -> type[nn.Module] | None:
        """
        Select optimal attention kernel for current hardware.

        Args:
            head_dim: Attention head dimension
            precision: Desired precision (auto-detected if None)

        Returns:
            Optimal attention kernel class, or None if no suitable kernel found
        """
        # Return None if CUDA is not available
        if not torch.cuda.is_available():
            return None

        if not self.config.kernel.enabled or not self.config.kernel.flash_attention_enabled:
            return None

        # Auto-detect precision if not specified
        if precision is None:
            if self.supports_fp8 and self.config.kernel.fp8_attention:
                precision = PrecisionFormat.FP8_E4M3
            else:
                precision = PrecisionFormat.BF16

        # Get optimal kernel from registry
        kernel_metadata = self._kernel_registry.get_optimal_kernel(
            kernel_type=KernelType.ATTENTION,
            device=self.device,
            precision=precision,
            prefer_backend=KernelBackend.CUDA
        )

        if kernel_metadata:
            return kernel_metadata.kernel_fn

        return None

    def prepare_model_with_custom_kernels(self, model: nn.Module) -> nn.Module:
        """
        Replace model layers with optimized custom kernels.

        This method scans the model and replaces compatible layers with
        fused kernel implementations for better performance.

        Args:
            model: PyTorch model to optimize

        Returns:
            Model with custom kernels applied
        """
        if not self.config.kernel.enabled or not self.is_cuda_available:
            return model

        try:
            from torchbridge.hardware.gpu.custom_kernels import (
                FusedLinearGELU,
                FusedLinearSiLU,
            )

            # Track replacements
            replacements = []

            # Scan for patterns to replace
            for name, module in list(model.named_modules()):
                # Look for Linear + GELU pattern
                if self.config.kernel.fuse_linear_activation and self.config.kernel.fused_gelu_enabled:
                    if isinstance(module, nn.Sequential) and len(module) == 2:
                        if isinstance(module[0], nn.Linear) and isinstance(module[1], nn.GELU):
                            linear = module[0]
                            fused = FusedLinearGELU(
                                in_features=linear.in_features,
                                out_features=linear.out_features,
                                bias=linear.bias is not None
                            )
                            # Copy weights
                            fused.weight.data.copy_(linear.weight.data)
                            if linear.bias is not None:
                                fused.bias.data.copy_(linear.bias.data)

                            # Replace in parent
                            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                            child_name = name.split('.')[-1]
                            if parent_name:
                                parent = dict(model.named_modules())[parent_name]
                                setattr(parent, child_name, fused)
                            else:
                                setattr(model, child_name, fused)

                            replacements.append(f"{name}: Linear+GELU → FusedLinearGELU")

                # Look for Linear + SiLU pattern
                if self.config.kernel.fuse_linear_activation and self.config.kernel.fused_silu_enabled:
                    if isinstance(module, nn.Sequential) and len(module) == 2:
                        if isinstance(module[0], nn.Linear) and isinstance(module[1], nn.SiLU):
                            linear = module[0]
                            fused = FusedLinearSiLU(
                                in_features=linear.in_features,
                                out_features=linear.out_features,
                                bias=linear.bias is not None
                            )
                            # Copy weights
                            fused.weight.data.copy_(linear.weight.data)
                            if linear.bias is not None:
                                fused.bias.data.copy_(linear.bias.data)

                            # Replace in parent
                            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                            child_name = name.split('.')[-1]
                            if parent_name:
                                parent = dict(model.named_modules())[parent_name]
                                setattr(parent, child_name, fused)
                            else:
                                setattr(model, child_name, fused)

                            replacements.append(f"{name}: Linear+SiLU → FusedLinearSiLU")

            if replacements:
                logger.info("Applied %d custom kernel replacements", len(replacements))
                for r in replacements[:5]:  # Log first 5
                    logger.debug("  %s", r)
                if len(replacements) > 5:
                    logger.debug("  ... and %d more", len(replacements) - 5)

        except Exception as e:
            warnings.warn(f"Could not apply custom kernels: {e}", stacklevel=2)

        return model

    @property
    def kernel_registry(self) -> KernelRegistry:
        """Get kernel registry instance."""
        return self._kernel_registry
