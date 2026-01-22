"""
Base classes and utilities for vision model optimization.

This module provides the foundation for optimizing computer vision models
including ResNet, ViT, and Stable Diffusion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn


class VisionModelType(Enum):
    """Supported vision model types."""
    RESNET = "resnet"
    VIT = "vit"
    STABLE_DIFFUSION = "stable_diffusion"
    CUSTOM = "custom"


class OptimizationLevel(Enum):
    """Optimization levels for vision models."""
    O0 = "O0"  # No optimization
    O1 = "O1"  # Basic optimizations (operator fusion)
    O2 = "O2"  # Advanced optimizations (memory efficient)
    O3 = "O3"  # Maximum optimizations (aggressive)


@dataclass
class VisionOptimizationConfig:
    """Configuration for vision model optimization."""

    # Model configuration
    model_type: VisionModelType = VisionModelType.CUSTOM
    optimization_level: OptimizationLevel = OptimizationLevel.O2

    # Inference optimization
    batch_size: int = 1
    enable_fusion: bool = True
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True

    # Memory optimization
    enable_gradient_checkpointing: bool = False
    enable_attention_slicing: bool = False
    attention_slice_size: Optional[int] = None
    enable_vae_tiling: bool = False
    vae_tile_size: int = 512

    # Precision
    use_fp16: bool = False
    use_bf16: bool = False
    use_int8: bool = False

    # Performance tuning
    num_inference_steps: int = 50
    channels_last: bool = True
    compile_model: bool = False

    # Device
    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration."""
        if self.use_fp16 and self.use_bf16:
            raise ValueError("Cannot use both FP16 and BF16 simultaneously")

        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"

    @classmethod
    def from_optimization_level(cls, level: OptimizationLevel, **kwargs) -> "VisionOptimizationConfig":
        """Create config from optimization level."""
        configs = {
            OptimizationLevel.O0: {
                "enable_fusion": False,
                "enable_cudnn_benchmark": False,
                "channels_last": False,
                "compile_model": False,
            },
            OptimizationLevel.O1: {
                "enable_fusion": True,
                "enable_cudnn_benchmark": True,
                "channels_last": False,
                "compile_model": False,
            },
            OptimizationLevel.O2: {
                "enable_fusion": True,
                "enable_cudnn_benchmark": True,
                "channels_last": True,
                "compile_model": False,
                "use_fp16": True,
            },
            OptimizationLevel.O3: {
                "enable_fusion": True,
                "enable_cudnn_benchmark": True,
                "channels_last": True,
                "compile_model": True,
                "use_fp16": True,
                "enable_attention_slicing": True,
            },
        }

        config_dict = configs.get(level, {})
        config_dict.update(kwargs)
        config_dict["optimization_level"] = level

        return cls(**config_dict)


class BaseVisionOptimizer(ABC):
    """Abstract base class for vision model optimizers."""

    def __init__(self, config: Optional[VisionOptimizationConfig] = None):
        """Initialize vision optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or VisionOptimizationConfig()
        self.device = torch.device(self.config.device)
        self.optimizations_applied: List[str] = []

    @abstractmethod
    def optimize(self, model: nn.Module) -> nn.Module:
        """Optimize the model.

        Args:
            model: Model to optimize

        Returns:
            Optimized model
        """
        pass

    def apply_precision_optimization(self, model: nn.Module) -> nn.Module:
        """Apply precision optimization (FP16/BF16/INT8).

        Args:
            model: Model to optimize

        Returns:
            Model with precision optimization
        """
        if self.config.use_fp16:
            model = model.half()
            self.optimizations_applied.append("fp16")
        elif self.config.use_bf16:
            model = model.bfloat16()
            self.optimizations_applied.append("bf16")

        return model

    def apply_memory_format_optimization(self, model: nn.Module) -> nn.Module:
        """Apply memory format optimization (channels_last).

        Args:
            model: Model to optimize

        Returns:
            Model with optimized memory format
        """
        if self.config.channels_last:
            model = model.to(memory_format=torch.channels_last)
            self.optimizations_applied.append("channels_last")

        return model

    def apply_cudnn_optimization(self) -> None:
        """Apply cuDNN optimizations."""
        if self.config.enable_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.optimizations_applied.append("cudnn_benchmark")

        if self.config.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.optimizations_applied.append("tf32")

    def apply_operator_fusion(self, model: nn.Module) -> nn.Module:
        """Apply operator fusion.

        Args:
            model: Model to optimize

        Returns:
            Model with fused operators
        """
        if not self.config.enable_fusion:
            return model

        # Fuse Conv+BN+ReLU patterns
        model = self._fuse_conv_bn_relu(model)

        # Fuse Linear+Activation patterns
        model = self._fuse_linear_activation(model)

        self.optimizations_applied.append("operator_fusion")
        return model

    def _fuse_conv_bn_relu(self, model: nn.Module) -> nn.Module:
        """Fuse Conv+BN+ReLU patterns.

        Args:
            model: Model to optimize

        Returns:
            Model with fused Conv+BN+ReLU
        """
        model.eval()  # Fusion only works in eval mode

        # Use PyTorch's built-in fusion for Conv+BN
        for name, module in list(model.named_modules()):
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                # Look for BN following Conv
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model

                # Check if next module is BatchNorm
                module_names = [n for n, _ in parent.named_children()]
                curr_idx = module_names.index(name.split('.')[-1])
                if curr_idx + 1 < len(module_names):
                    next_name = module_names[curr_idx + 1]
                    next_module = getattr(parent, next_name)

                    if isinstance(next_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        # Fuse using PyTorch utility
                        fused = torch.nn.utils.fusion.fuse_conv_bn_eval(module, next_module)
                        setattr(parent, name.split('.')[-1], fused)
                        # Remove BatchNorm
                        setattr(parent, next_name, nn.Identity())

        return model

    def _fuse_linear_activation(self, model: nn.Module) -> nn.Module:
        """Fuse Linear+Activation patterns.

        Args:
            model: Model to optimize

        Returns:
            Model with fused Linear+Activation
        """
        # This is a placeholder for potential future linear+activation fusion
        # Currently PyTorch doesn't have built-in fusion for this
        return model

    def apply_compilation(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile optimization.

        Args:
            model: Model to optimize

        Returns:
            Compiled model
        """
        if not self.config.compile_model:
            return model

        try:
            # Use torch.compile for PyTorch 2.0+
            model = torch.compile(model, mode="reduce-overhead")
            self.optimizations_applied.append("torch_compile")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Skipping compilation.")

        return model

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations.

        Returns:
            Dictionary containing optimization details
        """
        return {
            "model_type": self.config.model_type.value,
            "optimization_level": self.config.optimization_level.value,
            "device": str(self.device),
            "precision": self._get_precision_string(),
            "optimizations_applied": self.optimizations_applied,
            "batch_size": self.config.batch_size,
        }

    def _get_precision_string(self) -> str:
        """Get precision configuration as string."""
        if self.config.use_fp16:
            return "fp16"
        elif self.config.use_bf16:
            return "bf16"
        elif self.config.use_int8:
            return "int8"
        else:
            return "fp32"


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: Model to analyze

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_model_memory(
    model: nn.Module,
    batch_size: int = 1,
    input_size: Tuple[int, ...] = (3, 224, 224),
    precision: str = "fp32"
) -> Dict[str, float]:
    """Estimate model memory requirements.

    Args:
        model: Model to analyze
        batch_size: Batch size for inference
        input_size: Input tensor size (C, H, W)
        precision: Precision mode ("fp32", "fp16", "bf16", "int8")

    Returns:
        Dictionary with memory estimates in MB
    """
    # Parameter memory
    total_params, _ = count_parameters(model)

    # Bytes per parameter based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
    }

    param_bytes = bytes_per_param.get(precision, 4)
    param_memory = (total_params * param_bytes) / (1024 ** 2)

    # Activation memory (rough estimate)
    # Assume average activation size is ~2x input size per layer
    num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Module))
    input_elements = batch_size * input_size[0] * input_size[1] * input_size[2]
    activation_memory = (input_elements * num_layers * 2 * param_bytes) / (1024 ** 2)

    # Gradient memory (if training)
    gradient_memory = param_memory  # Same as parameters

    return {
        "parameter_memory_mb": param_memory,
        "activation_memory_mb": activation_memory,
        "gradient_memory_mb": gradient_memory,
        "total_inference_mb": param_memory + activation_memory,
        "total_training_mb": param_memory + activation_memory + gradient_memory,
    }
