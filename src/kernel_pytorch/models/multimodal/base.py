"""
Base classes and utilities for multi-modal model optimization.

This module provides the foundation for optimizing multi-modal models
including CLIP, LLaVA, and Whisper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn


class MultiModalType(Enum):
    """Supported multi-modal model types."""
    CLIP = "clip"
    LLAVA = "llava"
    WHISPER = "whisper"
    CUSTOM = "custom"


class OptimizationLevel(Enum):
    """Optimization levels for multi-modal models."""
    O0 = "O0"  # No optimization
    O1 = "O1"  # Basic optimizations
    O2 = "O2"  # Advanced optimizations (production)
    O3 = "O3"  # Maximum optimizations


class ModalityType(Enum):
    """Types of modalities."""
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalOptimizationConfig:
    """Configuration for multi-modal model optimization."""

    # Model configuration
    model_type: MultiModalType = MultiModalType.CUSTOM
    optimization_level: OptimizationLevel = OptimizationLevel.O2

    # Modalities
    modalities: List[ModalityType] = None

    # Inference optimization
    batch_size: int = 1
    enable_fusion: bool = True
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True

    # Cross-modal optimization
    enable_cross_attention_optimization: bool = True
    enable_modality_fusion: bool = True

    # Memory optimization
    enable_gradient_checkpointing: bool = False
    enable_attention_slicing: bool = False
    attention_slice_size: Optional[int] = None

    # Precision
    use_fp16: bool = False
    use_bf16: bool = False
    use_int8: bool = False

    # Performance tuning
    channels_last: bool = False
    compile_model: bool = False

    # Device
    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration."""
        if self.use_fp16 and self.use_bf16:
            raise ValueError("Cannot use both FP16 and BF16 simultaneously")

        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"

        if self.modalities is None:
            # Default to vision+text for multi-modal
            self.modalities = [ModalityType.VISION, ModalityType.TEXT]

    @classmethod
    def from_optimization_level(cls, level: OptimizationLevel, **kwargs) -> "MultiModalOptimizationConfig":
        """Create config from optimization level."""
        configs = {
            OptimizationLevel.O0: {
                "enable_fusion": False,
                "enable_cudnn_benchmark": False,
                "enable_cross_attention_optimization": False,
                "compile_model": False,
            },
            OptimizationLevel.O1: {
                "enable_fusion": True,
                "enable_cudnn_benchmark": True,
                "enable_cross_attention_optimization": True,
                "compile_model": False,
            },
            OptimizationLevel.O2: {
                "enable_fusion": True,
                "enable_cudnn_benchmark": True,
                "enable_cross_attention_optimization": True,
                "compile_model": False,
                "use_fp16": True,
            },
            OptimizationLevel.O3: {
                "enable_fusion": True,
                "enable_cudnn_benchmark": True,
                "enable_cross_attention_optimization": True,
                "compile_model": True,
                "use_fp16": True,
                "enable_attention_slicing": True,
            },
        }

        config_dict = configs.get(level, {})
        config_dict.update(kwargs)
        config_dict["optimization_level"] = level

        return cls(**config_dict)


class BaseMultiModalOptimizer(ABC):
    """Abstract base class for multi-modal model optimizers."""

    def __init__(self, config: Optional[MultiModalOptimizationConfig] = None):
        """Initialize multi-modal optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or MultiModalOptimizationConfig()
        self.device = torch.device(self.config.device)
        self.optimizations_applied: List[str] = []

    @abstractmethod
    def optimize(self, model: Union[nn.Module, Any]) -> Union[nn.Module, Any]:
        """Optimize the model.

        Args:
            model: Model or pipeline to optimize

        Returns:
            Optimized model or pipeline
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
            try:
                model = model.to(memory_format=torch.channels_last)
                self.optimizations_applied.append("channels_last")
            except Exception:
                # Not all models support channels_last
                pass

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
            model = torch.compile(model, mode="reduce-overhead")
            self.optimizations_applied.append("torch_compile")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Skipping compilation.")

        return model

    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory.

        Args:
            model: Model to optimize

        Returns:
            Model with gradient checkpointing
        """
        if not self.config.enable_gradient_checkpointing:
            return model

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            self.optimizations_applied.append("gradient_checkpointing")

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
            "modalities": [m.value for m in self.config.modalities],
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
    input_sizes: Dict[str, Tuple[int, ...]] = None,
    precision: str = "fp32"
) -> Dict[str, float]:
    """Estimate model memory requirements.

    Args:
        model: Model to analyze
        batch_size: Batch size for inference
        input_sizes: Dictionary of input tensor sizes per modality
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
    if input_sizes is None:
        input_sizes = {"default": (3, 224, 224)}

    total_input_elements = sum(
        batch_size * torch.Size(size).numel()
        for size in input_sizes.values()
    )

    num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Module))
    activation_memory = (total_input_elements * num_layers * 2 * param_bytes) / (1024 ** 2)

    # Gradient memory (if training)
    gradient_memory = param_memory

    return {
        "parameter_memory_mb": param_memory,
        "activation_memory_mb": activation_memory,
        "gradient_memory_mb": gradient_memory,
        "total_inference_mb": param_memory + activation_memory,
        "total_training_mb": param_memory + activation_memory + gradient_memory,
    }


class CrossModalAttention(nn.Module):
    """Cross-modal attention layer for vision-language interaction."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Initialize cross-modal attention.

        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query from one modality, Key/Value from another
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x_query: torch.Tensor,
        x_context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_query: Query tensor from one modality (B, N_q, D)
            x_context: Context tensor from another modality (B, N_c, D)
            mask: Optional attention mask

        Returns:
            Output tensor (B, N_q, D)
        """
        B, N_q, D = x_query.shape
        N_c = x_context.shape[1]

        # Project query and context
        q = self.q(x_query).reshape(B, N_q, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_context).reshape(B, N_c, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
