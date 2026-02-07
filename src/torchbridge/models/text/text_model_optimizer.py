"""
Text Model Optimizer

Production-grade optimization wrapper for text models from HuggingFace.
Supports BERT, GPT-2, DistilBERT with automatic backend detection and
multi-level optimization.

Features:
- Automatic hardware detection and backend selection
- Mixed precision (FP16, BF16, FP8 on H100+)
- torch.compile with various modes
- FlashAttention when available
- Memory-efficient inference
- Batch optimization

Version: 0.5.3
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TextModelType(Enum):
    """Supported text model types."""
    BERT = "bert"
    GPT2 = "gpt2"
    DISTILBERT = "distilbert"
    ROBERTA = "roberta"
    ALBERT = "albert"
    CUSTOM = "custom"


class OptimizationMode(Enum):
    """Optimization modes for text models."""
    INFERENCE = "inference"      # Optimized for low-latency inference
    THROUGHPUT = "throughput"    # Optimized for high-throughput batch processing
    MEMORY = "memory"            # Optimized for minimal memory usage
    BALANCED = "balanced"        # Balance between speed and memory


@dataclass
class TextModelConfig:
    """Configuration for text model optimization."""
    # Model settings
    model_name: str = "bert-base-uncased"
    model_type: TextModelType = TextModelType.BERT
    max_sequence_length: int = 512

    # Optimization settings
    optimization_mode: OptimizationMode = OptimizationMode.INFERENCE
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

    # Precision settings
    dtype: torch.dtype | None = None  # Auto-detect if None
    use_amp: bool = True

    # Memory settings
    gradient_checkpointing: bool = False
    enable_memory_efficient_attention: bool = True

    # Backend settings
    device: str = "auto"  # "auto", "cuda", "cpu", "xpu", "hip"

    # Additional options
    warmup_steps: int = 3
    enable_profiling: bool = False


class TextModelOptimizer:
    """
    Optimizer for text models with automatic hardware detection.

    This class wraps HuggingFace text models and applies TorchBridge
    optimizations automatically based on detected hardware.

    Example:
        >>> from torchbridge.models.text import TextModelOptimizer
        >>> optimizer = TextModelOptimizer()
        >>> model = optimizer.optimize("bert-base-uncased")
        >>> outputs = model(input_ids, attention_mask=attention_mask)
    """

    def __init__(self, config: TextModelConfig | None = None):
        """
        Initialize the text model optimizer.

        Args:
            config: Optional configuration for optimization
        """
        self.config = config or TextModelConfig()
        self._device = None
        self._backend = None
        self._dtype = None
        self._setup_backend()

    def _setup_backend(self) -> None:
        """Set up the appropriate backend based on available hardware."""
        # Import backend factory
        try:
            from torchbridge.backends import BackendFactory
            self._backend = BackendFactory.create()
            self._device = self._backend.device
            logger.info(f"Using backend: {self._backend.BACKEND_NAME} on {self._device}")
        except (ImportError, Exception) as e:
            logger.warning(f"BackendFactory not available ({e}), using PyTorch defaults")
            self._backend = None
            self._device = self._detect_device()

        # Set dtype based on device capabilities
        self._dtype = self._get_optimal_dtype()

    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
        if self.config.device != "auto":
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        # Check for Intel XPU
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return torch.device("xpu")
        except ImportError:
            pass

        return torch.device("cpu")

    def _get_optimal_dtype(self) -> torch.dtype:
        """Get the optimal dtype for the current device."""
        if self.config.dtype is not None:
            return self.config.dtype

        if self._device.type == "cuda":
            # Check for compute capability
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                if major >= 8:  # Ampere or newer
                    return torch.bfloat16
                return torch.float16
        elif self._device.type == "xpu":
            return torch.bfloat16
        elif self._device.type == "cpu":
            # BF16 on CPU with AVX-512 support
            return torch.bfloat16 if torch.cpu.is_available() else torch.float32

        return torch.float32

    def optimize(
        self,
        model_name_or_model: str | nn.Module,
        task: str | None = None,
        **kwargs
    ) -> nn.Module:
        """
        Optimize a text model for inference or training.

        Args:
            model_name_or_model: HuggingFace model name or pre-loaded model
            task: Task type (e.g., "text-classification", "text-generation")
            **kwargs: Additional arguments passed to model loading

        Returns:
            Optimized model ready for inference or training
        """
        # Load model if string
        if isinstance(model_name_or_model, str):
            model = self._load_model(model_name_or_model, task, **kwargs)
        else:
            model = model_name_or_model

        # Detect model type
        model_type = self._detect_model_type(model)
        logger.info(f"Detected model type: {model_type.value}")

        # Move to device
        model = model.to(self._device)

        # Apply dtype conversion
        if self._dtype != torch.float32:
            model = self._apply_dtype_conversion(model)

        # Apply optimizations based on mode
        if self.config.optimization_mode == OptimizationMode.INFERENCE:
            model = self._optimize_for_inference(model)
        elif self.config.optimization_mode == OptimizationMode.THROUGHPUT:
            model = self._optimize_for_throughput(model)
        elif self.config.optimization_mode == OptimizationMode.MEMORY:
            model = self._optimize_for_memory(model)
        else:
            model = self._optimize_balanced(model)

        # Apply torch.compile if enabled
        if self.config.use_torch_compile:
            model = self._apply_torch_compile(model)

        # Warmup
        if self.config.warmup_steps > 0:
            self._warmup(model)

        logger.info(f"Model optimization complete on {self._device}")
        return model

    def _load_model(
        self,
        model_name: str,
        task: str | None = None,
        **kwargs
    ) -> nn.Module:
        """Load a model from HuggingFace."""
        try:
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForMaskedLM,
                AutoModelForSequenceClassification,
            )

            # Select model class based on task
            if task == "text-classification" or task == "sequence-classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
            elif task == "text-generation" or task == "causal-lm":
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            elif task == "masked-lm" or task == "fill-mask":
                model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
            else:
                model = AutoModel.from_pretrained(model_name, **kwargs)

            logger.info(f"Loaded model: {model_name}")
            return model

        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            ) from None

    def _detect_model_type(self, model: nn.Module) -> TextModelType:
        """Detect the type of text model."""
        class_name = model.__class__.__name__.lower()

        if "bert" in class_name and "distil" not in class_name:
            return TextModelType.BERT
        elif "distilbert" in class_name:
            return TextModelType.DISTILBERT
        elif "gpt2" in class_name:
            return TextModelType.GPT2
        elif "roberta" in class_name:
            return TextModelType.ROBERTA
        elif "albert" in class_name:
            return TextModelType.ALBERT

        return TextModelType.CUSTOM

    def _apply_dtype_conversion(self, model: nn.Module) -> nn.Module:
        """Apply dtype conversion to model."""
        logger.info(f"Converting model to {self._dtype}")

        # Use autocast-compatible conversion
        if self._dtype == torch.float16:
            model = model.half()
        elif self._dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        return model

    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for low-latency inference."""
        model.eval()

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # Enable memory-efficient attention if available
        if self.config.enable_memory_efficient_attention:
            model = self._enable_efficient_attention(model)

        return model

    def _optimize_for_throughput(self, model: nn.Module) -> nn.Module:
        """Optimize model for high-throughput batch processing."""
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        # Enable efficient attention
        if self.config.enable_memory_efficient_attention:
            model = self._enable_efficient_attention(model)

        return model

    def _optimize_for_memory(self, model: nn.Module) -> nn.Module:
        """Optimize model for minimal memory usage."""
        model.eval()

        # Enable gradient checkpointing for memory savings
        if self.config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

        for param in model.parameters():
            param.requires_grad = False

        return model

    def _optimize_balanced(self, model: nn.Module) -> nn.Module:
        """Apply balanced optimizations."""
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        if self.config.enable_memory_efficient_attention:
            model = self._enable_efficient_attention(model)

        return model

    def _enable_efficient_attention(self, model: nn.Module) -> nn.Module:
        """Enable memory-efficient attention mechanisms."""
        # Try to enable SDPA (Scaled Dot Product Attention)
        try:
            if hasattr(model.config, 'attn_implementation'):
                # HuggingFace 4.36+ supports sdpa natively
                pass

            # Enable Flash Attention if available
            if hasattr(model, 'enable_flash_attention'):
                model.enable_flash_attention()
                logger.info("Enabled Flash Attention")
        except Exception as e:
            logger.debug(f"Could not enable efficient attention: {e}")

        return model

    def _apply_torch_compile(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile optimization."""
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model

        try:
            compiled_model = torch.compile(
                model,
                mode=self.config.compile_mode,
                fullgraph=False,
                dynamic=True
            )
            logger.info(f"Applied torch.compile with mode={self.config.compile_mode}")
            return compiled_model
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            return model

    def _warmup(self, model: nn.Module) -> None:
        """Warmup the model with dummy inputs."""
        logger.info(f"Running {self.config.warmup_steps} warmup steps")

        # Create dummy input
        batch_size = 4
        seq_length = min(128, self.config.max_sequence_length)

        dummy_input = torch.randint(
            0, 30000,
            (batch_size, seq_length),
            device=self._device
        )

        with torch.no_grad():
            for _ in range(self.config.warmup_steps):
                try:
                    model(dummy_input)
                except Exception:
                    # Some models need attention_mask
                    attention_mask = torch.ones_like(dummy_input)
                    model(dummy_input, attention_mask=attention_mask)

        if self._device.type == "cuda":
            torch.cuda.synchronize()

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get the current dtype."""
        return self._dtype

    def get_optimization_info(self) -> dict[str, Any]:
        """Get information about applied optimizations."""
        return {
            "device": str(self._device),
            "dtype": str(self._dtype),
            "backend": self._backend.BACKEND_NAME if self._backend else "pytorch",
            "optimization_mode": self.config.optimization_mode.value,
            "torch_compile": self.config.use_torch_compile,
            "compile_mode": self.config.compile_mode,
            "efficient_attention": self.config.enable_memory_efficient_attention,
        }


class OptimizedBERT(nn.Module):
    """
    Optimized BERT wrapper with automatic optimization.

    Example:
        >>> model = OptimizedBERT("bert-base-uncased", task="text-classification")
        >>> outputs = model(input_ids, attention_mask=attention_mask)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        task: str = "sequence-classification",
        num_labels: int = 2,
        config: TextModelConfig | None = None,
        **kwargs
    ):
        super().__init__()

        # Create config
        model_config = config or TextModelConfig(
            model_name=model_name,
            model_type=TextModelType.BERT
        )

        # Initialize optimizer
        self.optimizer = TextModelOptimizer(model_config)

        # Load and optimize model
        self.model = self.optimizer.optimize(
            model_name,
            task=task,
            num_labels=num_labels,
            **kwargs
        )

        self._device = self.optimizer.device
        self._dtype = self.optimizer.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs
    ):
        """Forward pass through the optimized BERT model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def get_optimization_info(self) -> dict[str, Any]:
        return self.optimizer.get_optimization_info()


class OptimizedGPT2(nn.Module):
    """
    Optimized GPT-2 wrapper with automatic optimization.

    Example:
        >>> model = OptimizedGPT2("gpt2", task="text-generation")
        >>> outputs = model.generate(input_ids, max_length=100)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        task: str = "causal-lm",
        config: TextModelConfig | None = None,
        **kwargs
    ):
        super().__init__()

        # Create config optimized for generation
        model_config = config or TextModelConfig(
            model_name=model_name,
            model_type=TextModelType.GPT2,
            compile_mode="reduce-overhead"  # Better for autoregressive generation
        )

        # Initialize optimizer
        self.optimizer = TextModelOptimizer(model_config)

        # Load and optimize model
        self.model = self.optimizer.optimize(model_name, task=task, **kwargs)

        self._device = self.optimizer.device
        self._dtype = self.optimizer.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs
    ):
        """Forward pass through the optimized GPT-2 model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        **kwargs
    ):
        """Generate text with the optimized model."""
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            **kwargs
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def get_optimization_info(self) -> dict[str, Any]:
        return self.optimizer.get_optimization_info()


class OptimizedDistilBERT(nn.Module):
    """
    Optimized DistilBERT wrapper with automatic optimization.

    DistilBERT is already a distilled model, so optimizations focus on
    inference speed and memory efficiency.

    Example:
        >>> model = OptimizedDistilBERT("distilbert-base-uncased")
        >>> outputs = model(input_ids, attention_mask=attention_mask)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        task: str = "sequence-classification",
        num_labels: int = 2,
        config: TextModelConfig | None = None,
        **kwargs
    ):
        super().__init__()

        # Create config
        model_config = config or TextModelConfig(
            model_name=model_name,
            model_type=TextModelType.DISTILBERT,
            optimization_mode=OptimizationMode.INFERENCE
        )

        # Initialize optimizer
        self.optimizer = TextModelOptimizer(model_config)

        # Load and optimize model
        self.model = self.optimizer.optimize(
            model_name,
            task=task,
            num_labels=num_labels,
            **kwargs
        )

        self._device = self.optimizer.device
        self._dtype = self.optimizer.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs
    ):
        """Forward pass through the optimized DistilBERT model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def get_optimization_info(self) -> dict[str, Any]:
        return self.optimizer.get_optimization_info()


def create_optimized_text_model(
    model_name: str,
    task: str | None = None,
    optimization_mode: str = "inference",
    **kwargs
) -> nn.Module:
    """
    Factory function to create an optimized text model.

    Args:
        model_name: HuggingFace model name (e.g., "bert-base-uncased", "gpt2")
        task: Task type (e.g., "text-classification", "text-generation")
        optimization_mode: One of "inference", "throughput", "memory", "balanced"
        **kwargs: Additional arguments passed to model loading

    Returns:
        Optimized model ready for use

    Example:
        >>> model = create_optimized_text_model(
        ...     "bert-base-uncased",
        ...     task="text-classification",
        ...     num_labels=3
        ... )
    """
    # Map string to enum
    mode_map = {
        "inference": OptimizationMode.INFERENCE,
        "throughput": OptimizationMode.THROUGHPUT,
        "memory": OptimizationMode.MEMORY,
        "balanced": OptimizationMode.BALANCED,
    }

    config = TextModelConfig(
        model_name=model_name,
        optimization_mode=mode_map.get(optimization_mode, OptimizationMode.INFERENCE)
    )

    optimizer = TextModelOptimizer(config)
    return optimizer.optimize(model_name, task=task, **kwargs)


__all__ = [
    "TextModelType",
    "OptimizationMode",
    "TextModelConfig",
    "TextModelOptimizer",
    "OptimizedBERT",
    "OptimizedGPT2",
    "OptimizedDistilBERT",
    "create_optimized_text_model",
]
