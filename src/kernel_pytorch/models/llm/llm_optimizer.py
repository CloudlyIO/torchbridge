"""
LLM Optimizer

Production-grade optimization wrapper for large language models (1B-13B params).
Supports Llama, Mistral, Phi, and other autoregressive models with:
- KV-cache optimization
- Quantization (INT8, INT4, FP8)
- Flash Attention
- Memory-efficient inference

Version: 0.4.12
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LLMType(Enum):
    """Supported LLM types."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    QWEN = "qwen"
    GEMMA = "gemma"
    CUSTOM = "custom"


class QuantizationMode(Enum):
    """Quantization modes for LLMs."""
    NONE = "none"           # Full precision
    INT8 = "int8"           # INT8 dynamic quantization
    INT4 = "int4"           # INT4 weight quantization (GPTQ/AWQ)
    FP8 = "fp8"             # FP8 for H100+
    BNBT4 = "bnb_4bit"      # BitsAndBytes 4-bit


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    use_cache: bool = True


@dataclass
class LLMConfig:
    """Configuration for LLM optimization."""
    # Model settings
    model_name: str = "meta-llama/Llama-2-7b-hf"
    model_type: LLMType = LLMType.LLAMA
    max_sequence_length: int = 4096

    # Quantization
    quantization: QuantizationMode = QuantizationMode.NONE
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Optimization settings
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"

    # Precision
    dtype: torch.dtype | None = None  # Auto-detect
    compute_dtype: torch.dtype | None = None

    # KV-cache settings
    use_kv_cache: bool = True
    kv_cache_dtype: torch.dtype | None = None
    max_cache_length: int = 4096

    # Memory settings
    offload_to_cpu: bool = False
    use_bettertransformer: bool = True

    # Device settings
    device: str = "auto"
    device_map: str | None = "auto"

    # Generation defaults
    generation: GenerationConfig = field(default_factory=GenerationConfig)


class LLMOptimizer:
    """
    Optimizer for large language models with comprehensive optimization.

    Supports models from 1B to 13B parameters with automatic optimization
    for available hardware including NVIDIA, AMD, TPU, and Intel.

    Example:
        >>> from kernel_pytorch.models.llm import LLMOptimizer
        >>> optimizer = LLMOptimizer()
        >>> model, tokenizer = optimizer.optimize("meta-llama/Llama-2-7b-hf")
        >>> outputs = model.generate(input_ids, max_new_tokens=100)
    """

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize the LLM optimizer.

        Args:
            config: Optional configuration for optimization
        """
        self.config = config or LLMConfig()
        self._device = None
        self._backend = None
        self._dtype = None
        self._setup_backend()

    def _setup_backend(self) -> None:
        """Set up the appropriate backend."""
        try:
            from kernel_pytorch.backends import BackendFactory
            self._backend = BackendFactory.create()
            self._device = self._backend.device
            logger.info(f"Using backend: {self._backend.BACKEND_NAME} on {self._device}")
        except Exception as e:
            logger.warning(f"BackendFactory not available ({e}), using PyTorch defaults")
            self._backend = None
            self._device = self._detect_device()

        self._dtype = self._get_optimal_dtype()

    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
        if self.config.device != "auto":
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return torch.device("xpu")
        except ImportError:
            pass

        return torch.device("cpu")

    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype based on hardware and config."""
        if self.config.dtype is not None:
            return self.config.dtype

        if self._device.type == "cuda":
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                if major >= 8:
                    return torch.bfloat16
                return torch.float16
        elif self._device.type == "xpu":
            return torch.bfloat16

        return torch.float32

    def _get_quantization_config(self) -> dict[str, Any] | None:
        """Get quantization configuration based on settings."""
        if self.config.quantization == QuantizationMode.NONE:
            if self.config.load_in_8bit:
                return {"load_in_8bit": True}
            elif self.config.load_in_4bit:
                return {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": self._dtype,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                }
            return None

        if self.config.quantization == QuantizationMode.INT8:
            return {"load_in_8bit": True}

        if self.config.quantization == QuantizationMode.BNBT4:
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": self._dtype,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }

        return None

    def optimize(
        self,
        model_name_or_model: str | nn.Module,
        tokenizer: Any | None = None,
        **kwargs
    ) -> tuple[nn.Module, Any]:
        """
        Optimize an LLM for inference.

        Args:
            model_name_or_model: HuggingFace model name or pre-loaded model
            tokenizer: Optional pre-loaded tokenizer
            **kwargs: Additional model loading arguments

        Returns:
            Tuple of (optimized_model, tokenizer)
        """
        # Load model and tokenizer
        if isinstance(model_name_or_model, str):
            model, tokenizer = self._load_model(model_name_or_model, tokenizer, **kwargs)
        else:
            model = model_name_or_model
            if tokenizer is None:
                raise ValueError("tokenizer required when passing pre-loaded model")

        # Detect model type
        model_type = self._detect_model_type(model)
        logger.info(f"Detected model type: {model_type.value}")

        # Apply Flash Attention if available
        if self.config.use_flash_attention:
            model = self._enable_flash_attention(model)

        # Apply BetterTransformer if enabled
        if self.config.use_bettertransformer:
            model = self._apply_bettertransformer(model)

        # Set model to eval mode
        model.eval()

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # Apply torch.compile
        if self.config.use_torch_compile and not self._is_quantized(model):
            model = self._apply_torch_compile(model)

        logger.info(f"Model optimization complete on {self._device}")
        return model, tokenizer

    def _load_model(
        self,
        model_name: str,
        tokenizer: Any | None = None,
        **kwargs
    ) -> tuple[nn.Module, Any]:
        """Load model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": self._dtype,
            }

            # Add device map for automatic distribution
            if self.config.device_map:
                model_kwargs["device_map"] = self.config.device_map

            # Add quantization config
            quant_config = self._get_quantization_config()
            if quant_config:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_config)
                except ImportError:
                    logger.warning("BitsAndBytes not available for quantization")
                    model_kwargs.update(quant_config)

            # Add Flash Attention
            if self.config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            model_kwargs.update(kwargs)

            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            logger.info(f"Loaded model: {model_name}")
            return model, tokenizer

        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            ) from None

    def _detect_model_type(self, model: nn.Module) -> LLMType:
        """Detect the type of LLM."""
        class_name = model.__class__.__name__.lower()
        config_name = getattr(model, "config", None)
        if config_name:
            config_name = config_name.__class__.__name__.lower()

        if "llama" in class_name or (config_name and "llama" in config_name):
            return LLMType.LLAMA
        elif "mistral" in class_name or (config_name and "mistral" in config_name):
            return LLMType.MISTRAL
        elif "phi" in class_name or (config_name and "phi" in config_name):
            return LLMType.PHI
        elif "qwen" in class_name or (config_name and "qwen" in config_name):
            return LLMType.QWEN
        elif "gemma" in class_name or (config_name and "gemma" in config_name):
            return LLMType.GEMMA

        return LLMType.CUSTOM

    def _enable_flash_attention(self, model: nn.Module) -> nn.Module:
        """Enable Flash Attention if supported."""
        try:
            if hasattr(model.config, 'attn_implementation'):
                pass  # Already set during loading
            elif hasattr(model, '_attn_implementation'):
                model._attn_implementation = "flash_attention_2"
                logger.info("Enabled Flash Attention 2")
        except Exception as e:
            logger.debug(f"Could not enable Flash Attention: {e}")

        return model

    def _apply_bettertransformer(self, model: nn.Module) -> nn.Module:
        """Apply BetterTransformer optimization."""
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
            logger.info("Applied BetterTransformer optimization")
        except ImportError:
            logger.debug("optimum not available for BetterTransformer")
        except Exception as e:
            logger.debug(f"BetterTransformer failed: {e}")

        return model

    def _is_quantized(self, model: nn.Module) -> bool:
        """Check if model is quantized."""
        # Check for BitsAndBytes quantization
        for module in model.modules():
            module_type = type(module).__name__
            if "4bit" in module_type.lower() or "8bit" in module_type.lower():
                return True
            if "Linear8bitLt" in module_type or "Linear4bit" in module_type:
                return True
        return False

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

    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get current dtype."""
        return self._dtype

    def get_optimization_info(self) -> dict[str, Any]:
        """Get information about applied optimizations."""
        return {
            "device": str(self._device),
            "dtype": str(self._dtype),
            "backend": self._backend.BACKEND_NAME if self._backend else "pytorch",
            "quantization": self.config.quantization.value,
            "flash_attention": self.config.use_flash_attention,
            "torch_compile": self.config.use_torch_compile,
            "kv_cache": self.config.use_kv_cache,
            "max_seq_length": self.config.max_sequence_length,
        }

    def estimate_memory(self, model_name: str = None) -> dict[str, float]:
        """Estimate memory requirements for the model."""
        # Rough estimates based on parameter count (in FP16)
        param_estimates = {
            "7b": 14.0,   # GB in FP16
            "8b": 16.0,
            "13b": 26.0,
            "70b": 140.0,
        }

        model_name = model_name or self.config.model_name
        model_lower = model_name.lower()

        base_memory = 14.0  # Default 7B estimate
        for size, mem in param_estimates.items():
            if size in model_lower:
                base_memory = mem
                break

        # Apply quantization reduction FIRST (takes precedence)
        if self.config.quantization == QuantizationMode.INT8:
            base_memory /= 2
        elif self.config.quantization in [QuantizationMode.INT4, QuantizationMode.BNBT4]:
            base_memory /= 4
        elif self.config.quantization == QuantizationMode.FP8:
            base_memory /= 2
        # Then adjust for dtype if not quantized
        elif self._dtype == torch.float32:
            base_memory *= 2
        elif self._dtype == torch.bfloat16 or self._dtype == torch.float16:
            pass  # Base estimate is already for FP16

        # KV-cache overhead (rough estimate)
        kv_overhead = base_memory * 0.1 if self.config.use_kv_cache else 0

        return {
            "model_memory_gb": base_memory,
            "kv_cache_gb": kv_overhead,
            "total_gb": base_memory + kv_overhead,
            "dtype": str(self._dtype),
            "quantization": self.config.quantization.value,
        }


class OptimizedLlama(nn.Module):
    """
    Optimized Llama wrapper for inference.

    Example:
        >>> model = OptimizedLlama("meta-llama/Llama-2-7b-hf")
        >>> outputs = model.generate(input_ids, max_new_tokens=100)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        config: LLMConfig | None = None,
        **kwargs
    ):
        super().__init__()

        model_config = config or LLMConfig(
            model_name=model_name,
            model_type=LLMType.LLAMA
        )

        self.optimizer = LLMOptimizer(model_config)
        self.model, self.tokenizer = self.optimizer.optimize(model_name, **kwargs)

        self._device = self.optimizer.device
        self._dtype = self.optimizer.dtype

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass."""
        return self.model(input_ids=input_ids, **kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate text."""
        return self.model.generate(input_ids=input_ids, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

    def get_optimization_info(self) -> dict[str, Any]:
        return self.optimizer.get_optimization_info()


class OptimizedMistral(nn.Module):
    """Optimized Mistral wrapper."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        config: LLMConfig | None = None,
        **kwargs
    ):
        super().__init__()

        model_config = config or LLMConfig(
            model_name=model_name,
            model_type=LLMType.MISTRAL,
            max_sequence_length=8192  # Mistral supports longer context
        )

        self.optimizer = LLMOptimizer(model_config)
        self.model, self.tokenizer = self.optimizer.optimize(model_name, **kwargs)

        self._device = self.optimizer.device
        self._dtype = self.optimizer.dtype

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.model(input_ids=input_ids, **kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids=input_ids, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

    def get_optimization_info(self) -> dict[str, Any]:
        return self.optimizer.get_optimization_info()


class OptimizedPhi(nn.Module):
    """Optimized Phi wrapper."""

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        config: LLMConfig | None = None,
        **kwargs
    ):
        super().__init__()

        model_config = config or LLMConfig(
            model_name=model_name,
            model_type=LLMType.PHI,
            max_sequence_length=2048
        )

        self.optimizer = LLMOptimizer(model_config)
        self.model, self.tokenizer = self.optimizer.optimize(model_name, **kwargs)

        self._device = self.optimizer.device
        self._dtype = self.optimizer.dtype

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.model(input_ids=input_ids, **kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids=input_ids, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

    def get_optimization_info(self) -> dict[str, Any]:
        return self.optimizer.get_optimization_info()


def create_optimized_llm(
    model_name: str,
    quantization: str = "none",
    max_sequence_length: int = 4096,
    **kwargs
) -> tuple[nn.Module, Any]:
    """
    Factory function to create an optimized LLM.

    Args:
        model_name: HuggingFace model name
        quantization: "none", "int8", "int4", "fp8", "bnb_4bit"
        max_sequence_length: Maximum sequence length
        **kwargs: Additional arguments

    Returns:
        Tuple of (optimized_model, tokenizer)
    """
    quant_map = {
        "none": QuantizationMode.NONE,
        "int8": QuantizationMode.INT8,
        "int4": QuantizationMode.INT4,
        "fp8": QuantizationMode.FP8,
        "bnb_4bit": QuantizationMode.BNBT4,
    }

    config = LLMConfig(
        model_name=model_name,
        quantization=quant_map.get(quantization, QuantizationMode.NONE),
        max_sequence_length=max_sequence_length,
    )

    optimizer = LLMOptimizer(config)
    return optimizer.optimize(model_name, **kwargs)


__all__ = [
    "LLMType",
    "QuantizationMode",
    "GenerationConfig",
    "LLMConfig",
    "LLMOptimizer",
    "OptimizedLlama",
    "OptimizedMistral",
    "OptimizedPhi",
    "create_optimized_llm",
]
