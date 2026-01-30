"""
Large Model Optimizer for Distributed Inference and Training

This module provides the main interface for optimizing large models
like Llama-70B, Falcon-180B, and Mixtral across multiple GPUs.

Features:
- Automatic parallelism strategy selection
- Memory-efficient inference
- Optimized generation with KV-cache sharding
- Integration with HuggingFace transformers
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from .model_sharding import (
    ModelSharder,
    ShardingConfig,
    ShardingStrategy,
    WeightDistributor,
    automatic_sharding,
)
from .pipeline_parallel import (
    PipelineParallelConfig,
    ScheduleType,
    create_pipeline_stages,
)
from .tensor_parallel import TensorParallelConfig, apply_tensor_parallelism

logger = logging.getLogger(__name__)


class LargeModelType(Enum):
    """Supported large model types."""
    LLAMA_70B = "llama-70b"
    LLAMA_405B = "llama-405b"
    FALCON_40B = "falcon-40b"
    FALCON_180B = "falcon-180b"
    MIXTRAL = "mixtral"
    MIXTRAL_8X22B = "mixtral-8x22b"
    QWEN_72B = "qwen-72b"
    DBRX = "dbrx"
    CUSTOM = "custom"


class ParallelismStrategy(Enum):
    """Parallelism strategy for large models."""
    TENSOR_PARALLEL = "tensor_parallel"  # Split layers across GPUs
    PIPELINE_PARALLEL = "pipeline_parallel"  # Split stages across GPUs
    HYBRID = "hybrid"  # Tensor + Pipeline parallel
    FSDP = "fsdp"  # Fully sharded data parallel
    AUTO = "auto"  # Automatic selection


@dataclass
class DistributedConfig:
    """Configuration for distributed large model optimization.

    Args:
        world_size: Total number of GPUs
        rank: Current GPU rank
        tensor_parallel_size: GPUs for tensor parallelism
        pipeline_parallel_size: GPUs for pipeline parallelism
        data_parallel_size: GPUs for data parallelism
        strategy: Parallelism strategy
        dtype: Data type for model weights
        quantization: Quantization mode (none, int8, int4, fp8)
        use_flash_attention: Enable flash attention
        use_kv_cache: Enable KV-cache for generation
        max_sequence_length: Maximum sequence length
        max_batch_size: Maximum batch size
        activation_checkpointing: Enable activation checkpointing
        cpu_offload: Enable CPU offloading
        nvlink_available: Whether NVLink is available
    """
    world_size: int = 1
    rank: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    strategy: ParallelismStrategy = ParallelismStrategy.AUTO
    dtype: torch.dtype = torch.bfloat16
    quantization: str = "none"  # none, int8, int4, fp8
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    max_sequence_length: int = 4096
    max_batch_size: int = 1
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    nvlink_available: bool = True

    def __post_init__(self):
        # Validate parallelism dimensions
        total = self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        if total != self.world_size:
            logger.warning(
                f"Parallelism sizes ({self.tensor_parallel_size} * {self.pipeline_parallel_size} * "
                f"{self.data_parallel_size} = {total}) != world_size ({self.world_size}). Adjusting."
            )
            self._auto_adjust_parallelism()

    def _auto_adjust_parallelism(self):
        """Auto-adjust parallelism dimensions to match world_size."""
        if self.strategy == ParallelismStrategy.TENSOR_PARALLEL:
            self.tensor_parallel_size = self.world_size
            self.pipeline_parallel_size = 1
            self.data_parallel_size = 1
        elif self.strategy == ParallelismStrategy.PIPELINE_PARALLEL:
            self.tensor_parallel_size = 1
            self.pipeline_parallel_size = self.world_size
            self.data_parallel_size = 1
        else:
            # Hybrid: prioritize tensor parallelism up to 8 GPUs
            if self.world_size <= 8:
                self.tensor_parallel_size = self.world_size
                self.pipeline_parallel_size = 1
            else:
                self.tensor_parallel_size = 8
                self.pipeline_parallel_size = self.world_size // 8
            self.data_parallel_size = 1


class DistributedLLMOptimizer:
    """Optimizer for large language models across multiple GPUs.

    This class handles:
    - Automatic parallelism strategy selection
    - Model loading and distribution
    - Optimized inference with batching
    - Memory management across devices
    """

    # Model specifications (parameters in billions, layers, hidden size)
    MODEL_SPECS = {
        LargeModelType.LLAMA_70B: {"params": 70, "layers": 80, "hidden": 8192, "heads": 64},
        LargeModelType.LLAMA_405B: {"params": 405, "layers": 126, "hidden": 16384, "heads": 128},
        LargeModelType.FALCON_40B: {"params": 40, "layers": 60, "hidden": 8192, "heads": 64},
        LargeModelType.FALCON_180B: {"params": 180, "layers": 80, "hidden": 14848, "heads": 232},
        LargeModelType.MIXTRAL: {"params": 47, "layers": 32, "hidden": 4096, "heads": 32, "experts": 8},
        LargeModelType.MIXTRAL_8X22B: {"params": 141, "layers": 56, "hidden": 6144, "heads": 48, "experts": 8},
        LargeModelType.QWEN_72B: {"params": 72, "layers": 80, "hidden": 8192, "heads": 64},
        LargeModelType.DBRX: {"params": 132, "layers": 40, "hidden": 6144, "heads": 48, "experts": 16},
    }

    def __init__(
        self,
        model_name: str,
        config: DistributedConfig | None = None,
        model_type: LargeModelType | None = None,
    ):
        """Initialize distributed LLM optimizer.

        Args:
            model_name: HuggingFace model name or path
            config: Distributed configuration
            model_type: Model type (auto-detected if None)
        """
        self.model_name = model_name
        self.config = config or DistributedConfig()
        self.model_type = model_type or self._detect_model_type(model_name)

        self._model: nn.Module | None = None
        self._tokenizer: Any | None = None
        self._device_map: dict[str, str] | None = None

        # Initialize sub-components
        self._tp_config: TensorParallelConfig | None = None
        self._pp_config: PipelineParallelConfig | None = None
        self._sharder: ModelSharder | None = None
        self._distributor: WeightDistributor | None = None

        # Select parallelism strategy
        self._select_strategy()

    def _detect_model_type(self, model_name: str) -> LargeModelType:
        """Detect model type from model name.

        Args:
            model_name: Model name or path

        Returns:
            Detected model type
        """
        name_lower = model_name.lower()

        if "llama" in name_lower:
            if "405b" in name_lower:
                return LargeModelType.LLAMA_405B
            elif "70b" in name_lower:
                return LargeModelType.LLAMA_70B
        elif "falcon" in name_lower:
            if "180b" in name_lower:
                return LargeModelType.FALCON_180B
            elif "40b" in name_lower:
                return LargeModelType.FALCON_40B
        elif "mixtral" in name_lower:
            if "22b" in name_lower:
                return LargeModelType.MIXTRAL_8X22B
            else:
                return LargeModelType.MIXTRAL
        elif "qwen" in name_lower and "72b" in name_lower:
            return LargeModelType.QWEN_72B
        elif "dbrx" in name_lower:
            return LargeModelType.DBRX

        return LargeModelType.CUSTOM

    def _select_strategy(self) -> None:
        """Select optimal parallelism strategy based on model and hardware."""
        if self.config.strategy != ParallelismStrategy.AUTO:
            logger.info(f"Using specified strategy: {self.config.strategy.value}")
            return

        # Get model specs
        specs = self.MODEL_SPECS.get(self.model_type, {"params": 70, "layers": 80})
        params_gb = specs["params"] * 2  # Assume FP16, 2 bytes per param

        # Estimate GPU memory available
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory_gb = 80  # Assume A100

        total_gpu_memory = gpu_memory_gb * self.config.world_size

        # Strategy selection heuristics
        if params_gb <= gpu_memory_gb * 0.7:
            # Model fits on single GPU
            self.config.strategy = ParallelismStrategy.TENSOR_PARALLEL
            self.config.tensor_parallel_size = 1
            logger.info("Model fits on single GPU, using no parallelism")

        elif params_gb <= total_gpu_memory * 0.5 and self.config.world_size <= 8:
            # Model fits with tensor parallelism
            self.config.strategy = ParallelismStrategy.TENSOR_PARALLEL
            self.config.tensor_parallel_size = self.config.world_size
            logger.info(f"Using tensor parallelism with {self.config.tensor_parallel_size} GPUs")

        elif self.config.nvlink_available and self.config.world_size > 8:
            # Use hybrid for very large models
            self.config.strategy = ParallelismStrategy.HYBRID
            self.config.tensor_parallel_size = min(8, self.config.world_size)
            self.config.pipeline_parallel_size = self.config.world_size // self.config.tensor_parallel_size
            logger.info(
                f"Using hybrid parallelism: TP={self.config.tensor_parallel_size}, "
                f"PP={self.config.pipeline_parallel_size}"
            )

        else:
            # Fall back to pipeline parallelism
            self.config.strategy = ParallelismStrategy.PIPELINE_PARALLEL
            self.config.pipeline_parallel_size = self.config.world_size
            logger.info(f"Using pipeline parallelism with {self.config.pipeline_parallel_size} stages")

    def load_model(
        self,
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
    ) -> nn.Module:
        """Load and distribute the model.

        Args:
            trust_remote_code: Trust remote code in HuggingFace models
            low_cpu_mem_usage: Use memory-efficient loading

        Returns:
            Distributed model
        """
        logger.info(f"Loading model: {self.model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            HAS_TRANSFORMERS = True
        except ImportError:
            HAS_TRANSFORMERS = False
            logger.warning("transformers not available, using mock model")

        if HAS_TRANSFORMERS:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code,
            )

            # Prepare loading kwargs
            load_kwargs = {
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "torch_dtype": self.config.dtype,
            }

            # Add quantization config if needed
            if self.config.quantization == "int8":
                load_kwargs["load_in_8bit"] = True
            elif self.config.quantization == "int4":
                load_kwargs["load_in_4bit"] = True

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
        else:
            # Create mock model for testing
            self._model = self._create_mock_model()

        # Apply parallelism
        self._apply_parallelism()

        return self._model

    def _create_mock_model(self) -> nn.Module:
        """Create a mock model for testing without transformers."""
        specs = self.MODEL_SPECS.get(self.model_type, {"layers": 80, "hidden": 8192})

        class MockTransformerLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.attention = nn.Linear(hidden_size, hidden_size)
                self.mlp = nn.Linear(hidden_size, hidden_size)
                self.norm = nn.LayerNorm(hidden_size)

            def forward(self, x):
                x = x + self.attention(self.norm(x))
                x = x + self.mlp(self.norm(x))
                return x

        class MockLargeModel(nn.Module):
            def __init__(self, num_layers, hidden_size):
                super().__init__()
                self.embed = nn.Embedding(32000, hidden_size)
                self.layers = nn.ModuleList([
                    MockTransformerLayer(hidden_size) for _ in range(num_layers)
                ])
                self.head = nn.Linear(hidden_size, 32000)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.head(x)

        return MockLargeModel(specs["layers"], specs["hidden"])

    def _apply_parallelism(self) -> None:
        """Apply selected parallelism strategy to the model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        strategy = self.config.strategy

        if strategy == ParallelismStrategy.TENSOR_PARALLEL:
            self._apply_tensor_parallelism()

        elif strategy == ParallelismStrategy.PIPELINE_PARALLEL:
            self._apply_pipeline_parallelism()

        elif strategy == ParallelismStrategy.HYBRID:
            self._apply_tensor_parallelism()
            self._apply_pipeline_parallelism()

        elif strategy == ParallelismStrategy.FSDP:
            self._apply_fsdp()

        # Move to GPU
        if torch.cuda.is_available() and self.config.world_size == 1:
            self._model = self._model.cuda()

        logger.info(f"Applied {strategy.value} parallelism")

    def _apply_tensor_parallelism(self) -> None:
        """Apply tensor parallelism to the model."""
        self._tp_config = TensorParallelConfig(
            world_size=self.config.tensor_parallel_size,
            rank=self.config.rank % self.config.tensor_parallel_size,
            sequence_parallel=True,
        )

        # Apply to model
        self._model = apply_tensor_parallelism(
            self._model,
            self._tp_config,
            linear_layer_names=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            embedding_layer_names=["embed_tokens", "lm_head"],
        )

    def _apply_pipeline_parallelism(self) -> None:
        """Apply pipeline parallelism to the model."""
        self._pp_config = PipelineParallelConfig(
            num_stages=self.config.pipeline_parallel_size,
            num_micro_batches=self.config.max_batch_size,
            stage_id=self.config.rank // self.config.tensor_parallel_size,
            schedule=ScheduleType.INTERLEAVED,
            activation_checkpointing=self.config.activation_checkpointing,
        )

        # Create pipeline stages
        self._stages = create_pipeline_stages(self._model, self._pp_config)

    def _apply_fsdp(self) -> None:
        """Apply FSDP (Fully Sharded Data Parallel) to the model."""
        shard_config = ShardingConfig(
            strategy=ShardingStrategy.FULL_SHARD,
            world_size=self.config.world_size,
            rank=self.config.rank,
            cpu_offload=self.config.cpu_offload,
            mixed_precision=self.config.dtype in [torch.float16, torch.bfloat16],
        )

        self._model, info = automatic_sharding(self._model, shard_config)
        logger.info(f"FSDP sharding: {info}")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text with the distributed model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs [batch_size, seq_len + new_tokens]
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use HuggingFace generate if available
        if hasattr(self._model, "generate"):
            return self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id if self._tokenizer else 0,
                **kwargs
            )

        # Manual generation loop for mock models
        return self._manual_generate(
            input_ids, max_new_tokens, temperature, top_p, top_k, do_sample
        )

    def _manual_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """Manual generation loop for models without generate method."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self._model(generated)

                # Get logits for last position
                if isinstance(outputs, tuple):
                    logits = outputs[0][:, -1, :]
                else:
                    logits = outputs[:, -1, :]

                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                # Sample or greedy
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)

        return generated

    def estimate_memory(self) -> dict[str, float]:
        """Estimate memory requirements per GPU.

        Returns:
            Dictionary with memory estimates in GB
        """
        specs = self.MODEL_SPECS.get(self.model_type, {"params": 70, "layers": 80, "hidden": 8192})
        params_b = specs["params"]

        # Base memory (parameters)
        if self.config.quantization == "int8":
            bytes_per_param = 1
        elif self.config.quantization == "int4":
            bytes_per_param = 0.5
        elif self.config.quantization == "fp8":
            bytes_per_param = 1
        elif self.config.dtype == torch.float32:
            bytes_per_param = 4
        else:  # fp16/bf16
            bytes_per_param = 2

        total_param_memory_gb = (params_b * 1e9 * bytes_per_param) / (1024**3)

        # Memory per GPU after parallelism
        param_memory_per_gpu = total_param_memory_gb / self.config.tensor_parallel_size
        param_memory_per_gpu /= self.config.pipeline_parallel_size

        # KV-cache memory
        hidden = specs.get("hidden", 8192)
        heads = specs.get("heads", 64)
        layers = specs.get("layers", 80)
        head_dim = hidden // heads

        kv_cache_per_layer = (
            2 *  # K and V
            self.config.max_batch_size *
            self.config.max_sequence_length *
            heads // self.config.tensor_parallel_size *
            head_dim *
            bytes_per_param
        )
        kv_cache_total_gb = (kv_cache_per_layer * layers / self.config.pipeline_parallel_size) / (1024**3)

        # Activation memory (rough estimate)
        activation_memory_gb = (
            self.config.max_batch_size *
            self.config.max_sequence_length *
            hidden *
            bytes_per_param *
            4  # Intermediate activations
        ) / (1024**3)

        return {
            "total_params_gb": total_param_memory_gb,
            "params_per_gpu_gb": param_memory_per_gpu,
            "kv_cache_gb": kv_cache_total_gb,
            "activation_gb": activation_memory_gb,
            "total_per_gpu_gb": param_memory_per_gpu + kv_cache_total_gb + activation_memory_gb,
            "recommended_gpus": max(1, int(total_param_memory_gb / 40)),  # Assuming 40GB GPUs
        }


# Convenience wrappers for specific models
class DistributedLlama70B(DistributedLLMOptimizer):
    """Optimized wrapper for Llama-2-70B."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-70b-hf",
        config: DistributedConfig | None = None,
    ):
        if config is None:
            config = DistributedConfig(
                world_size=8,  # Recommended for 70B
                tensor_parallel_size=8,
                dtype=torch.bfloat16,
                use_flash_attention=True,
            )
        super().__init__(model_name, config, LargeModelType.LLAMA_70B)


class DistributedFalcon(DistributedLLMOptimizer):
    """Optimized wrapper for Falcon-40B/180B."""

    def __init__(
        self,
        model_name: str = "tiiuae/falcon-40b",
        config: DistributedConfig | None = None,
    ):
        model_type = LargeModelType.FALCON_180B if "180b" in model_name.lower() else LargeModelType.FALCON_40B
        if config is None:
            world_size = 16 if model_type == LargeModelType.FALCON_180B else 4
            config = DistributedConfig(
                world_size=world_size,
                tensor_parallel_size=min(8, world_size),
                dtype=torch.bfloat16,
            )
        super().__init__(model_name, config, model_type)


class DistributedMixtral(DistributedLLMOptimizer):
    """Optimized wrapper for Mixtral MoE models."""

    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-v0.1",
        config: DistributedConfig | None = None,
    ):
        model_type = LargeModelType.MIXTRAL_8X22B if "22b" in model_name.lower() else LargeModelType.MIXTRAL
        if config is None:
            config = DistributedConfig(
                world_size=8,
                tensor_parallel_size=8,
                dtype=torch.bfloat16,
                use_flash_attention=True,
            )
        super().__init__(model_name, config, model_type)


def create_distributed_llm(
    model_name: str,
    world_size: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    quantization: str = "none",
    strategy: ParallelismStrategy = ParallelismStrategy.AUTO,
    **kwargs
) -> DistributedLLMOptimizer:
    """Create distributed LLM optimizer with automatic configuration.

    Args:
        model_name: HuggingFace model name or path
        world_size: Number of GPUs to use
        dtype: Data type for model weights
        quantization: Quantization mode (none, int8, int4, fp8)
        strategy: Parallelism strategy
        **kwargs: Additional configuration options

    Returns:
        Configured DistributedLLMOptimizer
    """
    config = DistributedConfig(
        world_size=world_size,
        dtype=dtype,
        quantization=quantization,
        strategy=strategy,
        **kwargs
    )

    return DistributedLLMOptimizer(model_name, config)


def estimate_gpu_requirements(
    model_name: str,
    max_sequence_length: int = 4096,
    max_batch_size: int = 1,
    quantization: str = "none",
    gpu_memory_gb: float = 80.0,
) -> dict[str, Any]:
    """Estimate GPU requirements for a large model.

    Args:
        model_name: Model name to analyze
        max_sequence_length: Maximum sequence length
        max_batch_size: Maximum batch size
        quantization: Quantization mode
        gpu_memory_gb: Memory per GPU in GB

    Returns:
        Dictionary with GPU requirements
    """
    # Create temporary optimizer to get estimates
    config = DistributedConfig(
        world_size=1,
        max_sequence_length=max_sequence_length,
        max_batch_size=max_batch_size,
        quantization=quantization,
    )
    optimizer = DistributedLLMOptimizer(model_name, config)

    # Get memory estimate
    memory = optimizer.estimate_memory()
    total_memory_needed = memory["total_params_gb"] + memory["kv_cache_gb"]

    # Calculate required GPUs
    min_gpus = max(1, int(total_memory_needed / (gpu_memory_gb * 0.8)))  # 80% utilization target

    # Recommend parallelism configuration
    if min_gpus <= 8:
        recommended_tp = min_gpus
        recommended_pp = 1
    else:
        recommended_tp = 8
        recommended_pp = min_gpus // 8

    return {
        "model_name": model_name,
        "model_type": optimizer.model_type.value,
        "memory_estimates": memory,
        "min_gpus": min_gpus,
        "recommended_gpus": max(min_gpus, memory["recommended_gpus"]),
        "recommended_tensor_parallel": recommended_tp,
        "recommended_pipeline_parallel": recommended_pp,
        "quantization_suggestion": (
            "Consider int8 or int4 quantization to reduce GPU requirements"
            if min_gpus > 4 and quantization == "none"
            else None
        ),
    }
