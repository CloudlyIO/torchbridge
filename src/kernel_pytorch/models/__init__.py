"""
KernelPyTorch Model Integration Module

Provides optimized wrappers for popular pre-trained models from HuggingFace
and other sources. Supports automatic backend selection and optimization
across NVIDIA, AMD, TPU, and Intel hardware.

Model Categories:
- text: BERT, GPT-2, DistilBERT, and other text models (v0.4.11)
- llm: Llama, Mistral, Phi, and other LLMs (v0.4.12)
- distributed: Large-scale distributed models (v0.4.13 - IN PROGRESS)
- vision: ResNet, ViT, Stable Diffusion (v0.4.14 - PLANNED)
- multimodal: CLIP, LLaVA, Whisper (v0.4.15 - PLANNED)

Version: 0.4.12
"""

from .text import (
    TextModelOptimizer,
    OptimizedBERT,
    OptimizedGPT2,
    OptimizedDistilBERT,
    create_optimized_text_model,
)

from .llm import (
    LLMOptimizer,
    LLMConfig,
    OptimizedLlama,
    OptimizedMistral,
    OptimizedPhi,
    create_optimized_llm,
    QuantizationMode,
    GenerationConfig,
    KVCacheManager,
    PagedKVCache,
    SlidingWindowCache,
)

# Distributed models (v0.4.13)
from .distributed import (
    # Tensor Parallelism
    TensorParallelConfig,
    TensorParallelLinear,
    TensorParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
    apply_tensor_parallelism,
    # Pipeline Parallelism
    PipelineParallelConfig,
    PipelineStage,
    PipelineScheduler,
    GPipeScheduler,
    InterleavedScheduler,
    create_pipeline_stages,
    estimate_pipeline_memory,
    # Model Sharding
    ShardingStrategy,
    ShardingConfig,
    ModelSharder,
    WeightDistributor,
    automatic_sharding,
    # Large Model Optimizer
    DistributedLLMOptimizer,
    DistributedConfig,
    DistributedLlama70B,
    DistributedFalcon,
    DistributedMixtral,
    create_distributed_llm,
    estimate_gpu_requirements,
    LargeModelType,
    ParallelismStrategy,
)

__all__ = [
    # Text models (v0.4.11)
    "TextModelOptimizer",
    "OptimizedBERT",
    "OptimizedGPT2",
    "OptimizedDistilBERT",
    "create_optimized_text_model",
    # LLM models (v0.4.12)
    "LLMOptimizer",
    "LLMConfig",
    "OptimizedLlama",
    "OptimizedMistral",
    "OptimizedPhi",
    "create_optimized_llm",
    "QuantizationMode",
    "GenerationConfig",
    "KVCacheManager",
    "PagedKVCache",
    "SlidingWindowCache",
    # Distributed models (v0.4.13)
    "TensorParallelConfig",
    "TensorParallelLinear",
    "TensorParallelEmbedding",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "apply_tensor_parallelism",
    "PipelineParallelConfig",
    "PipelineStage",
    "PipelineScheduler",
    "GPipeScheduler",
    "InterleavedScheduler",
    "create_pipeline_stages",
    "estimate_pipeline_memory",
    "ShardingStrategy",
    "ShardingConfig",
    "ModelSharder",
    "WeightDistributor",
    "automatic_sharding",
    "DistributedLLMOptimizer",
    "DistributedConfig",
    "DistributedLlama70B",
    "DistributedFalcon",
    "DistributedMixtral",
    "create_distributed_llm",
    "estimate_gpu_requirements",
    "LargeModelType",
    "ParallelismStrategy",
]
