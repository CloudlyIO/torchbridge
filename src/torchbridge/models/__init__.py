"""
TorchBridge Model Integration Module

Provides HAL-aware utilities for model training and inference across
NVIDIA, AMD, Trainium, and TPU backends.

Modules:
- llm: LLM inference utilities (KV cache, generation config, quantization)
- distributed: Distributed training primitives (tensor/pipeline parallelism, FSDP, sharding)
"""

# Distributed models
from .distributed import (
    ColumnParallelLinear,
    DistributedConfig,
    DistributedFalcon,
    DistributedLlama70B,
    # Large Model Optimizer
    DistributedLLMOptimizer,
    DistributedMixtral,
    GPipeScheduler,
    InterleavedScheduler,
    LargeModelType,
    ModelSharder,
    ParallelismStrategy,
    # Pipeline Parallelism
    PipelineParallelConfig,
    PipelineScheduler,
    PipelineStage,
    RowParallelLinear,
    ShardingConfig,
    # Model Sharding
    ShardingStrategy,
    # Tensor Parallelism
    TensorParallelConfig,
    TensorParallelEmbedding,
    TensorParallelLinear,
    WeightDistributor,
    apply_tensor_parallelism,
    automatic_sharding,
    create_distributed_llm,
    create_pipeline_stages,
    estimate_gpu_requirements,
    estimate_pipeline_memory,
)
from .llm import (
    GenerationConfig,
    KVCacheManager,
    LLMConfig,
    LLMOptimizer,
    OptimizedLlama,
    OptimizedMistral,
    OptimizedPhi,
    PagedKVCache,
    QuantizationMode,
    SlidingWindowCache,
    create_optimized_llm,
)

__all__ = [
    # LLM models
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
    # Distributed models
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
