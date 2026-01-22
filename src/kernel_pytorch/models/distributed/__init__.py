"""
KernelPyTorch Distributed Model Support (v0.4.13)

This module provides distributed inference and training support for large models
like Llama-70B, Falcon-180B, and other models that exceed single-GPU memory.

Features:
- Tensor Parallelism: Split model layers across GPUs
- Pipeline Parallelism: Split model stages across GPUs
- Model Sharding: Automatic weight distribution strategies
- Multi-GPU Coordination: Efficient communication patterns
- Memory Optimization: Activation checkpointing + offloading

Example:
    from kernel_pytorch.models.distributed import (
        DistributedLLMOptimizer,
        TensorParallelConfig,
        create_distributed_llm,
    )

    # Create distributed optimizer for Llama-70B
    optimizer = DistributedLLMOptimizer(
        model_name="meta-llama/Llama-2-70b-hf",
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
    )

    # Or use factory function
    model = create_distributed_llm(
        "meta-llama/Llama-2-70b-hf",
        world_size=8,
    )
"""

from .tensor_parallel import (
    TensorParallelConfig,
    TensorParallelLinear,
    TensorParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
    apply_tensor_parallelism,
)

from .pipeline_parallel import (
    PipelineParallelConfig,
    PipelineStage,
    PipelineScheduler,
    GPipeScheduler,
    InterleavedScheduler,
    create_pipeline_stages,
    estimate_pipeline_memory,
)

from .model_sharding import (
    ShardingStrategy,
    ShardingConfig,
    ModelSharder,
    WeightDistributor,
    automatic_sharding,
)

from .large_model_optimizer import (
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
    # Tensor Parallelism
    "TensorParallelConfig",
    "TensorParallelLinear",
    "TensorParallelEmbedding",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "apply_tensor_parallelism",

    # Pipeline Parallelism
    "PipelineParallelConfig",
    "PipelineStage",
    "PipelineScheduler",
    "GPipeScheduler",
    "InterleavedScheduler",
    "create_pipeline_stages",
    "estimate_pipeline_memory",

    # Model Sharding
    "ShardingStrategy",
    "ShardingConfig",
    "ModelSharder",
    "WeightDistributor",
    "automatic_sharding",

    # Large Model Optimizer
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
