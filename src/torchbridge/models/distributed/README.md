# TorchBridge Distributed Models (v0.4.13)

Multi-GPU training and inference support for large language models (70B+ parameters).

## Features

### Tensor Parallelism
Split model layers across GPUs for memory efficiency:
- **ColumnParallelLinear**: Splits output dimension
- **RowParallelLinear**: Splits input dimension
- **TensorParallelEmbedding**: Splits embedding dimension
- Automatic all-gather/reduce operations

### Pipeline Parallelism
Split model stages across GPUs with micro-batching:
- **GPipe**: Simple pipeline with flush after each batch
- **Interleaved**: Reduced bubble overhead with interleaving
- Automatic stage partitioning
- Memory estimation tools

### Model Sharding
Intelligent parameter distribution:
- Automatic sharding based on parameter size
- Row/column/table sharding strategies
- Multi-device weight distribution
- FSDP-style fully sharded support

### Large Model Optimizer
Complete optimizer for 70B+ models:
- Llama-2-70B, Falcon-180B, Mixtral-8x7B
- Automatic strategy selection (TP/PP/Sharding)
- GPU requirement estimation
- Memory-efficient loading

## Quick Start

```python
from torchbridge.models.distributed import (
    create_distributed_llm,
    DistributedConfig,
)

# Automatic configuration
model, tokenizer = create_distributed_llm(
    "meta-llama/Llama-2-70b-hf",
    world_size=4,  # 4 GPUs
)

# Manual configuration
config = DistributedConfig(
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    enable_gradient_checkpointing=True,
)
optimizer = DistributedLLMOptimizer("meta-llama/Llama-2-70b-hf", config)
model, tokenizer = optimizer.optimize()
```

## Hardware Requirements

| Model | Parameters | Memory (FP16) | Recommended GPUs |
|-------|------------|---------------|------------------|
| Llama-2-13B | 13B | ~26GB | 2x A100 40GB |
| Llama-2-70B | 70B | ~140GB | 4x A100 40GB or 2x A100 80GB |
| Mixtral-8x7B | 46.7B | ~93GB | 4x A100 40GB |
| Falcon-180B | 180B | ~360GB | 8x A100 80GB |

## Strategies

### Tensor Parallelism (TP)
Best for: Models that fit in memory but need faster computation
- Splits each layer across GPUs
- Low communication overhead
- Linear scaling up to 8 GPUs

### Pipeline Parallelism (PP)
Best for: Very large models that don't fit in single GPU
- Splits model into stages
- Higher latency but lower memory
- Handles 100B+ models

### Hybrid (TP + PP)
Best for: Maximum scale (70B-180B models)
- Combines both strategies
- TP within nodes, PP across nodes
- Optimal for 8-64 GPUs

## Examples

See `examples/models/large/llama_70b_distributed.py` for complete examples.

## Testing

Run distributed tests:
```bash
pytest tests/test_distributed_integration.py
```

Note: Full multi-GPU tests require actual distributed environment.
Single-GPU tests use mocked distributed operations.

## Architecture

```
┌─────────────────────────────────────┐
│   DistributedLLMOptimizer           │
│   ├── Model Type Detection          │
│   ├── Strategy Selection            │
│   └── GPU Requirement Estimation    │
└─────────────────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───┐    ┌───▼────┐    ┌──────────┐
│  TP   │    │   PP   │    │ Sharding │
│Config │    │Config  │    │ Config   │
└───┬───┘    └───┬────┘    └────┬─────┘
    │            │              │
┌───▼───────────▼──────────────▼──┐
│   Distributed Model Wrapper      │
│   ├── Communication Groups       │
│   ├── Memory Management          │
│   └── Gradient Synchronization   │
└──────────────────────────────────┘
```

## Version

Part of TorchBridge v0.4.13 - Large Model Integration

For more details, see the main documentation.
