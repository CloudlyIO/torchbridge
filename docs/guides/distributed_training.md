# Distributed Training Guide

> **Version**: v0.4.24 | **Status**: Production Ready | **Last Updated**: January 2026

This guide covers distributed training strategies for large language models using KernelPyTorch, including tensor parallelism, pipeline parallelism, FSDP, and hybrid approaches.

## Overview

KernelPyTorch provides multiple parallelism strategies for training models that don't fit on a single GPU:

| Strategy | Memory Efficiency | Communication | Best For |
|----------|-------------------|---------------|----------|
| Tensor Parallel | Medium | High | Wide models (many heads) |
| Pipeline Parallel | High | Low | Deep models (many layers) |
| FSDP | Very High | Medium | Any large model |
| Hybrid | Highest | Medium | 70B+ models |

## Quick Start

### Automatic Strategy Selection

```python
from kernel_pytorch.models.distributed import (
    DistributedLLMOptimizer,
    ParallelismStrategy,
)

# Auto-select best strategy for your hardware
optimizer = DistributedLLMOptimizer(
    model_name="meta-llama/Llama-2-70b-hf",
    num_gpus=8,
    strategy=ParallelismStrategy.AUTO,
)

# Load and optimize
model = optimizer.load_model()
```

### Memory Estimation

Before loading a model, estimate GPU requirements:

```python
from kernel_pytorch.models.distributed import estimate_gpu_requirements

# Check if your hardware can handle it
requirements = estimate_gpu_requirements(
    "meta-llama/Llama-2-70b-hf",
    max_sequence_length=4096,
    max_batch_size=1,
    quantization="none",  # or "int8", "int4"
)

print(f"GPUs needed: {requirements['num_gpus_required']}")
print(f"Memory per GPU: {requirements['memory_per_gpu_gb']:.1f} GB")
```

## Tensor Parallelism

Tensor parallelism splits individual layers across GPUs, ideal for models with large hidden dimensions.

### How It Works

```
Single GPU:                    Tensor Parallel (2 GPUs):
┌─────────────────┐           ┌────────┐  ┌────────┐
│  Linear Layer   │           │ GPU 0  │  │ GPU 1  │
│  [4096, 4096]   │    →      │[4096,  │  │[4096,  │
│                 │           │ 2048]  │  │ 2048]  │
└─────────────────┘           └────────┘  └────────┘
```

### Usage

```python
from kernel_pytorch.models.distributed import (
    TensorParallelConfig,
    apply_tensor_parallelism,
    ColumnParallelLinear,
    RowParallelLinear,
)

# Configure for 8 GPUs
config = TensorParallelConfig(
    world_size=8,
    rank=0,  # Set per process
    sequence_parallel=True,  # Also parallelize sequence dimension
)

# Apply to existing model
parallel_model = apply_tensor_parallelism(model, config)

# Or build with parallel layers directly
class ParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, config):
        super().__init__()
        # Q, K, V projections split across GPUs
        self.qkv = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, config,
            gather_output=False,
        )
        # Output projection with parallel input
        self.out = RowParallelLinear(
            hidden_size, hidden_size, config,
            input_is_parallel=True,
        )
```

### When to Use

- Models with large hidden dimensions (4096+)
- When you need all GPUs to work on the same sample
- For inference with low latency requirements

## Pipeline Parallelism

Pipeline parallelism splits the model by layers across GPUs, processing micro-batches in a pipeline.

### Schedulers

**GPipe (Fill-Drain)**
- Simple implementation
- Higher memory usage (stores all activations)
- Good for small number of stages

**1F1B Interleaved**
- Lower memory (~4x reduction)
- More complex scheduling
- Better for many stages

```
GPipe:
GPU 0: [F1][F2][F3][F4]        [B4][B3][B2][B1]
GPU 1:    [F1][F2][F3][F4]    [B4][B3][B2][B1]

1F1B Interleaved:
GPU 0: [F1][F2][F3][F4][B1][B2][B3][B4]
GPU 1:    [F1][F2][B1][F3][B2][F4][B3][B4]
```

### Usage

```python
from kernel_pytorch.models.distributed import (
    PipelineParallelConfig,
    create_pipeline_stages,
    InterleavedScheduler,
)

# Configure pipeline
config = PipelineParallelConfig(
    num_stages=4,
    num_micro_batches=8,
    stage_id=0,  # Set per process
)

# Create stages from model
stages = create_pipeline_stages(model, config)

# Create scheduler
scheduler = InterleavedScheduler(stages, config)

# Training step
micro_batches = split_batch(batch, num_micro_batches=8)

def loss_fn(output):
    return criterion(output, labels)

total_loss = scheduler.run_forward_backward(micro_batches, loss_fn)
```

### Memory Estimation

```python
from kernel_pytorch.models.distributed import estimate_pipeline_memory

gpipe_mem, interleaved_mem = estimate_pipeline_memory(
    model_size_gb=140,  # 70B model
    num_stages=8,
    num_micro_batches=16,
    activation_per_microbatch_gb=0.5,
)

print(f"GPipe peak memory: {gpipe_mem:.1f} GB")
print(f"1F1B peak memory: {interleaved_mem:.1f} GB")
```

### When to Use

- Deep models with many layers
- When minimizing communication is important
- For training (gradient accumulation across micro-batches)

## FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across GPUs.

### Sharding Strategies

| Strategy | Shards | Memory | Communication |
|----------|--------|--------|---------------|
| FULL_SHARD | Params + Grads + Optim | Lowest | Highest |
| SHARD_GRAD_OP | Grads + Optim | Medium | Medium |
| HYBRID_SHARD | Full within node | Medium | Lower cross-node |
| NO_SHARD | Nothing | Highest | Lowest |

### Usage

```python
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

# Initialize distributed
dist.init_process_group("nccl")

# Configure mixed precision
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)

# Wrap model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
)

# Training is standard after wrapping
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### With KernelPyTorch

```python
from kernel_pytorch.models.distributed import (
    ModelSharder,
    ShardingStrategy,
)

sharder = ModelSharder(
    strategy=ShardingStrategy.FULL_SHARD,
    num_gpus=8,
)

# Analyze model for optimal sharding
shard_plan = sharder.analyze_model(model)

# Apply sharding
sharded_model = sharder.apply(model)
```

### When to Use

- Any model that doesn't fit on one GPU
- When you want simple distributed training
- For maximum memory efficiency

## Hybrid Parallelism

For very large models (70B+), combine multiple strategies:

```python
from kernel_pytorch.models.distributed import (
    DistributedLLMOptimizer,
    ParallelismStrategy,
)

# Hybrid: Tensor parallel within node, pipeline across nodes
optimizer = DistributedLLMOptimizer(
    model_name="meta-llama/Llama-2-70b-hf",
    num_gpus=16,  # 2 nodes × 8 GPUs
    strategy=ParallelismStrategy.HYBRID,
    tensor_parallel_size=8,  # Within node
    pipeline_parallel_size=2,  # Across nodes
)
```

### Recommended Configurations

| Model Size | GPUs | Strategy | TP | PP |
|------------|------|----------|----|----|
| 7B | 1-2 | FSDP | - | - |
| 13B | 2-4 | FSDP or TP | 2-4 | - |
| 70B | 8 | TP | 8 | - |
| 70B | 4 | TP + FSDP | 4 | - |
| 405B | 64 | Hybrid | 8 | 8 |

## Training Example

### Single Node Training

```bash
# 4 GPUs on one machine
torchrun --nproc_per_node=4 train_llama_7b_fsdp.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --mixed_precision \
    --activation_checkpointing
```

### Multi-Node Training

```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=$MASTER_IP --master_port=29500 \
    train_llama_7b_fsdp.py --sharding_strategy HYBRID_SHARD

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_IP --master_port=29500 \
    train_llama_7b_fsdp.py --sharding_strategy HYBRID_SHARD
```

## Memory Optimization Tips

### 1. Activation Checkpointing

Trade compute for memory by recomputing activations during backward:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

# Apply to transformer layers
apply_activation_checkpointing(
    model,
    check_fn=lambda m: isinstance(m, TransformerLayer),
)
```

### 2. Mixed Precision Training

Use BF16/FP16 to halve memory usage:

```python
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
```

### 3. Gradient Accumulation

Simulate larger batches without more memory:

```python
effective_batch_size = batch_size * gradient_accumulation_steps * world_size
```

### 4. CPU Offloading

For extremely large models:

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),
)
```

## Debugging Tips

### Check Memory Usage

```python
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
```

### Verify Distribution

```python
import torch.distributed as dist

def check_distributed():
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()}/{dist.get_world_size()}")
        print(f"Backend: {dist.get_backend()}")
```

### Common Issues

1. **NCCL Timeout**: Increase timeout with `NCCL_TIMEOUT=1800`
2. **OOM during forward**: Enable activation checkpointing
3. **OOM during backward**: Use smaller micro-batches
4. **Slow training**: Check network bandwidth, use HYBRID_SHARD for multi-node

## API Reference

### DistributedLLMOptimizer

```python
DistributedLLMOptimizer(
    model_name: str,
    num_gpus: int,
    strategy: ParallelismStrategy = AUTO,
    tensor_parallel_size: int = None,
    pipeline_parallel_size: int = None,
    quantization: str = "none",
)
```

### TensorParallelConfig

```python
TensorParallelConfig(
    world_size: int = 1,
    rank: int = 0,
    sequence_parallel: bool = False,
    async_tensor_model_parallel_allreduce: bool = False,
)
```

### PipelineParallelConfig

```python
PipelineParallelConfig(
    num_stages: int,
    num_micro_batches: int,
    stage_id: int,
    activation_checkpointing: bool = False,
)
```

## See Also

- [Efficient Attention Guide](efficient_attention_guide.md) - Memory-efficient attention
- [Quantization Guide](quantization_guide.md) - Reduce model size
- [Backend Selection](../backends/README.md) - Hardware backends
