# Distributed Training

Multi-GPU and multi-node training with TorchBridge's unified distributed API.

## Strategy Overview

| Strategy | Use Case | Memory | Communication |
|----------|----------|--------|---------------|
| **Data Parallel (DDP)** | Multiple GPUs, same model | Full model per GPU | Gradient sync |
| **FSDP** | Large models, limited memory | Sharded model + optimizer | Parameter sync |
| **Tensor Parallel** | Very wide layers | Split within layers | All-reduce per layer |
| **Pipeline Parallel** | Very deep models | Subset of layers per GPU | Activation passing |
| **Hybrid** | 70B+ models | Combined strategies | Mixed |

## Quick Start

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group (launched via torchrun)
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = YourModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

## Data Parallel (DDP)

Best for: models that fit on a single GPU, training on 2-8 GPUs.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")  # "nccl" for NVIDIA, "gloo" for CPU/AMD
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = YourModel().to(local_rank)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```

Launch:
```bash
torchrun --nproc_per_node=4 train.py
```

## FSDP (Fully Sharded Data Parallel)

Best for: models too large for single-GPU memory.

```python
# Use PyTorch FSDP directly — TorchBridge provides config recommendations via tb-advisor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=None,  # configure as needed
    cpu_offload=None,
)
```

Use `tb-advisor --model-params <B> --world-size <N>` to get backend-specific FSDP config recommendations.

Sharding strategies:
- **FULL_SHARD**: Maximum memory savings, shard parameters + gradients + optimizer
- **SHARD_GRAD_OP**: Shard gradients + optimizer only (faster, more memory)
- **NO_SHARD**: Like DDP (baseline comparison)

## Tensor Parallelism

Best for: models with very wide layers (large hidden dimensions). Use PyTorch's native `torch.distributed.tensor.parallel` API directly.

```bash
# TorchBridge advisor shows optimal parallelism strategy for your hardware
torchbridge advisor --model-params 70e9 --world-size 8
```

## Pipeline Parallelism

Best for: very deep models (many layers). Use PyTorch's native `torch.distributed.pipelining` API directly.

```bash
torchbridge advisor --model-params 70e9 --world-size 16
```

## Memory Estimation

Before choosing a strategy, estimate memory requirements:

```python
# Quick parameter-count-based estimate
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
optimizer_bytes = param_bytes * 2  # Adam stores 2 states per param
print(f"Model: {param_bytes / 1e9:.1f} GB")
print(f"Optimizer: {optimizer_bytes / 1e9:.1f} GB")
```

## Multi-Node Training

Launch multi-node jobs directly with `torchrun` — no wrapper needed.

### Single-Node, Multi-GPU

```bash
torchrun --nproc_per_node=8 train.py
```

### Multi-Node

```bash
# Node 0 (master)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 train.py

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 train.py
```

## Memory Optimization

Use PyTorch's built-in gradient checkpointing to reduce activation memory:

```python
from torch.utils.checkpoint import checkpoint

# Apply to specific layers
def forward_with_checkpointing(layer, x):
    return checkpoint(layer, x)
```

For FSDP with CPU offload:
```python
from torch.distributed.fsdp import CPUOffload
model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
```

## Backend Compatibility

| Strategy | NVIDIA | AMD | Trainium | TPU |
|----------|--------|-----|----------|-----|
| DDP | NCCL | Gloo/RCCL | NeuronX | XLA |
| FSDP | Yes | Yes | Yes | Partial |
| Tensor Parallel | Yes | Yes | Yes | Partial |
| Pipeline Parallel | Yes | Yes | Yes | Partial |

## Debugging

```bash
# Check distributed setup
torchrun --nproc_per_node=2 -c "
import torch.distributed as dist
dist.init_process_group('nccl')
print(f'Rank {dist.get_rank()} of {dist.get_world_size()}')
"

# Common issues:
# - NCCL timeout: check network connectivity between nodes
# - OOM: reduce batch size or enable gradient checkpointing
# - Hangs: ensure all ranks execute the same operations
```

## See Also

- [Backends Overview](../backends/overview.md)
- [Backend Selection](backend-selection.md)
- [Deployment](deployment.md)
