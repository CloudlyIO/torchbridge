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
from torchbridge.distributed_scale import AdvancedFSDPManager

fsdp_manager = AdvancedFSDPManager(
    sharding_strategy="FULL_SHARD",  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    mixed_precision=True,
    cpu_offload=False,
    activation_checkpointing=True,
)
model = fsdp_manager.wrap_model(model)
```

Sharding strategies:
- **FULL_SHARD**: Maximum memory savings, shard parameters + gradients + optimizer
- **SHARD_GRAD_OP**: Shard gradients + optimizer only (faster, more memory)
- **NO_SHARD**: Like DDP (baseline comparison)

## Tensor Parallelism

Best for: models with very wide layers (large hidden dimensions).

```python
from torchbridge.models.distributed import ColumnParallelLinear, RowParallelLinear

# Replace linear layers with tensor-parallel equivalents
class ParallelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)))
```

## Pipeline Parallelism

Best for: very deep models (many layers).

```python
from torchbridge.models.distributed import PipelineStage

# Assign model layers to pipeline stages
stages = [
    PipelineStage(layers=model.layers[:8], device="cuda:0"),
    PipelineStage(layers=model.layers[8:16], device="cuda:1"),
    PipelineStage(layers=model.layers[16:24], device="cuda:2"),
    PipelineStage(layers=model.layers[24:], device="cuda:3"),
]
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

TorchBridge provides `MultiNodeTrainingManager` for coordinating multi-node jobs:

```python
from torchbridge.distributed_scale import MultiNodeTrainingManager

manager = MultiNodeTrainingManager()
manager.setup(num_nodes=2, gpus_per_node=8)
```

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

Combine distributed training with memory optimization:

```python
from torchbridge.advanced_memory import SelectiveGradientCheckpointing

# Gradient checkpointing reduces activation memory
checkpointing = SelectiveGradientCheckpointing(model, checkpoint_ratio=0.5)

# CPU offloading for optimizer states
fsdp_manager = AdvancedFSDPManager(cpu_offload=True)
```

## Backend Compatibility

| Strategy | NVIDIA | AMD | Intel | TPU |
|----------|--------|-----|-------|-----|
| DDP | NCCL | Gloo/RCCL | Gloo | XLA |
| FSDP | Yes | Yes | Partial | Partial |
| Tensor Parallel | Yes | Yes | Partial | Partial |
| Pipeline Parallel | Yes | Yes | Partial | Partial |

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
