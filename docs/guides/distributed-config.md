# Distributed Training Configuration

TorchBridge provides topology-aware distributed training configuration
that auto-selects optimal parallelism strategies per hardware backend.

## Quick Start

```python
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture
from torchbridge.distributed import DistributedConfig

# Auto-configure for 70B model on 16 H100 GPUs across 2 nodes
config = DistributedConfig.auto(
    model_params=int(70e9),
    backend=HardwareBackend.CUDA,
    architecture=NVIDIAArchitecture.HOPPER,
    world_size=16,
    gpus_per_node=8,
)

# Export as TOML for reproducibility
print(config.to_toml())
```

## CLI Advisor

The `torchbridge advisor` command recommends parallelism configuration:

```bash
# Basic recommendation
torchbridge advisor --model-params 7e9 --world-size 8 --backend nvidia

# JSON output for CI
torchbridge advisor --model-params 70e9 --world-size 16 --gpus-per-node 8 --ci

# TOML config file
torchbridge advisor --model-params 32e9 --world-size 8 --toml > dist_config.toml

# Detect cluster topology
torchbridge advisor --model-params 1e9 --topology
```

## Components

### FSDP2 Configuration

Backend-aware FSDP2 with automatic mixed precision and sharding strategy:

| Backend | Architecture | Mixed Precision | Float8 All-Gather |
|---------|-------------|-----------------|-------------------|
| NVIDIA | Blackwell DC | FP8 | Yes |
| NVIDIA | Hopper | BF16 | Yes |
| NVIDIA | Ampere | BF16 | No |
| NVIDIA | Turing | FP16 | No |
| AMD | CDNA3/CDNA4 | BF16 | No |
| Trainium | TRN2/TRN3 | BF16 | No |
| TPU | v5e/v7 | BF16 | No |
| CPU | — | FP32 | No |

Multi-node training automatically uses hybrid sharding (full shard within
node, replicate across nodes).

### Pipeline Schedules

TorchBridge selects the optimal pipeline schedule based on hardware:

| Schedule | Bubble Ratio | Requires Async | Hardware |
|----------|-------------|----------------|----------|
| Zero Bubble | ~0 | Yes | Hopper+, Blackwell |
| ZBV Zero Bubble | 0 | Yes | Hopper+ (4+ stages) |
| Interleaved 1F1B | (p-1)/(m*(v+1)) | No | All |
| Looped BFS | (p-1)/(m*v) | No | CUDA, AMD |
| GPipe | (p-1)/m | No | All |

### Communication Backends

Automatically selects the optimal collective library:

| Hardware | Collective | GPU Direct | Symmetric Memory |
|----------|-----------|------------|-----------------|
| NVIDIA | NCCL | Yes | Yes (Hopper+) |
| AMD | RCCL | Yes | No |
| Trainium | Neuron CC | No | No |
| TPU | XLA | No | No |
| CPU | Gloo | No | No |

### Topology Detection

Automatically detects cluster topology from environment:

1. **SLURM** — reads `SLURM_NNODES`, `SLURM_GPUS_ON_NODE`
2. **Kubernetes** — reads `WORLD_SIZE`, `LOCAL_WORLD_SIZE`
3. **torch.distributed** — reads standard env vars
4. **Single-node** — falls back to local GPU count

## Parallelism Advisor

The advisor recommends tensor parallel degree, pipeline stages, and
FSDP strategy based on:

- **Model size** — parameters determine memory requirements
- **GPU memory** — per-architecture memory capacity
- **World size** — total ranks available
- **Topology** — single-node vs multi-node

### Heuristics

- **TP > 1** when model > 3B params and multiple GPUs per node
- **PP > 1** when model > 30B params and world size >= 8
- **Hybrid shard** when training spans multiple nodes
- **Memory warning** when estimated memory/rank exceeds 90% of GPU capacity
