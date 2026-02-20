# Checkpointing & Fault Tolerance

TorchBridge provides backend-aware checkpoint management with async DCP
integration, cross-backend portability, and health-triggered saves.

## Quick Start

```python
from torchbridge.checkpoint import CheckpointConfig, CheckpointManager
from torchbridge.core.config import HardwareBackend

# Configure checkpoint manager
config = CheckpointConfig(
    storage_path="./checkpoints",
    async_save=True,
    plan_caching=True,
    max_checkpoints_to_keep=3,
)
manager = CheckpointManager(config=config, backend=HardwareBackend.CUDA)

# Save checkpoint
model_state = model.state_dict()
ckpt_id = manager.save(model_state, model_params=int(7e9))

# Load checkpoint
loaded_state = {}
metadata = manager.load(loaded_state, ckpt_id)
model.load_state_dict(loaded_state)
```

## Storage Backends

| Backend | URI Prefix | Package | Use Case |
|---------|-----------|---------|----------|
| LOCAL | `./path` | Built-in | Single-node, NFS |
| S3 | `s3://bucket/` | `s3fs` | AWS training |
| GCS | `gs://bucket/` | `gcsfs` | GCP training |
| Azure | `az://container/` | `adlfs` | Azure training |

Install cloud storage support:
```bash
pip install torchbridge-ml[checkpoint]
```

## Async Checkpointing

Async saves overlap checkpoint I/O with training, minimizing step time impact:

```python
config = CheckpointConfig(
    async_save=True,                # Enable async saving
    process_based_async=True,       # Use process (not thread) to avoid GIL
    pinned_memory_staging=True,     # Pinned memory for GPU→CPU transfer
    plan_caching=True,              # Cache DCP plans for 6x faster saves
)
```

| Mode | Step Impact | GIL Contention |
|------|-----------|----------------|
| Synchronous | Full save time | N/A |
| Thread async | ~5x staging time | Yes |
| Process async | ~3x staging time | No |
| Process + pinned | ~2.5x staging time | No |

## Cross-Backend Portability

Save on one backend, load on another:

```python
# Save on NVIDIA
cuda_manager = CheckpointManager(
    config=CheckpointConfig(normalize_on_save=True),
    backend=HardwareBackend.CUDA,
)
cuda_manager.save(model.state_dict(), checkpoint_id="./ckpt_cuda")

# Load on AMD
amd_manager = CheckpointManager(
    config=CheckpointConfig(),
    backend=HardwareBackend.AMD,
)
state = {}
metadata = amd_manager.load(
    state,
    "./ckpt_cuda",
    target_backend=HardwareBackend.AMD,
    target_device=torch.device("cuda:0"),
)
```

Normalization handles:
- FP8 dtypes (NVIDIA-only) → FP16 on save, restore if target supports FP8
- Device placement → CPU on save, target device on load
- Metadata sidecar records source backend, dtypes, and hardware provenance

## Checkpoint Frequency Advisor

Uses Young's formula to recommend optimal checkpoint interval:

```bash
# CLI usage
torchbridge checkpoint advisor --world-size 64
torchbridge checkpoint advisor --world-size 256 --checkpoint-time 120 --ci
```

```python
# Python API
from torchbridge.checkpoint import CheckpointFrequencyAdvisor

advisor = CheckpointFrequencyAdvisor()
rec = advisor.recommend(
    world_size=64,
    checkpoint_time_seconds=60,
    step_time_seconds=0.5,
)
print(f"Interval: {rec.interval_minutes} min ({rec.interval_steps} steps)")
print(f"Risk: {rec.risk_level}, Overhead: {rec.optimal_overhead_pct:.1f}%")
```

| Cluster Size | Default MTBF | Risk Level |
|-------------|-------------|------------|
| 1-8 GPUs | 168h (1 week) | Low |
| 9-64 GPUs | 48h (2 days) | Medium |
| 65-256 GPUs | 12h | Medium |
| 257+ GPUs | 4h | High |

## Health-Triggered Checkpointing

Automatically trigger checkpoint saves when hardware degradation is detected:

```python
from torchbridge.checkpoint import CheckpointHealthTrigger

trigger = CheckpointHealthTrigger(
    health_threshold_temp_c=85.0,
    trigger_on_degrading=True,
)

# Check single device
should_save, reason = trigger.should_checkpoint(device_health)

# Check entire cluster
should_save, reason = trigger.evaluate_cluster_health(all_device_health)

if should_save:
    manager.save(model.state_dict())
```

Trigger conditions:
- Temperature exceeds threshold (default 85C)
- Health trend transitions to DEGRADING or CRITICAL
- Memory errors detected (ECC corrections)
- Sudden GPU utilization drop (>50% below average)

## CLI Usage

```bash
# Show checkpoint metadata
torchbridge checkpoint info ./checkpoints/checkpoint_20260219

# List all checkpoints
torchbridge checkpoint list ./checkpoints

# Get frequency recommendation (JSON for CI)
torchbridge checkpoint advisor --world-size 128 --ci
```
