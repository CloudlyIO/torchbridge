# Adapter Training Guide

TorchBridge provides a unified adapter training API that automatically selects the optimal fine-tuning method for your hardware. Four methods are supported: **LoRA**, **QLoRA**, **DoRA**, and **QDoRA**.

## Quick Start

```python
import torch
from torchbridge.adapters import AdapterConfig, AdapterEngine, AdapterMethod
from torchbridge.core.config import HardwareBackend

# Configure adapter
config = AdapterConfig(
    method=AdapterMethod.LORA,
    rank=16,
    alpha=32.0,
    target_modules=["q_proj", "v_proj"],
)

# Inject into your model
engine = AdapterEngine(config, backend=HardwareBackend.CUDA)
result = engine.inject(model)

print(f"Adapted {result.modules_adapted} modules")
print(f"Trainable: {result.trainable_params:,} / {result.total_params:,} "
      f"({result.trainable_ratio:.2%})")
```

## Adapter Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **LoRA** | Low-rank adaptation of linear layers | All hardware, general fine-tuning |
| **QLoRA** | 4-bit quantized base + LoRA adapters | NVIDIA/AMD with INT4 support |
| **DoRA** | Weight-decomposed LoRA (magnitude + direction) | Low-rank regimes (r=8-16) |
| **QDoRA** | 4-bit quantized base + DoRA adapters | Blackwell/CDNA4 with FP8+INT4 |

## Backend Compatibility

TorchBridge automatically selects the optimal method per backend:

| Backend | Optimal | Supported Methods |
|---------|---------|-------------------|
| NVIDIA (Hopper+) | QLoRA | QLoRA, QDoRA, DoRA, LoRA |
| NVIDIA (Blackwell DC) | QDoRA | QDoRA, QLoRA, DoRA, LoRA |
| NVIDIA (Turing) | LoRA | LoRA, DoRA |
| AMD (CDNA3+) | QLoRA | QLoRA, DoRA, LoRA |
| AMD (CDNA2) | LoRA | LoRA, DoRA |
| Trainium | LoRA | LoRA |
| TPU | LoRA | LoRA, DoRA |
| CPU | LoRA | LoRA, DoRA |

Use the CLI to check recommendations for your hardware:

```bash
torchbridge adapter recommend --backend nvidia
torchbridge adapter info
```

## Configuration

```python
from torchbridge.adapters import AdapterConfig, AdapterMethod
from torchbridge.adapters.config import InitMethod

config = AdapterConfig(
    method=AdapterMethod.LORA,       # lora, qlora, dora, qdora
    rank=16,                          # Low-rank dimension (r)
    alpha=32.0,                       # Scaling factor (output scaled by alpha/rank)
    dropout=0.05,                     # Dropout on adapter path
    target_modules=["q_proj", "v_proj"],  # Module name suffixes to adapt
    init_method=InitMethod.KAIMING,   # kaiming, gaussian, zeros
    merge_on_save=False,              # Merge adapter into base before saving
)
```

### Target Modules

The `target_modules` parameter specifies which `nn.Linear` layers receive adapters. Modules are matched by name suffix. Common patterns:

- `["q_proj", "v_proj"]` — Attention queries and values (default, most efficient)
- `["q_proj", "k_proj", "v_proj", "o_proj"]` — All attention projections
- `["gate_proj", "up_proj", "down_proj"]` — MLP layers
- All of the above combined for maximum capacity

### Rank Selection

| Rank | Parameters | Use Case |
|------|-----------|----------|
| 4 | Minimal | Quick experiments, small datasets |
| 8-16 | Moderate | General fine-tuning (recommended default: 16) |
| 32-64 | Large | Complex tasks, large datasets |
| 128-256 | Very large | Approaching full fine-tuning capacity |

## Merge for Deployment

After training, merge adapter weights into the base model for zero-overhead inference:

```python
# Before deployment
model.eval()
merged_count = engine.merge(model)
print(f"Merged {merged_count} adapter layers")

# Now model is a plain PyTorch model with no adapter overhead
torch.save(model.state_dict(), "merged_model.pt")
```

## Saving and Loading Adapter Weights

Save only the adapter parameters (typically <1% of model size):

```python
# Save adapter weights
params = engine.get_adapter_params(model)
torch.save(params, "adapter_weights.pt")

# Load into a fresh adapted model
params = torch.load("adapter_weights.pt")
loaded = engine.load_adapter_params(model, params)
print(f"Loaded {loaded} adapter parameters")
```

## Multi-Adapter Serving

Serve multiple adapters on a single base model with LRU caching:

```python
from torchbridge.adapters import MultiAdapterManager

# Create manager with LRU cache
mgr = MultiAdapterManager(model, max_loaded=4)

# Load adapters
mgr.load_adapter("task_a", params_a, config_a)
mgr.load_adapter("task_b", params_b, config_b)

# Hot-swap between adapters
mgr.activate("task_a")
output_a = model(input_ids)

mgr.activate("task_b")
output_b = model(input_ids)

# Deactivate for base-model-only inference
mgr.deactivate()

# List loaded adapters
for adapter in mgr.list_adapters():
    print(f"  {adapter['name']}: {adapter['method']} rank={adapter['rank']} "
          f"active={adapter['active']}")
```

## Automatic Fallback

If your requested method isn't supported on the target hardware, TorchBridge automatically falls back to the next best option:

```python
# Requesting QLoRA on Trainium (not supported)
config = AdapterConfig(method=AdapterMethod.QLORA)
engine = AdapterEngine(config, HardwareBackend.TRAINIUM)
result = engine.inject(model)

print(result.method_requested)  # qlora
print(result.method_applied)    # lora (automatic fallback)
print(result.used_fallback)     # True
```

## CLI Reference

```bash
# Recommend adapter method for hardware
torchbridge adapter recommend --backend cuda
torchbridge adapter recommend --backend amd --rank 8

# JSON output for CI pipelines
torchbridge adapter recommend --backend nvidia --ci

# Show compatibility matrix for all backends
torchbridge adapter info
torchbridge adapter info --ci
```
