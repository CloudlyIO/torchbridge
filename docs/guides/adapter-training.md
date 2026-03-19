# Adapter Training Guide

TorchBridge provides a compatibility matrix that selects the optimal fine-tuning adapter method for your hardware. Four methods are defined: **LoRA**, **QLoRA**, **DoRA**, and **QDoRA**. For the actual training loop and parameter injection use PEFT, torchao, or Unsloth directly.

## Backend Compatibility

TorchBridge auto-selects the optimal method per backend:

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

## Query the Compatibility Matrix

```python
from torchbridge.adapters import AdapterCompatibilityMatrix, AdapterMethod
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture

# Get optimal method for hardware
optimal = AdapterCompatibilityMatrix.get_optimal_method(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → AdapterMethod.QLORA

# Get all supported methods
supported = AdapterCompatibilityMatrix.get_supported_methods(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → [QLORA, QDORA, DORA, LORA]

# Check if a specific method is supported
AdapterCompatibilityMatrix.is_method_supported(
    AdapterMethod.QLORA, HardwareBackend.CPU
)
# → False

# Get fallback chain (what to use if method unavailable)
chain = AdapterCompatibilityMatrix.get_fallback_chain(
    AdapterMethod.QLORA, HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
)
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
