# Model Optimization Guide

TorchBridge provides two forms of model optimization: **quantization** (reduce precision to lower
memory and increase throughput) and **adapter training** (efficient fine-tuning via LoRA/QLoRA/DoRA/QDoRA).
Both use a backend-aware compatibility matrix that auto-selects the optimal approach for your hardware.

## Quantization

### Quick Start

```python
from torchbridge.precision.quantization import QuantizationEngine

engine = QuantizationEngine()
result = engine.quantize(model, format="auto")
print(f"Applied: {result.format_applied.value}")
print(f"Memory reduction: {result.memory_reduction_pct:.1f}%")
```

### Supported Formats

| Format | Bits | Memory Reduction | Calibration | torchao Required |
|--------|------|-----------------|-------------|------------------|
| INT8 Dynamic | 8 | 50% | No | No |
| INT8 SmoothQuant | 8 | 50% | Yes | Yes |
| INT4 Weight-Only | 4 | 75% | No | Yes |
| FP8 E4M3 | 8 | 50% | No | No |
| FP8 E5M2 | 8 | 50% | No | No |
| NVFP4 | 4 | 87.5% | No | No |
| BF16 | 16 | 50% | No | No |

### Backend Compatibility Matrix

| Backend | Architecture | Optimal Format | Fallback Chain |
|---------|-------------|---------------|----------------|
| NVIDIA | Blackwell DC | NVFP4 | FP8 E4M3 > INT8 |
| NVIDIA | Blackwell Consumer | FP8 E4M3 | INT8 > INT4 |
| NVIDIA | Hopper | FP8 E4M3 | INT8 SmoothQuant > INT4 |
| NVIDIA | Ampere/Ada | INT8 SmoothQuant | FP8 E4M3 > INT4 |
| AMD | CDNA3/CDNA4 | FP8 E4M3 | INT8 > INT4 |
| AMD | CDNA2 | INT8 Dynamic | INT4 |
| Trainium | TRN2/TRN3 | FP8 E4M3 | BF16 |
| Trainium | TRN1/INF2 | BF16 | -- |
| TPU | v5+/v7 | FP8 E4M3 | BF16 |
| TPU | v4 | BF16 | -- |
| CPU | any | INT8 Dynamic | INT4 Weight-Only |

### Python API

```python
from torchbridge.precision.quantization import QuantizationEngine, QuantizationCompatibilityMatrix
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture

engine = QuantizationEngine()

# Auto-select optimal format for detected hardware
result = engine.quantize(model, format="auto")

# Explicit format
result = engine.quantize(model, format="int8_dynamic")

# Query compatibility matrix
formats = QuantizationCompatibilityMatrix.get_supported_formats(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
optimal = QuantizationCompatibilityMatrix.get_optimal_format(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
```

When a requested format is unsupported, the engine automatically falls back:

```python
result = engine.quantize(model, format="nvfp4")  # on CPU
# result.used_fallback == True
# result.format_applied == QuantizationFormat.INT8_DYNAMIC
# result.warnings contains fallback message
```

### CLI

```bash
# Auto-select format for detected hardware
tb-quantize --model model.pt

# Explicit format
tb-quantize --model model.pt --format int8_dynamic

# Override backend
tb-quantize --model model.pt --backend nvidia --format fp8_e4m3

# Save quantized model
tb-quantize --model model.pt --output quantized.pt

# Validate output quality after quantization
tb-quantize --model model.pt --validate

# CI mode (JSON output)
tb-quantize --model model.pt --ci
```

### torchao Integration

Some formats (INT4 Weight-Only, SmoothQuant) require torchao:

```bash
pip install torchbridge-ml[quantization]
# or directly:
pip install torchao
```

When torchao is not installed, INT4/SmoothQuant fall back to INT8 Dynamic automatically.

---

## Adapter Training

TorchBridge provides a compatibility matrix that selects the optimal fine-tuning adapter
method for your hardware. Four methods are defined: **LoRA**, **QLoRA**, **DoRA**, and
**QDoRA**. For the actual training loop and parameter injection, use PEFT, torchao, or
Unsloth directly.

### Backend Compatibility

| Backend | Optimal | Supported Methods |
|---------|---------|-------------------|
| NVIDIA (Blackwell DC) | QDoRA | QDoRA, QLoRA, DoRA, LoRA |
| NVIDIA (Hopper+) | QLoRA | QLoRA, QDoRA, DoRA, LoRA |
| NVIDIA (Turing) | LoRA | LoRA, DoRA |
| AMD (CDNA3+) | QLoRA | QLoRA, DoRA, LoRA |
| AMD (CDNA2) | LoRA | LoRA, DoRA |
| Trainium | LoRA | LoRA |
| TPU | LoRA | LoRA, DoRA |
| CPU | LoRA | LoRA, DoRA |

### Query the Compatibility Matrix

```python
from torchbridge.adapters import AdapterCompatibilityMatrix, AdapterMethod
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture

# Get optimal method for hardware
optimal = AdapterCompatibilityMatrix.get_optimal(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → AdapterMethod.QLORA

# Get fallback chain (ordered list of methods, best first)
chain = AdapterCompatibilityMatrix.get_fallback_chain(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → [QLORA, QDORA, DORA, LORA]

# Check if a specific method is supported
AdapterCompatibilityMatrix.supports_method(
    HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE, AdapterMethod.QLORA
)
# → True
```

### Configuration

```python
from torchbridge.adapters import AdapterConfig, AdapterMethod
from torchbridge.adapters.config import InitMethod

config = AdapterConfig(
    method=AdapterMethod.LORA,
    rank=16,                           # Low-rank dimension (r)
    alpha=32.0,                        # Scaling factor (output scaled by alpha/rank)
    dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Module name suffixes to adapt
    init_method=InitMethod.KAIMING,
    merge_on_save=False,
)
```

**Target modules** — common patterns:

- `["q_proj", "v_proj"]` — Attention queries and values (default, most efficient)
- `["q_proj", "k_proj", "v_proj", "o_proj"]` — All attention projections
- `["gate_proj", "up_proj", "down_proj"]` — MLP layers

**Rank selection:**

| Rank | Use Case |
|------|----------|
| 4 | Quick experiments, small datasets |
| 8–16 | General fine-tuning (recommended default: 16) |
| 32–64 | Complex tasks, large datasets |
| 128–256 | Maximum capacity |

### CLI

```bash
# Recommend adapter method for detected hardware
tb-adapter recommend --backend nvidia
tb-adapter recommend --backend amd --rank 8

# JSON output for CI pipelines
tb-adapter recommend --backend nvidia --ci

# Show compatibility matrix for all backends
tb-adapter info
tb-adapter info --ci
```

## See Also

- [Backend Selection](backend-selection.md)
- [Performance Tuning](performance-tuning.md)
- [Checkpointing](checkpointing.md)
