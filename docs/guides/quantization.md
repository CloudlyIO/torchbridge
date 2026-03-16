# Backend-Aware Quantization Guide

TorchBridge v0.5.23 introduces backend-aware quantization that automatically
selects the optimal quantization format based on detected hardware. This guide
covers format selection, Python API, CLI usage, and troubleshooting.

## Quick Start

```python
from torchbridge.precision.quantization import QuantizationEngine

engine = QuantizationEngine()
result = engine.quantize(model, format="auto")
print(f"Applied: {result.format_applied.value}")
print(f"Memory reduction: {result.memory_reduction_pct:.1f}%")
```

## Supported Formats

| Format | Bits | Memory Reduction | Calibration | torchao Required |
|--------|------|-----------------|-------------|------------------|
| INT8 Dynamic | 8 | 50% | No | No |
| INT8 SmoothQuant | 8 | 50% | Yes | Yes |
| INT4 Weight-Only | 4 | 75% | No | Yes |
| FP8 E4M3 | 8 | 50% | No | No |
| FP8 E5M2 | 8 | 50% | No | No |
| NVFP4 | 4 | 87.5% | No | No |
| BF16 | 16 | 50% | No | No |

## Backend Compatibility Matrix

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

## Python API

### Auto-Select Format

```python
from torchbridge.precision.quantization import QuantizationEngine

engine = QuantizationEngine()

# Auto-select optimal format for detected hardware
result = engine.quantize(model, format="auto")
```

### Explicit Format

```python
result = engine.quantize(model, format="int8_dynamic")
```

### Query Compatibility

```python
from torchbridge.precision.quantization import QuantizationCompatibilityMatrix
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture

formats = QuantizationCompatibilityMatrix.get_supported_formats(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
optimal = QuantizationCompatibilityMatrix.get_optimal_format(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
```

### Fallback Chains

When a requested format is unsupported, the engine automatically falls back:

```python
result = engine.quantize(model, format="nvfp4")  # on CPU
# result.used_fallback == True
# result.format_applied == QuantizationFormat.INT8_DYNAMIC
# result.warnings contains fallback message
```

## CLI Usage

```bash
# Auto-select format
tb-quantize --model model.pt

# Explicit format
tb-quantize --model model.pt --format int8_dynamic

# Override backend
tb-quantize --model model.pt --backend nvidia --format fp8_e4m3

# Save quantized model
tb-quantize --model model.pt --output quantized.pt

# Validate quality after quantization
tb-quantize --model model.pt --validate

# CI mode (JSON output)
tb-quantize --model model.pt --ci
```

## torchao Integration

Some formats (INT4, SmoothQuant) require torchao. Install it as an optional dependency:

```bash
pip install torchbridge-ml[quantization]
# or directly:
pip install torchao
```

When torchao is not installed:
- INT8 Dynamic falls back to PyTorch native `torch.quantization.quantize_dynamic`
- INT4/SmoothQuant emit a warning and fall back to INT8 Dynamic
- FP8 requires torchao (INT4/FP8 formats not available without it)

## Troubleshooting

### "torchao is required for this quantization format"

Install torchao: `pip install torchao`

### Fallback warnings

If you see "Requested X unavailable, using Y", your hardware doesn't support
the requested format. Use `engine.get_supported_formats()` to see what's available.

### Low cosine similarity after quantization

Some formats (INT4) have higher perplexity impact. Check the `FormatSpec.perplexity_tolerance_pct`
to understand expected quality tradeoffs.
