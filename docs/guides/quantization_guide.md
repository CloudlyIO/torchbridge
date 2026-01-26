# Quantization Guide

This guide covers how to use KernelPyTorch's quantization features to reduce model size and improve inference performance while maintaining quality.

## Overview

Quantization reduces model precision from FP32 to lower bit-widths, providing:
- **Memory reduction**: 2-4x smaller models
- **Faster inference**: Better hardware utilization
- **Lower costs**: Smaller models = cheaper deployment

## Supported Quantization Modes

| Mode | Precision | Memory Reduction | Quality Impact | Best For |
|------|-----------|------------------|----------------|----------|
| INT8 | 8-bit integer | ~50% | <2% perplexity | General inference |
| INT4 | 4-bit integer | ~75% | <5% perplexity | Memory-constrained |
| FP8 | 8-bit float | ~50% | <1% perplexity | H100/Blackwell GPUs |

## Quick Start

### INT8 Dynamic Quantization

Best for: CPU inference, general use cases

```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Apply INT8 quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Use normally
outputs = quantized_model(**inputs)
```

### FP8 Quantization (H100+)

Best for: NVIDIA H100/Blackwell GPUs

```python
from kernel_pytorch.precision import convert_model_to_native_fp8

# Convert to FP8
fp8_model = convert_model_to_native_fp8(model, device="cuda")

# Prepare inputs (also converts to FP8)
from kernel_pytorch.precision import FP8InferenceEngine

engine = FP8InferenceEngine(fp8_model)
outputs = engine.forward(inputs)
```

### INT4 via BitsAndBytes

Best for: Large models (7B+) on consumer GPUs

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # or "fp4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

## Integration with Model Optimizers

### LLM Optimizer

```python
from kernel_pytorch.models.llm import LLMOptimizer, LLMConfig, QuantizationMode

config = LLMConfig(
    quantization=QuantizationMode.INT8,  # or INT4, FP8
    load_in_8bit=True,
)

optimizer = LLMOptimizer(config)
model = optimizer.load_model("gpt2")
```

### Text Model Optimizer

```python
from kernel_pytorch.models.text import TextModelOptimizer, TextModelConfig

config = TextModelConfig(
    optimization_mode=OptimizationMode.MEMORY,
    # Quantization applied post-optimization
)

optimizer = TextModelOptimizer(config)
optimized = optimizer.optimize(model, task="causal-lm")

# Then apply quantization
quantized = torch.quantization.quantize_dynamic(optimized, {torch.nn.Linear}, torch.qint8)
```

## Quality Targets

### Perplexity Thresholds

| Mode | Max Perplexity Increase | Typical Range |
|------|------------------------|---------------|
| INT8 | <2% | 0.5-1.5% |
| INT4 (NF4) | <5% | 2-4% |
| INT4 (AWQ) | <3% | 1-2.5% |
| FP8 | <1% | 0.1-0.5% |

### When to Use Each Mode

**INT8 (Recommended Default)**
- General purpose inference
- CPU deployment
- When memory is a concern but not critical
- Production workloads requiring stability

**INT4 (Memory Critical)**
- Running 7B+ models on consumer GPUs (16GB VRAM)
- Edge deployment
- When 75% memory reduction is needed
- Acceptable quality trade-off

**FP8 (High Performance)**
- H100 or newer GPUs
- Maximum throughput needed
- Training with reduced memory
- Near-FP32 quality required

## Benchmarking Quantization

Run the quantization benchmark to validate quality:

```bash
# Basic benchmark
python benchmarks/quantization_accuracy.py --model gpt2 --modes int8 fp8

# Full benchmark with more samples
python benchmarks/quantization_accuracy.py \
    --model gpt2 \
    --modes int8 int4 fp8 \
    --samples 500 \
    --device cuda
```

Example output:
```
Quantization Accuracy Benchmark
============================================================
Model: gpt2
Device: cuda
Quantization modes: ['int8', 'fp8']
============================================================

Benchmarking int8...
  INT8 Results [PASS]:
    Perplexity: 29.45 -> 29.89 (+1.49%)
    Memory: 487.2MB -> 243.6MB (50.0% reduction)
    Latency: 12.34ms -> 8.56ms (1.44x speedup)

Benchmarking fp8...
  FP8 Results [PASS]:
    Perplexity: 29.45 -> 29.52 (+0.24%)
    Memory: 487.2MB -> 243.6MB (50.0% reduction)
    Latency: 12.34ms -> 6.12ms (2.02x speedup)
```

## Advanced: Entropy-Based Adaptive Quantization

For optimal quality/size trade-off, use adaptive precision:

```python
from kernel_pytorch.precision import (
    UltraPrecisionModule,
    PrecisionConfig,
    AllocationStrategy
)

config = PrecisionConfig(
    allocation_strategy=AllocationStrategy.ENTROPY_BASED,
    target_memory_reduction=0.5,  # 50% reduction target
)

# Analyze and apply adaptive quantization
precision_module = UltraPrecisionModule(model, config)
optimized_model = precision_module.apply_precision_allocation()
```

## Troubleshooting

### INT8 Quality Degradation

If INT8 quality is worse than expected:
1. Check if model has unusual activation ranges
2. Try calibration with representative data
3. Consider mixed-precision (keep some layers in FP32)

### FP8 Not Available

FP8 requires:
- PyTorch 2.1+ (basic support)
- PyTorch 2.4+ (torch._scaled_mm)
- NVIDIA H100/Blackwell for hardware acceleration

Check availability:
```python
from kernel_pytorch.precision import is_fp8_available, get_fp8_info
print(is_fp8_available())
print(get_fp8_info())
```

### INT4 OOM Errors

If INT4 still causes OOM:
1. Use `device_map="auto"` for automatic sharding
2. Enable gradient checkpointing
3. Consider CPU offloading for largest layers

## Best Practices

1. **Always benchmark first**: Run quality tests before deploying quantized models
2. **Use calibration data**: For static quantization, use representative samples
3. **Monitor in production**: Track quality metrics after deployment
4. **Test edge cases**: Quantization can affect rare inputs differently
5. **Version your quantized models**: Keep track of quantization settings

## API Reference

### Core Functions

```python
# INT8
torch.quantization.quantize_dynamic(model, {nn.Linear}, torch.qint8)

# FP8
from kernel_pytorch.precision import (
    quantize_to_fp8,
    dequantize_from_fp8,
    convert_model_to_native_fp8,
    NativeFP8Linear,
    FP8InferenceEngine,
)

# Ultra Precision
from kernel_pytorch.precision import (
    UltraPrecisionModule,
    PrecisionConfig,
    InformationEntropyAnalyzer,
    AdaptivePrecisionAllocator,
)
```

### Configuration Classes

```python
from kernel_pytorch.precision import (
    FP8Config,
    FP8Format,  # E4M3, E5M2
    PrecisionFormat,  # FP32, FP16, BF16, INT8, INT4, FP8, etc.
    AllocationStrategy,  # ENTROPY_BASED, GRADIENT_WEIGHTED, etc.
)
```

## See Also

- [Precision Module Reference](../api/precision.md)
- [Model Optimization Guide](./optimization_guide.md)
- [Hardware Backend Guide](./backend_selection.md)
