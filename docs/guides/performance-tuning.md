# Performance Tuning Guide

This guide covers techniques for maximizing inference and training performance
with TorchBridge across different hardware backends.

## Quick Wins

### 1. Use the Right Optimization Level

TorchBridge supports four optimization levels:

| Level | Alias | Description | Use Case |
|-------|-------|-------------|----------|
| O0 | DEBUG | No optimizations | Debugging, correctness checks |
| O1 | CONSERVATIVE | Safe optimizations only | Production (latency-sensitive) |
| O2 | BALANCED | Moderate optimizations | Most workloads (recommended) |
| O3 | AGGRESSIVE | Maximum optimization | Throughput-critical batch jobs |

```python
from torchbridge.core.config import OptimizationLevel, TorchBridgeConfig

config = TorchBridgeConfig(optimization_level=OptimizationLevel.O2)
```

### 2. Enable Mixed Precision

Mixed precision (FP16/BF16) can 2-4x throughput on modern GPUs:

```python
from torchbridge.core.config import TorchBridgeConfig, PrecisionConfig, PrecisionFormat

config = TorchBridgeConfig(
    precision=PrecisionConfig(default_format=PrecisionFormat.BF16),  # BF16 preferred on Ampere+
)
```

**Backend-specific notes:**
- **NVIDIA Ampere+**: Use BF16 for best numerical stability + speed
- **AMD MI300X**: BF16 and FP16 both well-supported via ROCm
- **TPU**: BF16 is the native format, always preferred
- **Trainium**: BF16 via NeuronX compiler auto-cast

### 3. Batch Size Tuning

Larger batches improve GPU utilization but increase memory:

```bash
tb-benchmark --model your-model --batch-sizes 1,4,8,16,32
```

**Rules of thumb:**
- Start with batch size 1, double until OOM or throughput plateaus
- GPU utilization below 80%? Increase batch size
- Latency matters more than throughput? Use smaller batches

## Backend-Specific Tuning

### NVIDIA GPUs

**torch.compile** (PyTorch 2.0+):
```python
import torch
model = torch.compile(model, mode="reduce-overhead")
```

**CUDA Graphs** (for fixed input shapes):
```python
# TorchBridge enables CUDA graphs at O3 optimization level
config = TorchBridgeConfig(optimization_level=OptimizationLevel.O3)
```

**Tensor Cores** — enabled automatically for FP16/BF16 when matrix dimensions
are multiples of 8 (Ampere) or 16 (Turing).

### AMD GPUs (ROCm)

**hipBLAS tuning:**
```bash
export HIPBLASLT_TUNING_FILE=/path/to/tuning.db
```

**Flash Attention:** Supported on MI250X/MI300X via composable kernel.
TorchBridge enables it automatically when available.

### TPU (XLA)

**Compilation caching:**
```bash
export XLA_FLAGS="--xla_gpu_persistent_cache_dir=/tmp/xla_cache"
```

**Batch padding:** TPU performance degrades with ragged batches.
Pad sequences to fixed lengths for optimal XLA compilation.

### Trainium (NeuronX)

**Compiler flags:**
```bash
export NEURON_CC_FLAGS="--auto-cast=matmul --auto-cast-type=bf16 --model-type=transformer"
```

**Static shapes:** NeuronX compiles for fixed shapes. Avoid dynamic
shapes in the critical path.

## Memory Optimization

### Gradient Checkpointing

Trade compute for memory during training:

```python
model.gradient_checkpointing_enable()
```

### Model Sharding

For models too large for a single GPU, use PyTorch's native FSDP. TorchBridge's
`tb-advisor` generates the recommended FSDP config for your hardware topology:

```bash
tb-advisor
```

## Profiling

Use PyTorch's built-in profiler to find bottlenecks:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for Chrome trace viewer
prof.export_chrome_trace("profile.json")
```

## Benchmarking

Compare performance across backends:

```bash
# Quick benchmark
tb-benchmark --model model.pt --quick

# Full suite with latency percentiles
tb-benchmark --model model.pt --iterations 1000 --warmup 50
```

## Common Pitfalls

1. **CPU fallback without knowing it** — Run `tb-doctor` to verify
   GPU detection. If CUDA/ROCm isn't found, inference silently falls back to CPU.

2. **Small batch size on large GPUs** — An A100 with batch size 1 wastes 95%
   of its compute. Profile and increase batch size.

3. **FP32 on Ampere+** — Tensor Cores are idle in FP32 mode. Use BF16/FP16
   unless you need full precision for numerical validation.

4. **Dynamic shapes on TPU/Trainium** — XLA and NeuronX compile for specific
   shapes. Each new shape triggers recompilation. Use fixed shapes or bucketing.

5. **Ignoring data loading** — GPU may be idle waiting for data. Use
   `num_workers > 0` in DataLoader and `pin_memory=True` for GPU training.
