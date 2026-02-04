# TPU Backend

TorchBridge TPU backend provides unified access to Google Cloud TPUs via PyTorch/XLA through the `BaseBackend` interface.

## Supported Hardware

| Generation | Chip | HBM | Key Features |
|-----------|------|-----|--------------|
| **v7** | Latest | 128GB | Highest throughput |
| **v6e** | Trillium | 32GB | Cost-efficient inference |
| **v5p** | Viperlight | 95GB | Large model training |
| **v5e** | Pufferfish | 16GB | Cost-efficient training |
| **v4** | Jf | 32GB | General purpose |

## Quick Start

```python
from torchbridge.backends.tpu import TPUBackend

backend = TPUBackend()
print(backend.get_device_info())

model = backend.prepare_model(your_model)
```

## Core Components

- **`TPUBackend`** -- main backend implementing `BaseBackend`
- **`TPUOptimizer`** -- XLA-specific optimization strategies
- **`XLACompiler`** -- XLA graph compilation with caching
- **`TPUMemoryManager`** -- HBM monitoring and management

## Configuration

```python
from torchbridge.core.config import TPUConfig

# Inference configuration
config = TPUConfig(
    enable_xla_cache=True,
    cache_max_size=100,
    compilation_timeout_seconds=300,
    memory_fraction=0.9,
    precision="bfloat16",
)

backend = TPUBackend(config)
```

### Configuration Modes

```python
# Development: fast iteration
config = TPUConfig(precision="float32", mixed_precision=False)

# Training: balanced
config = TPUConfig(precision="bfloat16", enable_xla_cache=True)

# Inference: maximum throughput
config = TPUConfig(precision="bfloat16", enable_xla_cache=True, cache_max_size=200)
```

## Precision Support

| Dtype | Notes |
|-------|-------|
| BF16 | Default and recommended for TPU |
| FP32 | Supported but slower |

TPUs are optimized for BF16. TorchBridge auto-converts models to BF16 by default.

```python
# Disable auto-conversion if needed
config.hardware.tpu.precision = "float32"
config.hardware.tpu.mixed_precision = False
```

## XLA Compilation

TPU operations require XLA graph compilation. First iterations are slower while graphs are compiled and cached:

```python
# Enable caching for faster subsequent runs
config = TPUConfig(
    enable_xla_cache=True,
    cache_max_size=100,
)

# Monitor compilation
compiler = XLACompiler(config)
stats = compiler.get_compilation_stats()
print(f"Cached graphs: {stats['cached_graphs']}")
```

### Static Shapes

XLA works best with static tensor shapes. Avoid dynamic shapes when possible:

```python
# Pad to fixed sizes for best performance
max_length = 512
inputs = torch.nn.functional.pad(inputs, (0, max_length - inputs.size(1)))
```

## Memory Management

```python
backend = TPUBackend(config)

stats = backend.get_memory_stats()
print(f"Allocated: {stats['allocated_memory']}MB")
print(f"Available: {stats['available_memory']}MB")

# Reduce memory usage
config.hardware.tpu.memory_fraction = 0.8

# Clear memory pools
from torchbridge.backends.tpu import TPUMemoryManager
memory_mgr = TPUMemoryManager(config)
memory_mgr.clear_memory_pools()
```

## TPU Pods

For multi-host TPU pod training, TorchBridge integrates with PyTorch/XLA's distributed training:

```python
# Multi-host TPU pod setup is handled through distributed training
# See: docs/guides/distributed-training.md
```

## Requirements

- Google Cloud TPU environment (GCE VM, Colab, or Kaggle)
- PyTorch/XLA

```bash
pip install torch_xla[tpu]~=2.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Verify
python3 -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

## Error Handling

- `DeviceNotAvailableError` -- PyTorch/XLA not installed or no TPU detected (from base exceptions)
- `OutOfMemoryError` -- HBM exhausted (from base exceptions)
- `CompilationError` -- graph compilation failure or timeout (from base exceptions)

## Troubleshooting

See [Troubleshooting](../getting-started/troubleshooting.md#tpu-issues) for common TPU issues.

## See Also

- [Backends Overview](overview.md)
- [Hardware Setup](../guides/hardware-setup.md)
- [Hardware Matrix](../reference/hardware-matrix.md)
