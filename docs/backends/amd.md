# AMD Backend

TorchBridge AMD backend provides unified access to AMD GPUs via ROCm/HIP through the `BaseBackend` interface.

## Supported Hardware

| Architecture | GPUs | Key Features |
|-------------|------|--------------|
| **CDNA3** | MI300X, MI300A | Matrix Cores, HBM3, 192GB |
| **CDNA2** | MI200, MI250X | Matrix Cores, HBM2e |
| **RDNA3** | RX 7900 XTX | Compute units, gaming + compute |
| **RDNA2** | RX 6900 XT | Compute units, limited ML |

Matrix Cores (for accelerated matmul) are only available on CDNA architectures (MI series).

## Quick Start

```python
from torchbridge.backends.amd import AMDBackend

backend = AMDBackend()
print(backend.get_device_info())

model = backend.prepare_model(your_model)
```

## Core Components

- **`AMDBackend`** -- main backend implementing `BaseBackend`
- **`AMDOptimizer`** -- ROCm-specific optimization strategies
- **`ROCmCompiler`** -- HIP kernel compilation with caching
- **`AMDMemoryManager`** -- GPU memory pooling and monitoring
- **`HIPUtilities`** -- low-level HIP runtime utilities

## Configuration

```python
from torchbridge.core.config import AMDConfig, AMDArchitecture

config = AMDConfig(
    architecture=AMDArchitecture.CDNA3,
    enable_matrix_cores=True,
    memory_pool_size_gb=8.0,
    enable_memory_pooling=True,
)

backend = AMDBackend(config)
```

### Optimization Levels

```python
# O0: No optimization
backend = AMDBackend(optimization_level="O0")

# O1: Basic (operator fusion)
backend = AMDBackend(optimization_level="O1")

# O2: Aggressive (mixed precision, HIP compilation)
backend = AMDBackend(optimization_level="O2")

# O3: Maximum (all techniques)
backend = AMDBackend(optimization_level="O3")
```

## Precision Support

| Dtype | Hardware | Notes |
|-------|----------|-------|
| BF16 | CDNA2+ | Recommended for training |
| FP16 | All | Mixed-precision training |
| FP32 | All | Baseline |

FP8 is not yet supported on AMD hardware through TorchBridge.

## HIP Kernel Compilation

The AMD backend compiles HIP kernels with caching for fast subsequent runs:

```python
from torchbridge.backends.amd import ROCmCompiler

compiler = ROCmCompiler(config)
stats = compiler.get_compilation_stats()
print(f"Cache hit rate: {stats['cache_hit_rate_percent']:.1f}%")

# Clear cache if needed
compiler.clear_cache()
```

## Memory Management

```python
backend = AMDBackend()

# Monitor memory
info = backend.get_device_info()
stats = backend.get_memory_stats()

# Configure memory pool
config = AMDConfig(
    memory_pool_size_gb=8.0,
    enable_memory_pooling=True,
)

# Clear cache
backend.empty_cache()
```

## Requirements

- ROCm 5.6+ (6.0+ recommended)
- PyTorch with ROCm support
- Linux (ROCm is Linux-only)

```bash
# Install PyTorch with ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Verify
python3 -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

## Error Handling

- `DeviceNotAvailableError` -- ROCm not installed (from base exceptions)
- `CompilationError` -- HIP kernel compilation failure (from base exceptions)
- `OutOfMemoryError` -- GPU memory exhausted (from base exceptions)

## Troubleshooting

See [Troubleshooting](../getting_started/troubleshooting.md#amd-issues) for common AMD issues.

## See Also

- [Backends Overview](overview.md)
- [Hardware Setup](../guides/hardware-setup.md)
- [Hardware Matrix](../reference/hardware-matrix.md)
