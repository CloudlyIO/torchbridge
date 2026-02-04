# NVIDIA Backend

TorchBridge NVIDIA backend provides unified access to CUDA GPUs through the `BaseBackend` interface.

## Supported Hardware

| Architecture | GPUs | Compute Capability | Key Features |
|-------------|------|-------------------|--------------|
| **Hopper** | H100, H200 | sm_90 | FP8, Transformer Engine |
| **Ada Lovelace** | L4, L40, RTX 4090 | sm_89 | FP8 (inference), DLSS |
| **Ampere** | A100, A10G, RTX 3090 | sm_80/86 | BF16, Tensor Cores Gen3 |
| **Turing** | T4, RTX 2080 | sm_75 | FP16, Tensor Cores Gen2 |
| **Volta** | V100 | sm_70 | FP16, Tensor Cores Gen1 |

## Quick Start

```python
from torchbridge.backends.nvidia import NVIDIABackend

backend = NVIDIABackend()
print(backend.get_device_info())

model = backend.prepare_model(your_model)
```

## Core Components

- **`NVIDIABackend`** -- main backend implementing `BaseBackend`
- **`NVIDIAOptimizer`** -- architecture-aware optimization strategies
- **`NVIDIAMemoryManager`** -- GPU memory monitoring and management
- **`CUDADeviceManager`** -- multi-GPU device handling

## Configuration

```python
from torchbridge import TorchBridgeConfig

config = TorchBridgeConfig.for_training()

# The config auto-detects NVIDIA architecture
print(f"Architecture: {config.hardware.nvidia.architecture}")
print(f"FP8 enabled: {config.hardware.nvidia.fp8_enabled}")
```

### Optimization Levels

```python
# O0: No optimization (debugging)
backend = NVIDIABackend(optimization_level="O0")

# O1: Basic (operator fusion)
backend = NVIDIABackend(optimization_level="O1")

# O2: Aggressive (mixed precision + torch.compile)
backend = NVIDIABackend(optimization_level="O2")

# O3: Maximum (all techniques including custom kernels)
backend = NVIDIABackend(optimization_level="O3")
```

## Precision Support

| Dtype | Hardware | Use Case |
|-------|----------|----------|
| FP8 (E4M3/E5M2) | H100+ | Training and inference |
| BF16 | Ampere+ | General training |
| FP16 | All | Mixed-precision training |
| FP32 | All | Baseline |

```python
import torch

# Use PyTorch native AMP with auto-detected device
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input_tensor)
```

## Memory Management

```python
backend = NVIDIABackend()

# Monitor memory
stats = backend.get_memory_stats()
print(f"Allocated: {stats['allocated']}MB")
print(f"Reserved: {stats['reserved']}MB")
print(f"Free: {stats['free']}MB")

# Clear cache
backend.empty_cache()
backend.reset_peak_memory_stats()

# OOM protection
config.hardware.nvidia.enable_oom_protection = True
```

## Multi-GPU

```python
from torchbridge.backends.nvidia import CUDADeviceManager

manager = CUDADeviceManager()
print(f"Available GPUs: {manager.device_count()}")

# Backend handles multi-GPU automatically with distributed training
```

## Error Handling

The NVIDIA backend provides specific exceptions:

- `CUDANotAvailableError` -- CUDA not installed or no GPU detected
- `OutOfMemoryError` -- GPU memory exhausted
- `CUDADeviceError` -- driver version mismatch
- `KernelCompilationError` -- custom kernel build failure (from base exceptions)

## Troubleshooting

See [Troubleshooting](../getting-started/troubleshooting.md#nvidia-issues) for common NVIDIA issues.

## See Also

- [Backends Overview](overview.md)
- [Hardware Setup](../guides/hardware-setup.md)
- [Hardware Matrix](../reference/hardware-matrix.md)
