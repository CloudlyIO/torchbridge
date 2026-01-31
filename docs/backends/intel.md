# Intel Backend

TorchBridge Intel backend provides unified access to Intel GPUs and accelerators via IPEX (Intel Extension for PyTorch) through the `BaseBackend` interface.

## Supported Hardware

| Architecture | Hardware | Key Features |
|-------------|----------|--------------|
| **Ponte Vecchio (PVC)** | Data Center GPU Max | XMX cores, HBM2e, 128GB |
| **Arc (Alchemist)** | Arc A770, A750 | Xe cores, ray tracing |
| **Flex** | Flex 170, 140 | Media + compute |
| **Integrated** | Intel UHD, Iris Xe | Basic compute |

## Quick Start

```python
from torchbridge.backends.intel import IntelBackend

backend = IntelBackend()
print(backend.get_device_info())

model = backend.prepare_model(your_model)
```

## Core Components

- **`IntelBackend`** -- main backend implementing `BaseBackend`
- **`IntelOptimizer`** -- IPEX-based optimization strategies
- **`IntelMemoryManager`** -- XPU memory monitoring
- **`XPUDeviceManager`** -- low-level XPU device utilities

## Configuration

```python
from torchbridge.core.config import IntelConfig

config = IntelConfig(
    enable_ipex=True,
    enable_onednn=True,
    enable_amx=True,  # Advanced Matrix Extensions (Sapphire Rapids+)
)

backend = IntelBackend(config)
```

### Optimization Levels

```python
# O0: No optimization
backend = IntelBackend(optimization_level="O0")

# O1: Basic (oneDNN fusion)
backend = IntelBackend(optimization_level="O1")

# O2: Aggressive (IPEX optimization + BF16)
backend = IntelBackend(optimization_level="O2")

# O3: Maximum (all techniques including XMX)
backend = IntelBackend(optimization_level="O3")
```

## Precision Support

| Dtype | Hardware | Notes |
|-------|----------|-------|
| BF16 | PVC, Arc | Recommended for training |
| FP16 | PVC, Arc | Mixed-precision |
| FP32 | All | Baseline |

```python
# IPEX handles precision optimization automatically
model = backend.prepare_model(model)
```

## IPEX Integration

The Intel backend leverages Intel Extension for PyTorch for optimizations:

```python
# Automatic IPEX optimization
model = backend.prepare_model(model)

# This applies:
# - oneDNN operator fusion
# - AMX/XMX acceleration (where available)
# - Memory layout optimization
# - Graph optimization
```

## Requirements

- Intel Extension for PyTorch (IPEX)
- oneAPI Base Toolkit (for XPU support)
- Linux or Windows

```bash
# Install IPEX
pip install intel-extension-for-pytorch

# For XPU (discrete GPU) support
pip install intel-extension-for-pytorch[xpu]

# Verify
python3 -c "import intel_extension_for_pytorch; print('IPEX available')"
```

## Memory Management

```python
backend = IntelBackend()

stats = backend.get_memory_stats()
print(f"Allocated: {stats['allocated']}MB")
print(f"Available: {stats['available']}MB")

backend.empty_cache()
```

## Error Handling

- `DeviceNotAvailableError` -- IPEX not installed or no Intel GPU detected (from base exceptions)
- `OutOfMemoryError` -- device memory exhausted (from base exceptions)

## Troubleshooting

Common issues:
- **IPEX import fails:** Install with `pip install intel-extension-for-pytorch`
- **XPU not detected:** Install oneAPI Base Toolkit and set `source /opt/intel/oneapi/setvars.sh`
- **Poor performance:** Ensure oneDNN fusion is enabled (`config.enable_onednn = True`)

## See Also

- [Backends Overview](overview.md)
- [Hardware Setup](../guides/hardware-setup.md)
- [Hardware Matrix](../reference/hardware-matrix.md)
