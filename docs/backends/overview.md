# Backends Overview

TorchBridge provides a hardware abstraction layer (HAL) that lets you write PyTorch code once and run it on any supported accelerator. This document explains how the backend system works.

## Architecture

Every hardware backend implements the same `BaseBackend` interface:

```
BaseBackend (abstract)
├── NVIDIABackend   → CUDA
├── AMDBackend      → ROCm / HIP
├── IntelBackend    → IPEX / oneAPI
├── TPUBackend      → XLA
└── CPUBackend      → fallback
```

Your code interacts with the unified API. TorchBridge dispatches to the correct vendor-specific implementation underneath.

## BaseBackend Interface

All backends implement these methods:

```python
class BaseBackend(ABC):
    @abstractmethod
    def prepare_model(self, model) -> nn.Module:
        """Apply backend-specific optimizations to a model."""

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """Return hardware details (name, memory, capabilities)."""

    @abstractmethod
    def get_memory_stats(self) -> dict:
        """Return current memory allocation and availability."""

    @abstractmethod
    def empty_cache(self) -> None:
        """Release cached memory back to the device."""

    @abstractmethod
    def synchronize(self) -> None:
        """Wait for all pending operations to complete."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the torch device for this backend."""
```

## Automatic Backend Selection

`BackendFactory` detects available hardware and creates the appropriate backend:

```python
from torchbridge.backends import BackendFactory, detect_best_backend

# Auto-detect: returns "cuda", "rocm", "xpu", "tpu", or "cpu"
backend = BackendFactory.create(detect_best_backend())
```

Detection priority:
1. NVIDIA CUDA (if `torch.cuda.is_available()`)
2. AMD ROCm (if ROCm runtime detected)
3. Intel XPU (if IPEX available)
4. Google TPU (if PyTorch/XLA available)
5. CPU (fallback)

## Optimization Levels

All backends support a consistent set of optimization levels:

| Level | Description | Use Case |
|-------|-------------|----------|
| **O0** | No optimization | Debugging |
| **O1** | Basic optimizations (operator fusion) | Development |
| **O2** | Aggressive optimization (mixed precision, compilation) | Training |
| **O3** | Maximum optimization (all available techniques) | Production inference |

```python
from torchbridge.backends.nvidia import NVIDIABackend

backend = NVIDIABackend(optimization_level="O2")
model = backend.prepare_model(model)
```

## Standardized Data Types

Backends return consistent data structures:

```python
@dataclass
class DeviceInfo:
    name: str              # e.g., "NVIDIA H100"
    backend: str           # "cuda", "rocm", "xpu", "tpu", "cpu"
    memory_total_gb: float
    memory_available_gb: float
    compute_capability: str
    supported_dtypes: list[str]
```

## Cross-Backend Usage

### Same Code, Different Hardware

```python
from torchbridge.backends import BackendFactory, detect_best_backend

backend = BackendFactory.create(detect_best_backend())
device = backend.device

model = YourModel().to(device)
model = backend.prepare_model(model)

# This runs identically on NVIDIA, AMD, Intel, or TPU
for batch in dataloader:
    inputs = batch.to(device)
    output = model(inputs)
```

### Querying Capabilities

```python
backend = BackendFactory.create("cuda")
info = backend.get_device_info()

print(f"Device: {info.name}")
print(f"Memory: {info.memory_total_gb:.1f} GB")
print(f"Supported dtypes: {info.supported_dtypes}")
```

### Memory Management

```python
# Works the same on all backends
stats = backend.get_memory_stats()
print(f"Allocated: {stats['allocated']}MB")
print(f"Free: {stats['free']}MB")

backend.empty_cache()
```

## Backend-Specific Documentation

Each backend has unique capabilities and configuration options:

- [NVIDIA](nvidia.md) -- CUDA, Tensor Cores, FP8, FlashAttention
- [AMD](amd.md) -- ROCm, HIP, Matrix Cores
- [Intel](intel.md) -- IPEX, oneDNN, AMX/XMX
- [TPU](tpu.md) -- XLA, TPU pods, BF16 optimization

## Migration Between Backends

To switch backends, change the backend name. No model code changes needed:

```python
# Development on CPU
backend = BackendFactory.create("cpu")

# Training on NVIDIA
backend = BackendFactory.create("cuda")

# Cost optimization on AMD
backend = BackendFactory.create("rocm")

# TPU pods
backend = BackendFactory.create("tpu")
```

Checkpoints are portable across backends -- save on one, load on another. TorchBridge handles device mapping and dtype conversion automatically.

## See Also

- [Backend Selection Guide](../guides/backend-selection.md) -- how to choose the right backend
- [Hardware Setup](../guides/hardware-setup.md) -- driver and toolkit installation
- [Hardware Matrix](../reference/hardware-matrix.md) -- full support table
