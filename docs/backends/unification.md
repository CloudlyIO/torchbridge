# Backend Unification Guide

**Version**: v0.4.8
**Last Updated**: January 2026

## Overview

KernelPyTorch v0.4.8 introduces a unified backend architecture that provides a consistent interface across all hardware backends (NVIDIA, AMD, TPU, Intel, CPU). This guide covers the new abstractions, the BackendFactory for automatic hardware detection, and best practices for cross-platform development.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [BaseBackend Interface](#basebackend-interface)
- [BackendFactory](#backendfactory)
- [Optimization Levels](#optimization-levels)
- [DeviceInfo Standardization](#deviceinfo-standardization)
- [BaseOptimizer Interface](#baseoptimizer-interface)
- [Migration Guide](#migration-guide)
- [Best Practices](#best-practices)
- [Examples](#examples)

---

## Quick Start

### Automatic Backend Selection (Recommended)

```python
from kernel_pytorch.backends import (
    BackendFactory,
    BackendType,
    get_backend,
    detect_best_backend,
    OptimizationLevel,
)

# Automatic detection and selection
backend = get_backend()  # Auto-selects best available backend
print(f"Using backend: {backend.BACKEND_NAME}")

# Prepare model with unified interface
model = backend.prepare_model(your_model, OptimizationLevel.O2)

# Optimize for inference
model = backend.optimize_for_inference(model)

# Get standardized device info
info = backend.get_device_info()
print(f"Device: {info.device_name} ({info.device_type})")
```

### Explicit Backend Selection

```python
from kernel_pytorch.backends import BackendFactory, BackendType

# Create specific backend
nvidia_backend = BackendFactory.create(BackendType.NVIDIA)
amd_backend = BackendFactory.create(BackendType.AMD)
tpu_backend = BackendFactory.create(BackendType.TPU)
intel_backend = BackendFactory.create(BackendType.INTEL)
cpu_backend = BackendFactory.create(BackendType.CPU)

# Or using string names
backend = BackendFactory.create("nvidia")
```

---

## Architecture Overview

### Unified Class Hierarchy

```
BaseBackend (Abstract)
├── CPUBackend (Concrete fallback)
├── NVIDIABackend
├── AMDBackend
├── TPUBackend
└── IntelBackend

BaseOptimizer (Abstract)
├── CPUOptimizer (Concrete fallback)
├── NVIDIAOptimizer
├── AMDOptimizer
└── IntelOptimizer
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `BaseBackend` | Abstract base class defining unified interface |
| `BackendFactory` | Factory for automatic backend detection and creation |
| `BackendType` | Enum for backend selection (AUTO, NVIDIA, AMD, etc.) |
| `OptimizationLevel` | Enum for optimization levels (O0, O1, O2, O3) |
| `DeviceInfo` | Standardized device information dataclass |
| `OptimizationResult` | Standardized optimization result dataclass |
| `BaseOptimizer` | Abstract base class for optimizers |

---

## BaseBackend Interface

All backends inherit from `BaseBackend` and implement these abstract methods:

### Required Methods

```python
from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Optional, Union

class BaseBackend(ABC):
    """Abstract base class for all hardware backends."""

    BACKEND_NAME: str = "base"  # Override in subclasses

    @abstractmethod
    def _setup_environment(self) -> None:
        """Set up backend-specific environment."""
        pass

    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if backend hardware is available."""
        pass

    @abstractmethod
    def _get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Get device information in unified format."""
        pass

    @abstractmethod
    def prepare_model(
        self,
        model: nn.Module,
        optimization_level: Optional[Union[str, OptimizationLevel]] = None
    ) -> nn.Module:
        """Prepare model for execution on this backend."""
        pass

    @abstractmethod
    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Module:
        """Optimize model for inference."""
        pass

    @abstractmethod
    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Union[nn.Module, Tuple[nn.Module, torch.optim.Optimizer]]:
        """Optimize model for training."""
        pass
```

### Common Properties and Methods

All backends provide:

```python
# Properties
backend.device           # Current torch.device
backend.is_available     # Whether hardware is available
backend.device_count     # Number of available devices
backend.BACKEND_NAME     # Backend identifier string

# Methods
backend.get_device_info(device_id=0)  # Returns DeviceInfo
backend.to_device(tensor)             # Move tensor to backend device
backend.synchronize()                 # Sync backend operations
backend.empty_cache()                 # Clear backend cache
backend.set_device(device_id)         # Change current device

# Context manager support
with backend as b:
    model = b.prepare_model(my_model)
```

---

## BackendFactory

### Creating Backends

```python
from kernel_pytorch.backends import BackendFactory, BackendType

# Auto-select best available backend
backend = BackendFactory.create(BackendType.AUTO)

# Create specific backend type
nvidia = BackendFactory.create(BackendType.NVIDIA)
amd = BackendFactory.create(BackendType.AMD)
tpu = BackendFactory.create(BackendType.TPU)
intel = BackendFactory.create(BackendType.INTEL)
cpu = BackendFactory.create(BackendType.CPU)

# Create using string name
backend = BackendFactory.create("nvidia")

# Pass configuration
from kernel_pytorch.core.config import KernelPyTorchConfig
config = KernelPyTorchConfig()
backend = BackendFactory.create(BackendType.NVIDIA, config=config)
```

### Querying Available Backends

```python
from kernel_pytorch.backends import (
    BackendFactory,
    detect_best_backend,
    list_available_backends,
)

# Get list of available backend names
available = list_available_backends()
# ['cpu', 'nvidia'] (depends on hardware)

# Get recommended backend name
best = detect_best_backend()
# 'nvidia' or 'amd' or 'tpu' or 'cpu'

# Get detailed info about all backends
info = BackendFactory.get_all_backend_info()
# {
#     'nvidia': {'available': True, 'class': 'NVIDIABackend', ...},
#     'amd': {'available': False, 'class': 'AMDBackend', ...},
#     ...
# }
```

### Backend Selection Priority

The AUTO selection follows this priority:

1. **NVIDIA** - If CUDA is available
2. **AMD** - If ROCm is available
3. **TPU** - If PyTorch/XLA is available
4. **Intel** - If Intel XPU is available
5. **CPU** - Always available fallback

---

## Optimization Levels

### OptimizationLevel Enum

```python
from kernel_pytorch.backends import OptimizationLevel

class OptimizationLevel(Enum):
    O0 = "O0"  # No optimization (device placement only)
    O1 = "O1"  # Conservative (safe optimizations)
    O2 = "O2"  # Balanced (recommended default)
    O3 = "O3"  # Aggressive (maximum performance)
```

### String Aliases

Aliases map to levels for convenience:

| Alias | Level |
|-------|-------|
| `"conservative"` | O1 |
| `"balanced"` | O2 |
| `"aggressive"` | O3 |
| `"none"` | O0 |

```python
# These are equivalent:
OptimizationLevel.from_string("O2")
OptimizationLevel.from_string("balanced")
OptimizationLevel.from_string("BALANCED")  # Case insensitive
```

### Level Descriptions

| Level | Description | Use Case |
|-------|-------------|----------|
| **O0** | Device placement only | Debugging, compatibility testing |
| **O1** | Safe optimizations (mixed precision, basic fusion) | Production with strict numerical requirements |
| **O2** | Balanced (torch.compile, gradient checkpointing) | General production use (recommended) |
| **O3** | Maximum performance (FP8, aggressive fusion, kernel compilation) | Maximum throughput, less numerical precision |

### Usage

```python
from kernel_pytorch.backends import get_backend, OptimizationLevel

backend = get_backend()

# Using enum
model = backend.prepare_model(model, OptimizationLevel.O2)

# Using string
model = backend.prepare_model(model, "balanced")

# Using alias string
model = backend.prepare_model(model, "aggressive")
```

---

## DeviceInfo Standardization

### DeviceInfo Dataclass

All backends return standardized device information:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class DeviceInfo:
    backend: str                        # "nvidia", "amd", "tpu", "intel", "cpu"
    device_type: str                    # "cuda:0", "xla:0", "xpu:0", "cpu"
    device_id: int                      # Device index
    device_name: str                    # Human-readable name
    compute_capability: Optional[str]   # e.g., "9.0" for H100
    total_memory_bytes: int             # Total device memory
    driver_version: Optional[str]       # Driver/runtime version
    is_available: bool                  # Whether device is usable
    properties: Dict[str, Any]          # Backend-specific properties

    @property
    def total_memory_gb(self) -> float:
        """Total memory in gigabytes."""
        return self.total_memory_bytes / (1024 ** 3)

    @property
    def total_memory_mb(self) -> float:
        """Total memory in megabytes."""
        return self.total_memory_bytes / (1024 ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...
```

### Example Usage

```python
from kernel_pytorch.backends import get_backend, DeviceInfo

backend = get_backend()
info: DeviceInfo = backend.get_device_info()

print(f"Backend: {info.backend}")
print(f"Device: {info.device_name}")
print(f"Memory: {info.total_memory_gb:.1f} GB")
print(f"Available: {info.is_available}")

# Backend-specific properties
if info.backend == "nvidia":
    print(f"Compute Capability: {info.compute_capability}")
    print(f"FP8 Support: {info.properties.get('fp8_supported', False)}")
elif info.backend == "tpu":
    print(f"TPU Version: {info.properties.get('version')}")
    print(f"Topology: {info.properties.get('topology')}")
```

---

## BaseOptimizer Interface

### Creating Optimizers

```python
from kernel_pytorch.backends import get_optimizer, OptimizationLevel

# Get optimizer for auto-selected backend
optimizer = get_optimizer()

# Get optimizer for specific backend
nvidia_optimizer = get_optimizer("nvidia")
cpu_optimizer = get_optimizer("cpu")

# Optimize model
optimized_model, result = optimizer.optimize(
    model,
    level=OptimizationLevel.O2,
    sample_input=sample_input
)

print(f"Applied: {result.optimizations_applied}")
print(f"Success: {result.success}")
```

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    success: bool
    model: nn.Module
    level: OptimizationLevel
    optimizations_applied: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        ...
```

### Available Strategies

```python
optimizer = get_optimizer("cpu")
strategies = optimizer.get_available_strategies()

for strategy in strategies:
    print(f"{strategy.name}: {strategy.description}")
    print(f"  Speedup: {strategy.speedup_estimate}x")
    print(f"  Applicable levels: {strategy.applicable_levels}")
```

---

## Migration Guide

### From v0.3.x to v0.4.8

#### Old Pattern (v0.3.x)

```python
# Old: Direct backend instantiation
from kernel_pytorch.backends.nvidia import NVIDIABackend
from kernel_pytorch.core.config import KernelPyTorchConfig

config = KernelPyTorchConfig()
backend = NVIDIABackend(config)
model = backend.prepare_model(model)

# Old: Getting device info (returned dict)
info = backend.get_device_info()  # Dict[str, Any]
```

#### New Pattern (v0.4.8)

```python
# New: Use BackendFactory
from kernel_pytorch.backends import (
    BackendFactory,
    BackendType,
    get_backend,
    OptimizationLevel,
)

# Option 1: Explicit factory
backend = BackendFactory.create(BackendType.NVIDIA, config=config)

# Option 2: Convenience function
backend = get_backend("nvidia", config=config)

# Option 3: Auto-detection
backend = get_backend()  # Selects best available

# New: prepare_model with optimization level
model = backend.prepare_model(model, OptimizationLevel.O2)

# New: Getting device info (returns DeviceInfo object)
info = backend.get_device_info()  # DeviceInfo dataclass
print(info.device_name)  # Attribute access
print(info.to_dict())    # Convert to dict if needed
```

#### Legacy Method Renames

Some methods were renamed for clarity (old methods still work but emit warnings):

| Old Method (v0.3.x) | New Method (v0.4.8) |
|---------------------|---------------------|
| `backend.get_device_info()` (dict) | `backend.get_device_info_dict()` |
| - | `backend.get_device_info()` (DeviceInfo) |
| `optimizer.optimize()` | `optimizer.optimize_legacy()` |
| - | `optimizer.optimize()` (returns tuple) |

---

## Best Practices

### 1. Use Factory for Portability

```python
# DO: Use factory for cross-platform code
from kernel_pytorch.backends import get_backend

backend = get_backend()  # Works on any hardware
model = backend.prepare_model(model)
```

### 2. Check Availability Before Use

```python
from kernel_pytorch.backends import BackendFactory, BackendType

# Check if specific backend is available
if BackendFactory.is_available(BackendType.NVIDIA):
    backend = BackendFactory.create(BackendType.NVIDIA)
else:
    backend = BackendFactory.create(BackendType.CPU)
```

### 3. Use Context Manager for Cleanup

```python
from kernel_pytorch.backends import get_backend

with get_backend() as backend:
    model = backend.prepare_model(model)
    model = backend.optimize_for_inference(model)
    output = model(input_tensor)
# Cache automatically cleared
```

### 4. Handle Fallback Gracefully

```python
from kernel_pytorch.backends import get_backend

backend = get_backend()

if backend.BACKEND_NAME == "cpu":
    print("Warning: Running on CPU, performance may be limited")
    # Adjust batch size, etc.
```

### 5. Use Standardized DeviceInfo

```python
from kernel_pytorch.backends import get_backend

backend = get_backend()
info = backend.get_device_info()

# Always available fields
print(f"Backend: {info.backend}")
print(f"Device: {info.device_name}")
print(f"Available: {info.is_available}")

# Memory info (may be 0 for CPU)
if info.total_memory_bytes > 0:
    print(f"Memory: {info.total_memory_gb:.1f} GB")
```

---

## Examples

### Cross-Platform Training Loop

```python
from kernel_pytorch.backends import get_backend, OptimizationLevel
import torch

# Auto-select backend
backend = get_backend()
print(f"Training on: {backend.BACKEND_NAME}")

# Prepare model
model = backend.prepare_model(my_model, OptimizationLevel.O2)
model, optimizer = backend.optimize_for_training(model, my_optimizer)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = backend.to_device(batch['input'])
        targets = backend.to_device(batch['target'])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    backend.synchronize()  # Ensure completion
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### Backend Comparison

```python
from kernel_pytorch.backends import BackendFactory, BackendType, OptimizationLevel
import time

backends_to_test = [BackendType.CPU, BackendType.NVIDIA, BackendType.AMD]

for backend_type in backends_to_test:
    try:
        backend = BackendFactory.create(backend_type)
        if not backend.is_available:
            continue

        model = create_test_model()
        model = backend.prepare_model(model, OptimizationLevel.O2)
        model = backend.optimize_for_inference(model)

        x = backend.to_device(torch.randn(8, 64, 256))

        # Warmup
        for _ in range(3):
            _ = model(x)
        backend.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        backend.synchronize()
        elapsed = time.perf_counter() - start

        info = backend.get_device_info()
        print(f"{backend.BACKEND_NAME}: {elapsed*10:.2f}ms/iter on {info.device_name}")

    except Exception as e:
        print(f"{backend_type}: Not available ({e})")
```

### Inference Pipeline

```python
from kernel_pytorch.backends import get_backend, OptimizationLevel

def create_inference_pipeline(model, sample_input=None):
    """Create optimized inference pipeline."""
    backend = get_backend()

    # Prepare and optimize
    model = backend.prepare_model(model, OptimizationLevel.O3)
    model = backend.optimize_for_inference(model, sample_input=sample_input)

    def predict(inputs):
        with torch.no_grad():
            inputs = backend.to_device(inputs)
            outputs = model(inputs)
            backend.synchronize()
            return outputs

    return predict, backend.get_device_info()

# Usage
predict, device_info = create_inference_pipeline(my_model)
print(f"Running inference on: {device_info.device_name}")

results = predict(my_inputs)
```

---

## Related Documentation

- [Backend Selection Guide](../guides/backend_selection.md)
- [NVIDIA Backend](nvidia.md)
- [AMD Backend](amd.md)
- [TPU Backend](tpu.md)
- [Troubleshooting](../getting-started/troubleshooting.md)

---

## Summary

The v0.4.8 backend unification provides:

- **Unified Interface**: All backends implement the same `BaseBackend` interface
- **Automatic Detection**: `BackendFactory` auto-selects the best available hardware
- **Standardized Types**: `DeviceInfo`, `OptimizationLevel`, and `OptimizationResult`
- **Cross-Platform Code**: Write once, run on NVIDIA, AMD, TPU, Intel, or CPU
- **Backward Compatibility**: Old patterns still work with deprecation warnings
- **Convenience Functions**: `get_backend()`, `get_optimizer()`, `detect_best_backend()`

Use the unified interface for portable, maintainable ML code that works across all supported hardware platforms.
