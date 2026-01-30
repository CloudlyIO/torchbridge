# Intel XPU Backend Guide - v0.4.10

**Status**: Production-Ready (95%+)
**Last Updated**: January 22, 2026
**Version**: v0.4.10 (Intel Documentation + Cloud Validation)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [Usage Examples](#usage-examples)
7. [Performance Optimization](#performance-optimization)
8. [Memory Management](#memory-management)
9. [Error Handling](#error-handling)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [Cloud Testing](#cloud-testing)

---

## Overview

The Intel XPU Backend provides production-ready support for Intel GPUs through Intel Extension for PyTorch (IPEX). It offers comprehensive optimization for data center GPUs (Ponte Vecchio, Data Center Max), consumer GPUs (Arc series), and integrated graphics.

### Supported Hardware

| Architecture | GPUs | Memory | AMX/XMX | Use Case |
|--------------|------|--------|---------|----------|
| **Ponte Vecchio (PVC)** | Data Center GPU Max 1100/1550 | HBM2e (48-128GB) | Yes | Data Center Training/Inference |
| **Arc (DG2)** | A770, A750, A580, A380 | GDDR6 (8-16GB) | Limited | Consumer/Gaming/Inference |
| **Flex** | Flex 140, Flex 170 | GDDR6 (12GB) | Yes | Media/Inference |
| **Integrated** | Iris Xe, UHD | Shared | No | Light Inference |

### Key Features

- **IPEX Integration**: Full Intel Extension for PyTorch support
- **oneDNN Fusion**: Automatic operator fusion via oneDNN
- **AMX/XMX Acceleration**: Matrix extensions for accelerated compute
- **BF16/FP16 Support**: Native mixed precision training
- **Memory Management**: XPU-optimized pooling and allocation
- **Multi-Device Support**: Automatic detection and coordination
- **CPU Fallback**: Graceful degradation when XPU unavailable

---

## Quick Start

### Basic Usage

```python
from torchbridge.backends.intel import IntelBackend
import torch.nn as nn

# Initialize backend (auto-detects hardware)
backend = IntelBackend()

# Prepare model for Intel XPU
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, 10)
)
prepared_model = backend.prepare_model(model)

print(f"Using device: {backend.device}")
print(f"Device info: {backend.get_device_info()}")
```

### With Intel Optimizer

```python
from torchbridge.backends.intel import IntelBackend, IntelOptimizer
from torchbridge.core.config import TorchBridgeConfig

# Initialize with configuration
config = TorchBridgeConfig()
backend = IntelBackend(config)

# Initialize optimizer
optimizer = IntelOptimizer(config)

# Optimize model
model = nn.Linear(1024, 1024)
optimized_model = optimizer.optimize(model, level="O2")

# Get optimization summary
summary = optimizer.get_optimization_summary()
print(f"Optimization level: {summary['level']}")
print(f"oneDNN fusion: {summary['onednn_fusion']}")
```

### Inference Optimization with IPEX

```python
from torchbridge.backends.intel import IntelBackend
import torch

backend = IntelBackend()
model = backend.prepare_model(your_model)

# Optimize for inference (uses IPEX.optimize)
sample_input = torch.randn(1, 512).to(backend.device)
optimized_model = backend.optimize_for_inference(
    model,
    sample_input=sample_input,
    dtype=torch.bfloat16  # Use BF16 for best performance
)

# Run inference
with torch.no_grad():
    output = optimized_model(sample_input)
```

---

## Installation

### Prerequisites

1. **Intel GPU Driver**: Install the latest Intel GPU driver
2. **oneAPI Base Toolkit**: Required for XPU support
3. **Intel Extension for PyTorch (IPEX)**: PyTorch optimization library

### Installation Steps

```bash
# 1. Install oneAPI Base Toolkit (includes Level Zero runtime)
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

# 2. Install PyTorch with XPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 3. Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# 4. Install TorchBridge
pip install torchbridge
```

### Verify Installation

```python
import torch
import intel_extension_for_pytorch as ipex

# Check XPU availability
print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")
print(f"IPEX version: {ipex.__version__}")

# Check TorchBridge Intel backend
from torchbridge.backends.intel import is_xpu_available, get_ipex_version
print(f"TorchBridge XPU: {is_xpu_available()}")
print(f"TorchBridge IPEX: {get_ipex_version()}")
```

---

## Configuration

### IntelConfig Options

```python
from torchbridge.core.config import TorchBridgeConfig

config = TorchBridgeConfig()

# Intel-specific settings via hardware config
config.hardware.intel = {
    'device_id': 0,                    # Default XPU device
    'enable_onednn_fusion': True,      # Enable oneDNN operator fusion
    'enable_amx': True,                # Enable AMX acceleration (if available)
    'default_dtype': 'bf16',           # Default precision (fp32, fp16, bf16)
    'memory_pool_size_gb': 8.0,        # Memory pool size
    'enable_jit': True,                # Enable JIT compilation
}
```

### Optimization Levels

| Level | Name | Description | Use Case |
|-------|------|-------------|----------|
| **O0** | None | No optimization | Debugging |
| **O1** | Conservative | Basic fusion, memory layout | General use |
| **O2** | Balanced | Full oneDNN fusion, BF16 | Production |
| **O3** | Aggressive | Maximum optimization, JIT | High performance |

```python
from torchbridge.backends.intel import IntelOptimizer

optimizer = IntelOptimizer(config)

# Apply different optimization levels
model_o1 = optimizer.optimize(model, level="O1")  # Conservative
model_o2 = optimizer.optimize(model, level="O2")  # Balanced (recommended)
model_o3 = optimizer.optimize(model, level="O3")  # Aggressive
```

---

## Core Components

### IntelBackend

The main backend class for Intel XPU device management.

```python
from torchbridge.backends.intel import IntelBackend

backend = IntelBackend()

# Properties
backend.device           # Current XPU device
backend.devices          # All available XPU devices
backend.device_name      # GPU name (e.g., "Intel Data Center GPU Max 1550")
backend.device_type      # Type: "data_center", "consumer", "integrated"
backend.is_xpu_available # Whether XPU is available
backend.is_data_center   # True if Ponte Vecchio/Max
backend.supports_bf16    # BF16 capability
backend.supports_amx     # AMX capability

# Methods
backend.prepare_model(model)              # Prepare for XPU
backend.optimize_for_inference(model)     # Optimize for inference
backend.optimize_for_training(model, opt) # Optimize for training
backend.get_memory_stats()                # Memory statistics
backend.synchronize()                     # Synchronize device
backend.empty_cache()                     # Clear memory cache
```

### IntelOptimizer

Advanced optimization using IPEX capabilities.

```python
from torchbridge.backends.intel import IntelOptimizer

optimizer = IntelOptimizer(config)

# Optimize model
optimized = optimizer.optimize(model, level="O2")

# Get summary
summary = optimizer.get_optimization_summary()
# Returns: {'level': 'O2', 'onednn_fusion': True, 'dtype': 'bfloat16', ...}
```

### IntelMemoryManager

XPU-specific memory management with pooling.

```python
from torchbridge.backends.intel import IntelMemoryManager

mem_manager = IntelMemoryManager(config, device_id=0)

# Allocate tensor
tensor = mem_manager.allocate_tensor(
    shape=(1024, 1024),
    dtype=torch.float32,
    purpose="activations"
)

# Get memory stats
stats = mem_manager.get_memory_stats()
print(f"Allocated: {stats.allocated_mb:.2f} MB")
print(f"Utilization: {stats.utilization:.1%}")

# Cleanup
mem_manager.cleanup()
```

### XPUUtilities

Helper utilities for XPU operations.

```python
from torchbridge.backends.intel import (
    is_xpu_available,
    is_ipex_available,
    get_ipex_version,
    get_xpu_device_count,
    XPU_AVAILABLE,
    IPEX_AVAILABLE,
)

# Check availability
print(f"XPU: {XPU_AVAILABLE}")
print(f"IPEX: {IPEX_AVAILABLE}")
print(f"Devices: {get_xpu_device_count()}")
```

---

## Usage Examples

### Training with IPEX

```python
import torch
import torch.nn as nn
from torchbridge.backends.intel import IntelBackend

# Initialize
backend = IntelBackend()
device = backend.device

# Create model and move to XPU
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Optimize for training with IPEX
model, optimizer = backend.optimize_for_training(model, optimizer, dtype=torch.bfloat16)

# Training loop
model.train()
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    backend.synchronize()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

### Inference with TorchScript

```python
from torchbridge.backends.intel import IntelBackend
import torch

backend = IntelBackend()
model = backend.prepare_model(your_model)
model.eval()

# Create sample input
sample_input = torch.randn(1, 512).to(backend.device)

# Optimize for inference
optimized_model = backend.optimize_for_inference(
    model,
    sample_input=sample_input,
    dtype=torch.bfloat16
)

# Export to TorchScript
scripted = torch.jit.trace(optimized_model, sample_input)
scripted.save("model_xpu.pt")

# Benchmark
import time
warmup = 10
iterations = 100

for _ in range(warmup):
    _ = optimized_model(sample_input)
backend.synchronize()

start = time.perf_counter()
for _ in range(iterations):
    _ = optimized_model(sample_input)
backend.synchronize()
elapsed = time.perf_counter() - start

print(f"Throughput: {iterations / elapsed:.2f} inferences/sec")
```

### Multi-GPU Training

```python
from torchbridge.backends.intel import IntelBackend
import torch.distributed as dist

backend = IntelBackend()

# Check available devices
print(f"Available XPU devices: {backend.device_count}")

# Set specific device
backend.set_device(device_id=0)

# For distributed training, use torch.distributed with XPU backend
if backend.device_count > 1:
    dist.init_process_group(backend="ccl")  # Intel oneCCL
    # Use DistributedDataParallel as usual
```

---

## Performance Optimization

### oneDNN Fusion

oneDNN automatically fuses common operator patterns:

```python
import intel_extension_for_pytorch as ipex

# Enable oneDNN fusion (enabled by default)
ipex.enable_onednn_fusion(True)

# Common fused patterns:
# - Conv2d + BatchNorm + ReLU
# - Linear + GELU
# - LayerNorm + Linear
# - Attention (Q, K, V) patterns
```

### AMX (Advanced Matrix Extensions)

For Sapphire Rapids CPUs and PVC GPUs with AMX support:

```python
from torchbridge.backends.intel import IntelBackend

backend = IntelBackend()

if backend.supports_amx:
    print("AMX is available - using accelerated matrix operations")
    # AMX is automatically used for BF16 matrix multiplications
    # on supported hardware
```

### Memory Layout Optimization

```python
# Intel XPU benefits from channels_last format
model = model.to(memory_format=torch.channels_last)

# Or use automatic optimization
from torchbridge.backends.intel import IntelOptimizer
optimizer = IntelOptimizer(config)
model = optimizer.optimize(model, level="O2")  # Includes layout optimization
```

### Dimension Alignment

Intel XPU performs best with dimensions divisible by 16:

```python
# Optimal dimensions for Intel XPU
optimal_hidden_size = 256   # 256 % 16 == 0
optimal_batch_size = 64     # 64 % 16 == 0

# The backend will warn about suboptimal dimensions
model = nn.Linear(256, 512)  # Good
model = nn.Linear(100, 200)  # Warning: not optimal
```

---

## Memory Management

### Memory Statistics

```python
from torchbridge.backends.intel import IntelBackend

backend = IntelBackend()

# Get current memory usage
stats = backend.get_memory_stats()
print(f"Allocated: {stats['allocated']:.2f} GB")
print(f"Reserved: {stats['reserved']:.2f} GB")
print(f"Total: {stats['total']:.2f} GB")
print(f"Utilization: {stats.get('utilization', 0):.1%}")

# Human-readable summary
print(backend.get_memory_summary())
```

### Cache Management

```python
# Empty cache to free reserved memory
backend.empty_cache()

# Force synchronization
backend.synchronize()
```

### Memory Pooling

```python
from torchbridge.backends.intel import IntelMemoryManager

# Initialize with custom pool size
mem_manager = IntelMemoryManager(
    config=config,
    device_id=0
)

# Allocate from pool (faster than standard allocation)
tensor = mem_manager.allocate_tensor(
    shape=(2048, 2048),
    dtype=torch.bfloat16,
    purpose="weights"
)

# Return to pool
mem_manager.free_tensor(tensor)

# Cleanup all allocations
mem_manager.cleanup()
```

---

## Error Handling

### Exception Hierarchy

```python
from torchbridge.backends.intel.intel_exceptions import (
    IntelBackendError,      # Base exception
    XPUNotAvailableError,   # XPU not detected
    IPEXNotInstalledError,  # IPEX not installed
    XPUDeviceError,         # Device-specific errors
    XPUMemoryError,         # Out of memory
    XPUOptimizationError,   # Optimization failures
)

try:
    backend = IntelBackend()
    model = backend.prepare_model(model)
except XPUNotAvailableError:
    print("No Intel XPU found, using CPU")
    model = model.to('cpu')
except IPEXNotInstalledError:
    print("IPEX not installed, limited optimization")
except XPUMemoryError as e:
    print(f"Out of memory: {e}")
    backend.empty_cache()
```

---

## Troubleshooting

### Common Issues

#### 1. XPU Not Detected

```bash
# Check driver installation
ls /dev/dri/render*

# Check Level Zero
export ZE_ENABLE_VALIDATION_LAYER=1
python -c "import torch; print(torch.xpu.is_available())"

# Verify oneAPI environment
source /opt/intel/oneapi/setvars.sh
```

#### 2. IPEX Import Error

```bash
# Ensure compatible versions
pip install intel-extension-for-pytorch==<version_matching_pytorch>

# Check IPEX version compatibility
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

#### 3. Out of Memory

```python
# Monitor memory usage
stats = backend.get_memory_stats()
print(f"Memory: {stats['allocated']:.2f}/{stats['total']:.2f} GB")

# Clear cache
backend.empty_cache()

# Reduce batch size or use gradient checkpointing
```

#### 4. Performance Lower Than Expected

```python
# Ensure oneDNN fusion is enabled
import intel_extension_for_pytorch as ipex
ipex.enable_onednn_fusion(True)

# Use BF16 for best performance on supported hardware
model = backend.optimize_for_inference(model, dtype=torch.bfloat16)

# Check dimension alignment (multiples of 16)
```

---

## Best Practices

### 1. Always Use IPEX Optimization

```python
# For inference
model = backend.optimize_for_inference(model, dtype=torch.bfloat16)

# For training
model, optimizer = backend.optimize_for_training(model, optimizer)
```

### 2. Use BF16 on Supported Hardware

```python
if backend.supports_bf16:
    dtype = torch.bfloat16
else:
    dtype = torch.float32
```

### 3. Align Dimensions to 16

```python
# Good
hidden_size = 256  # 256 % 16 == 0
batch_size = 64    # 64 % 16 == 0

# Avoid
hidden_size = 100  # 100 % 16 != 0
```

### 4. Use channels_last for CNNs

```python
model = model.to(memory_format=torch.channels_last)
input = input.to(memory_format=torch.channels_last)
```

### 5. Enable oneDNN Verbose for Debugging

```bash
export ONEDNN_VERBOSE=1
python your_script.py
```

---

## Cloud Testing

### Intel DevCloud

Intel DevCloud provides free access to Intel hardware for testing.

**Access**: https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html

```bash
# On DevCloud, run validation
cd torchbridge
bash scripts/cloud-deployment/intel_devcloud/run_validation.sh
```

### Test Script

```bash
# Run Intel backend tests
python -m pytest tests/test_intel_backend.py -v

# Run Intel demo
python demos/intel_xpu_demo.py

# Run benchmarks
python benchmarks/intel_benchmark.py
```

### Expected Results

| Test Category | Count | Expected |
|--------------|-------|----------|
| Configuration | 10 | 100% pass |
| Backend | 15 | 100% pass |
| Optimizer | 12 | 100% pass |
| Memory Manager | 10 | 100% pass |
| Integration | 14 | 100% pass |
| **Total** | **61** | **100% pass** |

---

## Comparison with Other Backends

| Feature | Intel XPU | NVIDIA CUDA | AMD ROCm |
|---------|-----------|-------------|----------|
| Framework | IPEX | cuDNN | ROCm/HIP |
| Matrix Accel | AMX/XMX | Tensor Cores | Matrix Cores |
| Precision | FP32/BF16/FP16 | FP32/TF32/FP16/FP8 | FP32/BF16/FP16 |
| Memory | HBM2e/GDDR6 | HBM3/GDDR6X | HBM3/HBM2e |
| Fusion | oneDNN | cuDNN | MIOpen |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.4.10 | 2026-01-22 | Full documentation, cloud validation |
| v0.4.8 | 2026-01-20 | BaseBackend integration |
| v0.4.7 | 2026-01-19 | Initial Intel XPU backend |

---

## Additional Resources

- [Intel Extension for PyTorch Documentation](https://intel.github.io/intel-extension-for-pytorch/)
- [oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [Intel DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)
- [oneDNN Documentation](https://oneapi-src.github.io/oneDNN/)
