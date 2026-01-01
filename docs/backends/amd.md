# AMD ROCm Backend Guide - v0.3.6

**Status**: Production-Ready (90%+)
**Last Updated**: December 31, 2025
**Version**: v0.3.6 (Phase 4C-Pre Week 6 Complete)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [Usage Examples](#usage-examples)
7. [Performance Optimization](#performance-optimization)
8. [Error Handling](#error-handling)
9. [Troubleshooting](#troubleshooting)
10. [Known Limitations](#known-limitations)
11. [Best Practices](#best-practices)

---

## Overview

The AMD ROCm Backend provides production-ready support for AMD GPUs through the ROCm/HIP platform. It offers comprehensive optimization for data center GPUs (MI200, MI300 series) and consumer GPUs (RX 6000, RX 7000 series).

### Supported Architectures

| Architecture | GPUs | Matrix Cores | Memory | Use Case |
|--------------|------|--------------|--------|----------|
| **CDNA3** | MI300A, MI300X | Yes (v2) | HBM3 | Data Center Training |
| **CDNA2** | MI210, MI250, MI250X | Yes | HBM2e | Data Center Training |
| **CDNA** | MI50, MI60 | No | HBM2 | Legacy Data Center |
| **RDNA3** | RX 7000 series | No | GDDR6 | Consumer/Gaming |
| **RDNA2** | RX 6000 series | No | GDDR6 | Consumer/Gaming |

### Key Features

- **Multi-GPU Support**: Automatic detection and coordination of multiple AMD GPUs
- **Matrix Core Acceleration**: Optimized for CDNA2/CDNA3 Matrix Cores
- **HIP Kernel Compilation**: With LRU caching for fast reloads
- **Memory Management**: HBM-optimized pooling and OOM protection
- **Optimization Levels**: Conservative, Balanced, Aggressive modes
- **Profiling Integration**: Built-in profiling with HIPUtilities
- **CPU Fallback**: Graceful degradation when ROCm unavailable
- **Error Handling**: Comprehensive exception hierarchy

---

## Quick Start

### Basic Usage

```python
from kernel_pytorch.backends.amd import AMDBackend
from kernel_pytorch.core.config import AMDConfig
import torch.nn as nn

# Initialize backend (auto-detects architecture)
backend = AMDBackend()

# Prepare model for AMD GPU
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, 10)
)
prepared_model = backend.prepare_model(model)

print(f"Using device: {backend.device}")
print(f"Device info: {backend.get_device_info()}")
```

### With AMD Optimizer

```python
from kernel_pytorch.backends.amd import AMDBackend, AMDOptimizer
from kernel_pytorch.core.config import AMDConfig, AMDArchitecture

# Configure for MI300X (CDNA3)
config = AMDConfig(
    architecture=AMDArchitecture.CDNA3,
    optimization_level="aggressive",
    enable_matrix_cores=True,
    enable_mixed_precision=True,
)

# Initialize optimizer
optimizer = AMDOptimizer(config)

# Optimize model
model = nn.Linear(1024, 1024)
optimized_model = optimizer.optimize(model)

# Get optimization summary
summary = optimizer.get_optimization_summary()
print(f"Architecture: {summary['architecture']}")
print(f"Matrix Cores: {summary['matrix_cores_enabled']}")
```

---

## Installation

### Prerequisites

1. **AMD GPU**: CDNA2, CDNA3, RDNA2, or RDNA3 architecture
2. **ROCm**: Version 5.7+ (recommended: 6.0+)
3. **PyTorch**: Built with ROCm support

### Installing ROCm

```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

# Set environment
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
```

### Installing PyTorch with ROCm

```bash
# PyTorch 2.2+ with ROCm 6.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

---

## Configuration

### AMDConfig Options

```python
from kernel_pytorch.core.config import AMDConfig, AMDArchitecture

config = AMDConfig(
    # Hardware
    architecture=AMDArchitecture.AUTO,  # AUTO, CDNA, CDNA2, CDNA3, RDNA2, RDNA3
    device_id=0,  # GPU device ID

    # Optimization
    optimization_level="balanced",  # conservative, balanced, aggressive
    enable_matrix_cores=True,  # Enable Matrix Core acceleration (CDNA2/3)
    enable_mixed_precision=True,  # Enable mixed precision training
    enable_operator_fusion=True,  # Enable operator fusion

    # Memory
    enable_memory_pooling=True,  # Enable memory pooling
    memory_pool_size_gb=8.0,  # Memory pool size
    max_memory_fraction=0.9,  # Maximum memory fraction to use

    # Compiler
    hip_kernel_cache_enabled=True,  # Enable HIP kernel caching
    hip_compiler_cache_size=100,  # Max cached kernels
    hip_compiler_cache_dir="/tmp/hip_cache",  # Cache directory

    # Libraries
    rocblas_enabled=True,  # Enable rocBLAS
    enable_rocblas_tuning=True,  # Enable rocBLAS tuning
    miopen_enabled=True,  # Enable MIOpen (AMD's cuDNN)
    miopen_find_mode="NORMAL",  # NORMAL, FAST, HYBRID

    # Precision
    default_precision="fp32",  # fp32, fp16, bf16
    allow_fp16=True,
    allow_bf16=True,

    # Profiling
    enable_profiling=False,  # Enable profiling

    # Validation
    enable_strict_validation=False,  # Strict validation mode
    enable_oom_protection=True,  # OOM protection
)
```

### Architecture-Specific Defaults

```python
# CDNA3 (MI300) - Maximum performance
config = AMDConfig(
    architecture=AMDArchitecture.CDNA3,
    optimization_level="aggressive",
    enable_matrix_cores=True,
    default_precision="bf16",
)

# CDNA2 (MI200) - Balanced performance
config = AMDConfig(
    architecture=AMDArchitecture.CDNA2,
    optimization_level="balanced",
    enable_matrix_cores=True,
    default_precision="fp16",
)

# RDNA3 (Consumer) - Conservative
config = AMDConfig(
    architecture=AMDArchitecture.RDNA3,
    optimization_level="conservative",
    enable_matrix_cores=False,  # Not available on RDNA
)
```

---

## Core Components

### 1. AMDBackend

Main orchestrator for AMD GPU operations.

```python
from kernel_pytorch.backends.amd import AMDBackend
from kernel_pytorch.core.config import AMDConfig

config = AMDConfig()
backend = AMDBackend(config)

# Check availability
if backend.is_available():
    print(f"Device: {backend.device}")
    print(f"Info: {backend.get_device_info()}")

# Prepare model
model = nn.Linear(512, 256)
prepared = backend.prepare_model(model)

# Synchronize operations
backend.synchronize()

# Cleanup
backend.cleanup()
```

### 2. AMDOptimizer

Multi-level optimization for AMD GPUs.

```python
from kernel_pytorch.backends.amd import AMDOptimizer
from kernel_pytorch.core.config import AMDConfig

config = AMDConfig(optimization_level="balanced")
optimizer = AMDOptimizer(config)

# Optimize with specific level
optimized = optimizer.optimize(model, level="aggressive")

# Get summary
summary = optimizer.get_optimization_summary()
print(f"Applied optimizations: {summary['optimizations_applied']}")
```

**Optimization Levels:**
- **Conservative**: Minimal changes, maximum compatibility
- **Balanced**: Good performance with reasonable compatibility
- **Aggressive**: Maximum performance, may affect compatibility

### 3. ROCmCompiler

HIP kernel compilation with caching.

```python
from kernel_pytorch.backends.amd import ROCmCompiler
from kernel_pytorch.core.config import AMDConfig

config = AMDConfig()
compiler = ROCmCompiler(config)

# Compile kernel
kernel_source = "__global__ void add(float* a, float* b, float* c) { ... }"
kernel = compiler.compile_kernel(kernel_source, "add_kernel")

print(f"Kernel: {kernel.name}")
print(f"Architecture: {kernel.architecture}")

# Get compilation stats
stats = compiler.get_compilation_stats()
print(f"Cache hit rate: {stats['cache_hit_rate_percent']}%")

# Clear cache
compiler.clear_cache()
```

### 4. AMDMemoryManager

GPU memory management and pooling.

```python
from kernel_pytorch.backends.amd import AMDMemoryManager
from kernel_pytorch.core.config import AMDConfig

config = AMDConfig(memory_pool_size_gb=8.0)
manager = AMDMemoryManager(config)

# Get memory stats
stats = manager.get_memory_stats()
print(f"Total: {stats.total_mb}MB")
print(f"Allocated: {stats.allocated_mb}MB")
print(f"Free: {stats.free_mb}MB")

# Get allocation summary
summary = manager.get_allocation_summary()
```

### 5. HIPUtilities

Device coordination and profiling.

```python
from kernel_pytorch.backends.amd import HIPUtilities
from kernel_pytorch.core.config import AMDConfig

config = AMDConfig(enable_profiling=True)
utils = HIPUtilities(config)

# Create stream
stream = utils.create_stream("compute", priority=0)

# Profile operations
with utils.profile_region("forward_pass"):
    output = model(input_data)

# Get profiling summary
summary = utils.get_profiling_summary()
for name, data in summary.get("regions", {}).items():
    print(f"{name}: {data['avg_ms']:.3f}ms")

# Cleanup
utils.cleanup()
```

---

## Usage Examples

### Training Loop

```python
from kernel_pytorch.backends.amd import AMDBackend, AMDOptimizer
from kernel_pytorch.core.config import AMDConfig, AMDArchitecture
import torch
import torch.nn as nn

# Configure for MI300X
config = AMDConfig(
    architecture=AMDArchitecture.CDNA3,
    optimization_level="balanced",
    enable_matrix_cores=True,
    default_precision="bf16",
)

# Initialize
backend = AMDBackend(config)
amd_optimizer = AMDOptimizer(config)

# Model and optimizer
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.GELU(),
    nn.Linear(2048, 1024),
)
model = backend.prepare_model(model)
model = amd_optimizer.optimize(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training
for epoch in range(10):
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(backend.device)
        targets = targets.to(backend.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

    backend.synchronize()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

### Cross-Backend Portability

```python
from kernel_pytorch.backends.amd import AMDBackend
from kernel_pytorch.backends.nvidia import NVIDIABackend
from kernel_pytorch.core.config import AMDConfig, KernelPyTorchConfig

# Train on NVIDIA
nvidia_config = KernelPyTorchConfig()
nvidia_backend = NVIDIABackend(nvidia_config)
nvidia_model = nvidia_backend.prepare_model(model)

# ... train ...

# Save checkpoint (CPU tensors)
checkpoint = {
    'model_state_dict': {k: v.cpu() for k, v in nvidia_model.state_dict().items()},
    'epoch': 10,
}

# Load on AMD
amd_config = AMDConfig()
amd_backend = AMDBackend(amd_config)
amd_model = amd_backend.prepare_model(model)
amd_model.load_state_dict(checkpoint['model_state_dict'])

# Continue training or inference on AMD
```

---

## Performance Optimization

### 1. Enable Matrix Cores (CDNA2/CDNA3)

```python
config = AMDConfig(
    architecture=AMDArchitecture.CDNA3,
    enable_matrix_cores=True,
    default_precision="bf16",  # Best for Matrix Cores
)
```

### 2. Use Aggressive Optimization

```python
config = AMDConfig(
    optimization_level="aggressive",
    enable_operator_fusion=True,
    enable_rocblas_tuning=True,
)
```

### 3. Optimize Memory

```python
config = AMDConfig(
    enable_memory_pooling=True,
    memory_pool_size_gb=16.0,  # Large pool for MI300X
    max_memory_fraction=0.9,
)
```

### 4. Use Profiling

```python
config = AMDConfig(enable_profiling=True)
utils = HIPUtilities(config)

with utils.profile_region("critical_path"):
    # Profile this code
    pass

summary = utils.get_profiling_summary()
```

### 5. Tune for Architecture

| Architecture | Precision | Optimization | Matrix Cores |
|--------------|-----------|--------------|--------------|
| CDNA3 | bf16 | aggressive | Yes |
| CDNA2 | fp16 | balanced | Yes |
| RDNA3 | fp32 | conservative | No |

---

## Error Handling

### Exception Hierarchy

```python
from kernel_pytorch.backends.amd.amd_exceptions import (
    AMDBackendError,      # Base exception
    ROCmNotAvailableError,  # ROCm not installed
    AMDDeviceError,       # Device-related errors
    HIPCompilationError,  # Kernel compilation failed
    ROCmMemoryError,      # Memory allocation failed
    MatrixCoreError,      # Matrix Core issues
    AMDOptimizationError, # Optimization failed
    AMDConfigurationError, # Invalid configuration
)

try:
    backend = AMDBackend(config)
    model = backend.prepare_model(model)
except ROCmNotAvailableError:
    print("ROCm not available - using CPU fallback")
except AMDDeviceError as e:
    print(f"Device error: {e}")
except ROCmMemoryError as e:
    print(f"Memory error: {e}")
```

### CPU Fallback Mode

```python
# AMDBackend automatically falls back to CPU when ROCm unavailable
backend = AMDBackend()

if backend.device.type == "cpu":
    print("Running in CPU fallback mode")
else:
    print(f"Running on AMD GPU: {backend.get_device_info()['name']}")
```

---

## Troubleshooting

### ROCm Not Detected

```bash
# Check ROCm installation
rocm-smi

# Verify environment
echo $ROCM_HOME
which hipcc

# Check PyTorch ROCm support
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues

```python
# Enable OOM protection
config = AMDConfig(
    enable_oom_protection=True,
    max_memory_fraction=0.8,  # Use less memory
)

# Clear cache periodically
backend.synchronize()
torch.cuda.empty_cache()
```

### Kernel Compilation Errors

```python
# Check compiler cache
compiler = ROCmCompiler(config)
stats = compiler.get_compilation_stats()

# Clear cache if corrupted
compiler.clear_cache()
```

### Performance Issues

```python
# Profile to find bottlenecks
config = AMDConfig(enable_profiling=True)
utils = HIPUtilities(config)

with utils.profile_region("suspect_code"):
    # Your code here
    pass

summary = utils.get_profiling_summary()
for name, data in summary.get("regions", {}).items():
    print(f"{name}: {data['avg_ms']:.3f}ms (min={data['min_ms']:.3f}, max={data['max_ms']:.3f})")
```

---

## Known Limitations

1. **ROCm Required**: Full functionality requires ROCm installation
2. **Matrix Cores**: Only available on CDNA2/CDNA3 (MI200/MI300)
3. **FP8 Support**: Limited to CDNA3 (MI300), experimental
4. **Consumer GPUs**: RDNA2/RDNA3 have limited optimization options
5. **XLA**: No XLA support (use TPU backend for XLA workloads)

---

## Best Practices

### 1. Use Architecture-Specific Configuration

```python
# Always specify architecture for optimal defaults
config = AMDConfig(architecture=AMDArchitecture.CDNA3)
```

### 2. Profile Before Optimizing

```python
# Identify bottlenecks first
config = AMDConfig(enable_profiling=True)
```

### 3. Start Conservative, Then Optimize

```python
# Start with conservative, increase if stable
config = AMDConfig(optimization_level="conservative")
# ... test ...
config = AMDConfig(optimization_level="balanced")
# ... test ...
config = AMDConfig(optimization_level="aggressive")
```

### 4. Monitor Memory Usage

```python
# Check memory regularly
manager = AMDMemoryManager(config)
stats = manager.get_memory_stats()
print(f"Memory usage: {stats.allocated_mb / stats.total_mb * 100:.1f}%")
```

### 5. Handle CPU Fallback Gracefully

```python
backend = AMDBackend()
if backend.device.type == "cpu":
    # Adjust batch size, disable certain features
    batch_size = batch_size // 4
```

---

## Next Steps

- **Backend Selection**: See [Backend Selection Guide](../guides/backend_selection.md)
- **Troubleshooting**: See [Troubleshooting Guide](../guides/troubleshooting.md)
- **NVIDIA Backend**: See [NVIDIA Backend](nvidia.md)
- **TPU Backend**: See [TPU Backend](tpu.md)
- **Testing Guide**: See [Testing Guide](../guides/testing_guide.md)

---

## Summary

The AMD ROCm Backend provides:

- **CDNA3 (MI300)**: Maximum performance with Matrix Cores v2, BF16
- **CDNA2 (MI200)**: Excellent data center performance
- **RDNA3/RDNA2**: Consumer GPU support with CPU fallback

**Quick Configuration Guide:**

| Use Case | Architecture | Precision | Optimization |
|----------|--------------|-----------|--------------|
| LLM Training | CDNA3 | bf16 | aggressive |
| General Training | CDNA2 | fp16 | balanced |
| Inference | CDNA2/3 | fp16/bf16 | balanced |
| Development | RDNA3/CPU | fp32 | conservative |

For questions or issues, see the troubleshooting guide or file an issue.
