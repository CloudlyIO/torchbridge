# Backend Selection Guide

**Version**: v0.3.3
**Last Updated**: December 29, 2025

## Overview

KernelPyTorch supports multiple hardware backends (NVIDIA GPU, TPU, CPU) with automatic detection and intelligent selection. This guide helps you choose the right backend for your workload and configure it optimally.

## Table of Contents

- [Quick Start](#quick-start)
- [Automatic Backend Selection](#automatic-backend-selection)
- [Manual Backend Selection](#manual-backend-selection)
- [Backend Comparison](#backend-comparison)
- [Configuration Guide](#configuration-guide)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

---

## Quick Start

### Automatic Selection (Recommended)

Let KernelPyTorch automatically detect and select the best backend:

```python
from kernel_pytorch.core.hardware_detector import HardwareDetector
from kernel_pytorch.core.config import KernelPyTorchConfig

# Automatic detection
detector = HardwareDetector()
backend_name = detector.get_optimal_backend()
print(f"Selected backend: {backend_name}")

# Use detected backend
config = KernelPyTorchConfig()
# Backend will be automatically initialized based on detection
```

### Manual Selection

Explicitly choose a specific backend:

```python
from kernel_pytorch.backends.nvidia import NVIDIABackend
from kernel_pytorch.backends.tpu import TPUBackend
from kernel_pytorch.core.config import KernelPyTorchConfig

config = KernelPyTorchConfig()

# Use NVIDIA backend
nvidia_backend = NVIDIABackend(config)
model = nvidia_backend.prepare_model(your_model)

# Or use TPU backend
tpu_backend = TPUBackend(config)
model = tpu_backend.prepare_model(your_model)
```

---

## Automatic Backend Selection

### How It Works

The `HardwareDetector` automatically:

1. **Detects Available Hardware**
   - Checks for NVIDIA CUDA GPUs
   - Checks for Google Cloud TPUs (via PyTorch/XLA)
   - Falls back to CPU if neither is available

2. **Profiles Hardware Capabilities**
   - GPU architecture (H100, A100, etc.)
   - TPU version (v4, v5e, v5p, etc.)
   - Memory capacity
   - Compute capabilities

3. **Selects Optimal Backend**
   - Prioritizes based on workload characteristics
   - Considers hardware generation
   - Accounts for availability

### Detection Example

```python
from kernel_pytorch.core.hardware_detector import HardwareDetector

detector = HardwareDetector()

# Get complete hardware profile
profile = detector.detect()

print(f"Hardware Type: {profile.hardware_type}")
print(f"Device Name: {profile.device_name}")
print(f"Device Count: {profile.device_count}")
print(f"Total Memory: {profile.total_memory_gb}GB")

# Get optimal backend name
backend = detector.get_optimal_backend()
print(f"Recommended Backend: {backend}")
```

### Selection Priority

Default priority order:
1. **NVIDIA H100/Blackwell** - Best for large-scale training
2. **TPU v5p/v6e/v7** - Best for XLA-optimized workloads
3. **NVIDIA A100** - Great for most training tasks
4. **TPU v4/v5e** - Good for TPU-optimized inference
5. **Other NVIDIA GPUs** - Fallback for CUDA workloads
6. **CPU** - Universal fallback

---

## Manual Backend Selection

### When to Use Manual Selection

- **Specific Hardware Requirements**: Need exact control over device
- **Multi-Backend Workflows**: Training on GPU, inference on TPU
- **Debugging**: Testing backend-specific behavior
- **Benchmarking**: Comparing backends

### NVIDIA Backend

**Best For:**
- Training large language models (LLMs)
- Models with custom CUDA kernels
- FlashAttention-3 optimization
- FP8 training (H100/Blackwell)

**Configuration:**

```python
from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAConfig, NVIDIAArchitecture

config = KernelPyTorchConfig()
config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER  # H100
config.hardware.nvidia.fp8_enabled = True
config.hardware.nvidia.flash_attention_version = "3"

from kernel_pytorch.backends.nvidia import NVIDIABackend
backend = NVIDIABackend(config)

# Prepare model
model = backend.prepare_model(your_model)

# Optional: Enable custom kernels
model_with_kernels = backend.prepare_model_with_custom_kernels(your_model)
```

**Key Methods:**
- `prepare_model(model)` - Move model to GPU
- `prepare_model_with_custom_kernels(model)` - Enable custom CUDA kernels
- `get_memory_stats()` - Get GPU memory usage
- `synchronize()` - Synchronize CUDA operations
- `empty_cache()` - Clear GPU cache

### TPU Backend

**Best For:**
- Large batch training
- XLA-optimized models
- Google Cloud TPU pods
- BFloat16 precision training

**Configuration:**

```python
from kernel_pytorch.core.config import KernelPyTorchConfig, TPUConfig, TPUVersion

config = KernelPyTorchConfig()
config.hardware.tpu.version = TPUVersion.V5E
config.hardware.tpu.precision = "bfloat16"
config.hardware.tpu.mixed_precision = True
config.hardware.tpu.compilation_mode = "torch_xla"

from kernel_pytorch.backends.tpu import TPUBackend
backend = TPUBackend(config)

# Prepare model
model = backend.prepare_model(your_model)
```

**Key Methods:**
- `prepare_model(model)` - Move model to TPU with XLA compilation
- `get_memory_stats()` - Get TPU memory usage
- `synchronize()` - Synchronize XLA operations

### CPU Backend (Fallback)

**Best For:**
- Development and testing
- Small models
- When GPU/TPU unavailable

Both NVIDIA and TPU backends automatically fall back to CPU when hardware is unavailable.

---

## Backend Comparison

### Feature Matrix

| Feature | NVIDIA | TPU | CPU |
|---------|--------|-----|-----|
| **Custom CUDA Kernels** | ✅ Full Support | ❌ Not Applicable | ❌ No |
| **FlashAttention-3** | ✅ H100/Blackwell | ❌ No | ❌ No |
| **FP8 Training** | ✅ H100/Blackwell | ❌ No | ❌ No |
| **XLA Compilation** | ⚠️ Limited | ✅ Native | ❌ No |
| **BFloat16** | ✅ Yes | ✅ Native | ✅ Yes |
| **Multi-GPU** | ✅ Native | ❌ Use Pods | ⚠️ Limited |
| **Dynamic Shapes** | ✅ Yes | ⚠️ Limited | ✅ Yes |

### Performance Characteristics

**NVIDIA GPU:**
- **Strengths**: Custom kernels, FlashAttention, FP8, dynamic shapes
- **Weaknesses**: Memory constraints on consumer GPUs
- **Best Workloads**: LLM training, custom CUDA ops, mixed precision

**TPU:**
- **Strengths**: Large batch sizes, XLA optimization, cost-effective for scale
- **Weaknesses**: Static shapes, limited custom ops, bfloat16 only
- **Best Workloads**: Large-scale training, XLA-friendly architectures

**CPU:**
- **Strengths**: Universal availability, good for development
- **Weaknesses**: Slow for large models
- **Best Workloads**: Testing, debugging, small models

---

## Configuration Guide

### NVIDIA Configuration

```python
from kernel_pytorch.core.config import NVIDIAConfig, NVIDIAArchitecture

nvidia_config = NVIDIAConfig(
    # Hardware
    architecture=NVIDIAArchitecture.HOPPER,  # H100
    device_id=0,  # GPU device ID

    # Optimization
    fp8_enabled=True,  # Enable FP8 (H100/Blackwell only)
    flash_attention_version="3",  # FlashAttention version
    tensor_core_version=4,  # Tensor Core generation

    # Memory
    memory_pool_init_mb=1024,  # Initial memory pool size
    memory_growth_enabled=True,  # Allow dynamic growth
    max_memory_fraction=0.9,  # Use up to 90% of GPU memory

    # Precision
    precision="float16",  # Default precision
    allow_fp32=True,  # Allow FP32 fallback
)
```

### TPU Configuration

```python
from kernel_pytorch.core.config import TPUConfig, TPUVersion, TPUTopology

tpu_config = TPUConfig(
    # Hardware
    version=TPUVersion.V5E,
    topology=TPUTopology.SINGLE,  # or POD, SUPERPOD

    # Compilation
    compilation_mode="torch_xla",  # XLA compilation mode
    enable_xla_cache=True,  # Cache XLA compilations
    cache_max_size=100,  # Max cached compilations

    # Precision
    precision="bfloat16",  # TPU native precision
    mixed_precision=True,  # Enable mixed precision

    # Memory
    memory_fraction=0.9,  # Use 90% of TPU HBM
    allocation_history_retention_seconds=3600,  # 1 hour

    # Monitoring
    monitoring_interval_seconds=1.0,
    monitoring_duration_seconds=60.0,
)
```

---

## Performance Optimization

### NVIDIA Optimization Tips

1. **Use FlashAttention for Attention Layers**
   ```python
   # Automatically enabled in H100/Blackwell
   config.hardware.nvidia.flash_attention_version = "3"
   ```

2. **Enable FP8 Training (H100/Blackwell)**
   ```python
   config.hardware.nvidia.fp8_enabled = True
   ```

3. **Use Custom CUDA Kernels**
   ```python
   model = backend.prepare_model_with_custom_kernels(model)
   ```

4. **Optimize Memory**
   ```python
   # Clear cache between runs
   backend.empty_cache()

   # Monitor memory
   stats = backend.get_memory_stats()
   print(f"Allocated: {stats['allocated']}MB")
   ```

### TPU Optimization Tips

1. **Use Static Shapes**
   ```python
   # TPUs work best with fixed input shapes
   # Avoid dynamic shapes when possible
   ```

2. **Leverage XLA Compilation**
   ```python
   # XLA automatically optimizes computation graphs
   config.hardware.tpu.enable_xla_cache = True
   ```

3. **Use Large Batch Sizes**
   ```python
   # TPUs excel with large batches
   # Aim for batch sizes of 128+ when possible
   ```

4. **Optimize for BFloat16**
   ```python
   # Use bfloat16 as TPU's native precision
   config.hardware.tpu.precision = "bfloat16"
   ```

---

## Best Practices

### 1. Use Automatic Detection for Production

```python
# Let the system choose the best backend
detector = HardwareDetector()
backend_name = detector.get_optimal_backend()
```

### 2. Validate Backend Compatibility

```python
from kernel_pytorch.validation.unified_validator import UnifiedValidator

validator = UnifiedValidator(config)

# Validate for NVIDIA
nvidia_result = validator.validate_nvidia_compatibility(model)

# Validate for TPU
tpu_result = validator.validate_tpu_compatibility(model)
```

### 3. Test on Multiple Backends

```python
# Test your model works on all backends
backends = [NVIDIABackend(config), TPUBackend(config)]

for backend in backends:
    model = backend.prepare_model(your_model)
    # Run validation tests
```

### 4. Use Backend-Specific Optimizations

```python
# Don't use NVIDIA-specific features on TPU
if backend_name == "nvidia":
    # Use FlashAttention, FP8, custom kernels
    pass
elif backend_name == "tpu":
    # Use large batches, static shapes, bfloat16
    pass
```

### 5. Monitor Performance

```python
# Track memory and performance
stats = backend.get_memory_stats()
print(f"Memory: {stats}")

# Synchronize for accurate timing
backend.synchronize()
```

### 6. Handle Cross-Backend Workflows

```python
# Train on NVIDIA, infer on TPU
nvidia_backend = NVIDIABackend(config)
nvidia_model = nvidia_backend.prepare_model(model)

# Train...
# Save checkpoint
checkpoint = {
    'model_state_dict': {k: v.cpu() for k, v in nvidia_model.state_dict().items()}
}

# Load on TPU
tpu_backend = TPUBackend(config)
tpu_model = tpu_backend.prepare_model(model)
tpu_model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Next Steps

- **Troubleshooting**: See [Troubleshooting Guide](troubleshooting.md)
- **Hardware Details**: See [NVIDIA Backend](../backends/nvidia.md) and [TPU Backend](../backends/tpu.md)
- **API Reference**: See [CLI Reference](../capabilities/cli_reference.md)
- **Testing**: See [Testing Guide](testing_guide.md)

---

## Summary

**Quick Selection Guide:**

- **Large-scale LLM Training** → NVIDIA H100 with FP8
- **Cost-effective Scale** → TPU v5e/v5p
- **Custom Kernels** → NVIDIA GPU
- **XLA-optimized Models** → TPU
- **Development/Testing** → CPU Fallback

**Remember:**
- Use automatic detection for production
- Validate on target backend before deployment
- Monitor memory and performance
- Test cross-backend compatibility for portability

For detailed backend-specific guidance, see the individual backend documentation.
