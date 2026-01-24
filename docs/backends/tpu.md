# TPU Backend Guide - v0.4.18

**Status**: ✅ Production-Ready (90%+)
**Last Updated**: December 29, 2025
**Version**: v0.4.18 (Phase 4C-Pre Week 2 Complete)

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

The TPU Backend provides production-ready support for Google Cloud TPUs through PyTorch/XLA integration. It offers:

- **Multi-TPU Support**: v4, v5e, v5p, v6e, v7
- **XLA Compilation**: Automatic and optimized compilation
- **Memory Management**: TPU-specific memory optimization and pooling
- **Distributed Training**: TPU Pod support
- **LRU Caching**: Bounded cache with automatic eviction (v0.4.18+)
- **Structured Logging**: Production-grade logging (v0.4.18+)
- **Custom Exceptions**: Comprehensive error hierarchy (v0.4.18+)

### Key Features (v0.4.18)

✅ **Structured Logging** - All operations logged with proper levels
✅ **LRU Cache Management** - Prevents unbounded memory growth
✅ **Configurable Parameters** - 8 new configuration options
✅ **Custom Exceptions** - 13 exception classes for better error handling
✅ **Strict Validation Mode** - Optional strict error checking
✅ **79+ Tests Passing** - Comprehensive test coverage

---

## Quick Start

```python
from kernel_pytorch.backends.tpu import TPUBackend, TPUOptimizer
from kernel_pytorch.core.config import KernelPyTorchConfig
import torch.nn as nn

# Initialize TPU backend
config = KernelPyTorchConfig()
backend = TPUBackend(config)

# Prepare model for TPU
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, 10)
)
prepared_model = backend.prepare_model(model)

# Optimize for TPU
optimizer = TPUOptimizer(config)
result = optimizer.optimize(
    model,
    sample_inputs=torch.randn(32, 512),
    optimization_level="balanced"
)

print(f"Optimized model ready for TPU execution")
```

---

## Installation

### Prerequisites

1. **PyTorch/XLA** - Required for TPU support:
   ```bash
   pip install torch torch_xla
   ```

2. **Google Cloud TPU** - Access to TPU hardware via:
   - Google Cloud Platform
   - Kaggle TPU kernels
   - Google Colab (TPU runtime)

### Verify Installation

```python
from kernel_pytorch.backends.tpu import TPUBackend
from kernel_pytorch.core.config import KernelPyTorchConfig

config = KernelPyTorchConfig()
backend = TPUBackend(config)

print(f"TPU Device: {backend.device}")
print(f"World Size: {backend.world_size}")
```

---

## Configuration

### Basic Configuration

```python
from kernel_pytorch.core.config import KernelPyTorchConfig, TPUConfig, TPUVersion

config = KernelPyTorchConfig()

# TPU-specific configuration
config.hardware.tpu.version = TPUVersion.V5P
config.hardware.tpu.memory_fraction = 0.90
config.hardware.tpu.mixed_precision = True
config.hardware.tpu.precision = "bfloat16"
```

### New Configuration Options (v0.4.18)

```python
# Cache management
config.hardware.tpu.cache_max_size = 100  # Max cached models/compilations
config.hardware.tpu.compilation_timeout_seconds = 300  # XLA timeout

# Memory management
config.hardware.tpu.allocation_history_retention_seconds = 3600  # 1 hour
config.hardware.tpu.v6e_memory_gb = 32.0  # Override v6e capacity
config.hardware.tpu.v7_memory_gb = 128.0  # Override v7 capacity

# Validation
config.hardware.tpu.enable_strict_validation = False  # Strict error mode

# Monitoring
config.hardware.tpu.monitoring_interval_seconds = 1.0
config.hardware.tpu.monitoring_duration_seconds = 60.0
```

### Configuration Modes

#### Inference Mode
```python
config = KernelPyTorchConfig(mode="inference")
config.hardware.tpu.gradient_checkpointing = False
config.hardware.tpu.mixed_precision = True
```

#### Training Mode
```python
config = KernelPyTorchConfig(mode="training")
config.hardware.tpu.gradient_checkpointing = True
config.hardware.tpu.mixed_precision = True
config.hardware.tpu.xla_optimization_level = 2
```

#### Development Mode
```python
config = KernelPyTorchConfig(mode="development")
config.hardware.tpu.enable_strict_validation = True
config.hardware.tpu.xla_optimization_level = 0  # Debug mode
```

---

## Core Components

### 1. TPUBackend

Main interface for TPU device management:

```python
from kernel_pytorch.backends.tpu import TPUBackend

backend = TPUBackend(config)

# Device information
print(f"Device: {backend.device}")
print(f"World Size: {backend.world_size}")
print(f"Rank: {backend.rank}")
print(f"Is Distributed: {backend.is_distributed}")

# Model preparation
prepared_model = backend.prepare_model(model)

# Data preparation
prepared_data = backend.prepare_data(data)

# Synchronization
backend.synchronize()

# Memory stats
stats = backend.get_memory_stats()
print(f"Cache Stats: {stats['model_cache_stats']}")
```

### 2. TPUOptimizer

High-level model optimization:

```python
from kernel_pytorch.backends.tpu import TPUOptimizer

optimizer = TPUOptimizer(config)

# Optimize model
result = optimizer.optimize(
    model,
    sample_inputs=sample_input,
    optimization_level="balanced"  # "conservative", "balanced", "aggressive"
)

# Access optimized model
optimized_model = result.optimized_model

# View metrics
print(f"Optimization Time: {result.optimization_time:.2f}s")
print(f"Performance Metrics: {result.performance_metrics}")
```

### 3. XLACompiler

XLA compilation and caching:

```python
from kernel_pytorch.backends.tpu import XLACompiler

compiler = XLACompiler(config.hardware.tpu)

# Compile model
compiled_model = compiler.compile_model(
    model,
    sample_inputs=sample_input,
    use_cache=True
)

# Get compilation stats
stats = compiler.get_compilation_stats()
print(f"Cache Hit Rate: {stats['compilation_cache']['hit_rate']:.2%}")
print(f"Evictions: {stats['compilation_cache']['evictions']}")
```

### 4. TPUMemoryManager

TPU memory management and pooling:

```python
from kernel_pytorch.backends.tpu import TPUMemoryManager

manager = TPUMemoryManager(config.hardware.tpu)

# Allocate tensor
tensor = manager.allocate_tensor((1000, 1000), dtype=torch.float32)

# Create memory pool
pool_id = manager.create_memory_pool(
    pool_size=10,
    tensor_size=(512, 512)
)

# Get tensor from pool
tensor = manager.get_tensor_from_pool(pool_id)

# Return to pool
manager.return_tensor_to_pool(pool_id, tensor)

# Memory stats
stats = manager.get_memory_stats()
print(f"Allocated: {stats.allocated_memory / 1e9:.2f} GB")
```

### 5. XLA Integration

Distributed training and device management:

```python
from kernel_pytorch.backends.tpu import create_xla_integration

device_manager, distributed, optimizations = create_xla_integration(config.hardware.tpu)

# Device management
print(f"Devices: {device_manager.devices}")
print(f"Current Device: {device_manager.device}")

# Distributed training
if distributed.is_distributed:
    wrapped_model = distributed.wrap_model(model)

    # All-reduce operation
    reduced_tensor = distributed.all_reduce(tensor, op="sum")
```

---

## Usage Examples

### Example 1: Basic Model Training

```python
import torch
import torch.nn as nn
from kernel_pytorch.backends.tpu import TPUBackend
from kernel_pytorch.core.config import KernelPyTorchConfig

# Setup
config = KernelPyTorchConfig(mode="training")
backend = TPUBackend(config)

# Model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
model = backend.prepare_model(model)

# Training loop
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    for batch in dataloader:
        data, target = backend.prepare_data(batch)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        backend.synchronize()  # Important for TPU
```

### Example 2: Model Optimization

```python
from kernel_pytorch.backends.tpu import TPUOptimizer

optimizer = TPUOptimizer(config)

# Conservative (safest, good for debugging)
result = optimizer.optimize_for_inference(model, sample_inputs)

# Balanced (recommended for most use cases)
result = optimizer.optimize(model, sample_inputs, optimization_level="balanced")

# Aggressive (maximum performance)
result = optimizer.optimize_for_training(model, sample_inputs)
```

### Example 3: Distributed Training

```python
from kernel_pytorch.backends.tpu import TPUBackend, XLADistributedTraining

backend = TPUBackend(config)

if backend.is_distributed:
    # Initialize distributed training
    from kernel_pytorch.backends.tpu import XLADeviceManager

    device_manager = XLADeviceManager(config.hardware.tpu)
    distributed = XLADistributedTraining(device_manager)

    # Wrap model
    model = distributed.wrap_model(model)

    # Training with gradient synchronization
    for batch in dataloader:
        # ... forward/backward ...

        # Synchronize gradients across TPUs
        for param in model.parameters():
            if param.grad is not None:
                param.grad = distributed.all_reduce(param.grad, op="mean")
```

### Example 4: Memory Pool Usage

```python
from kernel_pytorch.backends.tpu import TPUMemoryManager

manager = TPUMemoryManager(config.hardware.tpu)

# Create pools for different tensor sizes
activation_pool = manager.create_memory_pool(
    pool_size=50,
    tensor_size=(32, 512, 512)  # batch_size, seq_len, hidden_dim
)

# Use pooled tensors
for batch in dataloader:
    tensor = manager.get_tensor_from_pool(activation_pool)

    # Use tensor...

    # Return to pool
    manager.return_tensor_to_pool(activation_pool, tensor)

# Check pool utilization
pool_stats = manager.get_pool_stats()
print(f"Pool Utilization: {pool_stats['pool_details'][activation_pool]['utilization']:.2%}")
```

---

## Performance Optimization

### Optimization Levels

| Level | Use Case | Optimizations | Risk |
|-------|----------|---------------|------|
| **Conservative** | Debugging, First Run | Basic TPU prep, bfloat16 on Linear only | Low |
| **Balanced** | Production (Recommended) | +Gradient checkpointing, Layer fusion hints | Medium |
| **Aggressive** | Maximum Performance | +All layers to bfloat16, Model-specific opts | Higher |

### Best Practices

1. **Use Mixed Precision (bfloat16)**
   ```python
   config.hardware.tpu.mixed_precision = True
   config.hardware.tpu.precision = "bfloat16"  # TPU native precision
   ```

2. **Enable Gradient Checkpointing**
   ```python
   config.hardware.tpu.gradient_checkpointing = True  # Saves memory
   ```

3. **Batch Size Guidelines**
   - Use powers of 2 (32, 64, 128, 256)
   - Larger is better for TPU utilization
   - v4: 64-128, v5e: 128-256, v5p: 256-512+

4. **Synchronization**
   ```python
   # Call after each training step
   backend.synchronize()  # Ensures XLA ops complete
   ```

5. **Cache Management**
   ```python
   # Set appropriate cache size based on model variety
   config.hardware.tpu.cache_max_size = 100  # Prevents OOM
   ```

---

## Error Handling

### Custom Exceptions (v0.4.18)

```python
from kernel_pytorch.backends.tpu.tpu_exceptions import (
    TPUBackendError,           # Base exception
    TPUNotAvailableError,      # TPU/XLA not available
    XLACompilationError,       # Compilation failed
    TPUMemoryError,            # Memory-related errors
    TPUOutOfMemoryError,       # OOM during allocation
    TPUValidationError,        # Validation failed
    TPUDistributedError,       # Distributed ops failed
)

# Usage
try:
    backend = TPUBackend(config)
    model = backend.prepare_model(my_model)
except TPUNotAvailableError:
    print("TPU not available, falling back to CPU")
except TPUValidationError as e:
    print(f"Model validation failed: {e}")
except TPUBackendError as e:
    print(f"TPU backend error: {e}")
```

### Strict Validation Mode

```python
# Enable strict mode for development/debugging
config.hardware.tpu.enable_strict_validation = True

# In strict mode, warnings become exceptions
optimizer = TPUOptimizer(config)

try:
    result = optimizer.optimize(model, invalid_inputs)
except TPUValidationError as e:
    print(f"Caught validation error: {e}")
    # Fix inputs and retry
```

---

## Troubleshooting

### Common Issues

#### 1. PyTorch/XLA Not Available

**Symptom**: `TPUNotAvailableError` or CPU fallback warnings

**Solution**:
```bash
# Install PyTorch/XLA
pip install torch torch_xla

# Verify installation
python3 -c "import torch_xla; print(torch_xla.__version__)"
```

#### 2. Out of Memory Errors

**Symptom**: `TPUOutOfMemoryError` during training

**Solutions**:
```python
# Reduce memory fraction
config.hardware.tpu.memory_fraction = 0.80

# Enable gradient checkpointing
config.hardware.tpu.gradient_checkpointing = True

# Reduce batch size
batch_size = 64  # Instead of 128

# Use smaller cache
config.hardware.tpu.cache_max_size = 50
```

#### 3. Slow Compilation

**Symptom**: First iteration very slow

**Solutions**:
```python
# This is normal - XLA compiles on first run
# Use caching to avoid recompilation
compiler.compile_model(model, use_cache=True)

# Increase timeout if needed
config.hardware.tpu.compilation_timeout_seconds = 600
```

#### 4. Cache Growth

**Symptom**: Memory usage increases over time

**Solution**:
```python
# Set cache limits (v0.4.18+)
config.hardware.tpu.cache_max_size = 100  # LRU eviction

# Or manually clear caches
backend.clear_cache()
compiler.clear_cache()
```

### Debug Logging

```python
import logging

# Enable debug logging for TPU backend
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now all TPU operations will log detailed information
backend = TPUBackend(config)
```

---

## Known Limitations

### Current Limitations

1. **XLA-Dependent**: Requires PyTorch/XLA installation
2. **First Run Compilation**: Initial runs slower due to XLA compilation
3. **Hardware Specific**: Some optimizations only apply to newer TPU versions

### Version-Specific Notes

| TPU Version | Memory (GB) | Notes |
|-------------|-------------|-------|
| v4 | 32 | Verified capacity |
| v5e | 16 | Verified, cost-optimized |
| v5p | 95 | Verified, high-performance |
| v6e | 32 | Configurable (default 32) |
| v7 | 128 | Configurable (default 128) |

---

## Best Practices

### 1. Configuration Management

```python
# Use environment-specific configs
if os.getenv("ENVIRONMENT") == "production":
    config = KernelPyTorchConfig(mode="training")
    config.hardware.tpu.enable_strict_validation = False
else:
    config = KernelPyTorchConfig(mode="development")
    config.hardware.tpu.enable_strict_validation = True
```

### 2. Monitoring

```python
# Monitor memory usage
manager = TPUMemoryManager(config.hardware.tpu)
stats_history = manager.monitor_memory(
    interval=1.0,  # Check every second
    duration=60.0  # For 60 seconds
)

# Analyze peak usage
peak_memory = max(s.allocated_memory for s in stats_history)
print(f"Peak Memory: {peak_memory / 1e9:.2f} GB")
```

### 3. Model Checkpointing

```python
# Save TPU model with metadata
backend.save_model(
    model,
    "model_checkpoint.pth",
    save_optimizer=True,
    optimizer=optimizer
)

# Load with proper device placement
model = backend.load_model(
    model,
    "model_checkpoint.pth",
    load_optimizer=True,
    optimizer=optimizer
)
```

### 4. Testing

```python
# Run comprehensive tests
pytest tests/test_tpu_backend.py tests/test_tpu_config.py -v

# Specific error path tests (v0.4.18+)
pytest tests/test_tpu_backend.py::TestTPUErrorPaths -v
```

---

## Performance Benchmarks

### Expected Performance (CPU Fallback Mode)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Backend Creation | ~0.13 ms | One-time initialization |
| Model Preparation | <0.001 ms | Cached after first run |
| XLA Compilation | 1-10 s | First run only (cached) |
| Memory Allocation | ~0.001 ms | Very fast |
| Synchronization | <0.01 ms | Per step |

### With Real TPU Hardware

Expected 5-20x speedup over CPU for:
- Large batch training (256+)
- Transformer models
- Matrix-heavy operations

---

## Additional Resources

- **Test Files**: `tests/test_tpu_backend.py` (79 tests)
- **Examples**: `demos/tpu_integration_demo.py`
- **Benchmarks**: `benchmarks/tpu_integration_benchmark.py`
- **PyTorch/XLA Docs**: https://pytorch.org/xla/

---

## Changelog

### v0.4.18 (December 29, 2025) - TPU Backend Hardening

**Major Improvements**:
- ✅ Structured logging (35 print() → logging)
- ✅ LRU cache with configurable size limits
- ✅ 8 new configuration parameters
- ✅ 13 custom exception classes
- ✅ Strict validation mode
- ✅ 16 new error path tests (79 total tests)
- ✅ 749 tests passing (100% success rate)

**Performance**:
- Cache hit rate tracking
- Memory allocation history cleanup
- Automatic LRU eviction prevents OOM

**Production Readiness**: 65% → 90%+

---

**Last Updated**: December 29, 2025
**Status**: Production-Ready (90%+)
**Version**: v0.4.18
