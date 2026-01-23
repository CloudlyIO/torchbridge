# Backend Troubleshooting Guide

**Version**: v0.3.6
**Last Updated**: December 31, 2025

## Overview

This guide helps you diagnose and resolve common issues with NVIDIA GPU, AMD ROCm, and TPU backends in KernelPyTorch.

## Table of Contents

- [Common Issues](#common-issues)
- [NVIDIA Backend Issues](#nvidia-backend-issues)
- [AMD Backend Issues](#amd-backend-issues)
- [TPU Backend Issues](#tpu-backend-issues)
- [Performance Issues](#performance-issues)
- [Debugging Tools](#debugging-tools)

---

## Common Issues

### Issue: Backend Not Detected

**Symptoms:**
- `HardwareDetector` returns 'cpu' when GPU/TPU available
- Backend initializes but uses CPU fallback

**Solutions:**

1. **Check Hardware Installation**
   ```python
   # Check CUDA availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")

   # Check XLA availability
   try:
       import torch_xla
       import torch_xla.core.xla_model as xm
       print(f"XLA device: {xm.xla_device()}")
   except ImportError:
       print("PyTorch/XLA not installed")
   ```

2. **Verify Drivers**
   ```bash
   # NVIDIA: Check driver
   nvidia-smi

   # AMD: Check ROCm
   rocminfo
   rocm-smi

   # TPU: Check environment
   echo $TPU_NAME
   ```

3. **Check Dependencies**
   ```python
   # Verify installation
   from kernel_pytorch.core.hardware_detector import HardwareDetector

   detector = HardwareDetector()
   profile = detector.detect()
   print(f"Detected: {profile.hardware_type}")
   ```

---

### Issue: Import Errors

**Symptoms:**
- `ImportError: cannot import name 'NVIDIABackend'`
- `ModuleNotFoundError: No module named 'kernel_pytorch'`

**Solutions:**

1. **Check Installation**
   ```bash
   pip install -e .  # Development mode
   # or
   pip install kernel-pytorch
   ```

2. **Verify PYTHONPATH**
   ```bash
   export PYTHONPATH=/path/to/kernel-pytorch/src:$PYTHONPATH
   ```

3. **Check Import Path**
   ```python
   # Correct imports
   from kernel_pytorch.backends.nvidia import NVIDIABackend
   from kernel_pytorch.backends.amd import AMDBackend
   from kernel_pytorch.backends.tpu import TPUBackend
   ```

---

### Issue: Dtype Mismatches

**Symptoms:**
- `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`
- Model runs on NVIDIA but fails on TPU

**Cause:**
TPU backend automatically converts models to bfloat16 for optimal performance.

**Solutions:**

1. **Convert Input Data**
   ```python
   # Match model dtype
   if backend_name == "tpu":
       input_data = input_data.to(torch.bfloat16)
   ```

2. **Disable Automatic Conversion**
   ```python
   # Use float32 instead
   config.hardware.tpu.precision = "float32"
   config.hardware.tpu.mixed_precision = False
   ```

3. **Use Consistent Precision**
   ```python
   # Ensure model and inputs match
   model = model.to(backend.device).to(torch.bfloat16)
   inputs = inputs.to(backend.device).to(torch.bfloat16)
   ```

---

## NVIDIA Backend Issues

### Issue: CUDA Out of Memory (OOM)

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes after N iterations

**Solutions:**

1. **Monitor Memory Usage**
   ```python
   from kernel_pytorch.backends.nvidia import NVIDIABackend

   backend = NVIDIABackend(config)
   stats = backend.get_memory_stats()
   print(f"Allocated: {stats['allocated']}MB")
   print(f"Reserved: {stats['reserved']}MB")
   print(f"Free: {stats['free']}MB")
   ```

2. **Clear GPU Cache**
   ```python
   # Clear cache between runs
   backend.empty_cache()

   # Reset peak memory stats
   backend.reset_peak_memory_stats()
   ```

3. **Reduce Batch Size**
   ```python
   # Use smaller batches
   batch_size = 16  # Instead of 64
   ```

4. **Enable Gradient Checkpointing**
   ```python
   # Trade computation for memory
   from kernel_pytorch.advanced_memory import SelectiveGradientCheckpointing

   checkpointing = SelectiveGradientCheckpointing(model)
   ```

5. **Use OOM Protection**
   ```python
   # v0.3.1+: Automatic OOM protection
   config.hardware.nvidia.enable_oom_protection = True
   ```

---

### Issue: FlashAttention Not Working

**Symptoms:**
- No speedup with FlashAttention
- `flash-attn` import errors

**Solutions:**

1. **Check Hardware Compatibility**
   ```python
   # FlashAttention-3 requires H100/Blackwell
   from kernel_pytorch.core.hardware_detector import HardwareDetector

   detector = HardwareDetector()
   profile = detector.detect()

   if profile.is_nvidia_h100_or_better():
       print("FlashAttention-3 supported")
   else:
       print("Requires H100 or Blackwell")
   ```

2. **Install FlashAttention**
   ```bash
   pip install flash-attn>=2.3.0
   ```

3. **Enable in Configuration**
   ```python
   config.hardware.nvidia.flash_attention_version = "3"
   config.attention.enable_flash_attention = True
   ```

---

### Issue: Custom Kernels Not Loading

**Symptoms:**
- Custom kernels revert to PyTorch fallback
- No performance improvement

**Solutions:**

1. **Check NVCC Installation**
   ```bash
   nvcc --version
   ```

2. **Verify Compute Capability**
   ```python
   # Check if your GPU supports the kernel
   import torch
   cap = torch.cuda.get_device_capability()
   print(f"Compute capability: {cap}")

   # H100 requires sm_90
   ```

3. **Enable Custom Kernels**
   ```python
   # Use prepare_model_with_custom_kernels
   model = backend.prepare_model_with_custom_kernels(model)
   ```

---

## AMD Backend Issues

### Issue: ROCm Not Available

**Symptoms:**
- `RuntimeWarning: ROCm not available`
- AMD backend falls back to CPU
- HIP initialization errors

**Solutions:**

1. **Check ROCm Installation**
   ```bash
   # Verify ROCm is installed
   rocminfo

   # Check supported GPUs
   rocm-smi --showid

   # Verify ROCM_HOME
   echo $ROCM_HOME
   ```

2. **Install ROCm**
   ```bash
   # Ubuntu 22.04 with ROCm 6.x
   wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
       sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
   echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
       https://repo.radeon.com/rocm/apt/6.0 jammy main" | \
       sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update && sudo apt install rocm
   ```

3. **Set Environment Variables**
   ```bash
   export ROCM_HOME=/opt/rocm
   export PATH=$ROCM_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
   ```

4. **Install PyTorch with ROCm**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   ```

---

### Issue: HIP Kernel Compilation Errors

**Symptoms:**
- `HIPCompilationError`
- Kernel cache misses
- Slow first-run performance

**Solutions:**

1. **Check HIP Compiler**
   ```bash
   # Verify hipcc is available
   hipcc --version
   ```

2. **Clear Kernel Cache**
   ```python
   from kernel_pytorch.backends.amd import ROCmCompiler
   from kernel_pytorch.core.config import AMDConfig

   compiler = ROCmCompiler(AMDConfig())
   compiler.clear_cache()
   ```

3. **Enable Verbose Compilation**
   ```python
   config = AMDConfig(enable_profiling=True)
   compiler = ROCmCompiler(config)
   stats = compiler.get_compilation_stats()
   print(f"Cache hits: {stats['cache_hit_rate_percent']:.1f}%")
   ```

---

### Issue: AMD Memory Errors

**Symptoms:**
- `ROCm out of memory`
- Training crashes with memory allocation errors

**Solutions:**

1. **Monitor Memory Usage**
   ```python
   from kernel_pytorch.backends.amd import AMDBackend

   backend = AMDBackend()
   info = backend.get_device_info()
   print(f"Memory: {info}")
   ```

2. **Configure Memory Pool**
   ```python
   from kernel_pytorch.core.config import AMDConfig

   config = AMDConfig(
       memory_pool_size_gb=8.0,  # Reduce pool size
       enable_memory_pooling=True,
   )
   ```

3. **Clear GPU Cache**
   ```python
   backend.empty_cache()
   ```

4. **Use Gradient Checkpointing**
   ```python
   from kernel_pytorch.advanced_memory import SelectiveGradientCheckpointing

   checkpointing = SelectiveGradientCheckpointing(model)
   ```

---

### Issue: Matrix Cores Not Used

**Symptoms:**
- No performance improvement with CDNA GPUs
- Matrix operations not accelerated

**Solutions:**

1. **Check Architecture Support**
   ```python
   from kernel_pytorch.core.config import AMDConfig, AMDArchitecture

   # Only CDNA2 (MI200) and CDNA3 (MI300) have Matrix Cores
   config = AMDConfig(
       architecture=AMDArchitecture.CDNA3,
       enable_matrix_cores=True,
   )
   ```

2. **Verify GPU Type**
   ```bash
   # Check if you have a CDNA GPU (MI series)
   rocm-smi --showproductname
   # Matrix Cores only on: MI100, MI200, MI300 series
   ```

3. **Use Appropriate Data Types**
   ```python
   # Matrix Cores work best with FP16/BF16
   model = model.half()  # Convert to FP16
   ```

---

## TPU Backend Issues

### Issue: XLA Not Found

**Symptoms:**
- `RuntimeWarning: PyTorch/XLA not available`
- TPU backend falls back to CPU

**Solutions:**

1. **Install PyTorch/XLA**
   ```bash
   # For TPU v4/v5
   pip install torch_xla[tpu]~=2.0 -f https://storage.googleapis.com/libtpu-releases/index.html
   ```

2. **Check TPU Environment**
   ```bash
   # Verify TPU is accessible
   echo $TPU_NAME
   python3 -c "import torch_xla; import torch_xla.core.xla_model as xm; print(xm.xla_device())"
   ```

3. **Configure TPU Access**
   ```bash
   # Set TPU environment variable
   export TPU_NAME=your-tpu-name
   ```

---

### Issue: Slow XLA Compilation

**Symptoms:**
- First iteration takes very long
- Model compilation exceeds timeout

**Solutions:**

1. **Enable XLA Caching**
   ```python
   config.hardware.tpu.enable_xla_cache = True
   config.hardware.tpu.cache_max_size = 100
   ```

2. **Use Static Shapes**
   ```python
   # Avoid dynamic shapes
   # Bad: variable sequence lengths
   # Good: fixed sequence length

   # Pad to fixed size
   max_length = 512
   inputs = torch.nn.functional.pad(inputs, (0, max_length - inputs.size(1)))
   ```

3. **Increase Timeout**
   ```python
   config.hardware.tpu.compilation_timeout_seconds = 600  # 10 minutes
   ```

---

### Issue: TPU Memory Errors

**Symptoms:**
- `TPUOutOfMemoryError`
- Training crashes on TPU

**Solutions:**

1. **Monitor TPU Memory**
   ```python
   from kernel_pytorch.backends.tpu import TPUBackend

   backend = TPUBackend(config)
   stats = backend.get_memory_stats()
   print(f"Allocated: {stats['allocated_memory']}MB")
   print(f"Available: {stats['available_memory']}MB")
   ```

2. **Reduce Memory Fraction**
   ```python
   # Use less TPU HBM
   config.hardware.tpu.memory_fraction = 0.8  # Instead of 0.9
   ```

3. **Clear Memory Pools**
   ```python
   # For custom memory pools
   from kernel_pytorch.backends.tpu import TPUMemoryManager

   memory_mgr = TPUMemoryManager(config.hardware.tpu)
   memory_mgr.clear_memory_pools()
   ```

---

## Performance Issues

### Issue: Slow Training Speed

**Diagnosis Steps:**

1. **Profile Backend Operations**
   ```python
   import time

   # Measure model preparation
   start = time.perf_counter()
   model = backend.prepare_model(your_model)
   print(f"Preparation: {time.perf_counter() - start:.3f}s")

   # Measure forward pass
   backend.synchronize()
   start = time.perf_counter()
   output = model(input_data)
   backend.synchronize()
   print(f"Forward pass: {time.perf_counter() - start:.3f}s")
   ```

2. **Check Backend Selection**
   ```python
   # Ensure using correct backend
   from kernel_pytorch.core.hardware_detector import HardwareDetector

   detector = HardwareDetector()
   recommended = detector.get_optimal_backend()
   print(f"Recommended backend: {recommended}")
   ```

3. **Verify Optimizations Enabled**
   ```python
   # NVIDIA: Check custom kernels
   stats = backend.get_device_info()
   print(f"Custom kernels enabled: {stats.get('custom_kernels', False)}")

   # TPU: Check XLA compilation
   print(f"XLA available: {config.hardware.tpu.xla_available}")
   ```

---

### Issue: High Memory Usage

**Solutions:**

1. **Use Mixed Precision**
   ```python
   # Reduces memory usage by ~50%
   config.precision.precision_mode = "mixed"
   config.precision.compute_dtype = "float16"
   ```

2. **Enable Gradient Checkpointing**
   ```python
   from kernel_pytorch.advanced_memory import SelectiveGradientCheckpointing

   checkpointing = SelectiveGradientCheckpointing(
       model,
       checkpoint_ratio=0.5  # Checkpoint 50% of layers
   )
   ```

3. **Use CPU Offloading**
   ```python
   from kernel_pytorch.advanced_memory import CPUGPUHybridOptimizer

   hybrid_optimizer = CPUGPUHybridOptimizer(
       optimizer=optimizer,
       offload_ratio=0.5  # Offload 50% to CPU
   )
   ```

---

## Debugging Tools

### Enable Detailed Logging

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# For specific modules
logging.getLogger('kernel_pytorch.backends.nvidia').setLevel(logging.DEBUG)
logging.getLogger('kernel_pytorch.backends.amd').setLevel(logging.DEBUG)
logging.getLogger('kernel_pytorch.backends.tpu').setLevel(logging.DEBUG)
```

### Use Validation Tools

```python
from kernel_pytorch.validation.unified_validator import UnifiedValidator

validator = UnifiedValidator(config)

# Validate configuration
config_result = validator.validate_configuration()
print(f"Config valid: {config_result.passed}/{config_result.total_tests}")

# Validate model
model_result = validator.validate_model(model, input_shape=(32, 512))
print(f"Model valid: {model_result.passed}/{model_result.total_tests}")

# Check for warnings
for report in model_result.reports:
    if report.status == "warning":
        print(f"⚠️ {report.message}")
```

### Monitor Performance

```python
from kernel_pytorch.core.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()

# Record baseline
tracker.record_baseline(
    model=model,
    input_shape=(32, 512),
    batch_size=32
)

# After changes, check for regressions
result = tracker.check_regression(
    model=model,
    input_shape=(32, 512),
    batch_size=32
)

if result.has_regression:
    print(f"⚠️ Performance regression detected!")
    print(f"Latency: {result.latency_change:.1%}")
```

---

## Getting Help

### Check Documentation

- [Backend Selection Guide](backend_selection.md)
- [NVIDIA Backend Documentation](../backends/nvidia.md)
- [AMD Backend Documentation](../backends/amd.md)
- [TPU Backend Documentation](../backends/tpu.md)
- [Testing Guide](testing_guide.md)

### Report Issues

If you encounter a bug:

1. **Gather Information**
   ```python
   # System info
   import torch
   import sys
   print(f"Python: {sys.version}")
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.version.cuda}")

   # Hardware info
   from kernel_pytorch.core.hardware_detector import HardwareDetector
   detector = HardwareDetector()
   profile = detector.detect()
   print(f"Hardware: {profile.hardware_type}")
   ```

2. **Create Minimal Reproduction**
   ```python
   # Simplest code that reproduces the issue
   from kernel_pytorch.backends.nvidia import NVIDIABackend
   backend = NVIDIABackend(config)
   # ...reproduce issue...
   ```

3. **Report on GitHub**
   - Include system info
   - Provide error traceback
   - Share minimal reproduction code

### Enable Strict Validation

For development, enable strict validation mode:

```python
# Raise errors instead of warnings
config.hardware.nvidia.enable_strict_validation = True
config.hardware.tpu.enable_strict_validation = True
```

---

## Summary

**Quick Troubleshooting Checklist:**

- ✅ Hardware detected correctly?
- ✅ Dependencies installed? (CUDA/ROCm/XLA)
- ✅ Drivers up to date?
- ✅ Correct backend selected?
- ✅ Dtypes match?
- ✅ Sufficient memory?
- ✅ Logging enabled?

**Common Solutions:**

- Clear cache: `backend.empty_cache()`
- Reduce batch size
- Enable checkpointing
- Check dtype consistency
- Verify backend selection
- Monitor memory usage

**Backend-Specific Quick Fixes:**

- **NVIDIA**: Check `nvidia-smi`, verify CUDA version
- **AMD**: Check `rocminfo`, verify `$ROCM_HOME`
- **TPU**: Check `$TPU_NAME`, verify PyTorch/XLA installation

For persistent issues, enable debug logging and use validation tools to identify the root cause.
