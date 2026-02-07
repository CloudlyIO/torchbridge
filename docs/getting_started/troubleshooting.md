# Troubleshooting

Common issues and solutions for TorchBridge backends.

## Common Issues

### Backend Not Detected

**Symptoms:** `HardwareDetector` returns `cpu` when GPU/TPU is available.

**Solutions:**

1. Check hardware availability:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")
   ```

2. Verify drivers:
   ```bash
   nvidia-smi       # NVIDIA
   rocminfo         # AMD
   echo $TPU_NAME   # TPU
   ```

3. Check TorchBridge detection:
   ```python
   from torchbridge.backends import detect_best_backend
   print(f"Detected: {detect_best_backend()}")
   ```

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/path/to/torchbridge/src:$PYTHONPATH

# Or install in editable mode
pip install -e .
```

Correct import paths:
```python
from torchbridge.backends.nvidia import NVIDIABackend
from torchbridge.backends.amd import AMDBackend
from torchbridge.backends.intel import IntelBackend
from torchbridge.backends.tpu import TPUBackend
```

### Dtype Mismatches

**Symptoms:** `RuntimeError: mat1 and mat2 must have the same dtype`

This commonly happens when a backend auto-converts to BF16 but inputs remain FP32.

```python
# Match model and input dtypes
model = model.to(backend.device).to(torch.bfloat16)
inputs = inputs.to(backend.device).to(torch.bfloat16)
```

Or disable automatic conversion:
```python
config.hardware.tpu.precision = "float32"
config.hardware.tpu.mixed_precision = False
```

---

## NVIDIA Issues

### CUDA Out of Memory

```python
# Monitor memory
backend = NVIDIABackend(config)
stats = backend.get_memory_stats()
print(f"Allocated: {stats['allocated']}MB / Free: {stats['free']}MB")

# Clear cache
backend.empty_cache()
```

Mitigations:
- Reduce batch size
- Enable gradient checkpointing: `SelectiveGradientCheckpointing(model)`
- Enable OOM protection: `config.hardware.nvidia.enable_oom_protection = True`

### FlashAttention Not Working

FlashAttention v2 works on Ampere and newer GPUs (sm_80+). FlashAttention v3 requires H100/Blackwell (sm_90+). Install with `pip install flash-attn>=2.3.0`.

```python
# FlashAttention v2 (Ampere+: A100, RTX 3090, etc.)
config.hardware.nvidia.flash_attention_version = "2"
config.attention.enable_flash_attention = True

# FlashAttention v3 (Hopper+: H100, Blackwell, etc.)
config.hardware.nvidia.flash_attention_version = "3"
config.attention.enable_flash_attention = True
```

---

## AMD Issues

### ROCm Not Available

```bash
# Verify ROCm
rocminfo
rocm-smi --showid
echo $ROCM_HOME

# Set environment
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH

# Install PyTorch with ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### HIP Compilation Errors

```python
# Clear kernel cache
from torchbridge.backends.amd import ROCmCompiler
compiler = ROCmCompiler(config)
compiler.clear_cache()
```

### Matrix Cores Not Used

Only CDNA GPUs (MI100/MI200/MI300) have Matrix Cores. RDNA GPUs do not.

```python
config = AMDConfig(architecture=AMDArchitecture.CDNA3, enable_matrix_cores=True)
```

---

## TPU Issues

### XLA Not Found

```bash
pip install torch_xla[tpu]~=2.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

### Slow XLA Compilation

First iteration is slow due to XLA graph compilation. Mitigations:
- Use static shapes (pad to fixed lengths)
- Enable caching: `config.hardware.tpu.enable_xla_cache = True`
- Increase timeout: `config.hardware.tpu.compilation_timeout_seconds = 600`

### TPU Memory Errors

```python
backend = TPUBackend(config)
stats = backend.get_memory_stats()

# Reduce memory fraction
config.hardware.tpu.memory_fraction = 0.8
```

---

## Performance Issues

### Slow Training

1. Verify backend: `print(detect_best_backend())`
2. Profile operations:
   ```python
   backend.synchronize()
   start = time.perf_counter()
   output = model(input_data)
   backend.synchronize()
   print(f"Forward: {time.perf_counter() - start:.3f}s")
   ```

### High Memory Usage

- Enable mixed precision: `config.precision.precision_mode = "mixed"`
- Enable gradient checkpointing
- Use CPU offloading for optimizer states

---

## Debugging

### Enable Logging

```python
import logging
logging.getLogger("torchbridge.backends").setLevel(logging.DEBUG)
```

### Validation Tools

```python
from torchbridge.validation import UnifiedValidator

validator = UnifiedValidator(config)
results = validator.validate_model(model, input_shape=(32, 512))
for report in results.reports:
    if report.status == "warning":
        print(f"Warning: {report.message}")
```

### System Diagnostics

```bash
torchbridge doctor --full-report
```

---

## Quick Checklist

- Hardware detected correctly?
- Dependencies installed? (CUDA/ROCm/XLA/IPEX)
- Drivers up to date?
- Correct backend selected?
- Dtypes match between model and inputs?
- Sufficient memory?
- Logging enabled for debugging?

## Getting Help

1. Run `torchbridge doctor` for automated diagnostics
2. Check the [backend-specific docs](../backends/overview.md)
3. Report issues on GitHub with system info:
   ```python
   import torch, sys
   print(f"Python: {sys.version}")
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.version.cuda}")
   ```
