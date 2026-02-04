# Backend Selection Guide

How to choose and configure the right backend for your workload.

## Automatic Selection

TorchBridge auto-detects available hardware and selects the optimal backend:

```python
from torchbridge.backends import BackendFactory, detect_best_backend

backend_name = detect_best_backend()  # "cuda", "rocm", "xpu", "tpu", or "cpu"
backend = BackendFactory.create(backend_name)
```

Detection priority: NVIDIA CUDA > AMD ROCm > Intel XPU > Google TPU > CPU.

For most users, automatic selection is the right choice. Manual selection is useful when you have multiple accelerators or want to force a specific backend.

## Manual Selection

```python
# Force a specific backend
backend = BackendFactory.create("cuda")   # NVIDIA
backend = BackendFactory.create("rocm")   # AMD
backend = BackendFactory.create("xpu")    # Intel
backend = BackendFactory.create("tpu")    # Google TPU
backend = BackendFactory.create("cpu")    # CPU fallback
```

Via environment variable:

```bash
export TORCHBRIDGE_BACKEND=rocm
```

## Feature Matrix

| Feature | NVIDIA | AMD | Intel | TPU | CPU |
|---------|--------|-----|-------|-----|-----|
| FP8 training | H100+ | -- | -- | -- | -- |
| BF16 training | Ampere+ | CDNA2+ | PVC, Arc | All | Some |
| FP16 training | All | All | PVC, Arc | -- | -- |
| FlashAttention | Ampere+ | -- | -- | -- | -- |
| Distributed | Multi-GPU | Multi-GPU | Multi-GPU | Pods | -- |
| torch.compile | Yes | Yes | Yes | Partial | Yes |

## Choosing by Workload

### Training (Large Models)

**Best:** NVIDIA H100/A100, AMD MI300X, TPU v5p

These have the most HBM and highest matmul throughput.

```python
config = TorchBridgeConfig.for_training()
```

### Inference (Low Latency)

**Best:** NVIDIA L4/T4, TPU v5e, Intel Arc

Optimized for throughput per dollar.

```python
config = TorchBridgeConfig.for_inference()
```

### Development (Iteration Speed)

**Best:** Whatever you have locally.

```python
config = TorchBridgeConfig.for_development()
```

### Cost Optimization

Compare backends by cost-per-token on your workload:

| Backend | Typical Cloud Cost | Best For |
|---------|-------------------|----------|
| NVIDIA L4 | $0.70/hr (GCP) | Inference |
| NVIDIA A100 | $3.67/hr (GCP) | Training |
| AMD MI300X | ~$3.50/hr | Training (alternative) |
| TPU v5e | $1.20/hr (GCP) | Cost-effective training |
| CPU | $0.05-0.50/hr | Prototyping |

## Backend-Specific Configuration

### NVIDIA

```python
from torchbridge import TorchBridgeConfig

config = TorchBridgeConfig.for_training()
config.hardware.nvidia.fp8_enabled = True           # H100+
config.hardware.nvidia.flash_attention_version = "3" # H100+
config.hardware.nvidia.enable_oom_protection = True
```

### AMD

```python
from torchbridge.core.config import AMDConfig, AMDArchitecture

config = AMDConfig(
    architecture=AMDArchitecture.CDNA3,
    enable_matrix_cores=True,
    memory_pool_size_gb=8.0,
)
```

### Intel

```python
from torchbridge.core.config import IntelConfig

config = IntelConfig(
    enable_ipex=True,
    enable_onednn=True,
    enable_amx=True,
)
```

### TPU

```python
from torchbridge.core.config import TPUConfig

config = TPUConfig(
    precision="bfloat16",
    enable_xla_cache=True,
    cache_max_size=100,
)
```

## Validation

After selecting a backend, validate it works correctly:

```python
from torchbridge import UnifiedValidator

validator = UnifiedValidator()
results = validator.validate_model(model, input_shape=(1, 768))
print(f"Passed: {results.passed}/{results.total_tests}")
```

## Cross-Backend Testing

To ensure your model works on multiple backends:

```python
for backend_name in ["cuda", "rocm", "cpu"]:
    try:
        backend = BackendFactory.create(backend_name)
        model = YourModel().to(backend.device)
        model = backend.prepare_model(model)
        output = model(sample_input.to(backend.device))
        print(f"{backend_name}: OK")
    except Exception as e:
        print(f"{backend_name}: {e}")
```

## See Also

- [Backends Overview](../backends/overview.md)
- [Hardware Matrix](../reference/hardware-matrix.md)
- [Hardware Setup](hardware-setup.md)
