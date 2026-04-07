# Quick Start

Get running with TorchBridge in three steps: install, detect hardware, run your model.

## 1. Install

```bash
pip install torchbridge-ml
```

## 2. Detect Your Hardware

```python
from torchbridge.backends import BackendFactory, detect_best_backend

backend = BackendFactory.create(detect_best_backend())
print(f"Backend: {backend}")
```

TorchBridge automatically detects NVIDIA CUDA, AMD ROCm, AWS Trainium, Google TPU, or falls back to CPU.

## 3. Optimize and Run

```python
import torch
from torchbridge import TorchBridgeConfig, UnifiedManager

config = TorchBridgeConfig.for_training()
manager = UnifiedManager(config)

# Your model -- no hardware-specific code needed
model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768),
)

# Optimize for detected hardware
optimized_model = manager.optimize(model)
```

## Configuration Presets

TorchBridge provides presets for common workloads:

```python
# Development -- fast iteration, minimal optimization
config = TorchBridgeConfig.for_development()

# Training -- balanced speed and memory
config = TorchBridgeConfig.for_training()

# Inference -- maximum throughput
config = TorchBridgeConfig.for_inference()
```

## Training with AMP

A training loop using PyTorch native automatic mixed precision:

```python
import torch
from torchbridge.backends import BackendFactory, detect_best_backend

# Auto-detect hardware
backend = BackendFactory.create(detect_best_backend())
device = backend.device

model = YourModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler(device.type)

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.amp.autocast(device.type):
        loss = criterion(model(inputs), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

## Model Validation

Verify your model works correctly on the current backend:

```python
from torchbridge import UnifiedValidator

validator = UnifiedValidator()
results = validator.validate_model(model, input_shape=(1, 768))
print(f"Passed: {results.passed}/{results.total_tests}")
```

## Model Export

Use PyTorch's native APIs to export your model:

```python
import torch

sample = torch.randn(1, 768)

# TorchScript
traced = torch.jit.trace(model, sample)
traced.save("model.pt")

# ONNX
torch.onnx.export(model, sample, "model.onnx", opset_version=17)
```

## CLI Tools

```bash
# System diagnostics
tb-doctor

# Cross-backend validation
tb-validate --compare cuda cpu --model model.pt

# Benchmark
tb-benchmark --predefined optimization --quick
```

## Common Pitfalls

### GPU not detected (silent CPU fallback)

TorchBridge falls back to CPU without error if no GPU is found. Run diagnostics to verify:

```bash
tb-doctor
```

If the doctor reports CPU-only, check that your GPU drivers and the correct PyTorch build are installed.

### Device placement mismatch

Always use the detected backend's device for both model and data:

```python
backend = BackendFactory.create(detect_best_backend())
device = backend.device

model = model.to(device)
inputs = inputs.to(device)  # Must match model device
```

Mixing devices (e.g., model on CUDA, inputs on CPU) raises `RuntimeError`.

### Precision differences after optimization

Optimized models may produce slightly different numerical outputs due to mixed precision, kernel fusion, or operator reordering. This is expected. Use `tb-validate` to confirm outputs are within tolerance:

```bash
tb-validate --compare cuda cpu --model optimized_model.pt
```

### Batch size too small for GPU utilization

GPUs need enough parallel work to saturate compute. If throughput is lower than expected, increase the batch size (16--64 is a good starting range) and re-benchmark:

```bash
tb-benchmark --model model.pt --batch-sizes 1,16,32,64
```

### Missing optional dependencies

Some features require extras. Install everything at once:

```bash
pip install torchbridge-ml[all]
```

Or install only what you need: `torchbridge-ml[quantization]`, `torchbridge-ml[tracing]`.

### Forgetting `model.to(device)` after optimization

`manager.optimize()` returns a new model object. If you move it to a device afterward, use the returned reference:

```python
optimized = manager.optimize(model)
optimized = optimized.to(device)  # Use the optimized model, not the original
```

### Model optimization changes output slightly

Small numerical differences (typically < 1e-4) are normal after optimization and do not indicate a bug. Run `tb-validate` to verify that outputs remain within acceptable tolerance for your use case.

## Next Steps

- [Backends Overview](../backends/overview.md) -- how the backend system works
- [Backend Selection](../guides/backend-selection.md) -- choosing and configuring backends
- [Distributed Training](../guides/distributed-training.md) -- multi-GPU and multi-node
- [Deployment](../guides/deployment.md) -- serving and containerization
