# Quick Start

Get running with TorchBridge in three steps: install, detect hardware, run your model.

## 1. Install

```bash
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge
pip install -r requirements.txt
```

## 2. Detect Your Hardware

```python
from torchbridge.backends import BackendFactory, detect_best_backend

backend = BackendFactory.create(detect_best_backend())
print(f"Backend: {backend}")
```

TorchBridge automatically detects NVIDIA CUDA, AMD ROCm, Intel XPU, Google TPU, or falls back to CPU.

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

Export to portable formats for deployment:

```python
from torchbridge.deployment import export_to_torchscript, export_to_onnx, export_to_safetensors

sample = torch.randn(1, 768)

export_to_torchscript(model, sample, "model.pt")
export_to_onnx(model, sample, "model.onnx", opset_version=17)
export_to_safetensors(model, "model.safetensors")
```

## CLI Tools

```bash
# System diagnostics
torchbridge doctor

# Optimize a saved model
torchbridge optimize --model model.pt --level production

# Benchmark
torchbridge benchmark --model model.pt --batch-sizes 1,8,32
```

## Next Steps

- [Backends Overview](../backends/overview.md) -- how the hardware abstraction layer works
- [Backend Selection](../guides/backend-selection.md) -- choosing and configuring backends
- [Distributed Training](../guides/distributed-training.md) -- multi-GPU and multi-node
- [Deployment](../guides/deployment.md) -- serving and containerization
