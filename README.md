# TorchBridge

**Hardware Abstraction Layer for PyTorch** -- Write once, run on any accelerator.

[![Version](https://img.shields.io/badge/version-0.4.42-green)](./CHANGELOG.md) [![Tests](https://img.shields.io/badge/tests-1600%20passed-blue)](./docs/reference/hardware-matrix.md) [![Cloud GPU](https://img.shields.io/badge/cloud%20GPU-5%2F5%20passed-brightgreen)](./docs/reference/cloud-validation.md) [![AWS A10G](https://img.shields.io/badge/AWS%20A10G-PASS-brightgreen)](./docs/reference/cloud-validation.md) [![GCP L4](https://img.shields.io/badge/GCP%20L4-PASS-brightgreen)](./docs/reference/cloud-validation.md) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org)

## What is TorchBridge?

PyTorch lets you build models. TorchBridge lets you run them **anywhere**.

Most teams write hardware-specific code -- CUDA calls for NVIDIA, ROCm setup for AMD, XLA boilerplate for TPU. When the hardware changes, the code breaks. TorchBridge eliminates that problem with a **unified API** that detects your hardware and adapts automatically.

```
Your model code
      |
  TorchBridge HAL
      |
  +---------+---------+---------+---------+
  | NVIDIA  |   AMD   |  Intel  |   TPU   |
  | CUDA    |  ROCm   |  IPEX   |   XLA   |
  +---------+---------+---------+---------+
```

**What it does:**
- **Backend detection** -- automatically identifies available accelerators
- **Vendor adapters** -- translates unified API calls to vendor-specific operations
- **Precision management** -- handles FP32/FP16/BF16/FP8 across backends
- **Memory optimization** -- gradient checkpointing, activation offloading, memory pooling
- **Checkpoint portability** -- save on one backend, load on another
- **Distributed training** -- tensor/pipeline/data parallelism across backend types

## Quick Start

```bash
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge
pip install -r requirements.txt

# Verify
PYTHONPATH=src python3 -c "import torchbridge; print(f'TorchBridge v{torchbridge.__version__} ready')"
```

### Detect Hardware

```python
from torchbridge.backends import BackendFactory, detect_best_backend

backend_type = detect_best_backend()  # NVIDIA, AMD, INTEL, TPU, or CPU
backend = BackendFactory.create(backend_type)
print(backend.get_device_info())
```

### Optimize for Any Backend

```python
import torch
from torchbridge import TorchBridgeConfig, UnifiedManager

config = TorchBridgeConfig.for_training()
manager = UnifiedManager(config)

model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768),
)

optimized_model = manager.optimize(model)
```

### Validate

```python
from torchbridge import UnifiedValidator

validator = UnifiedValidator()
results = validator.validate_model(optimized_model, input_shape=(1, 768))
print(f"Validation: {results.passed}/{results.total_tests} tests passed")
```

## Supported Backends

| Backend | Hardware | Precision | Status |
|---------|----------|-----------|--------|
| **NVIDIA** | H100, A100, L4, T4, RTX | FP8, BF16, FP16, FP32 | Production |
| **AMD** | MI300X, MI200, RDNA3 | BF16, FP16, FP32 | Production |
| **Intel** | Ponte Vecchio, Arc, Flex | BF16, FP16, FP32 | Production |
| **TPU** | v4, v5e, v5p, v6e | BF16, FP32 | Production |
| **CPU** | x86, ARM (Apple Silicon) | FP32, BF16 | Fallback |

See [Hardware Matrix](./docs/reference/hardware-matrix.md) for full details.

## Key Features

### Backend Detection and Adaptation
Automatically identifies available hardware and selects the optimal backend. No code changes needed when moving between GPU vendors or cloud providers.

### Vendor Adapters
Each backend implements a common `BaseBackend` interface. Your code calls `manager.optimize(model)` and the correct vendor-specific operations execute underneath -- CUDA on NVIDIA, HIP on AMD, XLA on TPU.

### Precision Management
Configure precision once. TorchBridge handles the details per backend -- FP8 on H100, BF16 where supported, FP16 as fallback. Mixed-precision training with `torch.amp` autocast works across all backends.

### Memory Optimization
Gradient checkpointing, activation offloading, optimizer state sharding, and memory pooling. These work consistently whether you're on a single GPU or a multi-node cluster.

### Checkpoint Portability
Save a checkpoint on NVIDIA hardware, load it on AMD or TPU. TorchBridge handles device mapping and dtype conversion.

### Distributed Training
Tensor parallelism, pipeline parallelism, and FSDP with a unified API. The same distributed training script runs on NVIDIA DGX, AMD Instinct, or TPU pods.

## Code Examples

### Backend-Agnostic Training

```python
import torch
from torchbridge.backends import BackendFactory, detect_best_backend

backend = BackendFactory.create(detect_best_backend())
device = backend.device

model = YourModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Use PyTorch native AMP -- works on any backend
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

### Hardware Capability Queries

```python
from torchbridge.backends.nvidia import NVIDIABackend

nvidia = NVIDIABackend()
print(nvidia.get_device_info())  # GPU model, compute capability, memory
print(nvidia.supports_fp8())     # True on H100+
```

### Cross-Backend Model Export

```python
from torchbridge.deployment import export_to_torchscript, export_to_onnx, export_to_safetensors

sample_input = torch.randn(1, 768)

export_to_torchscript(model, output_path="model.pt", sample_input=sample_input)
export_to_onnx(model, output_path="model.onnx", sample_input=sample_input)
export_to_safetensors(model, output_path="model.safetensors")
```

## Project Structure

```
src/torchbridge/
├── backends/          # Vendor-specific backend implementations
│   ├── nvidia/        #   NVIDIA CUDA backend
│   ├── amd/           #   AMD ROCm backend
│   ├── intel/         #   Intel IPEX backend
│   └── tpu/           #   Google TPU/XLA backend
├── hardware/          # Hardware detection and abstraction
├── precision/         # FP8 training and precision management
├── attention/         # Attention mechanisms (unified API)
├── advanced_memory/   # Memory optimization strategies
├── distributed_scale/ # Distributed training
├── deployment/        # Model export and serving
├── monitoring/        # Metrics, logging, health checks
├── optimizations/     # Optimization patterns and strategies
├── core/              # Core config, management, optimized layers
├── cli/               # Command-line tools
├── models/            # Model implementations
├── mixture_of_experts/ # MoE layer support
├── validation/        # Cross-backend validation
└── utils/             # Utilities and profiling
```

## Cloud GPU Validation

All 5 use cases validated on real GPU hardware across AWS and GCP:

| Use Case | AWS A10G | GCP L4 | Description |
|----------|----------|--------|-------------|
| Export Pipeline | PASS | PASS | TorchScript, ONNX, SafeTensors export with validation |
| LLM Optimization | PASS | PASS | GPT-2 optimization with BetterTransformer |
| CI/CD Validation | PASS | PASS | Diagnostics, benchmarks, cross-backend checks |
| Backend Training | PASS | PASS | AMP training with auto backend detection |
| Cross-Backend Validation | PASS | PASS | Model, hardware, config, and output consistency |

**Platforms tested:**
- **AWS g5.xlarge** -- NVIDIA A10G 24GB, PyTorch 2.9.1+cu130
- **GCP g2-standard-4** -- NVIDIA L4 24GB, PyTorch 2.7.1+cu128

See [full validation report](./docs/reference/cloud-validation.md) for detailed benchmarks and results.

## Quality

- **1600+ tests** passing across all modules
- **0 ruff violations** -- clean linting
- **0 mypy errors** -- full type coverage
- **Cloud validated** on NVIDIA A10G (AWS) and L4 (GCP) -- 5/5 use cases pass
- **Cross-platform** tested on macOS, Linux, AWS, GCP

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
ruff check src/ tests/
```

## Use Cases

**Cross-vendor training** -- Train on NVIDIA in the cloud, fine-tune on AMD on-prem, infer on Intel at the edge. Same code throughout.

**Cost optimization** -- Switch between cloud GPU types based on spot pricing without rewriting training scripts.

**Hardware migration** -- Move from one GPU vendor to another without a code rewrite.

**Research portability** -- Share models and training code that colleagues can run on whatever hardware they have.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](./docs/getting-started/installation.md) | Setup and requirements |
| [Quick Start](./docs/getting-started/quickstart.md) | First steps with TorchBridge |
| [Troubleshooting](./docs/getting-started/troubleshooting.md) | Common issues and fixes |
| [Backends Overview](./docs/backends/overview.md) | How the HAL works |
| [Backend Selection](./docs/guides/backend-selection.md) | Choosing the right backend |
| [Hardware Setup](./docs/guides/hardware-setup.md) | Driver and toolkit installation |
| [Distributed Training](./docs/guides/distributed-training.md) | Multi-GPU and multi-node |
| [Deployment](./docs/guides/deployment.md) | Export, serve, containerize |
| [CLI Reference](./docs/guides/cli.md) | Command-line tools |
| [Hardware Matrix](./docs/reference/hardware-matrix.md) | Full hardware support table |
| [Contributing](./CONTRIBUTING.md) | Development and contribution guide |
| [Changelog](./CHANGELOG.md) | Version history |

## License

Open source -- see LICENSE file for details.
