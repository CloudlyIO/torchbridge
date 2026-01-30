# 🚀 Quick Start Guide (v0.4.18)

Get up and running with **TorchBridge** in under 5 minutes.

> For detailed installation options, see [Installation Guide](installation.md).

## 1. Setup

```bash
# Clone and install
git clone https://github.com/your-org/torchbridge.git
cd torchbridge
pip install -r requirements.txt

# Verify installation
PYTHONPATH=src python3 -c "import torchbridge; print(f'✅ v{torchbridge.__version__} ready!')"
```

## 2. Basic Usage

```python
import torch
from torchbridge import TorchBridgeConfig, UnifiedManager

# Your PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768)
)

# Optimize with unified manager
config = TorchBridgeConfig.for_development()
manager = UnifiedManager(config)
optimized_model = manager.optimize(model)

# Run inference
x = torch.randn(32, 128, 768)
y = optimized_model(x)
print(f"✓ Output shape: {y.shape}")
```

## 3. Configuration Presets

```python
from torchbridge import TorchBridgeConfig

# Development (debugging, verbose)
config = TorchBridgeConfig.for_development()

# Training (optimized for training loops)
config = TorchBridgeConfig.for_training()

# Inference (maximum performance)
config = TorchBridgeConfig.for_inference()

# Production (balanced, production-ready)
config = TorchBridgeConfig.for_production()
```

## 4. NVIDIA GPU Optimization

```python
from torchbridge import TorchBridgeConfig, UnifiedManager

# Auto-detects NVIDIA hardware
config = TorchBridgeConfig.for_training()

# Check detected configuration
print(f"Architecture: {config.hardware.nvidia.architecture.value}")
print(f"FP8 enabled: {config.hardware.nvidia.fp8_enabled}")

# Optimize model
manager = UnifiedManager(config)
optimized_model = manager.optimize(your_model)
```

> For detailed NVIDIA backend documentation, see [NVIDIA Backend](../backends/nvidia.md).

## 5. Validation

```python
from torchbridge.validation import UnifiedValidator

validator = UnifiedValidator()
results = validator.validate_model(model, input_shape=(32, 128, 768))
print(f"Validation: {results.passed}/{results.total_tests} passed")
```

## 6. Run Tests

```bash
# Quick validation
PYTHONPATH=src python3 -m pytest tests/test_configs.py -v

# Full test suite
PYTHONPATH=src python3 -m pytest tests/ --tb=short -q
```

## 7. Run Demos

```bash
# All demos
PYTHONPATH=src python3 demos/run_all_demos.py --quick

# NVIDIA demo
PYTHONPATH=src python3 demos/nvidia_configuration_demo.py --quick
```

## Next Steps

| Topic | Link |
|-------|------|
| Detailed installation | [Installation Guide](installation.md) |
| NVIDIA backend | [NVIDIA Backend](../backends/nvidia.md) |
| TPU backend | [TPU Backend](../backends/tpu.md) |
| Architecture overview | [Architecture](../capabilities/architecture.md) |
| Testing | [Testing Guide](../guides/testing_guide.md) |
| API reference | [API.md](../../API.md) |

---

**Need help?** See [Troubleshooting](troubleshooting.md) or check the [Testing Guide](../guides/testing_guide.md).
