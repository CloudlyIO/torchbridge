# 🚀 Quick Start Guide (v0.2.3)

Get up and running with **KernelPyTorch unified architecture** in under 5 minutes.

## 1. Installation

```bash
# Clone and setup KernelPyTorch v0.2.3
git clone <repository-url>
cd shahmod
pip install -r requirements.txt

# Verify unified architecture
PYTHONPATH=src python3 -c "
import kernel_pytorch
print(f'✅ KernelPyTorch v{kernel_pytorch.__version__} ready!')
"
```

## 2. Unified Architecture Quick Start

### Using the Unified Manager

```python
import torch
from kernel_pytorch import KernelPyTorchConfig, UnifiedManager

# Create your model
model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768),
    torch.nn.LayerNorm(768)
)

# Unified optimization approach (v0.2.3)
config = KernelPyTorchConfig.for_development()
manager = UnifiedManager(config)

# Automatically detect hardware and apply optimal optimizations
optimized_model = manager.optimize(model)

# Test performance
x = torch.randn(32, 128, 768)
y = optimized_model(x)

print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {y.shape}")
```

### NVIDIA Hardware Optimization (v0.2.3)

```python
from kernel_pytorch import KernelPyTorchConfig, UnifiedManager

# NVIDIA-optimized configuration with auto-detection
config = KernelPyTorchConfig.for_training()

# The unified system automatically detects:
print(f"Detected architecture: {config.hardware.nvidia.architecture.value}")
print(f"FP8 enabled: {config.hardware.nvidia.fp8_enabled}")
print(f"Tensor Core version: {config.hardware.nvidia.tensor_core_version}")
print(f"FlashAttention version: {config.hardware.nvidia.flash_attention_version}")

# Unified management with NVIDIA optimization
manager = UnifiedManager(config)
optimized_model = manager.optimize(your_model)

# The system automatically:
# - Detects NVIDIA architecture (H100, Blackwell, Ampere, etc.)
# - Enables FP8 training on supported hardware
# - Optimizes memory management and kernel fusion
# - Validates correctness and performance
```

### Production Configuration

```python
from kernel_pytorch import KernelPyTorchConfig, UnifiedManager

# Production-optimized configuration
config = KernelPyTorchConfig.for_inference()

# Unified management with automatic optimization
manager = UnifiedManager(config)
optimized_model = manager.optimize(your_model)

# The unified system automatically:
# - Detects hardware (NVIDIA, AMD, Intel)
# - Selects optimal optimization level
# - Applies appropriate precision settings
# - Validates correctness
```

## 3. Validation and Testing

### Unified Validation Framework

```python
from kernel_pytorch.validation import UnifiedValidator

# Comprehensive model validation
validator = UnifiedValidator()

# Validate model correctness and performance
results = validator.validate_model(model, input_shape=(32, 128, 768))
print(f"Validation passed: {results.passed}/{results.total_tests}")

# Validate configuration
config_results = validator.validate_configuration(config)
print(f"Config validation: {config_results.success_rate:.2%}")

# Hardware compatibility check
hw_results = validator.validate_hardware_compatibility(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Hardware compatibility: {hw_results.passed}/{hw_results.total_tests}")
```

### Test Suite Execution

```bash
# Run unified test suite (504 tests)
PYTHONPATH=src python3 -m pytest tests/ --tb=short -q

# Run specific test categories
PYTHONPATH=src python3 -m pytest tests/test_configs.py -v
PYTHONPATH=src python3 -m pytest tests/ -k "config or manager or validator" -v
```

## 4. Advanced Features

### Attention Systems

```python
from kernel_pytorch.attention import BaseAttention

# Advanced attention with unified config
attention_config = config.attention
attention_layer = BaseAttention(
    d_model=768,
    n_heads=12,
    config=attention_config
)

# Unified attention computation
x = torch.randn(32, 128, 768)  # batch, seq_len, embed_dim
output = attention_layer(x)
```

### Demos and Examples

```bash
# Run all demos with unified architecture
PYTHONPATH=src python3 demos/run_all_demos.py --quick

# NVIDIA configuration demo (v0.2.3)
PYTHONPATH=src python3 demos/nvidia_configuration_demo.py --quick

# NVIDIA benchmarks
PYTHONPATH=src python3 benchmarks/nvidia_config_benchmarks.py --quick

# Specific feature demos
PYTHONPATH=src python3 demos/attention/fusion.py --quick
PYTHONPATH=src python3 demos/memory/basic.py --quick
```

## 5. Next Steps

After completing this quick start:

1. **Deep Dive**: Read the [Architecture Guide](../capabilities/architecture.md) to understand the unified system
2. **Testing**: Follow the [Testing Guide](testing_guide.md) for comprehensive validation
3. **Hardware**: Check [Hardware Integration](../capabilities/hardware.md) for GPU optimization
4. **Production**: See [Unified Roadmap](../unified_roadmap.md) for NVIDIA/TPU hardware support

## Performance Expectations

With KernelPyTorch v0.2.3 unified architecture:
- **Memory Efficiency**: 30-60% memory reduction through unified management
- **Optimization Overhead**: Minimal overhead with unified optimization pipeline
- **Hardware Detection**: Automatic optimization selection based on detected hardware
- **Validation**: Comprehensive correctness checking with 504 test coverage

## Support

- **Issues**: Report bugs or questions in the project repository
- **Architecture**: See unified design in [Architecture docs](../capabilities/architecture.md)
- **API Reference**: Complete API documentation in root `API.md`
