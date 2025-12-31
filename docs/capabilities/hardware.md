# 🔧 Hardware Abstraction (v0.3.3)

**Multi-vendor GPU and TPU support through unified hardware management.**

## 🎯 Supported Hardware

### GPUs
- **NVIDIA**: RTX 4090, A100, H100, Blackwell (all generations)
- **AMD**: MI200 series, MI300 series, RX 7000 series
- **Intel**: Arc A-series, Xe-HPG, Data Center GPU

### CPUs
- **Intel**: x86-64 processors with optimized fallback support
- **Automatic detection**: Hardware capabilities and optimization selection

### TPUs (NEW in v0.2.4)
- **Google Cloud TPU**: v4, v5e, v5p, v6e, v7 with auto-detection
- **PyTorch/XLA**: Complete XLA compiler integration
- **Memory Optimization**: TPU-specific memory pooling and layout optimization

### Custom Hardware
- **ASICs**: Custom accelerator support framework
- **Neuromorphic**: Specialized computing hardware (research)

## 🏗️ Unified Hardware Management (v0.2.4)

### NVIDIA Hardware Auto-Detection (NEW in v0.2.3)
```python
from kernel_pytorch import KernelPyTorchConfig

# NVIDIA hardware auto-detection and configuration
config = KernelPyTorchConfig()

# Automatic architecture detection
nvidia_config = config.hardware.nvidia
print(f"Detected architecture: {nvidia_config.architecture.value}")
print(f"FP8 training support: {nvidia_config.fp8_enabled}")
print(f"Tensor Core version: {nvidia_config.tensor_core_version}")
print(f"FlashAttention version: {nvidia_config.flash_attention_version}")

# Auto-configured optimizations based on detected hardware:
# H100/Blackwell: FP8 training, Tensor Core 4.x, FlashAttention 3
# A100/Ampere: Mixed precision, Tensor Core 3.x, FlashAttention 2/3
# Pascal/Turing: Basic optimizations with compatibility mode
```

### TPU Hardware Auto-Detection (NEW in v0.2.4)
```python
from kernel_pytorch import KernelPyTorchConfig

# TPU hardware auto-detection and configuration
config = KernelPyTorchConfig()

# Automatic TPU detection and configuration
tpu_config = config.hardware.tpu
print(f"TPU version: {tpu_config.version.value}")
print(f"TPU topology: {tpu_config.topology.value}")
print(f"Compilation mode: {tpu_config.compilation_mode.value}")
print(f"Precision: {tpu_config.precision}")

# Auto-configured optimizations based on detected TPU:
# v6e/v7: bfloat16 precision, XLA compilation, memory-optimized kernels
# v5e/v5p: bfloat16 precision, optimized for large model training
# v4: Compatibility mode with standard PyTorch/XLA patterns
```

### Unified Architecture
```python
from kernel_pytorch import KernelPyTorchConfig, UnifiedManager

# Automatic hardware detection and optimization
config = KernelPyTorchConfig.for_production()
manager = UnifiedManager(config)

# Hardware is automatically detected and optimized
optimized_model = manager.optimize(model)
```

### Hardware Detection in Unified System
```python
# Hardware detection through unified manager
manager = UnifiedManager(config)

# Access hardware manager
hardware_info = manager.hardware_manager.get_hardware_info()
print(f"Detected hardware: {hardware_info['primary_device']}")

# Automatic optimization selection based on detected hardware
optimization_strategy = manager.hardware_manager.get_optimization_strategy()
```

## ⚡ Hardware-Specific Optimizations

### NVIDIA GPUs (Enhanced v0.2.3)
```python
from kernel_pytorch import KernelPyTorchConfig
from kernel_pytorch.core.config import NVIDIAArchitecture

# Automatic detection and optimization for NVIDIA hardware
config = KernelPyTorchConfig()
nvidia_config = config.hardware.nvidia

# Architecture-specific features (auto-detected):
if nvidia_config.architecture == NVIDIAArchitecture.HOPPER:  # H100, H200
    print(f"FP8 training: {nvidia_config.fp8_enabled}")  # True
    print(f"Tensor Core: v{nvidia_config.tensor_core_version}")  # 4
    print(f"Memory fraction: {nvidia_config.memory_fraction}")  # 0.95

elif nvidia_config.architecture == NVIDIAArchitecture.BLACKWELL:  # B100, B200
    print("Next-gen Blackwell optimizations enabled")

elif nvidia_config.architecture == NVIDIAArchitecture.AMPERE:  # A100, RTX 3000
    print(f"Mixed precision: {nvidia_config.mixed_precision_enabled}")  # True
    print(f"FP8 support: {nvidia_config.fp8_enabled}")  # False (A100 limitation)

# Generation-specific features accessible through unified config:
features = {
    'H100': nvidia_config.fp8_enabled and nvidia_config.tensor_core_version == 4,
    'FlashAttention': nvidia_config.flash_attention_enabled,
    'Memory_Pool': nvidia_config.memory_pool_enabled,
    'Kernel_Fusion': nvidia_config.kernel_fusion_enabled
}
```

### AMD GPUs
```python
# ROCm platform optimization
amd_config = {
    'MI300': {
        'rocm_version': '5.7+',
        'mixed_precision_support': True
    },
    'MI200': {
        'matrix_core_optimization': True,
        'infinity_fabric': True
    }
}
```

### Intel GPUs
```python
# Intel XPU optimization
intel_config = {
    'Arc_A770': {
        'xe_hpg_optimization': True,
        'dp4a_int8_acceleration': True
    },
    'Data_Center_GPU': {
        'enterprise_features': True,
        'multi_tile_scaling': True
    }
}
```

## 🔄 Kernel Mapping Strategy

### Optimization Selection
```python
class HardwareOptimizer:
    def select_optimal_kernel(self, operation, device):
        """Select best kernel implementation for hardware."""
        if device.vendor == 'nvidia':
            return self._select_nvidia_kernel(operation, device)
        elif device.vendor == 'amd':
            return self._select_amd_kernel(operation, device)
        elif device.vendor == 'intel':
            return self._select_intel_kernel(operation, device)
        else:
            return self._select_cpu_fallback(operation)
```

### Compiler Integration
```python
# Hardware-aware compilation
def compile_for_hardware(model, target_device):
    """Compile model with hardware-specific optimizations."""

    if target_device.supports_fp8():
        model = convert_to_fp8(model)

    if target_device.has_tensor_cores():
        model = optimize_for_tensor_cores(model)

    if target_device.supports_sparsity():
        model = apply_structured_sparsity(model)

    return torch.compile(model, backend=target_device.compiler_backend)
```

## 📊 Performance Characteristics

### Hardware Comparison
| Hardware | FP8 Support | Tensor Cores | Memory BW | Best Use Case |
|----------|-------------|--------------|-----------|---------------|
| H100 | ✅ Native | 4th Gen | 3.35 TB/s | Training |
| A100 | ⚡ Emulated | 3rd Gen | 1.96 TB/s | Inference |
| RTX 4090 | ❌ No | 3rd Gen | 1.01 TB/s | Development |
| AMD MI300 | ✅ Native | Matrix Cores | 5.3 TB/s | HPC |
| Intel Arc | ⚡ Partial | XMX | 560 GB/s | Edge |

### Optimization Impact
| Optimization | NVIDIA | AMD | Intel | CPU |
|--------------|--------|-----|-------|-----|
| Attention Fusion | 2.1x | 1.8x | 1.6x | 1.2x |
| FP8 Training | 1.9x | 1.7x | N/A | N/A |
| Sparsity | 1.4x | 1.3x | 1.5x | 1.1x |
| Multi-GPU | Linear | Linear | 1.8x | N/A |

## 🛠️ Development Guidelines

### Adding New Hardware
1. **Detect capabilities**: Implement device detection
2. **Create config**: Define hardware-specific parameters
3. **Implement kernels**: Hardware-optimized implementations
4. **Integrate compiler**: Backend-specific compilation
5. **Validate performance**: Comprehensive benchmarking

### Testing Strategy
```python
# Hardware-agnostic testing
@pytest.mark.parametrize("device", get_available_devices())
def test_attention_layer(device):
    layer = AttentionLayer(512, 8).to(device)
    x = torch.randn(1, 100, 512, device=device)
    output = layer(x)
    assert output.shape == x.shape
```

---

**For specific hardware setup instructions, see [Installation Guide](installation.md).**