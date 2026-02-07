# TorchBridge Vision Models (v0.5.3)

Optimized inference for computer vision models including ResNet, Vision Transformers, and Stable Diffusion.

## Features

### ResNet Optimization
Efficient inference for ResNet models (ResNet-50, ResNet-152):
- **Operator Fusion**: Conv+BN+ReLU fusion for reduced memory and faster inference
- **Memory Layout**: channels_last format for better cache utilization
- **Batch Inference**: Optimized batch processing
- **Precision**: FP16/BF16 support for 2x speedup on modern GPUs

### Vision Transformer Optimization
Efficient inference for ViT models (ViT-Base, ViT-Large):
- **Attention Slicing**: Reduce memory usage for large batch sizes
- **Gradient Checkpointing**: Trade computation for memory
- **Precision**: FP16/BF16 support
- **Batch Inference**: Optimized batch processing

### Stable Diffusion Optimization
Memory-efficient image generation:
- **Attention Slicing**: Reduce memory usage during generation
- **VAE Tiling**: Generate large images (1024x1024+) without OOM
- **xformers Integration**: Memory-efficient attention (if installed)
- **Faster Schedulers**: DPM-Solver++ for faster generation
- **Precision**: FP16 for 2x faster generation

## Quick Start

### ResNet Optimization

```python
from torchbridge.models.vision import (
    create_resnet50_optimized,
    OptimizationLevel,
)

# Create optimized ResNet-50
model, optimizer = create_resnet50_optimized(
    optimization_level=OptimizationLevel.O2,
    batch_size=32,
    device="cuda",
)

# Run inference
import torch
images = torch.randn(32, 3, 224, 224, device="cuda")
outputs = optimizer.optimize_batch_inference(model, images)
```

### Vision Transformer Optimization

```python
from torchbridge.models.vision import (
    create_vit_base_optimized,
    OptimizationLevel,
)

# Create optimized ViT-Base
model, optimizer = create_vit_base_optimized(
    optimization_level=OptimizationLevel.O2,
    batch_size=32,
    device="cuda",
    enable_attention_slicing=True,
)

# Run inference
import torch
images = torch.randn(32, 3, 224, 224, device="cuda")
outputs = optimizer.optimize_batch_inference(model, images)
```

### Stable Diffusion Optimization

```python
from torchbridge.models.vision import (
    create_sd_1_5_optimized,
    OptimizationLevel,
)

# Create optimized Stable Diffusion 1.5
pipeline, optimizer = create_sd_1_5_optimized(
    optimization_level=OptimizationLevel.O2,
    device="cuda",
    enable_attention_slicing=True,
    enable_vae_tiling=True,
)

# Generate image
output = optimizer.generate_optimized(
    prompt="A beautiful sunset over mountains",
    num_inference_steps=50,
    height=512,
    width=512,
)

# Save image
output.images[0].save("sunset.png")
```

## Optimization Levels

| Level | Optimizations | Use Case |
|-------|---------------|----------|
| **O0** | None | Debugging, accuracy verification |
| **O1** | Basic fusion, cuDNN benchmark | Baseline optimization |
| **O2** | + channels_last, FP16 | Production (recommended) |
| **O3** | + torch.compile, attention slicing | Maximum performance |

## Hardware Requirements

### ResNet

| Model | Parameters | Memory (FP32) | Memory (FP16) | Recommended GPU |
|-------|------------|---------------|---------------|-----------------|
| ResNet-50 | 25.6M | ~100MB | ~50MB | Any GPU with 2GB+ |
| ResNet-152 | 60.2M | ~240MB | ~120MB | Any GPU with 4GB+ |

### Vision Transformer

| Model | Parameters | Memory (FP32) | Memory (FP16) | Recommended GPU |
|-------|------------|---------------|---------------|-----------------|
| ViT-Base | 86M | ~350MB | ~175MB | GPU with 4GB+ |
| ViT-Large | 307M | ~1.2GB | ~600MB | GPU with 8GB+ |

### Stable Diffusion

| Model | Parameters | Memory (FP32) | Memory (FP16) | Recommended GPU |
|-------|------------|---------------|---------------|-----------------|
| SD 1.5 | 860M | ~3.5GB | ~2GB | GPU with 8GB+ |
| SD 2.1 | 865M | ~3.5GB | ~2GB | GPU with 8GB+ |
| SDXL | 6.6B | ~26GB | ~13GB | GPU with 24GB+ |

## Performance Benchmarks

Measured on NVIDIA A100 40GB (approximate):

### ResNet-50 (batch_size=32, 224x224)
- **O0**: 850 images/sec
- **O1**: 1,200 images/sec (+41%)
- **O2**: 2,400 images/sec (+182%)
- **O3**: 2,600 images/sec (+206%)

### ViT-Base (batch_size=32, 224x224)
- **O0**: 320 images/sec
- **O1**: 450 images/sec (+41%)
- **O2**: 850 images/sec (+166%)
- **O3**: 920 images/sec (+188%)

### Stable Diffusion 1.5 (512x512, 50 steps)
- **O0**: 1.2 sec/image
- **O1**: 0.9 sec/image (+25%)
- **O2**: 0.5 sec/image (+140%)
- **O3**: 0.45 sec/image (+167%)

## Advanced Features

### Custom Configuration

```python
from torchbridge.models.vision import (
    VisionOptimizationConfig,
    OptimizationLevel,
    VisionModelType,
    ResNetOptimizer,
)

# Create custom configuration
config = VisionOptimizationConfig(
    model_type=VisionModelType.RESNET,
    optimization_level=OptimizationLevel.O2,
    batch_size=64,
    enable_fusion=True,
    enable_cudnn_benchmark=True,
    enable_tf32=True,
    use_fp16=True,
    channels_last=True,
    compile_model=False,
    device="cuda",
)

# Create optimizer with custom config
optimizer = ResNetOptimizer(config)
model = optimizer.optimize(your_model)
```

### Benchmarking

```python
from torchbridge.models.vision import (
    create_resnet50_optimized,
    ResNetBenchmark,
)

# Create optimized model
model, optimizer = create_resnet50_optimized()

# Create benchmark
benchmark = ResNetBenchmark(model, optimizer)

# Run benchmark
results = benchmark.benchmark_inference(
    batch_size=32,
    num_iterations=100,
    warmup_iterations=10,
)

print(f"Throughput: {results['throughput_images_per_second']:.2f} images/sec")
print(f"Time per image: {results['time_per_image_ms']:.2f} ms")
```

### Memory Estimation

```python
from torchbridge.models.vision import estimate_model_memory
import torch.nn as nn

model = nn.Sequential(...)  # Your model

memory = estimate_model_memory(
    model,
    batch_size=32,
    input_size=(3, 224, 224),
    precision="fp16"
)

print(f"Parameter memory: {memory['parameter_memory_mb']:.2f} MB")
print(f"Total inference memory: {memory['total_inference_mb']:.2f} MB")
```

## Examples

See the `examples/models/vision/` directory for complete examples:
- `resnet_optimization.py` - ResNet optimization examples
- `vit_optimization.py` - Vision Transformer examples
- `stable_diffusion_optimization.py` - Stable Diffusion examples

## Testing

Run vision model tests:
```bash
pytest tests/test_vision_model_integration.py -v
```

## Dependencies

### Required
- PyTorch 2.0+
- torchvision (for ResNet models)

### Optional
- `timm` - For additional ViT models
- `diffusers` - For Stable Diffusion models
- `transformers` - For text encoders in Stable Diffusion
- `xformers` - For memory-efficient attention in Stable Diffusion

Install all dependencies:
```bash
pip install torch torchvision timm diffusers transformers xformers
```

## Architecture

```
┌─────────────────────────────────────┐
│   VisionOptimizationConfig          │
│   ├── Model Type (ResNet/ViT/SD)    │
│   ├── Optimization Level (O0-O3)    │
│   └── Memory/Precision Settings     │
└─────────────────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐    ┌──▼─────┐    ┌──────────┐
│ ResNet │    │  ViT   │    │   SD     │
│Optimizer    │Optimizer    │Optimizer │
└───┬────┘    └──┬─────┘    └────┬─────┘
    │            │               │
┌───▼────────────▼───────────────▼──┐
│   Optimization Passes              │
│   ├── Operator Fusion              │
│   ├── Memory Format                │
│   ├── Precision                    │
│   ├── Compilation                  │
│   └── Device Placement             │
└────────────────────────────────────┘
```

## Supported Models

### ResNet
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- ResNeXt variants
- Wide ResNet variants

### Vision Transformer
- ViT-Base/16, ViT-Large/16, ViT-Huge/14
- DeiT variants
- BEiT variants
- Any timm or torchvision ViT model

### Stable Diffusion
- Stable Diffusion 1.4, 1.5
- Stable Diffusion 2.0, 2.1
- Stable Diffusion XL
- Custom fine-tuned models

## Version

Part of TorchBridge v0.5.3 - Vision Model Integration

For more details, see the main documentation and guides.
