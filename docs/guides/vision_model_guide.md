# Vision Model Optimization Guide

Complete guide to optimizing computer vision models using KernelPyTorch (v0.4.14).

## Table of Contents

1. [Introduction](#introduction)
2. [ResNet Optimization](#resnet-optimization)
3. [Vision Transformer Optimization](#vision-transformer-optimization)
4. [Stable Diffusion Optimization](#stable-diffusion-optimization)
5. [Performance Tuning](#performance-tuning)
6. [Memory Optimization](#memory-optimization)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

This guide covers optimization strategies for three major types of computer vision models:

### Supported Models

| Category | Models | Parameters | Use Cases |
|----------|--------|------------|-----------|
| **Classification** | ResNet-50/152 | 25-60M | Image classification, feature extraction |
| **Transformers** | ViT-Base/Large | 86-307M | Classification, embeddings |
| **Generation** | Stable Diffusion | 860M-6.6B | Image generation, editing |

### Optimization Goals

- **Latency**: Reduce inference time per image
- **Throughput**: Maximize images processed per second
- **Memory**: Minimize GPU memory usage
- **Quality**: Maintain model accuracy

---

## ResNet Optimization

### Basic Usage

```python
from kernel_pytorch.models.vision import create_resnet50_optimized, OptimizationLevel

# Create optimized ResNet-50
model, optimizer = create_resnet50_optimized(
    optimization_level=OptimizationLevel.O2,
    batch_size=32,
    device="cuda",
)

# Run inference
import torch
images = torch.randn(32, 3, 224, 224, device="cuda")
with torch.no_grad():
    outputs = model(images)
```

### Operator Fusion

ResNet models benefit significantly from operator fusion:

**Conv+BN+ReLU Fusion**: Combines three operations into one
- Reduces memory bandwidth
- Eliminates intermediate allocations
- ~15-20% speedup

```python
from kernel_pytorch.models.vision import VisionOptimizationConfig, ResNetOptimizer

config = VisionOptimizationConfig(
    enable_fusion=True,  # Enable operator fusion
    device="cuda",
)

optimizer = ResNetOptimizer(config)
model = optimizer.optimize(resnet_model)

print(f"Optimizations applied: {optimizer.optimizations_applied}")
# Output: ['cudnn_benchmark', 'tf32', 'operator_fusion', 'channels_last', 'fp16']
```

### Memory Layout Optimization

**channels_last** format improves cache utilization:

```python
# Automatic with O2+
model, optimizer = create_resnet50_optimized(
    optimization_level=OptimizationLevel.O2,  # Enables channels_last
)

# Or manual
config = VisionOptimizationConfig(channels_last=True)
optimizer = ResNetOptimizer(config)
```

Performance impact:
- 10-15% speedup on modern GPUs
- Better for conv-heavy models
- No accuracy loss

### Batch Processing

Optimize for different batch sizes:

```python
from kernel_pytorch.models.vision import ResNetBenchmark

model, optimizer = create_resnet50_optimized(batch_size=32)
benchmark = ResNetBenchmark(model, optimizer)

# Test different batch sizes
for batch_size in [1, 8, 16, 32, 64]:
    results = benchmark.benchmark_inference(
        batch_size=batch_size,
        num_iterations=50,
    )
    print(f"Batch {batch_size}: {results['throughput_images_per_second']:.1f} img/s")
```

Typical results (A100):
- Batch 1: 450 img/s
- Batch 8: 1,800 img/s
- Batch 32: 2,400 img/s
- Batch 64: 2,500 img/s (marginal improvement)

### Precision Optimization

FP16 provides 2x speedup on modern GPUs:

```python
model, optimizer = create_resnet50_optimized(
    optimization_level=OptimizationLevel.O2,  # Enables FP16
    device="cuda",
)

# Check precision
print(next(model.parameters()).dtype)  # torch.float16
```

Precision comparison:
- **FP32**: Full precision, slowest
- **TF32**: Automatic on Ampere+, 20% speedup
- **FP16**: 2x speedup, minimal accuracy loss
- **BF16**: Better numerical stability than FP16

---

## Vision Transformer Optimization

### Basic Usage

```python
from kernel_pytorch.models.vision import create_vit_base_optimized

model, optimizer = create_vit_base_optimized(
    optimization_level=OptimizationLevel.O2,
    batch_size=32,
    device="cuda",
)

# Run inference
images = torch.randn(32, 3, 224, 224, device="cuda")
outputs = optimizer.optimize_batch_inference(model, images)
```

### Attention Slicing

Reduce memory usage for large batch sizes:

```python
model, optimizer = create_vit_base_optimized(
    enable_attention_slicing=True,
    attention_slice_size=8,  # Slice size (auto or int)
    batch_size=64,
)
```

**How it works**: Computes attention in slices instead of all at once
- **Memory**: Reduces peak memory by 30-50%
- **Speed**: 5-10% slower due to additional overhead
- **Use when**: Memory-constrained or large batch sizes

### Gradient Checkpointing

Trade computation for memory (useful for training):

```python
config = VisionOptimizationConfig(
    enable_gradient_checkpointing=True,
    device="cuda",
)

optimizer = ViTOptimizer(config)
model = optimizer.optimize(vit_model)
```

Impact:
- **Memory**: 50-60% reduction
- **Speed**: 20-30% slower (recomputation)
- **Use when**: Training large models with limited memory

### Patch Embedding Optimization

ViT processes images as patches:

```python
# Standard ViT: 224x224 image -> 196 patches (14x14)
# Each patch: 16x16 pixels

# Optimize patch size for your use case:
# - Larger patches: Faster but less detail
# - Smaller patches: More detail but slower
```

Default configurations:
- **ViT-Base/16**: 16x16 patches, 224x224 images
- **ViT-Large/16**: 16x16 patches, 224x224 images
- **ViT-Huge/14**: 14x14 patches, 224x224 images

### Benchmarking

```python
from kernel_pytorch.models.vision import ViTBenchmark

model, optimizer = create_vit_base_optimized()
benchmark = ViTBenchmark(model, optimizer)

results = benchmark.benchmark_inference(
    batch_size=32,
    num_iterations=100,
)

print(f"Throughput: {results['throughput_images_per_second']:.1f} img/s")
print(f"Latency: {results['time_per_image_ms']:.2f} ms")
```

---

## Stable Diffusion Optimization

### Basic Usage

```python
from kernel_pytorch.models.vision import create_sd_1_5_optimized

pipeline, optimizer = create_sd_1_5_optimized(
    optimization_level=OptimizationLevel.O2,
    device="cuda",
)

# Generate image
output = optimizer.generate_optimized(
    prompt="A beautiful sunset over mountains",
    num_inference_steps=50,
    height=512,
    width=512,
)

output.images[0].save("sunset.png")
```

### Memory Optimization

Stable Diffusion requires significant memory. Use these techniques:

#### 1. Attention Slicing

```python
pipeline, optimizer = create_sd_1_5_optimized(
    enable_attention_slicing=True,
    attention_slice_size="auto",  # or specific size like 8
)
```

Memory reduction: ~30-40%

#### 2. VAE Tiling

For generating large images:

```python
pipeline, optimizer = create_sd_1_5_optimized(
    enable_vae_tiling=True,
    vae_tile_size=512,
)

# Generate 1024x1024 image without OOM
output = optimizer.generate_optimized(
    prompt="Detailed landscape",
    height=1024,
    width=1024,
)
```

**How it works**: Decodes VAE latents in tiles
- Enables generation of very large images (2048x2048+)
- Minimal quality impact
- Slight slowdown (~10%)

#### 3. xformers

If installed, automatically uses memory-efficient attention:

```bash
pip install xformers
```

```python
# Automatically enabled if available
pipeline, optimizer = create_sd_1_5_optimized()

# Check if enabled
print(optimizer.optimizations_applied)
# Output includes 'xformers' if available
```

Memory reduction: 40-50%

### Speed Optimization

#### 1. Fewer Inference Steps

```python
# Standard: 50 steps
output = optimizer.generate_optimized(
    prompt="...",
    num_inference_steps=50,  # ~2.5 sec
)

# Faster: 25 steps
output = optimizer.generate_optimized(
    prompt="...",
    num_inference_steps=25,  # ~1.3 sec
)
```

Quality vs Speed:
- 50+ steps: Best quality
- 30-50 steps: Good quality (recommended)
- 20-30 steps: Acceptable quality, faster
- <20 steps: Fast but lower quality

#### 2. Faster Schedulers

DPM-Solver++ is automatically used:

```python
# Automatically uses DPMSolverMultistepScheduler
pipeline, optimizer = create_sd_1_5_optimized()

# 25 steps with DPM-Solver++ ≈ 50 steps with DDIM
```

#### 3. FP16 Precision

```python
pipeline, optimizer = create_sd_1_5_optimized(
    optimization_level=OptimizationLevel.O2,  # Enables FP16
)
```

Speedup: 2x faster generation

### Classifier-Free Guidance

Control how closely the model follows the prompt:

```python
output = optimizer.generate_optimized(
    prompt="A professional photograph of a vintage car",
    negative_prompt="blurry, low quality, distorted",
    guidance_scale=7.5,  # Default: 7.5
)
```

Guidance scale:
- **1.0**: Ignore prompt, random generation
- **3-5**: Loose interpretation
- **7-8**: Balanced (recommended)
- **10-15**: Strict following, may reduce diversity
- **20+**: Very strict, often artifacts

### Batch Generation

Generate multiple images efficiently:

```python
output = optimizer.generate_optimized(
    prompt="A cute robot",
    num_images_per_prompt=4,  # Generate 4 images
)

for i, image in enumerate(output.images):
    image.save(f"robot_{i}.png")
```

### Model Comparison

```python
# SD 1.5: Fast, good quality
pipeline_15, _ = create_sd_1_5_optimized()

# SD 2.1: Better quality, similar speed
pipeline_21, _ = create_sd_2_1_optimized()

# SDXL: Best quality, slower, requires more memory
pipeline_xl, _ = create_sdxl_optimized()
```

| Model | Speed | Quality | Memory | Use Case |
|-------|-------|---------|--------|----------|
| SD 1.5 | Fast | Good | 8GB | General use, fast iteration |
| SD 2.1 | Fast | Better | 8GB | Better text rendering |
| SDXL | Slow | Best | 24GB | High-quality final renders |

---

## Performance Tuning

### Optimization Level Guide

```python
# O0: No optimization (debugging)
model, _ = create_resnet50_optimized(optimization_level=OptimizationLevel.O0)

# O1: Basic optimization (fusion, cuDNN)
model, _ = create_resnet50_optimized(optimization_level=OptimizationLevel.O1)

# O2: Production (O1 + FP16 + channels_last) ← RECOMMENDED
model, _ = create_resnet50_optimized(optimization_level=OptimizationLevel.O2)

# O3: Maximum (O2 + compile + slicing)
model, _ = create_resnet50_optimized(optimization_level=OptimizationLevel.O3)
```

### Custom Configuration

```python
from kernel_pytorch.models.vision import VisionOptimizationConfig

config = VisionOptimizationConfig(
    model_type=VisionModelType.RESNET,
    optimization_level=OptimizationLevel.O2,

    # Batch configuration
    batch_size=32,

    # Operator fusion
    enable_fusion=True,

    # cuDNN settings
    enable_cudnn_benchmark=True,
    enable_tf32=True,

    # Memory layout
    channels_last=True,

    # Precision
    use_fp16=True,
    use_bf16=False,

    # Compilation
    compile_model=False,  # True for O3

    # Device
    device="cuda",
)
```

### Benchmarking Best Practices

```python
from kernel_pytorch.models.vision import ResNetBenchmark

model, optimizer = create_resnet50_optimized()
benchmark = ResNetBenchmark(model, optimizer)

# Use sufficient iterations for stable results
results = benchmark.benchmark_inference(
    batch_size=32,
    num_iterations=100,      # More iterations = more stable
    warmup_iterations=10,    # Warm up GPU
    image_size=224,
)
```

---

## Memory Optimization

### Memory Profiling

```python
from kernel_pytorch.models.vision import estimate_model_memory

memory = estimate_model_memory(
    model,
    batch_size=32,
    input_size=(3, 224, 224),
    precision="fp16"
)

print(f"Parameter memory: {memory['parameter_memory_mb']:.1f} MB")
print(f"Activation memory: {memory['activation_memory_mb']:.1f} MB")
print(f"Total inference: {memory['total_inference_mb']:.1f} MB")
```

### Reducing Memory Usage

1. **Use FP16/BF16**: Halves memory usage
2. **Reduce batch size**: Linear memory reduction
3. **Attention slicing**: 30-50% reduction for transformers
4. **VAE tiling**: Enable large image generation
5. **Gradient checkpointing**: 50-60% reduction (training only)

### Memory vs Speed Trade-offs

| Technique | Memory Saved | Speed Impact | Quality Impact |
|-----------|--------------|--------------|----------------|
| FP16 | 50% | +100% | Minimal |
| Attention slicing | 30-40% | -10% | None |
| VAE tiling | 40-50% | -10% | Minimal |
| Smaller batch | Variable | -20-50% | None |
| Gradient checkpoint | 50-60% | -25% | None |

---

## Production Deployment

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install dependencies
RUN pip install torch torchvision kernel-pytorch

# Copy model code
COPY app.py /app/app.py

# Run inference server
CMD ["python", "/app/app.py"]
```

### FastAPI Server

```python
from fastapi import FastAPI, File, UploadFile
from kernel_pytorch.models.vision import create_resnet50_optimized
import torch
from PIL import Image
import io

app = FastAPI()

# Load model once at startup
model, optimizer = create_resnet50_optimized(
    optimization_level=OptimizationLevel.O2,
    batch_size=1,
    device="cuda",
)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Load image
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess
    # ... (your preprocessing code)

    # Inference
    with torch.no_grad():
        output = model(tensor)

    # Return predictions
    return {"class": output.argmax().item()}
```

### Monitoring

```python
import time
import logging

class InferenceMonitor:
    def __init__(self):
        self.latencies = []

    def measure(self, func):
        start = time.time()
        result = func()
        latency = time.time() - start
        self.latencies.append(latency)
        return result

    def get_stats(self):
        return {
            "p50": np.percentile(self.latencies, 50),
            "p95": np.percentile(self.latencies, 95),
            "p99": np.percentile(self.latencies, 99),
        }
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size
2. Enable attention slicing
3. Enable VAE tiling (for SD)
4. Use FP16 instead of FP32
5. Reduce image resolution

```python
# Before (OOM)
model, optimizer = create_vit_large_optimized(batch_size=64)

# After (works)
model, optimizer = create_vit_large_optimized(
    batch_size=32,
    enable_attention_slicing=True,
    use_fp16=True,
)
```

### Slow Inference

**Symptoms**: Lower than expected throughput

**Solutions**:
1. Use higher optimization level
2. Increase batch size
3. Enable cuDNN benchmark
4. Check GPU utilization

```python
# Enable all optimizations
model, optimizer = create_resnet50_optimized(
    optimization_level=OptimizationLevel.O3,
    batch_size=32,
)
```

### Poor Quality

**Symptoms**: Generated images look wrong

**Solutions**:
1. Use more inference steps
2. Adjust guidance scale
3. Use better prompt
4. Try different seed

```python
# Better quality generation
output = optimizer.generate_optimized(
    prompt="detailed, high quality, professional photo",
    negative_prompt="blurry, low quality, distorted, ugly",
    num_inference_steps=50,  # More steps
    guidance_scale=7.5,
)
```

### Model Not Found

**Symptoms**: Import or load errors

**Solutions**:
1. Install required dependencies
2. Check model availability

```bash
# For ResNet
pip install torchvision

# For ViT
pip install timm

# For Stable Diffusion
pip install diffusers transformers

# For best performance
pip install xformers
```

---

## Additional Resources

- [Vision Module README](../../src/kernel_pytorch/models/vision/README.md)
- [ResNet Example](../../examples/models/vision/resnet_optimization.py)
- [ViT Example](../../examples/models/vision/vit_optimization.py)
- [Stable Diffusion Example](../../examples/models/vision/stable_diffusion_optimization.py)
- [Integration Tests](../../tests/test_vision_model_integration.py)

---

**Version**: KernelPyTorch v0.4.14
**Last Updated**: January 22, 2026
