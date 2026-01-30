# Small Model Optimization Guide

**TorchBridge v0.4.18 - Text Model Integration**

This guide covers optimization of small text models (50-150M parameters) including BERT, GPT-2, and DistilBERT. These models are ideal for single-GPU deployment with <8GB VRAM requirements.

## Supported Models

| Model | Parameters | VRAM (FP32) | VRAM (FP16) | Use Cases |
|-------|------------|-------------|-------------|-----------|
| **bert-base-uncased** | 110M | ~440MB | ~220MB | Classification, NER, Q&A |
| **bert-large-uncased** | 340M | ~1.3GB | ~650MB | Higher accuracy tasks |
| **distilbert-base-uncased** | 66M | ~265MB | ~130MB | Lightweight inference |
| **gpt2** | 124M | ~500MB | ~250MB | Text generation |
| **gpt2-medium** | 355M | ~1.4GB | ~700MB | Better generation quality |

## Quick Start

### 1. Install Dependencies

```bash
pip install torchbridge transformers
```

### 2. Optimize BERT for Classification

```python
from torchbridge.models.text import OptimizedBERT

# Create optimized BERT model
model = OptimizedBERT(
    model_name="bert-base-uncased",
    task="sequence-classification",
    num_labels=2
)

# Run inference
outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits
```

### 3. Optimize GPT-2 for Generation

```python
from torchbridge.models.text import OptimizedGPT2

# Create optimized GPT-2 model
model = OptimizedGPT2(model_name="gpt2", task="causal-lm")

# Generate text
outputs = model.generate(input_ids, max_length=100)
```

### 4. Using the Factory Function

```python
from torchbridge.models.text import create_optimized_text_model

# Create any text model with automatic optimization
model = create_optimized_text_model(
    "bert-base-uncased",
    task="text-classification",
    optimization_mode="inference",  # or "throughput", "memory", "balanced"
    num_labels=3
)
```

## Optimization Modes

### Inference Mode (Default)
Best for low-latency single-request inference.

```python
from torchbridge.models.text import TextModelOptimizer, TextModelConfig, OptimizationMode

config = TextModelConfig(
    optimization_mode=OptimizationMode.INFERENCE,
    use_torch_compile=True,
    compile_mode="reduce-overhead"  # Best for inference
)

optimizer = TextModelOptimizer(config)
model = optimizer.optimize("bert-base-uncased", task="text-classification")
```

### Throughput Mode
Best for batch processing with high throughput.

```python
config = TextModelConfig(
    optimization_mode=OptimizationMode.THROUGHPUT,
    compile_mode="max-autotune"  # Aggressive optimization
)
```

### Memory Mode
Best for memory-constrained environments.

```python
config = TextModelConfig(
    optimization_mode=OptimizationMode.MEMORY,
    gradient_checkpointing=True,
    dtype=torch.float16  # Use FP16 for memory savings
)
```

## Precision Options

### FP16 (Half Precision)
50% memory reduction, works on all modern GPUs.

```python
config = TextModelConfig(
    dtype=torch.float16,
    use_amp=True
)
```

### BF16 (Brain Float 16)
Better numerical stability, recommended for Ampere+ GPUs.

```python
config = TextModelConfig(
    dtype=torch.bfloat16
)
```

### FP8 (H100/Blackwell Only)
Maximum memory savings and speed on supported hardware.

```python
# FP8 is automatically enabled on H100+ GPUs
# Requires compute capability 9.0+
```

## Backend-Specific Optimization

### NVIDIA GPUs

```python
# Automatic optimization for NVIDIA
model = create_optimized_text_model(
    "bert-base-uncased",
    task="text-classification"
)
# Uses: torch.compile, SDPA, FP16/BF16, tensor cores
```

### AMD GPUs (ROCm)

```python
from torchbridge.models.text import TextModelConfig

config = TextModelConfig(
    device="hip",  # Or "auto" for detection
    compile_mode="reduce-overhead"
)
```

### Intel XPU

```python
config = TextModelConfig(
    device="xpu",
    dtype=torch.bfloat16  # Optimal for Intel
)
```

### CPU

```python
config = TextModelConfig(
    device="cpu",
    dtype=torch.bfloat16  # If AVX-512 supported
)
```

## Complete Examples

### Text Classification Pipeline

```python
import torch
from transformers import AutoTokenizer
from torchbridge.models.text import OptimizedBERT

# Initialize
model = OptimizedBERT("bert-base-uncased", task="sequence-classification", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare input
texts = ["This movie is great!", "I didn't like this at all."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

labels = ["Negative", "Positive"]
for text, pred in zip(texts, predictions):
    print(f"{text} -> {labels[pred.item()]}")
```

### Named Entity Recognition

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torchbridge.models.text import TextModelOptimizer

# Load NER model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

optimizer = TextModelOptimizer()
model = optimizer.optimize(
    "dslim/bert-base-NER",
    task="token-classification"
)

# Run NER
text = "John works at Google in New York."
inputs = tokenizer(text, return_tensors="pt").to(optimizer.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

### Text Generation with GPT-2

```python
from transformers import AutoTokenizer
from torchbridge.models.text import OptimizedGPT2

# Initialize
model = OptimizedGPT2("gpt2", task="causal-lm")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
```

### Batch Processing

```python
from torchbridge.models.text import TextModelOptimizer, TextModelConfig, OptimizationMode

# Configure for throughput
config = TextModelConfig(
    optimization_mode=OptimizationMode.THROUGHPUT,
    compile_mode="max-autotune"
)

optimizer = TextModelOptimizer(config)
model = optimizer.optimize("bert-base-uncased", task="text-classification", num_labels=2)

# Process large batch
batch_size = 32
texts = ["Sample text"] * batch_size

inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)
inputs = {k: v.to(optimizer.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
```

## Performance Benchmarks

### BERT-base Classification (batch=8, seq=128)

| Hardware | Baseline (ms) | Optimized (ms) | Speedup |
|----------|---------------|----------------|---------|
| NVIDIA A100 | 12.5 | 4.2 | 2.97x |
| NVIDIA V100 | 18.3 | 7.8 | 2.35x |
| AMD MI250 | 15.6 | 6.5 | 2.40x |
| Intel Arc A770 | 22.1 | 10.4 | 2.13x |
| CPU (AVX-512) | 145.2 | 89.3 | 1.63x |

### GPT-2 Generation (50 tokens)

| Hardware | Baseline (ms) | Optimized (ms) | Speedup |
|----------|---------------|----------------|---------|
| NVIDIA A100 | 85.3 | 38.2 | 2.23x |
| NVIDIA V100 | 112.5 | 56.8 | 1.98x |
| AMD MI250 | 98.7 | 52.1 | 1.89x |

## Troubleshooting

### Out of Memory (OOM)

```python
# Use memory-optimized configuration
config = TextModelConfig(
    optimization_mode=OptimizationMode.MEMORY,
    dtype=torch.float16,
    gradient_checkpointing=True,
    enable_memory_efficient_attention=True
)
```

### torch.compile Errors

```python
# Disable torch.compile if issues occur
config = TextModelConfig(
    use_torch_compile=False
)
```

### Slow First Inference

```python
# Increase warmup steps
config = TextModelConfig(
    warmup_steps=5  # Default is 3
)
```

### Backend Not Detected

```python
# Explicitly set device
config = TextModelConfig(
    device="cuda"  # or "cpu", "xpu", "hip"
)
```

## Best Practices

1. **Use BF16 on Ampere+ GPUs** - Better stability than FP16
2. **Enable torch.compile** - 1.5-3x speedup after warmup
3. **Use SDPA** - Memory-efficient attention enabled by default
4. **Warmup the model** - First few inferences are slower due to compilation
5. **Batch when possible** - Higher throughput with larger batches
6. **Profile first** - Use `enable_profiling=True` to identify bottlenecks

## Next Steps

- **v0.4.18**: Medium model optimization (Llama-7B, Mistral-7B)
- **v0.4.18**: Large model distributed training (Llama-70B)
- **v0.4.18**: Vision models (ViT, Stable Diffusion)
- **v0.4.15**: Multi-modal models (CLIP, LLaVA, Whisper)

## API Reference

See the full API documentation:
- `TextModelOptimizer` - Main optimizer class
- `TextModelConfig` - Configuration dataclass
- `OptimizedBERT` - BERT wrapper
- `OptimizedGPT2` - GPT-2 wrapper
- `OptimizedDistilBERT` - DistilBERT wrapper
- `create_optimized_text_model()` - Factory function
