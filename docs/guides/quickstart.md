# 🚀 Quick Start Guide

Get up and running with **KernelPyTorch** in under 5 minutes.

## 1. Installation

```bash
# Clone and setup KernelPyTorch
git clone <repository-url>
cd shahmod
pip install -r requirements.txt

# Verify installation
python3 -c "import kernel_pytorch; print('✅ KernelPyTorch ready!')"
```

## 2. Your First Optimization

### Optimize an Existing Model

```python
import torch
import kernel_pytorch as kpt

# Create or load your model
model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768),
    torch.nn.LayerNorm(768)
)

# Quick optimization with torch.compile
optimized_model = torch.compile(model, mode='max-autotune')

# Test performance
x = torch.randn(32, 128, 768)  # batch_size=32, seq_len=128
y = optimized_model(x)

print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {y.shape}")
```

### Use KernelPyTorch Components

```python
import kernel_pytorch as kpt

# Replace standard layers with optimized versions
optimized_model = torch.nn.Sequential(
    kpt.OptimizedLinear(768, 3072),           # 2-3x faster
    kpt.OptimizedGELU(),                      # Fused activation
    kpt.OptimizedLinear(3072, 768),
    kpt.OptimizedLayerNorm(768)               # Optimized normalization
)

# Forward pass
x = torch.randn(32, 128, 768)
y = optimized_model(x)
```

## 3. CLI Tools

### System Check

```bash
# Quick system diagnostics
kernelpytorch doctor

# Detailed hardware analysis
kernelpytorch doctor --category hardware --verbose
```

### Model Optimization via CLI

```bash
# Save a test model
python -c "
import torch
model = torch.nn.Linear(512, 256)
torch.save(model, 'test_model.pt')
"

# Optimize it
kernelpytorch optimize \
    --model test_model.pt \
    --level production \
    --validate \
    --benchmark
```

### Performance Benchmarking

```bash
# Quick performance benchmark
kernelpytorch benchmark --predefined optimization --quick

# Compare optimization levels
kernelpytorch benchmark \
    --model test_model.pt \
    --type compare \
    --levels basic,compile,triton
```

## 4. Advanced Features

### Attention Optimization

```python
import kernel_pytorch as kpt

# Optimized multi-head attention
attention = kpt.OptimizedMultiHeadAttention(
    embed_dim=768,
    num_heads=12,
    optimization_level='flash'  # Flash Attention 2
)

# Forward pass
query = torch.randn(32, 128, 768)  # batch, seq_len, embed_dim
key = value = query

output, attn_weights = attention(query, key, value)
print(f"✓ Attention output: {output.shape}")
```

### Optimization Assistant

```python
from kernel_pytorch.utils.compiler_optimization_assistant import CompilerOptimizationAssistant

# Analyze and optimize your model
assistant = CompilerOptimizationAssistant()
model = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512)
)

# Get optimization recommendations
result = assistant.optimize_model(model, interactive=False)

print(f"Found {len(result.optimization_opportunities)} optimization opportunities:")
for opp in result.optimization_opportunities[:3]:  # Show first 3
    print(f"  • {opp.title}: {opp.description}")
```

### Hardware-Aware Optimization

```python
import kernel_pytorch as kpt

# Automatic hardware detection and optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create hardware-optimized model
if torch.cuda.is_available():
    # GPU optimization
    model = kpt.create_optimized_model(
        layers=[768, 3072, 768],
        activation='gelu',
        hardware='cuda',
        optimization_level='high'
    )
else:
    # CPU optimization
    model = kpt.create_optimized_model(
        layers=[768, 3072, 768],
        activation='gelu',
        hardware='cpu',
        optimization_level='medium'
    )

model = model.to(device)
```

## 5. Real-World Example

### Transformer Block Optimization

```python
import torch
import kernel_pytorch as kpt

class OptimizedTransformerBlock(torch.nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072):
        super().__init__()
        # Use KernelPyTorch optimized components
        self.attention = kpt.OptimizedMultiHeadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            optimization_level='flash'
        )
        self.norm1 = kpt.OptimizedLayerNorm(d_model)
        self.norm2 = kpt.OptimizedLayerNorm(d_model)

        # Optimized feed-forward network
        self.ffn = torch.nn.Sequential(
            kpt.OptimizedLinear(d_model, d_ff),
            kpt.OptimizedGELU(),
            kpt.OptimizedLinear(d_ff, d_model)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

# Create and test the optimized block
block = OptimizedTransformerBlock()
x = torch.randn(2, 128, 768)  # batch_size=2, seq_len=128, d_model=768

# Apply torch.compile for additional optimization
optimized_block = torch.compile(block, mode='max-autotune')

# Forward pass
output = optimized_block(x)
print(f"✓ Transformer block optimized: {x.shape} -> {output.shape}")
```

### Benchmarking Your Model

```python
import time
import torch
import kernel_pytorch as kpt

def benchmark_model(model, input_data, warmup=10, runs=100):
    """Simple benchmarking function."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)

    # Synchronize for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_data)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_time_ms = (total_time / runs) * 1000
    throughput = runs / total_time

    return avg_time_ms, throughput

# Compare standard vs optimized
standard_block = torch.nn.TransformerEncoderLayer(d_model=768, nhead=12)
optimized_block = OptimizedTransformerBlock()

# Compile both for fair comparison
standard_block = torch.compile(standard_block)
optimized_block = torch.compile(optimized_block)

x = torch.randn(8, 128, 768)

# Benchmark both
std_time, std_throughput = benchmark_model(standard_block, x)
opt_time, opt_throughput = benchmark_model(optimized_block, x)

print(f"Standard TransformerBlock: {std_time:.2f} ms/inference")
print(f"Optimized TransformerBlock: {opt_time:.2f} ms/inference")
print(f"Speedup: {std_time/opt_time:.2f}x faster")
```

## 6. Production Tips

### Model Serving

```python
# Save optimized model for production
torch.save(optimized_model, 'production_model.pt')

# Load in production
model = torch.load('production_model.pt')
model.eval()

# Production inference
with torch.no_grad():
    output = model(input_batch)
```

### Memory Optimization

```python
# Enable memory-efficient attention for large sequences
attention = kpt.OptimizedMultiHeadAttention(
    embed_dim=768,
    num_heads=12,
    memory_efficient=True,  # Reduces memory usage
    max_sequence_length=4096
)
```

### Error Handling

```python
try:
    # Try GPU optimization first
    model = kpt.OptimizedLinear(512, 256).cuda()
    optimization_level = 'gpu'
except RuntimeError:
    # Fallback to CPU optimization
    model = kpt.OptimizedLinear(512, 256)
    optimization_level = 'cpu'

print(f"Using {optimization_level} optimization")
```

## 7. Next Steps

### Learn More
- 📖 **[CLI Reference](cli_reference.md)** - Complete command-line interface guide
- 🏗️ **[Architecture Guide](architecture.md)** - Deep dive into KernelPyTorch components
- 📊 **[Benchmarks](benchmarks.md)** - Performance comparisons and analysis
- 🐳 **[Docker Guide](docker.md)** - Containerized development and deployment

### Advanced Topics
- **[Custom Optimizations](tutorials/custom_optimizations.md)** - Build your own optimized layers
- **[Distributed Training](tutorials/distributed.md)** - Scale across multiple GPUs
- **[Production Deployment](tutorials/production.md)** - Best practices for serving models

### Community
- 🐛 **[Report Issues](https://github.com/KernelPyTorch/kernel-pytorch/issues)**
- 💬 **[Discussions](https://github.com/KernelPyTorch/kernel-pytorch/discussions)**
- 📚 **[Examples Repository](https://github.com/KernelPyTorch/examples)**

## Common Patterns

### Pattern 1: Drop-in Replacement
```python
# Replace this:
model = torch.nn.Linear(768, 768)

# With this:
model = kpt.OptimizedLinear(768, 768)
```

### Pattern 2: Model-wide Optimization
```python
# Apply to entire model
optimized_model = torch.compile(model, mode='max-autotune')
```

### Pattern 3: Hardware-aware Development
```python
# Check capabilities first
if torch.cuda.is_available() and kpt.cuda_capabilities.has_tensor_cores():
    optimization_level = 'maximum'
else:
    optimization_level = 'balanced'
```

---

**Ready to optimize?** Try `kernelpytorch optimize --model your_model.pt --level production` 🚀