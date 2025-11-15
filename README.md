# Kernel-Optimized PyTorch

A comprehensive educational project demonstrating how to design PyTorch neural network components that map cleanly to efficient GPU kernel patterns. This repository serves as both a practical implementation and an educational resource for understanding the relationship between machine learning semantics and GPU computation.

## 🎯 Project Overview

This project explores the fundamental question: **How can we design ML model components that align with GPU computation graphs while maintaining semantic clarity?**

We demonstrate that GPU kernels are **pure computation** while **control logic resides on the CPU**, and show how to write PyTorch components that exploit this architecture for maximum efficiency.

## 🏗️ Architecture Levels

The project implements the same ML semantics across **5 progressive optimization levels**:

### Level 1: PyTorch Native Optimizations
- Uses built-in operations that automatically map to optimized kernels (cuDNN, cuBLAS)
- Demonstrates kernel-friendly patterns and memory access optimization
- **Files**: `src/kernel_pytorch/components/basic_optimized.py`

### Level 2: TorchScript JIT Compilation
- Leverages JIT compilation for automatic kernel fusion
- Shows how to write fusion-friendly code patterns
- **Files**: `src/kernel_pytorch/components/jit_optimized.py`

### Level 3: torch.compile (Inductor Backend)
- Uses PyTorch 2.0's compilation for graph-level optimization
- Demonstrates modern optimization techniques
- **Integration**: Available in all components when PyTorch 2.0+ is used

### Level 4: Triton Kernels
- Python-based GPU kernel development for educational clarity
- Shows block-based parallel computation and memory tiling
- **Files**: `src/kernel_pytorch/triton_kernels/fused_ops.py`

### Level 5: Custom CUDA Kernels
- Maximum control with raw CUDA C++ implementation
- Demonstrates warp-level operations and shared memory usage
- **Files**: `src/kernel_pytorch/cuda_kernels/`

## 🧠 Semantic ML Models Included

### 1. Language Model (Autoregressive Generation)
```python
from kernel_pytorch.examples.semantic_ml_models import KernelOptimizedLanguageModel

model = KernelOptimizedLanguageModel(vocab_size=10000, dim=512)
# Demonstrates: Causal attention, autoregressive generation, token prediction
```

### 2. Vision Transformer (Spatial Attention)
```python
from kernel_pytorch.examples.semantic_ml_models import KernelOptimizedVisionTransformer

model = KernelOptimizedVisionTransformer(image_size=224, num_classes=1000)
# Demonstrates: Patch embedding, spatial positional encoding, global attention
```

### 3. Graph Neural Network (Message Passing)
```python
from kernel_pytorch.examples.semantic_ml_models import KernelOptimizedGraphNeuralNetwork

model = KernelOptimizedGraphNeuralNetwork(node_features=64, num_classes=10)
# Demonstrates: Message passing, graph pooling, relational reasoning
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd kernel-optimized-pytorch

# Install dependencies
pip install -r requirements.txt

# For CUDA support (optional but recommended)
pip install -e .
# This builds custom CUDA kernels if CUDA toolkit is available
```

### Basic Usage

```python
import torch
from kernel_pytorch.components.basic_optimized import OptimizedTransformerBlock

# Create an optimized transformer block
block = OptimizedTransformerBlock(dim=512, num_heads=8)

# Input tensor
x = torch.randn(4, 128, 512)  # [batch, sequence, features]

# Forward pass - automatically uses optimized kernels
output = block(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### Progressive Optimization Demo

```python
from kernel_pytorch.examples.progressive_optimization import run_progressive_optimization_demo

# Run comprehensive demonstration of all optimization levels
run_progressive_optimization_demo()
```

## 📊 Performance Benchmarking

### Benchmark Different Optimization Levels

```python
from kernel_pytorch.utils.profiling import KernelProfiler, compare_functions

# Compare implementations
implementations = {
    'pytorch_native': lambda x: torch.layer_norm(x, (512,)),
    'custom_optimized': optimized_layer_norm_function,
}

profiler = KernelProfiler()
results = profiler.compare_implementations(implementations, args=(test_tensor,))
profiler.plot_comparison(results)
```

### Memory Usage Analysis

```python
from kernel_pytorch.utils.profiling import profile_model_inference

# Analyze memory patterns
model = YourOptimizedModel()
input_data = torch.randn(4, 128, 512)

report = profile_model_inference(model, input_data)
print(report)
```

## 🎓 Educational Concepts Demonstrated

### GPU Kernel Alignment Principles

1. **Memory Coalescing**: Sequential access patterns for optimal bandwidth
2. **Kernel Fusion**: Combining operations to reduce memory transfers
3. **Parallel Reduction**: Tree-based algorithms for statistical computations
4. **Shared Memory Usage**: Exploiting GPU memory hierarchy
5. **Warp-Level Operations**: Utilizing GPU hardware primitives

### ML Semantic Preservation

- **Autoregressive Modeling**: Causal attention and sequential generation
- **Spatial Reasoning**: Vision transformers with patch-based attention
- **Relational Learning**: Graph neural networks with message passing
- **Representation Learning**: Embedding spaces and feature hierarchies

### Kernel Optimization Patterns

```python
# Example: Memory-efficient attention computation
def optimized_attention(q, k, v, scale):
    # Single kernel launch for all positions (good)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

# vs. Sequential computation (bad)
def inefficient_attention(q, k, v, scale):
    seq_len = q.size(1)
    outputs = []
    for i in range(seq_len):  # This loop prevents parallelization!
        qi = q[:, i:i+1]
        scores = torch.matmul(qi, k.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        outputs.append(torch.matmul(weights, v))
    return torch.cat(outputs, dim=1)
```

## 🔧 Development Workflow

### Building Custom CUDA Kernels

```bash
# Development build with CUDA extensions
python setup.py build_ext --inplace

# Verify installation
python -c "import kernel_pytorch; print('✓ Installation successful')"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific optimization level tests
python -m pytest tests/test_basic_optimized.py
python -m pytest tests/test_cuda_kernels.py
```

### Profiling Your Code

```python
# Profile a specific operation
from kernel_pytorch.utils.profiling import quick_benchmark

def my_operation(x):
    return my_optimized_function(x)

stats = quick_benchmark(my_operation, test_input)
print(f"Average time: {stats['mean_time']:.6f}s")
```

## 📁 Project Structure

```
src/kernel_pytorch/
├── components/           # Optimized neural network components
│   ├── basic_optimized.py    # Level 1: PyTorch native optimizations
│   └── jit_optimized.py      # Level 2: TorchScript JIT compilation
├── cuda_kernels/        # Level 5: Custom CUDA implementations
│   ├── fused_ops.cu          # CUDA kernel implementations
│   └── cuda_interface.cpp    # Python binding interface
├── triton_kernels/      # Level 4: Triton kernel implementations
│   └── fused_ops.py          # Python-based GPU kernels
├── examples/            # Complete model demonstrations
│   ├── progressive_optimization.py  # Optimization level comparison
│   └── semantic_ml_models.py       # Educational ML model examples
└── utils/               # Profiling and benchmarking tools
    └── profiling.py          # Performance analysis utilities
```

## 🎯 Learning Outcomes

After working with this project, you'll understand:

1. **How PyTorch operations map to GPU computation graphs**
2. **The relationship between ML semantics and kernel efficiency**
3. **Progressive optimization strategies from simple to advanced**
4. **Memory access patterns and their performance impact**
5. **How to design ML components that are both semantically clear and computationally efficient**

## 🔬 Advanced Examples

### Custom Attention Implementation
```python
from kernel_pytorch.cuda_kernels import flash_attention

# Use custom Flash Attention kernel
def efficient_attention(q, k, v):
    # This uses our custom CUDA implementation
    return flash_attention(q, k, v, scale=0.125)
```

### Triton Kernel Development
```python
from kernel_pytorch.triton_kernels.fused_ops import TritonLayerNorm

# Educational Triton implementation
norm = TritonLayerNorm(dim=512)
normalized = norm(input_tensor)
```

## 🤝 Contributing

This is an educational project designed to demonstrate kernel optimization concepts. Feel free to:

- Add new optimization examples
- Implement additional semantic ML models
- Improve profiling and benchmarking tools
- Enhance documentation and tutorials

## 📚 Further Reading

- **Flash Attention Paper**: Understanding memory-efficient attention
- **Triton Documentation**: Python-based GPU kernel development
- **PyTorch Internals**: How PyTorch dispatches to optimized kernels
- **CUDA Programming Guide**: Low-level GPU programming concepts

## ⚠️ Requirements

- **Python 3.8+**
- **PyTorch 2.0+** (for torch.compile support)
- **CUDA Toolkit** (optional, for custom kernels)
- **Triton** (optional, for Triton kernels)
- **GPU with Compute Capability 7.0+** (recommended)

## 🏷️ License

MIT License - Feel free to use this code for educational and research purposes.

---

**🎯 Remember**: The goal is to understand how ML semantics can be preserved while achieving optimal kernel efficiency. Each optimization level maintains the same semantic behavior while progressively improving performance through better alignment with GPU computation patterns.