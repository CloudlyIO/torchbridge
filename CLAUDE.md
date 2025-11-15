# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kernel-Optimized PyTorch** - An educational project demonstrating how to design PyTorch neural network components that map cleanly to efficient GPU kernel patterns. This repository serves as both a practical implementation and educational resource for understanding the relationship between ML semantics and GPU computation.

### Core Concept
The project explores how GPU kernels are **pure computation** while **control logic resides on the CPU**, showing how to write PyTorch components that exploit this architecture for maximum efficiency.

## Architecture & Code Organization

### Progressive Optimization Levels (5 levels)
1. **Level 1**: `src/kernel_pytorch/components/basic_optimized.py` - PyTorch native optimizations
2. **Level 2**: `src/kernel_pytorch/components/jit_optimized.py` - TorchScript JIT compilation
3. **Level 3**: torch.compile integration (PyTorch 2.0+)
4. **Level 4**: `src/kernel_pytorch/triton_kernels/` - Triton kernels (Python-based GPU programming)
5. **Level 5**: `src/kernel_pytorch/cuda_kernels/` - Custom CUDA kernels (C++/CUDA)

### Key Components
- **Transformer Models**: Language models with autoregressive generation
- **Vision Transformers**: Spatial attention and patch processing
- **Graph Neural Networks**: Message passing and aggregation
- **Profiling Tools**: `src/kernel_pytorch/utils/profiling.py` - Comprehensive benchmarking

## Development Commands

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Development install with CUDA extensions (if CUDA available)
pip install -e .

# Build CUDA extensions manually
python setup.py build_ext --inplace
```

### Testing & Validation
```bash
# Verify installation
python -c "import kernel_pytorch; print('✓ Installation successful')"

# Run progressive optimization demo
python src/kernel_pytorch/examples/progressive_optimization.py

# Run semantic ML models demo
python src/kernel_pytorch/examples/semantic_ml_models.py

# Quick profiling test
python src/kernel_pytorch/utils/profiling.py
```

### Benchmarking & Profiling
```bash
# Compare optimization levels
python -c "from kernel_pytorch.examples.progressive_optimization import run_progressive_optimization_demo; run_progressive_optimization_demo()"

# Memory usage analysis
python -c "from kernel_pytorch.utils.profiling import profile_model_inference; # [use with your model]"
```

## Code Patterns & Best Practices

### Kernel-Aligned Design Principles
1. **Memory Coalescing**: Use sequential access patterns, avoid strided access
2. **Kernel Fusion**: Design operations that can be combined into single kernels
3. **Batch Operations**: Group similar computations to launch fewer, larger kernels
4. **Minimize CPU-GPU Sync**: Avoid operations requiring CPU intervention

### Example Patterns
```python
# GOOD: Tensor-native operations that map to efficient kernels
def efficient_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

# BAD: Loops that prevent parallelization
def inefficient_attention(q, k, v):
    outputs = []
    for i in range(q.size(1)):  # This loop breaks GPU parallelism!
        # ... sequential processing
    return torch.cat(outputs, dim=1)
```

## Dependencies & Requirements

### Core Requirements
- Python 3.8+
- PyTorch 2.0+ (for torch.compile)
- NumPy 1.21+

### Optional (for full optimization levels)
- CUDA Toolkit (for custom CUDA kernels)
- Triton 2.0+ (for Triton kernels)
- GPU with Compute Capability 7.0+

### Development Tools
- pytest (testing)
- matplotlib (visualization)
- psutil (system profiling)

## Key Educational Concepts

### ML Semantics Preserved Across All Levels
- **Autoregressive Generation**: Causal attention patterns
- **Spatial Reasoning**: Vision transformer patch attention
- **Message Passing**: Graph neural network aggregation
- **Embedding Interactions**: Representation learning

### Optimization Strategies Demonstrated
- Progressive kernel fusion (Level 1→5)
- Memory hierarchy utilization
- Parallel reduction algorithms
- Shared memory usage patterns

## Important Notes for Development

### When Working with CUDA Kernels
- Always test CPU fallbacks exist
- Verify memory allocation patterns
- Use proper error checking (`CUDA_CHECK` macro)
- Test across different GPU architectures

### Performance Considerations
- Start with Level 1 (basic) optimizations
- Profile before moving to custom kernels
- Memory bandwidth often more important than compute
- Batch size significantly affects kernel efficiency

### Debugging Tips
- Use `torch.cuda.synchronize()` for accurate timing
- Enable CUDA error checking during development
- Profile memory usage with `torch.cuda.memory_allocated()`
- Compare outputs across optimization levels for correctness

## Project Goals

This repository is designed to teach:
1. How PyTorch operations map to GPU computation graphs
2. The relationship between ML semantics and kernel efficiency
3. Progressive optimization from simple to advanced techniques
4. Memory access patterns and performance impact
5. Designing components that are both semantically clear and computationally efficient