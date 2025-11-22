# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PyTorch GPU Compiler Optimization** - A practical guide for building PyTorch neural network components that achieve maximum GPU performance through compiler optimization. This repository provides production-ready optimized components and practical optimization techniques for real-world ML development.

### Core Concept
The project demonstrates how to write PyTorch code that **compiles efficiently** with `torch.compile` and **maps cleanly** to optimized GPU kernels, providing 2-4x speedups in production ML workloads through better compiler integration.

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
python3 setup.py build_ext --inplace
```

### Testing & Validation
```bash
# Verify installation
python3 -c "import kernel_pytorch; print('✓ Installation successful')"

# Run progressive optimization demo
python3 src/kernel_pytorch/examples/progressive_optimization.py

# Run progressive optimization demo
python3 demo_progressive_optimization.py

# Quick profiling test
python3 src/kernel_pytorch/utils/profiling.py
```

### Benchmarking & Profiling
```bash
# Compare optimization levels
python3 -c "from kernel_pytorch.examples.progressive_optimization import run_progressive_optimization_demo; run_progressive_optimization_demo()"

# Memory usage analysis
python3 -c "from kernel_pytorch.utils.profiling import profile_model_inference; # [use with your model]"
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

### Computational Correctness Preserved Across All Levels
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

## AI-Powered Optimization Assistant

### New Feature: Intelligent GPU Optimization
The project now includes an advanced compiler optimization assistant that can automatically analyze PyTorch models and provide intelligent optimization recommendations.

#### Key Components
- **Optimization Assistant**: `src/kernel_pytorch/utils/compiler_optimization_assistant.py` - AI-powered optimization analysis
- **Validation Framework**: `src/kernel_pytorch/utils/validation_framework.py` - Automated correctness testing
- **GPU Integration**: `src/kernel_pytorch/gpu_integration/` - Advanced GPU optimization tools

#### Usage Examples
```bash
# Run optimization assistant demo
python3 demos/02_compiler_optimizations/demo_compiler_optimization.py

# Interactive optimization analysis
PYTHONPATH=src python3 -c "
from kernel_pytorch.utils.compiler_optimization_assistant import CompilerOptimizationAssistant
import torch.nn as nn
assistant = CompilerOptimizationAssistant()
model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
result = assistant.optimize_model(model, interactive=True)
"
```

#### Optimization Analysis Capabilities
- **Automatic Model Analysis**: Identifies optimization opportunities in PyTorch models
- **Intelligent Recommendations**: Provides prioritized optimization suggestions with implementation guidance
- **Performance Prediction**: Estimates speedup potential for different optimizations
- **Educational Tutorials**: Explains optimization techniques with step-by-step guidance
- **Validation Integration**: Automatically tests optimization correctness

## Project Goals

This repository is designed to teach:
1. How PyTorch operations map to GPU computation graphs
2. The relationship between neural network computation and kernel efficiency
3. Progressive optimization from simple to advanced techniques
4. Memory access patterns and performance impact
5. Designing components that are both computationally correct and efficient
6. **NEW**: How AI can automatically analyze and optimize ML/AI code for maximum performance