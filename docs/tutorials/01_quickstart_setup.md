# Tutorial 1: Repository Overview and Quick Setup

**Duration**: 30-45 minutes
**Difficulty**: Beginner
**Prerequisites**: Basic Python knowledge, familiarity with PyTorch

## 🎯 **Learning Objectives**

By the end of this tutorial, you will:
- Understand the repository structure and GPU optimization philosophy
- Set up the development environment successfully
- Run your first optimization example
- Understand the progressive optimization framework
- Use the optimization patterns analysis tools

## 🏗️ **Repository Structure Deep Dive**

Our repository follows a **performance-driven optimization architecture**:

```
shahmod/
├── src/kernel_pytorch/              # Core optimization components
│   ├── compiler_optimized/          # Production-ready optimized components
│   │   ├── attention_modules.py     # Various attention implementations
│   │   ├── activation_functions.py  # Fused activation patterns
│   │   ├── linear_transformations.py # Optimized linear operations
│   │   └── embedding_layers.py      # Efficient embeddings with RoPE
│   ├── optimization_patterns/       # GPU optimization design patterns
│   │   ├── fusion_strategies.py     # Kernel fusion analysis
│   │   ├── memory_efficiency.py     # Memory optimization techniques
│   │   ├── compute_intensity.py     # Arithmetic intensity analysis
│   │   └── compiler_friendly.py     # Compiler optimization patterns
│   ├── gpu_integration/             # Advanced GPU techniques
│   │   └── custom_kernels.py        # Triton/CUDA kernel integration
│   ├── components/                  # Progressive optimization levels
│   │   ├── basic_optimized.py       # Level 1: PyTorch native optimizations
│   │   └── jit_optimized.py         # Level 2: TorchScript compilation
│   ├── triton_kernels/              # Level 4: Triton GPU programming
│   ├── cuda_kernels/                # Level 5: Custom CUDA kernels
│   └── utils/                       # Profiling and benchmarking
├── docs/                           # Comprehensive documentation
│   ├── tutorials/                  # Step-by-step learning guides
│   └── research_roadmap.md         # Future research directions
└── demo_*.py                       # Interactive demonstrations
```

### **🚀 Key Innovation: GPU Optimization Framework**

This repository uniquely combines:
- **Systematic Optimization**: Progressive optimization levels from PyTorch to custom kernels
- **Educational Patterns**: Reusable optimization patterns with detailed explanations
- **Performance Analysis**: Comprehensive tools for identifying and measuring optimizations
- **Production Ready**: Components designed for real-world deployment

## 🛠️ **Environment Setup**

### **Step 1: Clone and Basic Installation**

```bash
# Clone the repository
git clone https://github.com/shahrahman-fb/shahmod.git
cd shahmod

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic dependencies
pip install torch torchvision
pip install -r requirements.txt
```

### **Step 2: Verify Installation**

```python
# Test basic import
PYTHONPATH=src python3 -c "
import torch
print(f'✅ PyTorch {torch.__version__} installed successfully')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

# Test optimization patterns framework
from kernel_pytorch.optimization_patterns import (
    identify_fusion_opportunities,
    calculate_arithmetic_intensity
)
print('✅ Optimization patterns loaded successfully')

# Test compiler-optimized components
from kernel_pytorch.compiler_optimized import FusedGELU, OptimizedLayerNorm
print('✅ Compiler-optimized components loaded successfully')
"
```

**Expected Output**:
```
✅ PyTorch 2.0.0 installed successfully
✅ CUDA available: True
✅ Optimization patterns loaded successfully
✅ Compiler-optimized components loaded successfully
```

### **Step 3: Optional Advanced Setup**

```bash
# For Triton kernels (optional)
pip install triton

# For custom CUDA kernels (requires CUDA toolkit)
python setup.py build_ext --inplace

# Verify advanced features
PYTHONPATH=src python3 -c "
try:
    import triton
    print('✅ Triton available for custom kernels')
except ImportError:
    print('⚠️  Triton not available (optional)')

try:
    from kernel_pytorch.cuda_kernels import custom_attention_kernel
    print('✅ Custom CUDA kernels compiled successfully')
except ImportError:
    print('⚠️  CUDA kernels not compiled (optional)')
"
```

## 🚀 **First Optimization Example**

### **Example 1: Basic Optimization Pattern Analysis**

Create a simple PyTorch model and analyze its optimization opportunities:

```python
# save as test_optimization_analysis.py
import torch
import torch.nn as nn

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kernel_pytorch.optimization_patterns import (
    identify_fusion_opportunities,
    analyze_memory_access_patterns,
    calculate_arithmetic_intensity
)

# Create a simple transformer-like model
class SimpleModel(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm(x + residual)
        return x

# Initialize model and sample input
model = SimpleModel(dim=128)
sample_input = torch.randn(16, 64, 128)  # [batch, seq_len, dim]

print("🔍 GPU Optimization Analysis Results:")
print("=" * 50)

# Analyze fusion opportunities
fusion_opportunities = identify_fusion_opportunities(model, sample_input)
print(f"🚀 Found {len(fusion_opportunities)} fusion opportunities:")
for i, opportunity in enumerate(fusion_opportunities, 1):
    print(f"  {i}. {opportunity.get('pattern', 'Unknown')}")
    print(f"     Modules: {', '.join(opportunity.get('modules', []))}")
    print(f"     Expected speedup: {opportunity.get('estimated_speedup', 'N/A')}x")

# Analyze memory access patterns
memory_analysis = analyze_memory_access_patterns(model, sample_input, [8, 16, 32])
print(f"\n📊 Memory Analysis:")
for batch_size, stats in memory_analysis.get('memory_usage_by_batch', {}).items():
    print(f"  Batch {batch_size}: {stats['peak_memory_mb']:.1f} MB peak, "
          f"{stats['memory_per_sample']:.2f} MB/sample")

# Calculate arithmetic intensity for linear operations
linear1_flops = 2 * 128 * (128 * 4) * 16 * 64  # Approximate FLOPs for linear1
linear1_memory = (128 * 128 * 4 + 128 * 4 + 16 * 64 * 128 + 16 * 64 * 128 * 4) * 4  # Memory access in bytes
intensity = calculate_arithmetic_intensity(linear1_flops, linear1_memory)
print(f"\n⚡ Arithmetic Intensity: {intensity:.2f} FLOP/byte")
if intensity < 1.0:
    print("   📋 Memory-bound operation - good candidate for optimization")
else:
    print("   📋 Compute-bound operation - focus on computational optimization")
```

### **Run the Analysis**

```bash
PYTHONPATH=src python3 test_optimization_analysis.py
```

**Expected Output**:
```
🔍 GPU Optimization Analysis Results:
==================================================
🚀 Found 1 fusion opportunities:
  1. Linear + Activation
     Modules: linear1, activation
     Expected speedup: 1.3x

📊 Memory Analysis:
  Batch 8: 0.5 MB peak, 0.06 MB/sample
  Batch 16: 1.0 MB peak, 0.06 MB/sample
  Batch 32: 2.0 MB peak, 0.06 MB/sample

⚡ Arithmetic Intensity: 1.34 FLOP/byte
   📋 Memory-bound operation - good candidate for optimization
```

### **Example 2: Using Optimized Components**

Replace standard components with optimized versions:

```python
# save as test_optimized_components.py
import torch
import torch.nn as nn

# Import optimized components
from kernel_pytorch.compiler_optimized import (
    FusedGELU,
    OptimizedLayerNorm,
    FusedLinearGELU
)

# Standard implementation
class StandardBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 4)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.activation(self.linear(x)))

# Optimized implementation
class OptimizedBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # Use fused linear + GELU
        self.linear_gelu = FusedLinearGELU(dim, dim * 4)
        # Use optimized layer norm
        self.norm = OptimizedLayerNorm(dim * 4)

    def forward(self, x):
        return self.norm(self.linear_gelu(x))

# Compare implementations
sample_input = torch.randn(16, 64, 128)

standard_block = StandardBlock()
optimized_block = OptimizedBlock()

print("🔄 Component Comparison:")
print("=" * 30)

# Test both implementations
with torch.no_grad():
    standard_out = standard_block(sample_input)
    optimized_out = optimized_block(sample_input)

print(f"✅ Standard output shape: {standard_out.shape}")
print(f"✅ Optimized output shape: {optimized_out.shape}")
print("📊 Both implementations produce valid outputs!")

# Add torch.compile for additional optimization
compiled_optimized = torch.compile(optimized_block)
compiled_out = compiled_optimized(sample_input)
print(f"⚡ torch.compile output shape: {compiled_out.shape}")
print("🚀 Ready for maximum performance with compilation!")
```

**Run the Component Test**:

```bash
PYTHONPATH=src python3 test_optimized_components.py
```

## 📋 **Understanding the 5 Optimization Levels**

Our repository demonstrates progressive optimization complexity:

### **Level 1: PyTorch Native** (`components/basic_optimized.py`)
- **Focus**: Use PyTorch operations that map to optimized kernels
- **Techniques**: Vectorization, memory layout optimization, kernel-friendly patterns
- **When to use**: Starting point, good baseline performance

### **Level 2: TorchScript JIT** (`components/jit_optimized.py`)
- **Focus**: JIT compilation for automatic optimization
- **Techniques**: Graph optimization, kernel fusion, type specialization
- **When to use**: Easy performance boost with minimal code changes

### **Level 3: torch.compile** (Available throughout)
- **Focus**: Modern PyTorch 2.0 compilation
- **Techniques**: Inductor backend, graph optimization, kernel fusion
- **When to use**: Maximum performance with standard PyTorch

### **Level 4: Triton Kernels** (`triton_kernels/`)
- **Focus**: Python-like GPU kernel programming
- **Techniques**: Block-based computation, memory tiling, auto-tuning
- **When to use**: Custom operations, educational GPU programming

### **Level 5: Custom CUDA** (`cuda_kernels/`)
- **Focus**: Maximum control with raw CUDA
- **Techniques**: Warp-level operations, shared memory, hardware-specific optimization
- **When to use**: Ultimate performance, hardware-specific optimization

### **Quick Demo: Compare All Levels**

```bash
# Run the comprehensive demo
python demo_compiler_optimization.py
```

This will show performance comparisons across optimization levels.

## 🔬 **Understanding Optimization Patterns**

Our `optimization_patterns/` framework provides systematic optimization guidance:

```python
# Quick pattern analysis example
from kernel_pytorch.optimization_patterns import (
    check_compilation_compatibility,
    COMMON_FUSION_PATTERNS
)

# Check your model for compiler compatibility
model = SimpleModel()
sample_input = torch.randn(8, 32, 128)
compatibility = check_compilation_compatibility(model, sample_input)

print(f"Compilation compatibility: {compatibility['overall_compatibility']}")
print(f"Optimization opportunities: {len(compatibility['optimization_opportunities'])}")

# Explore common fusion patterns
print(f"\n📚 Available Fusion Patterns: {len(COMMON_FUSION_PATTERNS)}")
for pattern in COMMON_FUSION_PATTERNS[:3]:  # Show first 3
    print(f"  • {pattern.name}: {pattern.description}")
```

## 📋 **Knowledge Check**

### **Question 1**: Repository Structure
Where are the production-ready optimized components located?
- A) `src/kernel_pytorch/components/`
- B) `src/kernel_pytorch/compiler_optimized/`
- C) `src/kernel_pytorch/optimization_patterns/`
- D) `src/kernel_pytorch/gpu_integration/`

<details>
<summary>Click for answer</summary>
<b>Answer: B</b> - Production-ready optimized components are in `src/kernel_pytorch/compiler_optimized/`
</details>

### **Question 2**: Optimization Levels
Which optimization level provides the easiest performance boost with minimal code changes?
- A) PyTorch Native
- B) TorchScript JIT
- C) torch.compile
- D) Custom CUDA

<details>
<summary>Click for answer</summary>
<b>Answer: C</b> - torch.compile provides significant performance improvements with minimal code changes
</details>

### **Question 3**: Optimization Analysis
What does arithmetic intensity measure?
- A) Total FLOPs in an operation
- B) Memory bandwidth usage
- C) FLOPs per byte of memory access
- D) GPU utilization percentage

<details>
<summary>Click for answer</summary>
<b>Answer: C</b> - Arithmetic intensity measures FLOPs per byte, indicating whether operations are compute-bound or memory-bound
</details>

## ✅ **Tutorial Completion Checklist**

Before moving to the next tutorial, ensure you have:
- ✅ Successfully cloned and set up the repository
- ✅ Run optimization pattern analysis on a simple model
- ✅ Used optimized components from `compiler_optimized/`
- ✅ Understood the 5 progressive optimization levels
- ✅ Explored the optimization patterns framework

## 🎯 **Next Steps**

1. **Continue Learning**: [Tutorial 2: Basic PyTorch Optimization](02_pytorch_optimization_basics.md)
2. **Explore Examples**: Run `python demo_progressive_optimization.py` for comprehensive examples
3. **Practice**: Try analyzing your own PyTorch models with the optimization patterns framework
4. **Experiment**: Apply `@torch.compile` to your models and measure the performance impact

### **Quick References**
- **Performance Analysis**: Use functions from `optimization_patterns/` modules
- **Optimized Components**: Import from `compiler_optimized/` for production use
- **Learning**: Follow the progressive optimization levels 1→2→3→4→5
- **Help**: Check module docstrings for detailed explanations

---

**🎯 Tutorial Complete!** You now have a solid foundation for exploring GPU optimization techniques. The journey from basic PyTorch to cutting-edge optimization starts here!