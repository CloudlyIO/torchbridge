# Tutorial 1: Repository Overview and Quick Setup

**Duration**: 30-45 minutes
**Difficulty**: Beginner
**Prerequisites**: Basic Python knowledge, familiarity with PyTorch

## 🎯 **Learning Objectives**

By the end of this tutorial, you will:
- Understand the repository structure and optimization philosophy
- Set up the development environment successfully
- Run your first optimization example
- Understand the 5 progressive optimization levels
- Use the semantic code understanding agent

## 🏗️ **Repository Structure Deep Dive**

Our repository follows a **semantic-driven optimization architecture**:

```
shahmod/
├── src/kernel_pytorch/              # Core optimization components
│   ├── components/                  # 5-level optimization hierarchy
│   │   ├── basic_optimized.py      # Level 1: PyTorch native optimizations
│   │   └── jit_optimized.py        # Level 2: TorchScript compilation
│   ├── triton_kernels/              # Level 4: Triton GPU programming
│   ├── cuda_kernels/                # Level 5: Custom CUDA kernels
│   ├── semantic_agent/              # AI-powered code understanding
│   │   ├── architecture.py         # Core semantic analysis
│   │   ├── llm_understanding.py    # Advanced concept extraction
│   │   └── concept_mapping.py      # Code-to-concept mapping
│   └── utils/                       # Profiling and benchmarking
├── docs/                           # Comprehensive documentation
│   ├── tutorials/                  # Step-by-step learning guides
│   └── research_roadmap.md         # Future research directions
├── demo_*.py                       # Interactive demonstrations
└── interactive_semantic_demo.py    # AI-powered code analysis tool
```

### **🧠 Key Innovation: Semantic Code Understanding**

This repository uniquely combines:
- **Traditional Optimization**: Manual kernel optimization techniques
- **AI-Powered Analysis**: Semantic understanding of ML/AI code patterns
- **Educational Focus**: Learning-oriented design with comprehensive explanations

## 🛠️ **Environment Setup**

### **Step 1: Clone and Basic Installation**

```bash
# Clone the repository
git clone https://github.com/shahrahman-fb/shahmod.git
cd shahmod

# Basic installation (CPU-only initially)
pip install -r requirements.txt
pip install -e .
```

### **Step 2: Verify Installation**

```bash
# Test basic installation
python -c "import kernel_pytorch; print('✅ Basic installation successful')"

# Test semantic agent
PYTHONPATH=src python3 -c "
from kernel_pytorch.semantic_agent.architecture import SemanticCodeAgent
agent = SemanticCodeAgent()
print('✅ Semantic agent loaded successfully')
"
```

### **Step 3: GPU Setup (Optional but Recommended)**

```bash
# Check CUDA availability
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
"

# For advanced features (if CUDA available)
# Note: Custom CUDA kernels require CUDA toolkit
# pip install triton  # For Triton kernels (if not already installed)
```

## 🚀 **Your First Optimization Example**

### **Example 1: Progressive Optimization Demo**

Let's see the power of optimization levels in action:

```python
# Run this in Python or Jupyter notebook
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from kernel_pytorch.components.basic_optimized import OptimizedTransformerBlock
from kernel_pytorch.utils.profiling import quick_benchmark

# Create a transformer block
dim = 512
num_heads = 8
batch_size = 4
seq_len = 128

# Initialize optimized transformer
transformer_block = OptimizedTransformerBlock(dim=dim, num_heads=num_heads)

# Generate test input
input_tensor = torch.randn(batch_size, seq_len, dim)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_block = transformer_block.to(device)
input_tensor = input_tensor.to(device)

print(f"🎯 Running optimization example on: {device}")

# Benchmark the optimized implementation
def optimized_forward():
    with torch.no_grad():
        return transformer_block(input_tensor)

# Run benchmark
stats = quick_benchmark(optimized_forward, num_runs=100)

print(f"✅ Optimized Transformer Block Performance:")
print(f"   Average time: {stats['mean_time']*1000:.3f} ms")
print(f"   Std deviation: {stats['std_time']*1000:.3f} ms")
print(f"   Memory used: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB" if torch.cuda.is_available() else "   CPU execution")
```

**Expected Output**:
```
🎯 Running optimization example on: cuda:0
✅ Optimized Transformer Block Performance:
   Average time: 2.145 ms
   Std deviation: 0.023 ms
   Memory used: 156.3 MB
```

### **Example 2: Semantic Code Understanding**

Now let's use our AI agent to understand the optimization patterns:

```python
# Interactive semantic analysis
from kernel_pytorch.semantic_agent.architecture import SemanticCodeAgent

# Initialize semantic agent
agent = SemanticCodeAgent()

# Analyze the transformer code we just ran
transformer_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = OptimizedMultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
'''

# Analyze the code
print("🧠 Semantic Analysis Results:")
print("=" * 50)

result = agent.analyze_code(transformer_code)

print(f"📊 Detected {len(result['patterns'])} semantic patterns:")

for pattern in result['patterns']:
    print(f"\n🎯 Concept: {pattern['concept']}")
    print(f"   Confidence: {pattern['confidence']:.2f}")
    print(f"   Location: {pattern['location']}")
    print(f"   Evidence: {', '.join(pattern['evidence'][:2])}")
    if pattern.get('optimization_potential'):
        print(f"   💡 Optimization: {pattern['optimization_potential']}")

print(f"\n📋 Summary: {result['summary']}")

if result.get('optimization_suggestions'):
    print(f"\n🚀 Optimization Suggestions:")
    for suggestion in result['optimization_suggestions']:
        print(f"   • {suggestion}")
```

**Expected Output**:
```
🧠 Semantic Analysis Results:
==================================================
📊 Detected 3 semantic patterns:

🎯 Concept: transformer
   Confidence: 0.75
   Location: class:OptimizedTransformerBlock
   Evidence: class OptimizedTransformerBlock(nn.Module), self.attention
   💡 Optimization: Consider using torch.compile or custom CUDA kernels for performance

🎯 Concept: attention
   Confidence: 0.67
   Location: class:OptimizedTransformerBlock
   Evidence: OptimizedMultiHeadAttention, self.attention
   💡 Optimization: Consider using F.scaled_dot_product_attention for Flash Attention optimization

🎯 Concept: layer_normalization
   Confidence: 0.80
   Location: class:OptimizedTransformerBlock
   Evidence: nn.LayerNorm(dim), self.norm1
   💡 Optimization: Consider kernel fusion for LayerNorm + following operations

📋 Summary: Detected 3 semantic patterns. High confidence patterns: 2. Main concepts: transformer, attention, layer_normalization.

🚀 Optimization Suggestions:
   • class:OptimizedTransformerBlock: Consider using torch.compile or custom CUDA kernels for performance
   • class:OptimizedTransformerBlock: Consider using F.scaled_dot_product_attention for Flash Attention optimization
```

## 📊 **Understanding the 5 Optimization Levels**

Our repository demonstrates **progressive optimization** while maintaining **semantic equivalence**:

### **Level 1: PyTorch Native Optimizations** 📈
**Focus**: Kernel-friendly patterns using PyTorch built-ins
```python
# Example: Memory-coalesced attention computation
def level1_attention(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # Vectorized
    weights = F.softmax(scores, dim=-1)                    # cuDNN optimized
    return torch.matmul(weights, v)                        # Batch operation
```

### **Level 2: TorchScript JIT Compilation** ⚡
**Focus**: Automatic kernel fusion through compilation
```python
@torch.jit.script
def level2_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    # JIT compiler automatically fuses operations where possible
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

### **Level 3: torch.compile (PyTorch 2.0+)** 🚀
**Focus**: Advanced graph optimization with TorchInductor
```python
@torch.compile
def level3_attention(q, k, v, scale):
    # TorchInductor automatically generates optimized Triton kernels
    return F.scaled_dot_product_attention(q, k, v, scale=scale)
```

### **Level 4: Triton Kernels** 🔬
**Focus**: Educational GPU programming in Python
```python
import triton

@triton.jit
def level4_attention_kernel(q_ptr, k_ptr, v_ptr, output_ptr, ...):
    # Manual GPU kernel programming with Python syntax
    # Educational visibility into GPU programming concepts
    pass
```

### **Level 5: Custom CUDA Kernels** ⚙️
**Focus**: Maximum performance with C++/CUDA
```cpp
__global__ void level5_attention_kernel(
    const float* q, const float* k, const float* v,
    float* output, int seq_len, int head_dim
) {
    // Direct CUDA programming for maximum control and performance
    // Complete hardware feature access
}
```

## 🎮 **Interactive Exploration**

### **Using the Interactive Semantic Demo**

```bash
# Launch the interactive demo
PYTHONPATH=src python3 interactive_semantic_demo.py
```

**Available Commands**:
- `list` - Show available code examples
- `analyze basic_attention` - Analyze pre-loaded attention implementation
- `custom` - Analyze your own code input
- `explain attention` - Get detailed concept explanations
- `compare basic_attention flash_attention` - Compare implementations
- `mapping basic_attention` - Show concept mappings
- `optimize basic_attention` - Get optimization suggestions

### **Try This Interactive Exercise**

1. **Launch the demo**: `PYTHONPATH=src python3 interactive_semantic_demo.py`
2. **Run**: `analyze basic_attention`
3. **Run**: `explain transformer`
4. **Run**: `optimize basic_attention`
5. **Run**: `quit`

**What You Should Learn**:
- How the semantic agent identifies ML concepts in code
- The difference between basic and optimized implementations
- Specific optimization recommendations for your code

## 🔍 **Performance Comparison Exercise**

Let's compare optimization levels on a simple operation:

```python
import time
import torch
import torch.nn.functional as F

# Setup
batch_size, seq_len, dim = 4, 128, 512
q = torch.randn(batch_size, 8, seq_len, 64, device='cuda')
k = torch.randn(batch_size, 8, seq_len, 64, device='cuda')
v = torch.randn(batch_size, 8, seq_len, 64, device='cuda')

# Level 1: Manual implementation
def manual_attention(q, k, v):
    scale = (q.size(-1)) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

# Level 3: PyTorch optimized
def optimized_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

# Benchmark both (warm up first)
for _ in range(10):
    manual_attention(q, k, v)
    optimized_attention(q, k, v)

# Time manual implementation
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    result1 = manual_attention(q, k, v)
torch.cuda.synchronize()
manual_time = time.time() - start

# Time optimized implementation
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    result2 = optimized_attention(q, k, v)
torch.cuda.synchronize()
optimized_time = time.time() - start

print(f"🔍 Performance Comparison:")
print(f"   Manual implementation: {manual_time*1000:.2f} ms")
print(f"   Optimized implementation: {optimized_time*1000:.2f} ms")
print(f"   Speedup: {manual_time/optimized_time:.2f}x")
print(f"   Results match: {torch.allclose(result1, result2, atol=1e-6)}")
```

**Expected Output**:
```
🔍 Performance Comparison:
   Manual implementation: 45.23 ms
   Optimized implementation: 18.67 ms
   Speedup: 2.42x
   Results match: True
```

## ✅ **Knowledge Check**

Test your understanding with these questions:

### **Question 1**: Repository Structure
Which directory contains the AI-powered semantic code understanding agent?
- A) `src/kernel_pytorch/components/`
- B) `src/kernel_pytorch/semantic_agent/`
- C) `src/kernel_pytorch/utils/`
- D) `docs/tutorials/`

<details>
<summary>Click for answer</summary>
<b>Answer: B</b> - The semantic agent is located in `src/kernel_pytorch/semantic_agent/`
</details>

### **Question 2**: Optimization Levels
What is the main focus of Level 2 optimization?
- A) PyTorch native operations
- B) TorchScript JIT compilation
- C) Custom CUDA kernels
- D) Triton programming

<details>
<summary>Click for answer</summary>
<b>Answer: B</b> - Level 2 focuses on TorchScript JIT compilation for automatic kernel fusion
</details>

### **Question 3**: Semantic Understanding
What does the semantic agent provide besides concept detection?
- A) Only performance metrics
- B) Only bug detection
- C) Optimization suggestions and learning paths
- D) Only code formatting

<details>
<summary>Click for answer</summary>
<b>Answer: C</b> - The semantic agent provides optimization suggestions, learning paths, and educational insights
</details>

## 🎯 **Next Steps**

Congratulations! You've successfully:
- ✅ Set up the repository environment
- ✅ Run your first optimization example
- ✅ Used the semantic code understanding agent
- ✅ Understood the 5 optimization levels
- ✅ Performed a performance comparison

### **Continue Your Learning Journey**

1. **Next Tutorial**: [Tutorial 2: Basic PyTorch Optimization](02_pytorch_optimization_basics.md)
2. **Explore Examples**: Run `python demo_semantic_agent.py` for more examples
3. **Read Documentation**: Check out the component READMEs in `src/kernel_pytorch/`
4. **Try Your Own Code**: Use the interactive demo to analyze your own ML implementations

### **Recommended Practice**
- Experiment with different input sizes in the performance comparison
- Try analyzing your own PyTorch code with the semantic agent
- Explore the other demo scripts in the repository root

### **Get Help**
- **Documentation**: Check component READMEs for detailed explanations
- **Interactive Demo**: Use `help` command in the semantic demo
- **Community**: Create issues on GitHub for questions and suggestions

---

**🎯 Tutorial Complete!** You now have a solid foundation for exploring ML kernel optimization and semantic code understanding. The journey from PyTorch basics to cutting-edge research starts here!