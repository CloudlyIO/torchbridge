# 🤝 Contributing to KernelPyTorch

**Guide for developers contributing to the PyTorch GPU optimization framework.**

## 🚀 Quick Start for Contributors

### **Prerequisites**
- Python 3.9+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU features)
- Git with proper SSH/HTTPS setup

### **Development Setup**
```bash
# Clone and setup
git clone <repository-url>
cd shahmod
pip install -r requirements.txt

# Validate setup
python3 -c "import kernel_pytorch; print('✅ Setup successful')"

# Run tests
PYTHONPATH=src python3 -m pytest tests/ -v
```

## 🎯 Project Architecture

### **Core Components**
```
src/kernel_pytorch/
├── attention/              # Unified attention framework (Ring, Sparse, Context Parallel)
├── precision/              # FP8 training and quantization
├── hardware_abstraction/   # Multi-vendor GPU/CPU support
├── compiler_integration/   # FlashLight, PyGraph, TorchInductor
├── components/             # Core optimized layers
├── utils/                  # Profiling and optimization tools
└── testing_framework/      # Validation and benchmarking
```

### **Key Features Implemented**
- ✅ **Ring Attention**: Million-token sequences with O(N) memory
- ✅ **Sparse Attention**: 90% compute reduction with dynamic patterns
- ✅ **FP8 Training**: 2x speedup on H100/Blackwell hardware
- ✅ **Hardware Abstraction**: Multi-vendor GPU optimization
- ✅ **Compiler Integration**: FlashLight and PyGraph optimizations

## 🧪 Development Workflow

### **1. Making Changes**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes following naming conventions (see below)
# Classes: PascalCase (e.g., RingAttentionLayer)
# Functions: snake_case (e.g., create_ring_attention)
# Constants: UPPER_SNAKE_CASE (e.g., MAX_SEQUENCE_LENGTH)

# Add comprehensive tests
# Edit files in tests/ to cover new functionality
```

### **2. Testing Requirements**
```bash
# WORKING: Run specific test modules (recommended)
PYTHONPATH=src python3 -m pytest tests/test_advanced_memory.py -v          # 22/22 pass ✅
PYTHONPATH=src python3 -m pytest tests/test_advanced_memory_benchmarks.py -v  # 5/8 pass ⚠️

# WORKING: Run demos to validate integration
PYTHONPATH=src python3 demos/run_all_demos.py --quick                     # 3/5 working ✅
PYTHONPATH=src python3 demos/05_next_generation/run_next_gen_demos.py --device cpu --quick  # All working ✅
PYTHONPATH=src python3 demos/06_advanced_memory/simple_memory_demo.py --device cpu --quick  # Working ✅

# NOTE: Full test suite (PYTHONPATH=src python3 -m pytest tests/) has hanging tests
# Use specific modules instead for reliable testing
```

### **3. Code Quality Standards**

#### **Naming Conventions (PEP 8)**
```python
# ✅ Correct Examples
class RingAttentionLayer(nn.Module):          # Classes: PascalCase
    pass

def create_ring_attention() -> RingAttentionLayer:  # Functions: snake_case
    pass

MAX_SEQUENCE_LENGTH = 1_000_000               # Constants: UPPER_SNAKE_CASE

# ❌ Avoid
class ring_attention_layer:  # Wrong casing
def CreateRingAttention():   # Wrong casing
```

#### **Documentation Requirements**
```python
def create_ring_attention(d_model: int, num_heads: int) -> RingAttentionLayer:
    """
    Create Ring Attention layer for million-token sequences.

    Args:
        d_model: Model dimension size
        num_heads: Number of attention heads

    Returns:
        Configured RingAttentionLayer instance

    Example:
        >>> attention = create_ring_attention(512, 8)
        >>> output = attention(long_sequence)
    """
```

#### **Testing Requirements**
```python
class TestRingAttention:
    def test_linear_memory_complexity(self):
        """Test O(N) memory usage vs O(N²) baseline"""
        pass

    def test_million_token_support(self):
        """Test handling of 1M+ token sequences"""
        pass
```

### **4. Performance Requirements**
- **No regressions**: New code must not slow down existing functionality
- **Benchmark validation**: Run benchmarks to verify improvements
- **Memory efficiency**: Optimize for both speed and memory usage
- **Hardware compatibility**: Test on both CPU and GPU when available

## 📋 Submission Process

### **Before Submitting PR**
- [ ] Specific tests passing: `PYTHONPATH=src python -m pytest tests/test_advanced_memory.py -v`
- [ ] All demos working: `PYTHONPATH=src python demos/run_all_demos.py --quick`
- [ ] Core benchmarks operational: Check `demos/` performance output
- [ ] Code follows naming conventions
- [ ] Documentation updated for new features
- [ ] No hardcoded device assumptions (support both CPU/GPU)
- [ ] Verify imports work: Test your API imports before submission

### **PR Requirements**
1. **Clear description** of what the PR accomplishes
2. **Test results** showing all validations pass
3. **Performance impact** (if applicable) with benchmark results
4. **Breaking changes** clearly documented (if any)
5. **Documentation updates** included

### **Review Process**
- PRs require passing CI/CD checks
- Code review focuses on correctness, performance, and maintainability
- Large changes may require design discussion before implementation

## 🎯 Focus Areas for Contributors

### **High-Priority Areas**
1. **Ultra-Precision Quantization** (FP4/MXFP building on FP8 success)
2. **Structured Sparsity** (2:4 patterns for Tensor Core acceleration)
3. **Hardware-Specific Optimizations** (NVIDIA generation-aware tuning)
4. **Performance Benchmarking** (Comprehensive validation framework)

### **Technical Debt & Improvements**
1. **HAL Unification** (Consistent hardware abstraction across vendors)
2. **Demo Simplification** (Clearer, more focused examples)
3. **Documentation Consolidation** (Reduce verbosity, improve clarity)
4. **Test Coverage** (Expand edge case and integration testing)

## 🔧 Development Tools

### **Useful Commands**
```bash
# Test verified working imports
PYTHONPATH=src python -c "from kernel_pytorch.advanced_memory import DeepOptimizerStates; print('✅ Advanced Memory available')"
PYTHONPATH=src python -c "from kernel_pytorch.optimizations.next_gen import create_advanced_flex_attention; print('✅ Next-gen optimizations available')"
PYTHONPATH=src python -c "from kernel_pytorch.precision import create_fp8_trainer; print('✅ FP8 Training available')"

# Test import that may have issues
PYTHONPATH=src python -c "
try:
    from kernel_pytorch.attention import create_attention
    print('✅ Attention framework available')
except ImportError as e:
    print(f'⚠️ Attention import issue: {e}')
"

# Run working performance tests
PYTHONPATH=src python demos/06_advanced_memory/simple_memory_demo.py --device cpu --quick
```

### **IDE Setup Recommendations**
- **PyCharm/VSCode**: Configured with Python 3.9+ interpreter
- **Type checking**: Enable MyPy or Pylance for type validation
- **Code formatting**: Black or similar for consistent style
- **Import organization**: isort for clean import structure

## 🚨 Common Issues & Solutions

### **Import Errors**
```bash
# Always set PYTHONPATH for testing
export PYTHONPATH=src
# Or use inline: PYTHONPATH=src python <command>
```

### **CUDA/GPU Issues**
```bash
# Code must work on CPU-only systems
# Use device-agnostic patterns:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### **Test Failures**
```bash
# Run single test file for debugging
PYTHONPATH=src python -m pytest tests/test_specific_file.py::TestClass::test_method -v -s
```

## 📞 Getting Help

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Technical questions and design discussions
- **Code Review**: Submit PRs for collaborative development

## 📜 Code of Conduct

- **Respectful collaboration** in all interactions
- **Constructive feedback** in code reviews
- **Quality focus** over speed of delivery
- **Documentation** as important as code
- **Testing** as essential as implementation

---

**Ready to contribute?** Start with running the test suite and exploring the demos to understand the codebase architecture. 🚀