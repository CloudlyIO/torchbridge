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
python -c "import kernel_pytorch; print('✅ Setup successful')"

# Run tests
PYTHONPATH=src python -m pytest tests/ -v
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
# Run full test suite (required)
PYTHONPATH=src python -m pytest tests/ --tb=short

# Run specific test categories
PYTHONPATH=src python -m pytest tests/test_attention_compatibility.py -v
PYTHONPATH=src python -m pytest tests/test_fp8_training.py -v

# Run demos to validate integration
PYTHONPATH=../src python demos/run_all_demos.py --quick

# Run benchmarks to check performance
PYTHONPATH=src python benchmarks/run_comprehensive_benchmark.py --quick
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
- [ ] All tests passing: `PYTHONPATH=src python -m pytest tests/`
- [ ] All demos working: `python demos/run_all_demos.py --quick`
- [ ] Benchmarks operational: `python benchmarks/run_comprehensive_benchmark.py --quick`
- [ ] Code follows naming conventions
- [ ] Documentation updated for new features
- [ ] No hardcoded device assumptions (support both CPU/GPU)

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
# Profile import performance
PYTHONPATH=src python -c "from kernel_pytorch.utils.import_profiler import benchmark_lazy_loading_improvements; benchmark_lazy_loading_improvements()"

# Validate attention implementations
PYTHONPATH=src python -c "from kernel_pytorch.attention import create_ring_attention; print('✅ Ring Attention available')"

# Check FP8 training setup
PYTHONPATH=src python -c "from kernel_pytorch.precision import create_fp8_trainer; print('✅ FP8 Training available')"

# Run specific benchmark category
PYTHONPATH=src python benchmarks/framework/benchmark_runner.py
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