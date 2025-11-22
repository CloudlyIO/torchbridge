# 🧪 PyTorch Optimization Tests

Comprehensive test suite for validating PyTorch kernel and compiler optimizations. This directory contains tests for all optimization components, from basic functionality to advanced next-generation computing paradigms.

## 📋 Test Overview

| Test Module | Coverage | Focus Area | Runtime |
|-------------|----------|------------|---------|
| `test_priority1_compiler_integration.py` | Compiler Optimizations | FlashLight, PyGraph, Enhanced Fusion | 2-5 min |
| `test_advanced_optimizations.py` | Advanced Techniques | Attention patterns, GPU optimization | 3-7 min |
| `test_next_gen_optimizations.py` | Next-Generation | Neuromorphic, quantum-classical hybrid | 5-10 min |
| `test_distributed_scale.py` | Distributed Computing | Multi-GPU, scaling patterns | 3-8 min |
| `test_testing_framework.py` | Testing Infrastructure | Validation, benchmarking framework | 2-4 min |

**Total Test Coverage**: 5 test modules, ~15-35 minutes for complete suite

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure Python 3.8+ and PyTorch are installed
python --version
pip install torch torchvision pytest

# Optional: Install additional dependencies for full functionality
pip install numpy scipy matplotlib
```

### Running All Tests

```bash
# From repository root - run complete test suite
python -m pytest tests/ -v

# Run with detailed output and timing
python -m pytest tests/ -v --tb=short --durations=10

# Run tests in parallel (if pytest-xdist installed)
pip install pytest-xdist
python -m pytest tests/ -v -n auto
```

### Running Individual Test Modules

```bash
# Priority 1 compiler integration tests (recommended first)
python -m pytest tests/test_priority1_compiler_integration.py -v

# Advanced optimization tests
python -m pytest tests/test_advanced_optimizations.py -v

# Next-generation computing tests
python -m pytest tests/test_next_gen_optimizations.py -v

# Distributed scaling tests
python -m pytest tests/test_distributed_scale.py -v

# Testing framework validation
python -m pytest tests/test_testing_framework.py -v
```

### Running Specific Test Categories

```bash
# Run only FlashLight compiler tests
python -m pytest tests/test_priority1_compiler_integration.py::TestFlashLightCompiler -v

# Run attention pattern optimization tests
python -m pytest tests/test_advanced_optimizations.py::TestAdvancedAttentionOptimizations -v

# Run neuromorphic computing tests
python -m pytest tests/test_next_gen_optimizations.py::TestNeuromorphicComputing -v

# Run GPU integration tests
python -m pytest tests/test_distributed_scale.py::TestMultiGPUIntegration -v
```

## 📊 Test Categories

### 1. Priority 1 Compiler Integration (`test_priority1_compiler_integration.py`)
**Core compiler optimization features from 2025-2026 roadmap**

- **FlashLight Compiler**: Automatic attention kernel generation and compilation
- **PyGraph CUDA Graphs**: Revolutionary CUDA graph optimization with cost-benefit analysis
- **Enhanced Fusion**: Advanced TorchInductor fusion beyond standard boundaries
- **Integration Testing**: All compiler components working together

**Key Test Classes:**
- `TestFlashLightCompiler`: FlashLight compilation and caching
- `TestPyGraphOptimizer`: CUDA graph optimization
- `TestEnhancedFusion`: Advanced fusion techniques
- `TestCompilerIntegration`: End-to-end integration scenarios

### 2. Advanced Optimizations (`test_advanced_optimizations.py`)
**Advanced optimization techniques and attention patterns**

- **Attention Optimizations**: Custom patterns, memory efficiency, long sequences
- **Kernel Fusion**: Advanced fusion strategies and validation
- **Memory Management**: Efficient memory usage and optimization
- **Performance Benchmarking**: Comprehensive performance validation

**Key Test Classes:**
- `TestAdvancedAttentionOptimizations`: Attention pattern testing
- `TestAdvancedKernelFusion`: Kernel fusion validation
- `TestMemoryOptimizations`: Memory management testing
- `TestPerformanceBenchmarking`: Performance validation

### 3. Next-Generation Computing (`test_next_gen_optimizations.py`)
**2026+ computing paradigms and emerging technologies**

- **Neuromorphic Computing**: Spiking neural networks and brain-inspired computation
- **Quantum-Classical Hybrid**: Quantum algorithms integrated with classical optimization
- **Post-Transformer Architectures**: Beyond transformer models (Mamba, SSM, Liquid Networks)
- **Advanced Attention**: Ring attention, sparse patterns, ultra-long sequences

**Key Test Classes:**
- `TestNeuromorphicComputing`: Neuromorphic algorithm validation
- `TestQuantumClassicalHybrid`: Quantum-classical integration
- `TestPostTransformerArchitectures`: Next-gen model architectures
- `TestAdvancedFlexAttention`: Advanced attention mechanisms

### 4. Distributed Scale (`test_distributed_scale.py`)
**Multi-GPU and distributed computing optimization**

- **Multi-GPU Integration**: Scaling across multiple GPUs
- **Communication Optimization**: Efficient inter-GPU communication
- **Load Balancing**: Dynamic workload distribution
- **Scaling Patterns**: Large-scale deployment validation

**Key Test Classes:**
- `TestMultiGPUIntegration`: Multi-GPU functionality
- `TestCommunicationOptimization`: Communication efficiency
- `TestLoadBalancing`: Workload distribution
- `TestScalingPatterns`: Large-scale patterns

### 5. Testing Framework (`test_testing_framework.py`)
**Testing infrastructure and validation framework**

- **Optimization Validation**: Correctness testing for optimizations
- **Performance Monitoring**: Real-time performance tracking
- **Benchmarking Framework**: Comprehensive benchmarking tools
- **CI/CD Integration**: Continuous integration testing

**Key Test Classes:**
- `TestOptimizationValidation`: Optimization correctness
- `TestPerformanceMonitoring`: Performance tracking
- `TestBenchmarkingFramework`: Benchmarking infrastructure
- `TestCICDIntegration`: CI/CD pipeline testing

## ⚙️ Test Configuration

### Environment Variables

```bash
# Enable detailed test output
export PYTEST_VERBOSITY=2

# Set CUDA device for GPU tests (if available)
export CUDA_VISIBLE_DEVICES=0

# Enable torch compilation debugging
export TORCH_COMPILE_DEBUG=1

# Set PyTorch compilation cache directory
export TORCH_COMPILE_CACHE_DIR=/tmp/torch_compile_cache
```

### Hardware Requirements

**Minimum Requirements:**
- CPU: x86_64 or ARM64
- Memory: 8GB RAM
- Python: 3.8+
- PyTorch: 2.0+

**Recommended for Full Testing:**
- GPU: NVIDIA GPU with CUDA 11.8+
- Memory: 16GB+ RAM, 8GB+ GPU memory
- Storage: 5GB free space for compilation cache

### Test Modes

**Quick Testing (CPU-compatible):**
```bash
# Run core tests without GPU requirements
python -m pytest tests/test_priority1_compiler_integration.py::TestFlashLightCompiler::test_attention_compilation -v
```

**GPU Testing (if CUDA available):**
```bash
# Run GPU-specific optimizations
python -m pytest tests/test_distributed_scale.py -v -k "gpu or cuda"
```

**Comprehensive Testing:**
```bash
# Run all tests with extensive validation
python -m pytest tests/ -v --tb=long --capture=no
```

## 🔍 Test Output and Debugging

### Understanding Test Results

**Test Status Indicators:**
- `PASSED` ✅: Test executed successfully
- `FAILED` ❌: Test failed with error
- `SKIPPED` ⏭️: Test skipped (usually due to missing hardware/dependencies)
- `XFAIL` ⚠️: Expected failure (known limitations)

**Performance Metrics:**
- Tests include performance benchmarks and validation
- Look for speedup measurements in test output
- Memory usage and efficiency metrics are reported

### Common Test Failures and Solutions

**CUDA Compilation Errors:**
```bash
# Clear PyTorch compilation cache
rm -rf ~/.cache/torch/
export TORCH_COMPILE_DEBUG=1
python -m pytest tests/test_priority1_compiler_integration.py -v
```

**Memory Errors:**
```bash
# Reduce batch sizes for memory-constrained environments
export PYTORCH_TEST_CUDA_MEM_FRAC=0.5
python -m pytest tests/ -v
```

**Import Errors:**
```bash
# Ensure PYTHONPATH includes src directory
export PYTHONPATH=src:$PYTHONPATH
python -m pytest tests/ -v
```

**Timeout Issues:**
```bash
# Increase test timeout for slow hardware
python -m pytest tests/ -v --timeout=300
```

## 📈 Continuous Integration

### Running Tests in CI/CD

```yaml
# Example GitHub Actions workflow
name: PyTorch Optimization Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install torch torchvision pytest
    - name: Run tests
      run: |
        export PYTHONPATH=src:$PYTHONPATH
        python -m pytest tests/ -v --tb=short
```

### Test Coverage and Quality

**Coverage Analysis:**
```bash
# Install coverage tools
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

**Performance Regression Testing:**
```bash
# Run performance benchmarks
python -m pytest tests/test_testing_framework.py::TestPerformanceBenchmarking -v
```

## 🎯 Best Practices

### Writing New Tests

1. **Test Structure**: Follow the pattern in existing test files
2. **Mock External Dependencies**: Use mocks for hardware-specific features
3. **Performance Validation**: Include performance benchmarks where appropriate
4. **Error Handling**: Test both success and failure scenarios
5. **Documentation**: Add clear docstrings explaining test purpose

### Test Maintenance

1. **Regular Updates**: Keep tests aligned with code changes
2. **Performance Baselines**: Update expected performance metrics
3. **Hardware Compatibility**: Ensure tests work across different hardware
4. **Dependencies**: Keep test dependencies minimal and well-documented

## 🔧 Troubleshooting

### Common Issues

**Tests fail with compilation errors:**
- Check PyTorch version compatibility
- Clear compilation cache: `rm -rf ~/.cache/torch/`
- Verify CUDA installation (for GPU tests)

**Tests skip with "hardware not available":**
- Expected behavior on systems without specific hardware
- Use CPU-only test modes: `pytest -m "not gpu"`

**Performance tests show unexpected results:**
- Performance varies by hardware and system load
- Focus on relative improvements rather than absolute numbers
- Run multiple times for consistent baselines

### Getting Help

**Test-Specific Issues:**
```bash
# Run specific failing test with maximum verbosity
python -m pytest tests/test_priority1_compiler_integration.py::TestFlashLightCompiler::test_specific_function -vvv --tb=long
```

**Environment Debugging:**
```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify test environment
python -m pytest --collect-only tests/
```

## 🎉 Quick Test Validation

```bash
# Validate basic functionality (1-2 minutes)
python -m pytest tests/test_priority1_compiler_integration.py::TestEnhancedFusion::test_fusion_pattern_detection -v

# Test hardware simulation (1 minute)
python -m pytest tests/test_testing_framework.py::TestHardwareSimulator -v

# Test advanced optimizations (2-3 minutes, some may fail with compilation issues)
python -m pytest tests/test_advanced_optimizations.py -v --tb=short

# Full test suite (15-35 minutes, some tests may fail due to compilation/hardware limitations)
python -m pytest tests/ -v --tb=short
```

### Expected Test Results

**Note**: Some tests may fail due to:
- **Compilation Issues**: torch.compile compilation cache problems (clear with `rm -rf ~/.cache/torch/`)
- **Hardware Dependencies**: Tests requiring CUDA/specific hardware will be skipped on incompatible systems
- **Mock Limitations**: Some advanced features use mock implementations and may have expected limitations

**Successful Test Categories**:
- ✅ Hardware simulation and benchmarking framework
- ✅ Enhanced fusion pattern detection
- ✅ Basic compiler integration tests
- ✅ Testing framework validation

**Happy Testing!** 🧪✨