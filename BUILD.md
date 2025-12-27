# Building KernelPyTorch with Custom CUDA Kernels

This guide explains how to build KernelPyTorch with custom CUDA kernels (Phase 4A).

## 📋 Prerequisites

### Required

- **Python**: 3.8 or higher
- **PyTorch**: 2.0.0 or higher with CUDA support
- **CUDA Toolkit**: 11.8 or higher (12.0+ recommended for H100)
- **C++ Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **NVCC**: NVIDIA CUDA Compiler (comes with CUDA Toolkit)

### Optional

- **H100/Blackwell GPU**: For FP8 support
- **A100/V100 GPU**: For standard CUDA acceleration
- **pybind11**: For C++ bindings (auto-installed)

## 🔧 Quick Start

### 1. Verify CUDA Installation

```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Check NVCC compiler
nvcc --version

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA (if not already installed)
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install build dependencies
pip install numpy>=1.21.0 pybind11 ninja

# Optional: Install Triton for Level 4 optimizations
pip install triton>=2.0.0
```

### 3. Build CUDA Extensions

```bash
# Development build (in-place)
python setup.py build_ext --inplace

# Or: Editable install (recommended for development)
pip install -e .

# Or: Production install
pip install .
```

### 4. Verify Installation

```bash
# Test basic import
python3 -c "import kernel_pytorch; print(f'Version: {kernel_pytorch.__version__}')"

# Test CUDA kernel availability
python3 -c "
from kernel_pytorch.hardware.gpu.custom_kernels import FlashAttentionV3
print('✅ FlashAttention-3 available')
"

# Run comprehensive demo
python demos/custom_kernel_demo.py --quick
```

## 📦 What Gets Built

### CUDA Kernel Files

The build process compiles these custom CUDA kernels:

1. **`fused_ops.cu`** (Original)
   - Basic fused operations
   - Layer normalization
   - SwiGLU activation

2. **`flash_attention_v3.cu`** (Phase 4A - NEW)
   - FlashAttention-3 with online softmax
   - Head dimension templates (64, 128)
   - Split-K optimization for long sequences
   - FP8 accumulation (H100+)

3. **`fused_linear_activation.cu`** (Phase 4A - NEW)
   - Fused Linear+GELU
   - Fused Linear+SiLU
   - Fused Linear+ReLU
   - Template-based activation functors

4. **`cuda_interface.cpp`** (C++ Bindings)
   - PyBind11 interface
   - Input validation
   - CPU fallback implementations

### Build Outputs

After successful build:

- **Shared library**: `kernel_pytorch_cuda.so` (or `.pyd` on Windows)
- **Location**: `src/kernel_pytorch/` or site-packages
- **Import name**: Automatically available via `kernel_pytorch.hardware.gpu.custom_kernels`

## 🏗️ Build Configuration

### Compiler Flags

The build uses architecture-specific optimizations:

```python
# Supported GPU architectures
-gencode=arch=compute_70,code=sm_70  # V100 (Volta)
-gencode=arch=compute_75,code=sm_75  # T4, RTX 20xx (Turing)
-gencode=arch=compute_80,code=sm_80  # A100, RTX 30xx (Ampere)
-gencode=arch=compute_86,code=sm_86  # RTX 30xx (Ampere)
-gencode=arch=compute_89,code=sm_89  # RTX 40xx (Ada Lovelace)
-gencode=arch=compute_90,code=sm_90  # H100 (Hopper) with FP8
```

### FP8 Support

FP8 kernels are enabled with:

```python
-DENABLE_FP8                      # Enable FP8 code paths
-DENABLE_FLASH_ATTENTION_V3       # Enable FlashAttention-3
-DENABLE_FUSED_KERNELS            # Enable fused activation kernels
```

**Note**: FP8 operations require H100 (sm_90) or Blackwell GPUs.

### Optimization Flags

```python
-O3                                # Maximum optimization
--use_fast_math                    # Fast math (may reduce precision)
-Xptxas=-O3                        # PTX assembler optimization
--expt-relaxed-constexpr           # Relaxed constexpr rules
--extra-device-vectorization       # Additional vectorization
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "CUDA not available"

**Symptoms**: Build skips CUDA extensions

**Solutions**:
```bash
# Verify PyTorch CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. "nvcc: command not found"

**Symptoms**: Build fails with NVCC not found

**Solutions**:
```bash
# Add CUDA to PATH (Linux/macOS)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or set CUDA_HOME
export CUDA_HOME=/usr/local/cuda

# Windows
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
```

#### 3. "Unsupported GPU architecture"

**Symptoms**: Warnings about sm_XX not supported

**Solutions**:
- Your GPU is too old (pre-Volta)
- CPU fallback will be used automatically
- Consider upgrading GPU for Phase 4A features

#### 4. "Compilation error in .cu file"

**Symptoms**: NVCC compilation errors

**Solutions**:
```bash
# Clean build
python setup.py clean --all
rm -rf build/ dist/ *.egg-info

# Try with verbose output
python setup.py build_ext --inplace -v

# Check GCC version (needs 9+)
gcc --version

# Update CUDA Toolkit if needed
```

#### 5. "Import error after build"

**Symptoms**: `ImportError: cannot import name 'FlashAttentionV3'`

**Solutions**:
```bash
# Verify build succeeded
ls -la kernel_pytorch_cuda*.so

# Reinstall in editable mode
pip uninstall kernel-pytorch
pip install -e .

# Check Python can find the module
python3 -c "import kernel_pytorch_cuda; print('Module found')"
```

## 🧪 Testing the Build

### Basic Tests

```bash
# Run kernel registry tests
pytest tests/test_kernel_registry.py -v

# Run custom kernel tests (CPU fallback)
pytest tests/test_custom_kernels.py -v

# Run integration tests
pytest tests/test_kernel_integration.py -v
```

### CUDA Tests (Requires GPU)

```bash
# Run CUDA-only tests
pytest tests/test_custom_kernels.py -v -m "not slow"

# Run all tests including benchmarks
pytest tests/ -v --tb=short
```

### Benchmarks

```bash
# Run performance benchmarks
python benchmarks/custom_kernel_benchmark.py

# Expected output on CUDA GPU:
# - FlashAttention-3: 2-5x speedup vs PyTorch SDPA
# - Fused Linear+GELU: 1.8-2.5x speedup
```

## 📊 Performance Validation

### Expected Speedups (on CUDA GPU)

| Kernel | Baseline | Speedup | Hardware |
|--------|----------|---------|----------|
| FlashAttention-3 | PyTorch SDPA | 2-5x | V100+ |
| FlashAttention-3 (FP8) | PyTorch SDPA | 4-10x | H100+ |
| Fused Linear+GELU | Separate ops | 1.8-2.5x | V100+ |
| Fused Linear+SiLU | Separate ops | 1.8-2.5x | V100+ |

### Validation Commands

```bash
# Run demo with performance comparison
python demos/custom_kernel_demo.py

# Run comprehensive benchmark suite
python benchmarks/custom_kernel_benchmark.py

# Validate numerical accuracy
pytest tests/test_custom_kernels.py::TestFlashAttentionV3::test_numerical_accuracy -v
```

## 🔄 Rebuilding

### When to Rebuild

Rebuild is needed when:
- CUDA kernel files (`.cu`) are modified
- C++ interface files (`.cpp`) are modified
- CUDA Toolkit version changes
- PyTorch version changes

### Clean Build

```bash
# Remove all build artifacts
python setup.py clean --all
rm -rf build/ dist/ *.egg-info kernel_pytorch_cuda*.so

# Rebuild from scratch
python setup.py build_ext --inplace
```

### Incremental Build

```bash
# Only rebuild changed files
python setup.py build_ext --inplace
```

## 📝 Build Verification Checklist

After building, verify:

- [ ] `kernel_pytorch_cuda.so` exists in package directory
- [ ] `import kernel_pytorch` succeeds
- [ ] `import kernel_pytorch.hardware.gpu.custom_kernels` succeeds
- [ ] Demo runs without errors: `python demos/custom_kernel_demo.py --quick`
- [ ] Tests pass: `pytest tests/test_kernel_integration.py -v`
- [ ] Benchmarks run: `python benchmarks/custom_kernel_benchmark.py`

## 🚀 Advanced Build Options

### Custom CUDA Architectures

Edit `setup.py` to target specific GPUs:

```python
nvcc_flags = [
    # Only build for your GPU
    '-gencode=arch=compute_90,code=sm_90',  # H100 only
]
```

### Debug Build

```bash
# Add debug symbols
CFLAGS="-g -O0" python setup.py build_ext --inplace
```

### Ninja Build System (Faster)

```bash
# Install ninja
pip install ninja

# Build with ninja (automatic if installed)
python setup.py build_ext --inplace
```

## 📚 Additional Resources

- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **PyTorch C++ Extension**: https://pytorch.org/tutorials/advanced/cpp_extension.html
- **NVCC Compiler**: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- **KernelPyTorch Docs**: See `docs/` directory
- **Troubleshooting**: See GitHub Issues

## 🤝 Getting Help

If you encounter build issues:

1. Check this guide's troubleshooting section
2. Run diagnostic: `python demos/custom_kernel_demo.py --quick`
3. Review build logs carefully
4. Check CUDA/PyTorch versions compatibility
5. Open an issue on GitHub with:
   - Build command used
   - Full error message
   - `nvcc --version` output
   - `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"`
   - GPU model and compute capability

---

**Last Updated**: 2025-12-26 (Phase 4B)
**Version**: 0.3.0
