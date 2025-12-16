# 🔧 CUDA & Triton Setup Guide

**Complete setup instructions for CUDA and Triton development, including hardware simulation and testing.**

## 🎯 Overview

This guide covers:
1. **CUDA Toolkit Installation** - For GPU development and kernel compilation
2. **Triton Installation** - For high-performance GPU kernel development
3. **Testing Setup** - Both hardware simulation and real GPU testing
4. **Validation** - Ensuring everything works correctly

## 🚀 Quick Setup (Recommended)

### Prerequisites
- **Python 3.8+** (3.11+ recommended)
- **PyTorch 2.0+** with CUDA support
- **NVIDIA GPU** (optional but recommended)

### Basic Installation
```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Triton (GPU kernel development)
pip3 install triton

# Install our framework requirements
pip3 install -r requirements.txt
```

### Quick Validation
```bash
# Test basic CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Triton availability
python3 -c "import triton; print('Triton installed successfully')"

# Validate our framework
python3 demos/01_basic_optimizations.py --quick
```

## 🔧 Detailed Setup Instructions

### 1. CUDA Toolkit Installation

#### **Option A: System-wide Installation (Recommended)**

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH="/usr/local/cuda-12.1/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

**CentOS/RHEL:**
```bash
# Add repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install CUDA
sudo dnf install -y cuda-toolkit-12-1
```

**Windows:**
1. Download CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run installer with default options
3. Verify installation: `nvcc --version`

#### **Option B: Conda Installation**
```bash
# Using conda-forge
conda install -c conda-forge cudatoolkit=12.1 cudnn

# Or using nvidia channel
conda install -c nvidia cuda-toolkit=12.1
```

### 2. PyTorch with CUDA Installation

#### **Standard Installation**
```bash
# For CUDA 12.1 (recommended)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older systems)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Development Installation (from source)**
```bash
# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Set environment
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CUDA_HOME=/usr/local/cuda-12.1

# Build and install
python setup.py develop
```

### 3. Triton Installation

#### **Standard Installation**
```bash
# Install stable version
pip3 install triton

# Or install nightly for latest features
pip3 install -U --index-url https://download.pytorch.org/whl/nightly/cu121 triton
```

#### **Development Installation**
```bash
# Clone Triton
git clone https://github.com/openai/triton.git
cd triton

# Install in development mode
cd python
pip install -e .
```

### 4. Framework Dependencies

```bash
# Core dependencies
pip3 install numpy scipy matplotlib seaborn

# Testing and benchmarking
pip3 install pytest pytest-benchmark

# Optional: Advanced profiling
pip3 install py-spy memory-profiler

# Install our framework
pip3 install -e .
```

## ✅ Installation Verification

### 1. CUDA Verification
```bash
# Check CUDA compiler
nvcc --version

# Check GPU detection
nvidia-smi

# Test CUDA with PyTorch
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory // 1024**3}GB')
        print(f'    Compute: {props.major}.{props.minor}')
"
```

### 2. Triton Verification
```bash
# Test basic Triton functionality
python3 -c "
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

print('✅ Triton kernel compilation successful')
"
```

### 3. Framework Verification
```bash
# Quick functionality test
export PYTHONPATH=src:$PYTHONPATH
python3 -c "
from kernel_pytorch.components import FusedGELU
from kernel_pytorch.hardware_abstraction import NVIDIAAdapter
import torch

print('✅ Framework imports successful')

if torch.cuda.is_available():
    # Test optimized component
    gelu = FusedGELU()
    x = torch.randn(32, 768, device='cuda')
    y = gelu(x)
    print(f'✅ GPU optimization working: {y.shape}')

    # Test hardware abstraction
    adapter = NVIDIAAdapter()
    print(f'✅ Hardware adapter: CUDA available = {adapter.cuda_available}')
else:
    print('⚠️  No GPU available - CPU-only mode')
"

# Run comprehensive tests
python3 -m pytest tests/test_basic_functionality.py -v
```

## 🔧 Troubleshooting

### Common CUDA Issues

#### **CUDA Not Found**
```bash
# Check if CUDA is in PATH
which nvcc

# If not found, add manually
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Make permanent
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

#### **Driver Version Mismatch**
```bash
# Check driver version
nvidia-smi

# Check CUDA runtime version
nvcc --version

# Update driver if needed (Ubuntu)
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### **PyTorch CUDA Mismatch**
```bash
# Check PyTorch CUDA version
python3 -c "import torch; print(torch.version.cuda)"

# Reinstall matching version
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Common Triton Issues

#### **Compilation Errors**
```bash
# Update to latest version
pip install --upgrade triton

# Check for LLVM conflicts
python3 -c "
import triton
print(f'Triton version: {triton.__version__}')
import triton._C.libtriton as libtriton
print('✅ Triton backend loaded successfully')
"
```

#### **Runtime Errors**
```bash
# Clear Triton cache
rm -rf ~/.triton/cache

# Set debug environment
export TRITON_PRINT_AUTOTUNING=1
export CUDA_LAUNCH_BLOCKING=1

# Test simple kernel
python3 tests/test_triton_basic.py
```

### Performance Issues

#### **Slow Compilation**
```bash
# Enable parallel compilation
export TRITON_COMPILE_THREADS=8

# Use compilation cache
export TRITON_CACHE_DIR=~/.triton/cache
```

#### **Memory Issues**
```bash
# Check GPU memory
nvidia-smi

# Clear PyTorch cache
python3 -c "
import torch
torch.cuda.empty_cache()
print(f'GPU memory cleared')
"
```

## 🚀 Development Setup

### IDE Configuration

#### **VS Code Setup**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "files.associations": {
        "*.cu": "cuda-cpp"
    }
}
```

#### **PyCharm Setup**
1. Go to File → Settings → Project → Python Interpreter
2. Add new environment: `./venv/bin/python`
3. Enable CUDA syntax highlighting: Plugins → CUDA Support

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Configuration in .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
EOF
```

## 📊 Performance Validation

### Benchmark Setup
```bash
# Run basic benchmarks
python3 benchmarks/simple_benchmark_test.py

# Run comprehensive benchmarks
python3 -c "
import sys
import time
import torch

print('=== Performance Validation ===')

# CUDA basic operations
if torch.cuda.is_available():
    device = torch.device('cuda')

    # Matrix multiplication benchmark
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * size**3 * 100 / elapsed  # FLOPS calculation
    print(f'Matrix Multiply ({size}x{size}): {elapsed:.3f}s')
    print(f'Performance: {flops/1e12:.2f} TFLOPS')

else:
    print('⚠️  No CUDA available for performance testing')

print('✅ Performance validation complete')
"
```

## 🎯 Next Steps

1. **Basic Setup**: Follow Quick Setup for immediate start
2. **Development**: Configure IDE and development environment
3. **Testing**: Run verification scripts to ensure everything works
4. **Optimization**: Start with basic optimization demos
5. **Advanced**: Move to advanced features like multi-GPU and custom kernels

For more advanced setup and cloud deployment, see:
- [Cloud Testing Guide](cloud_testing_guide.md)
- [Hardware Abstraction Setup](hardware.md)
- [Performance Optimization Guide](performance.md)

**🚀 Ready for GPU-accelerated PyTorch development!**