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
python3 benchmarks/simple_benchmark_test.py
```

## 🔧 Detailed Setup Instructions

### 1. CUDA Toolkit Installation

#### **Option A: System-wide Installation (Recommended)**

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 12.1
sudo apt-get install cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**CentOS/RHEL:**
```bash
# Add NVIDIA repository
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install CUDA
sudo yum install cuda-toolkit-12-1

# Environment setup
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**macOS:**
```bash
# CUDA not natively supported on Apple Silicon
# Use CPU simulation mode or cloud GPU instances

# For Intel Macs (deprecated):
# Download CUDA toolkit from NVIDIA website
# Follow installer instructions
```

**Windows:**
```powershell
# Download CUDA Toolkit from NVIDIA Developer site
# https://developer.nvidia.com/cuda-toolkit
# Run installer and follow GUI instructions

# Add to PATH (usually automatic)
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
```

#### **Option B: Conda Installation**
```bash
# Create conda environment with CUDA
conda create -n gpu-optimization python=3.11
conda activate gpu-optimization

# Install CUDA toolkit via conda
conda install -c nvidia cuda-toolkit=12.1

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Triton Installation

#### **Standard Installation**
```bash
# Install latest Triton
pip3 install triton

# For development (latest features)
pip3 install git+https://github.com/openai/triton.git
```

#### **Compile from Source (Advanced)**
```bash
# Install dependencies
pip3 install cmake ninja

# Clone and build Triton
git clone https://github.com/openai/triton.git
cd triton/python
python3 setup.py build
python3 setup.py install
```

### 3. Driver and Runtime Setup

#### **NVIDIA Driver Installation**
```bash
# Ubuntu - Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535

# Verify installation
nvidia-smi
```

#### **Docker Setup (Recommended for Consistent Environment)**
```bash
# Pull NVIDIA PyTorch container
docker pull nvcr.io/nvidia/pytorch:23.10-py3

# Run with GPU access
docker run --gpus all -it \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:23.10-py3 \
  bash

# Install our framework inside container
cd /workspace
pip3 install -r requirements.txt
```

## 🧪 Testing Setup

### Hardware Simulation Mode

**Use Case**: Development without GPU hardware, CI/CD pipelines

```bash
# Set environment for simulation
export CUDA_SIMULATE=1
export TRITON_SIMULATE=1

# Run tests in simulation mode
python3 -c "
import torch
from kernel_pytorch.testing_framework import HardwareSimulator

# Create hardware simulator
simulator = HardwareSimulator(device_type='cuda', simulate=True)
result = simulator.simulate_kernel_execution(
    kernel_type='attention',
    input_shapes=[(1, 512, 768)],
    optimization_level='aggressive'
)
print(f'Simulation result: {result.summary}')
"
```

### Real Hardware Testing

**Use Case**: Performance validation, production deployment

```bash
# Verify GPU availability
python3 -c "
import torch
print(f'CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')
"

# Run hardware-specific tests
PYTHONPATH=src python3 -m pytest tests/ -m gpu -v

# Run specific CUDA tests
python3 benchmarks/next_gen/demo_cutting_edge_benchmark.py --validate
```

### Performance Testing

```bash
# Comprehensive GPU testing
python3 -c "
from kernel_pytorch.testing_framework import PerformanceBenchmarking, BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Run predefined benchmarks
results = suite.run_predefined_benchmarks()
print(f'Benchmark results: {results}')

# Generate performance report
suite.generate_report('gpu_performance_report.html')
"
```

## 🔍 Validation Commands

### Quick Health Check
```bash
# Complete framework validation
python3 -c "
import torch
import sys

# Check PyTorch CUDA
cuda_available = torch.cuda.is_available()
print(f'✅ PyTorch CUDA: {cuda_available}')

if cuda_available:
    print(f'   GPU Count: {torch.cuda.device_count()}')
    print(f'   Current GPU: {torch.cuda.get_device_name()}')
    print(f'   CUDA Version: {torch.version.cuda}')

# Check Triton
try:
    import triton
    print('✅ Triton: Available')
    print(f'   Version: {triton.__version__}')
except ImportError:
    print('❌ Triton: Not available')

# Check our framework
try:
    from kernel_pytorch.compiler_integration import FlashLightKernelCompiler
    compiler = FlashLightKernelCompiler()
    print('✅ FlashLight Compiler: Available')
except ImportError as e:
    print(f'❌ FlashLight Compiler: {e}')

# Check optimization components
try:
    from kernel_pytorch.next_gen_optimizations import AdaptivePrecisionAllocator
    allocator = AdaptivePrecisionAllocator(device=torch.device('cpu'))
    print('✅ Advanced Optimizations: Available')
except ImportError as e:
    print(f'❌ Advanced Optimizations: {e}')

print('\\n🎯 Setup validation complete!')
"
```

### Test Suite Execution

```bash
# Quick validation (30 seconds)
python3 benchmarks/simple_benchmark_test.py

# CPU tests only
PYTHONPATH=src python3 -m pytest tests/ -m "not gpu" --maxfail=5

# GPU tests (if hardware available)
PYTHONPATH=src python3 -m pytest tests/ -m gpu --maxfail=5

# Integration tests with realistic data
PYTHONPATH=src python3 -m pytest tests/ -m integration --maxfail=3

# Full test suite (comprehensive)
PYTHONPATH=src python3 -m pytest tests/ --maxfail=10
```

## 🐛 Troubleshooting

### Common Issues

#### **CUDA Out of Memory**
```bash
# Reduce batch sizes in tests
export CUDA_TEST_BATCH_SIZE=1

# Enable memory fraction limiting
export CUDA_MEMORY_FRACTION=0.8

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### **Triton Compilation Errors**
```bash
# Check Triton cache
ls ~/.triton/cache

# Clear Triton cache
rm -rf ~/.triton/cache

# Rebuild with debug info
TRITON_DEBUG=1 python3 your_script.py
```

#### **Driver/Runtime Mismatch**
```bash
# Check versions
nvidia-smi  # Driver version
nvcc --version  # CUDA runtime version

# Typical fix - update driver
sudo apt update && sudo apt upgrade nvidia-driver-*
```

#### **Import Errors**
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Add source directory
export PYTHONPATH=src:$PYTHONPATH

# Verify installation
pip3 list | grep torch
pip3 list | grep triton
```

### Performance Issues

#### **Slow Compilation**
```bash
# Enable parallel compilation
export TRITON_CACHE_MANAGER='MemoryMappedCache'

# Use fewer optimization passes
export TRITON_OPTIMIZE_LEVEL=1
```

#### **Memory Leaks**
```bash
# Enable memory debugging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Monitor memory usage
python3 -c "
import torch
import gc

def check_memory():
    if torch.cuda.is_available():
        print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB')
    print(f'Python Objects: {len(gc.get_objects())}')

check_memory()
# ... your code here ...
check_memory()
"
```

## 📚 Advanced Configuration

### Environment Variables

```bash
# CUDA Configuration
export CUDA_VISIBLE_DEVICES=0,1  # Specify GPU devices
export CUDA_LAUNCH_BLOCKING=1    # Synchronous execution (debugging)
export CUDA_CACHE_DISABLE=1      # Disable kernel caching

# Triton Configuration
export TRITON_CACHE_DIR=~/.triton/cache
export TRITON_DUMP_BINARY=1      # Save compiled kernels
export TRITON_PRINT_AUTOTUNING=1 # Show tuning progress

# PyTorch Configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_SHOW_CPP_STACKTRACES=1
```

### Custom Kernel Development

```python
# Example: Custom Triton kernel
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Custom addition kernel for learning"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

# Integration with our framework
from kernel_pytorch.testing_framework import HardwareSimulator

simulator = HardwareSimulator()
result = simulator.test_custom_kernel(add_kernel, input_shapes=[(1024,), (1024,)])
print(f'Custom kernel test: {result.passed}')
```

## 🎯 Next Steps

After successful setup:

1. **Run Quick Validation**: `python3 benchmarks/simple_benchmark_test.py`
2. **Try Cutting-Edge Benchmarks**: `python3 benchmarks/next_gen/demo_cutting_edge_benchmark.py --quick`
3. **Explore Optimizations**: Check `demos/` directory for examples
4. **Read Documentation**: See `docs/` for advanced usage

For questions or issues, refer to the [main README](README.md) or create an issue.

---

**🚀 Ready for high-performance GPU optimization development!**