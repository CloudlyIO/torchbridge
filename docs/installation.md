# 📦 Installation Guide

**KernelPyTorch** - Production-grade PyTorch GPU optimization framework

## Quick Install

```bash
pip install kernel-pytorch
```

## Detailed Installation

### System Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **PyTorch**: 2.0+ (2.1+ recommended)
- **Platform**: Linux, macOS, Windows
- **GPU**: CUDA 11.8+ (optional, for GPU acceleration)

### Installation Options

#### 1. Standard Installation

```bash
# Install base package
pip install kernel-pytorch

# Verify installation
kernelpytorch --version
kernelpytorch doctor
```

#### 2. Complete Installation (All Features)

```bash
# Install with all optional dependencies
pip install kernel-pytorch[all]
```

#### 3. Development Installation

```bash
# Clone repository
git clone https://github.com/KernelPyTorch/kernel-pytorch.git
cd kernel-pytorch

# Install in development mode
pip install -e .[dev,all]

# Verify development setup
python -m pytest tests/ -v
```

#### 4. Docker Installation

```bash
# Production container
docker run ghcr.io/kernelpytorch/kernel-pytorch:latest kernelpytorch doctor

# Development environment
docker run -it -v $(pwd):/workspace ghcr.io/kernelpytorch/kernel-pytorch:dev
```

## Optional Dependencies

### Core Extensions
```bash
pip install kernel-pytorch[all]  # All core features
```

### Cloud Integration
```bash
pip install kernel-pytorch[cloud]  # AWS, GCP, Azure support
```

### Serving & Deployment
```bash
pip install kernel-pytorch[serving]  # FastAPI, TorchServe integration
```

### Monitoring & Profiling
```bash
pip install kernel-pytorch[monitoring]  # Weights & Biases, MLflow
```

### Benchmarking Tools
```bash
pip install kernel-pytorch[benchmark]  # Performance analysis tools
```

## System Compatibility Check

After installation, run the system diagnostics:

```bash
# Quick system check
kernelpytorch doctor

# Comprehensive diagnostics
kernelpytorch doctor --full-report

# Save diagnostic report
kernelpytorch doctor --full-report --output system_report.json
```

## GPU Setup

### CUDA Installation

**Ubuntu/Linux:**
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Conda Environment:**
```bash
conda create -n kernelpytorch python=3.10
conda activate kernelpytorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install kernel-pytorch[all]
```

### Apple Silicon (M1/M2/M3)

```bash
# Install with Metal Performance Shaders support
pip install torch torchvision
pip install kernel-pytorch[all]

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Quick Start Verification

### 1. Test Core Functionality

```python
import kernel_pytorch as kpt
import torch

# Create optimized model
model = kpt.OptimizedLinear(512, 256)
x = torch.randn(32, 512)
y = model(x)

print(f"✓ KernelPyTorch {kpt.__version__} working!")
print(f"  Input: {x.shape}")
print(f"  Output: {y.shape}")
```

### 2. Test CLI Tools

```bash
# System diagnostics
kernelpytorch doctor --verbose

# Quick optimization
echo "import torch; torch.save(torch.nn.Linear(32, 16), 'test_model.pt')" | python
kernelpytorch optimize --model test_model.pt --level production

# Performance benchmark
kernelpytorch benchmark --predefined optimization --quick
```

### 3. Test Advanced Features

```python
import kernel_pytorch as kpt

# Test optimization assistant
from kernel_pytorch.utils.compiler_optimization_assistant import CompilerOptimizationAssistant

assistant = CompilerOptimizationAssistant()
model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768)
)

result = assistant.optimize_model(model, interactive=False)
print(f"✓ Found {len(result.optimization_opportunities)} optimization opportunities")
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Fix: Reinstall with all dependencies
pip uninstall kernel-pytorch
pip install kernel-pytorch[all] --no-cache-dir
```

#### CUDA Not Found
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Fix: Install CUDA-compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### CLI Commands Not Found
```bash
# Fix: Ensure package is installed correctly
pip show kernel-pytorch
which kernelpytorch

# Alternative: Use module syntax
python -m kernel_pytorch.cli doctor
```

#### Performance Issues
```bash
# Check system configuration
kernelpytorch doctor --category hardware

# Run performance regression test
kernelpytorch benchmark --type regression
```

### Getting Help

1. **System Diagnostics**: `kernelpytorch doctor --full-report`
2. **Documentation**: [https://kernel-pytorch.readthedocs.io](https://kernel-pytorch.readthedocs.io)
3. **Issues**: [GitHub Issues](https://github.com/KernelPyTorch/kernel-pytorch/issues)
4. **CLI Help**: `kernelpytorch --help`

## Environment Setup Examples

### Research Environment
```bash
pip install kernel-pytorch[all,dev]
jupyter lab  # Pre-configured notebooks available
```

### Production Environment
```bash
pip install kernel-pytorch[serving,monitoring]
# Use Docker for consistent deployment
```

### Cloud Deployment
```bash
pip install kernel-pytorch[cloud,serving]
# AWS/GCP/Azure integrations available
```

## Next Steps

After installation:

1. **📖 Read the [Quick Start Guide](quickstart.md)**
2. **🔧 Try the [Optimization Tutorial](tutorials/optimization.md)**
3. **📊 Explore [Performance Benchmarks](benchmarks.md)**
4. **🐳 Setup [Docker Development Environment](docker.md)**

---

**Need help?** Run `kernelpytorch doctor` for automated system diagnostics.