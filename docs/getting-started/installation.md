# 📦 Installation Guide

> **Version**: v0.4.18 | **Status**: ✅ Current | **Last Updated**: Jan 22, 2026

**TorchBridge** - Production-grade PyTorch GPU/TPU optimization framework

## Development Installation (Current)

```bash
# Clone repository
git clone https://github.com/your-org/torchbridge.git
cd torchbridge

# Install dependencies
pip install -r requirements.txt

# Verify installation
PYTHONPATH=src python3 -c "import torchbridge; print(f'✅ TorchBridge v{torchbridge.__version__} ready!')"
```

## System Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **PyTorch**: 2.0+ (2.1+ recommended)
- **Platform**: Linux, macOS, Windows
- **GPU**: CUDA 11.8+ (optional, for GPU acceleration)

### Unified Architecture with Multi-Backend Support (v0.4.18)

TorchBridge v0.4.18 is a production-ready framework with:
- **Single Configuration System**: `TorchBridgeConfig`
- **Unified Management**: `UnifiedManager`
- **Comprehensive Validation**: `UnifiedValidator`
- **NVIDIA Backend**: H100/Blackwell support with FP8 training (cloud validated)
- **TPU Backend**: Google Cloud TPU v4/v5e/v6 with XLA compilation (cloud validated)
- **AMD Backend**: MI200/MI300 with ROCm/HIP support
- **Cloud Validated**: GCP (L4, TPU v5e), AWS (A10G)

### Installation Options

#### 1. Basic Development Setup

```bash
# Clone and install
git clone https://github.com/your-org/torchbridge.git
cd torchbridge
pip install torch numpy

# Verify core functionality
PYTHONPATH=src python3 -c "
from torchbridge import TorchBridgeConfig, UnifiedManager
print('✅ Unified architecture ready!')
"
```

#### 2. Complete Development Setup

```bash
# Install all dependencies from pyproject.toml
pip install -e .[dev,all]

# Run comprehensive tests
PYTHONPATH=src python3 -m pytest tests/ --tb=short -q
```

#### 3. Verification Steps

```bash
# Test unified architecture
PYTHONPATH=src python3 -c "
from torchbridge import TorchBridgeConfig, UnifiedManager
from torchbridge.validation import UnifiedValidator

config = TorchBridgeConfig.for_development()
manager = UnifiedManager(config)
validator = UnifiedValidator()
print('✅ v0.4.18 unified architecture verified!')
"

# Run test suite
PYTHONPATH=src python3 -m pytest tests/ --tb=short -q

# Run demos
PYTHONPATH=src python3 demos/run_all_demos.py --quick
```

## Next Steps

After installation, see:
- [Quick Start Guide](quickstart.md) - Get started with unified architecture
- [Architecture Guide](../capabilities/architecture.md) - Understand the unified design
- [Testing Guide](../guides/testing_guide.md) - Run comprehensive validation

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH=/path/to/shahmod/src
```

**Version Mismatch**:
```bash
# Check version consistency
PYTHONPATH=src python3 -c "import torchbridge; print(torchbridge.__version__)"
# Should output: 0.4.3
```

**Test Failures**:
```bash
# Platform-specific skips are normal
# 905 tests should pass, 101 may be skipped (platform-specific)
```

## System Compatibility Check

After installation, run the system diagnostics:

```bash
# Quick system check
torchbridge doctor

# Comprehensive diagnostics
torchbridge doctor --full-report

# Save diagnostic report
torchbridge doctor --full-report --output system_report.json
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
conda create -n torchbridge python=3.10
conda activate torchbridge
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torchbridge[all]
```

### Apple Silicon (M1/M2/M3)

```bash
# Install with Metal Performance Shaders support
pip install torch torchvision
pip install torchbridge[all]

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Quick Start Verification

### 1. Test Core Functionality

```python
import torchbridge as kpt
import torch

# Create optimized model
model = kpt.OptimizedLinear(512, 256)
x = torch.randn(32, 512)
y = model(x)

print(f"✓ TorchBridge {kpt.__version__} working!")
print(f"  Input: {x.shape}")
print(f"  Output: {y.shape}")
```

### 2. Test CLI Tools

```bash
# System diagnostics
torchbridge doctor --verbose

# Quick optimization
echo "import torch; torch.save(torch.nn.Linear(32, 16), 'test_model.pt')" | python
torchbridge optimize --model test_model.pt --level production

# Performance benchmark
torchbridge benchmark --predefined optimization --quick
```

### 3. Test Unified Architecture

```python
from torchbridge import TorchBridgeConfig, UnifiedManager
from torchbridge.validation import UnifiedValidator
import torch

# Test unified optimization system
config = TorchBridgeConfig.for_production()
manager = UnifiedManager(config)

model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768)
)

# Unified optimization and validation
optimized_model = manager.optimize_model(model)
validator = UnifiedValidator()
results = validator.validate_model(optimized_model, (1, 768))

print(f"✓ Unified optimization completed")
print(f"✓ Validation: {results.passed}/{results.total_tests} tests passed")
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Fix: Reinstall with all dependencies
pip uninstall torchbridge
pip install torchbridge[all] --no-cache-dir
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
pip show torchbridge
which torchbridge

# Alternative: Use module syntax
python -m torchbridge.cli doctor
```

#### Performance Issues
```bash
# Check system configuration
torchbridge doctor --category hardware

# Run performance regression test
torchbridge benchmark --type regression
```

### Getting Help

1. **System Diagnostics**: `torchbridge doctor --full-report`
2. **Documentation**: [https://torchbridge.readthedocs.io](https://torchbridge.readthedocs.io)
3. **Issues**: [GitHub Issues](https://github.com/TorchBridge/torchbridge/issues)
4. **CLI Help**: `torchbridge --help`

## Environment Setup Examples

### Research Environment
```bash
pip install torchbridge[all,dev]
jupyter lab  # Pre-configured notebooks available
```

### Production Environment
```bash
pip install torchbridge[serving,monitoring]
# Use Docker for consistent deployment
```

### Cloud Deployment
```bash
pip install torchbridge[cloud,serving]
# AWS/GCP/Azure integrations available
```

## Next Steps

After installation:

1. **📖 Read the [Quick Start Guide](quickstart.md)**
2. **🔧 Try the [Guides](../guides/README.md)**
3. **📊 Explore [Performance](../capabilities/performance.md)**
4. **🐳 Setup [Docker Guide](../guides/docker.md)**

---

**Need help?** Run `torchbridge doctor` for automated system diagnostics.