# Installation

## Requirements

- **Python** 3.10+
- **PyTorch** 2.0+ (2.1+ recommended)
- **Platform**: Linux, macOS, Windows

GPU backends are optional. TorchBridge always falls back to CPU.

## Development Install

```bash
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge
pip install -r requirements.txt

# Verify
PYTHONPATH=src python3 -c "import torchbridge; print(f'TorchBridge v{torchbridge.__version__} ready')"
```

### Editable Install

```bash
pip install -e .[dev,all]
```

## Backend-Specific Setup

### NVIDIA (CUDA)

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Requires CUDA 11.8+ (12.0+ recommended for H100/Blackwell). See [Hardware Setup](../guides/hardware-setup.md) for full CUDA/NVCC installation.

### AMD (ROCm)

```bash
# Install PyTorch with ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# Verify
python3 -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

Requires ROCm 5.6+. Supported on MI200, MI300X, RDNA3.

### Intel (IPEX)

```bash
pip install intel-extension-for-pytorch

# Verify
python3 -c "import intel_extension_for_pytorch; print('IPEX available')"
```

Supported on Ponte Vecchio, Arc, Flex series.

### TPU (XLA)

```bash
pip install torch_xla

# Verify
python3 -c "import torch_xla; print('XLA available')"
```

Requires Google Cloud TPU environment. Supported on v4, v5e, v5p, v6e.

### Apple Silicon (MPS)

```bash
pip install torch torchvision

# Verify
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Conda Environment

```bash
conda create -n torchbridge python=3.10
conda activate torchbridge
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Verify Installation

```bash
# Core import
PYTHONPATH=src python3 -c "
from torchbridge import TorchBridgeConfig, UnifiedManager
from torchbridge.backends import detect_best_backend
print(f'Backend: {detect_best_backend()}')
print('TorchBridge ready')
"

# Run tests
PYTHONPATH=src python3 -m pytest tests/ -q

# System diagnostics
torchbridge doctor
```

## Next Steps

- [Quick Start](quickstart.md) -- get running with TorchBridge
- [Hardware Setup](../guides/hardware-setup.md) -- driver and toolkit installation
- [Troubleshooting](troubleshooting.md) -- common issues
