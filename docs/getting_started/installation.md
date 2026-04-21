# Installation

## Requirements

- **Python** 3.10+ (3.12 recommended)
- **PyTorch** 2.5+ (2.7 recommended)
- **Platform**: Linux, macOS, Windows

See the [Compatibility Matrix](../reference/compatibility-matrix.md) for full version details and known issues.

GPU backends are optional. TorchBridge always falls back to CPU.

## Development Install

```bash
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge
pip install -e ".[dev]"

# Verify
python3 -c "import torchbridge; print(f'TorchBridge v{torchbridge.__version__} ready')"
tb-doctor
```

To install all optional GPU/cloud extras as well (heavier, may require CUDA/ROCm):

```bash
pip install -e ".[dev,all]"
```

## Backend-Specific Setup

### NVIDIA (CUDA)

```bash
# Install PyTorch with CUDA 12.6 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Verify
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Requires CUDA 12.4+ (12.6 recommended). Note: CUDA 12.8 with driver 580 has known FP16/BF16 GEMM issues -- prefer cu126. See [Hardware Setup](../guides/backend-selection.md) for full CUDA/NVCC installation and the [Compatibility Matrix](../reference/compatibility-matrix.md) for details.

### AMD (ROCm)

```bash
# Install PyTorch with ROCm 6.2 (stable)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Or ROCm 7.2 (current)
# pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# Verify
python3 -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

Requires ROCm 6.2+ (6.2 stable, 7.0/7.2 also supported). Validated on MI200, MI300X, MI325X. ROCm 5.x is no longer supported.

### TPU (XLA)

```bash
pip install torch_xla

# Verify
python3 -c "import torch_xla; print('XLA available')"
```

Requires Google Cloud TPU environment. Supported on v4, v5e, v5p, v6e, v7 (Ironwood).

### AWS Trainium (NeuronX)

```bash
# On Trn1/Trn2 instances with NeuronX pre-installed
pip install torch-neuronx

# Verify
python3 -c "import torch_neuronx; print('NeuronX available')"
```

Requires AWS Trn1 or Trn2 instances with NeuronX runtime.

### Apple Silicon (MPS)

```bash
pip install torch torchvision

# Verify
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Conda Environment

```bash
conda create -n torchbridge python=3.12
conda activate torchbridge
conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia
pip install torchbridge-ml
```

Conda is a good option for managing Python versions and CUDA toolkit dependencies together. Use `pytorch-cuda=12.6` for the recommended CUDA version.

## Verify Installation

```bash
# Core import
python3 -c "
from torchbridge import TorchBridgeConfig, UnifiedManager
from torchbridge.backends import detect_best_backend
print(f'Backend: {detect_best_backend()}')
print('TorchBridge ready')
"

# System diagnostics
tb-doctor
```

For development installs (cloned repo), run the test suite to verify everything works:

```bash
# From the repo root (dev install only)
python3 -m pytest tests/ -q
```

After installation, run the system diagnostics command to verify your environment:

```bash
tb-doctor
```

This checks your Python version, PyTorch installation, available backends, and driver compatibility.

## Next Steps

- [Quick Start](quickstart.md) -- get running with TorchBridge
- [Compatibility Matrix](../reference/compatibility-matrix.md) -- supported versions and known issues
- [Backend Selection](../guides/backend-selection.md) -- driver setup and backend configuration
- [Troubleshooting](troubleshooting.md) -- common issues
