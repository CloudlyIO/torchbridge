# Hardware Setup

Driver and toolkit installation for each backend.

## NVIDIA (CUDA)

### CUDA Toolkit

**Ubuntu/Linux:**
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Verify
nvidia-smi
nvcc --version
```

**Conda:**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Environment variables:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
```

### PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}')"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

### Triton (Optional)

For Triton GPU kernels:

```bash
pip install triton>=2.0.0

python3 -c "import triton; print(f'Triton: {triton.__version__}')"
```

---

## AMD (ROCm)

### ROCm Installation

**Ubuntu 22.04:**
```bash
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/6.0 jammy main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm
```

**Environment variables:**
```bash
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
```

### PyTorch with ROCm

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### Verify

```bash
rocminfo
rocm-smi
python3 -c "import torch; print(f'ROCm: {torch.cuda.is_available()}')"
```

---

## Intel (IPEX / oneAPI)

### Intel Extension for PyTorch

```bash
pip install intel-extension-for-pytorch
```

### oneAPI Base Toolkit (for XPU)

```bash
# Install oneAPI
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/..../l_BaseKit.sh
sudo sh l_BaseKit.sh

# Set environment
source /opt/intel/oneapi/setvars.sh
```

### PyTorch with XPU

```bash
pip install intel-extension-for-pytorch[xpu]
```

### Verify

```bash
python3 -c "import intel_extension_for_pytorch; print('IPEX available')"
python3 -c "import torch; print(f'XPU devices: {torch.xpu.device_count()}')"
```

---

## Google TPU (XLA)

### PyTorch/XLA

TPU setup requires a Google Cloud TPU environment (GCE VM, Colab, or Kaggle).

```bash
pip install torch_xla[tpu]~=2.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

### Verify

```bash
python3 -c "
import torch_xla
import torch_xla.core.xla_model as xm
print(f'XLA device: {xm.xla_device()}')
"
```

### Environment

```bash
export TPU_NAME=your-tpu-name
```

---

## Apple Silicon (MPS)

No special setup needed -- MPS is included with PyTorch on macOS.

```bash
pip install torch torchvision

python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## IDE Configuration

### VS Code

```json
{
    "python.analysis.extraPaths": ["src"],
    "python.envFile": "${workspaceFolder}/.env"
}

```

Add to `.env`:
```
PYTHONPATH=src
```

### PyCharm

1. Settings > Project > Python Interpreter: select your env
2. Settings > Project > Project Structure: mark `src` as Sources Root

---

## Verification Checklist

After setup, run:

```bash
# TorchBridge detection
PYTHONPATH=src python3 -c "
from torchbridge.backends import detect_best_backend
print(f'Detected backend: {detect_best_backend()}')
"

# Full diagnostics
torchbridge doctor

# Run tests
PYTHONPATH=src python3 -m pytest tests/ -q
```

## See Also

- [Installation](../getting-started/installation.md)
- [Backend Selection](backend-selection.md)
- [Troubleshooting](../getting-started/troubleshooting.md)
