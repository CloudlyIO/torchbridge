# 🚀 Setup & Installation

**Quick setup guide for KernelPyTorch - for detailed CUDA/GPU setup, see [CUDA Setup Guide](cuda_setup.md)**

## ⚡ Quick Installation

```bash
# Clone repository
git clone <repository-url>
cd shahmod

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Verify installation
export PYTHONPATH=src:$PYTHONPATH
python -c "
import torch
from kernel_pytorch.components import FusedGELU
print(f'✅ KernelPyTorch ready!')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## ⚡ Quick Test

```bash
# Run basic optimization demo (30 seconds)
python demos/01_basic_optimizations.py --quick

# Expected output:
# ✅ Basic optimizations demo completed successfully
# 📊 Performance improvements: 2.8x-6.1x speedup demonstrated
```

## 📚 Next Steps

For detailed setup and configuration:
- **[CUDA Setup Guide](cuda_setup.md)** - Complete CUDA/Triton installation
- **[Cloud Testing](cloud_testing_guide.md)** - AWS/Azure/GCP deployment
- **[Hardware Guide](hardware.md)** - Multi-vendor GPU support
- **[Testing Guide](testing_guide.md)** - Comprehensive testing strategy

## 🔧 Troubleshooting

**CUDA not available?** → See [CUDA Setup Guide](cuda_setup.md)
**Import errors?** → Ensure `export PYTHONPATH=src:$PYTHONPATH`
**Memory issues?** → Use `--quick` flag in demos

**🚀 Ready to start optimizing PyTorch models!**