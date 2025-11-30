# 🚀 PyTorch Optimization Demos

**5 focused demonstrations showcasing 2-6x performance improvements.**

## ⚡ Quick Start

```bash
# Run all demos (5 minutes)
python3 demos/run_all_demos.py --quick

# Expected: 2.8x-6.1x speedup demonstrations
```

## 📁 Demo Structure

| Demo | Focus | Speedup | Time |
|------|-------|---------|------|
| [01_basic_optimizations.py](01_basic_optimizations.py) | Core optimizations | 2.8-5.1x | 2 min |
| [02_advanced_attention.py](02_advanced_attention.py) | Attention mechanisms | 3-10x | 3 min |
| [03_fp8_training.py](03_fp8_training.py) | FP8 precision | 2x | 2 min |
| [04_hardware_abstraction.py](04_hardware_abstraction.py) | Multi-vendor GPUs | 1.2-1.5x | 3 min |
| [05_production_deployment.py](05_production_deployment.py) | Production patterns | 3.6-6.4x | 4 min |

## 🎯 Demo Modes

- `--quick` - Fast validation (2-4 min per demo)
- `--validate` - Full testing with benchmarks
- `python3 demos/run_all_demos.py --help` - All options

## 🔧 Requirements

```bash
# Required
pip install torch torchvision torchaudio triton numpy

# GPU support (recommended)
nvidia-smi  # Verify CUDA availability

# Set path
export PYTHONPATH=src:$PYTHONPATH
```

## 🧪 Hardware Testing

**CPU Only:**
```bash
# All demos work on CPU
python3 demos/01_basic_optimizations.py --quick
```

**GPU (Recommended):**
```bash
# GPU-accelerated demos
python3 demos/04_hardware_abstraction.py --quick
```

For detailed setup: [Setup Guide](../docs/setup.md)

**🚀 Start with demo 01 to see immediate 2.8x speedup!**