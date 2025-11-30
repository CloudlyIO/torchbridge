# 🚀 Setup & Installation

**Complete setup guide for KernelPyTorch development and deployment.**

## ⚡ Quick Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU acceleration)

### Basic Setup
```bash
# Clone repository
git clone <repository-url>
cd shahmod

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import kernel_pytorch; print('✅ KernelPyTorch ready!')"
```

## 🖥️ Hardware Configuration

### GPU Setup (NVIDIA)
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Validate GPU setup
python scripts/validate_gpu_setup.py

# Run GPU benchmark
PYTHONPATH=src python benchmarks/simple_benchmark_test.py
```

### Multi-GPU Configuration
```bash
# Check available devices
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Multi-GPU benchmark
PYTHONPATH=src python benchmarks/run_comprehensive_benchmark.py --multi-gpu
```

### CPU Fallback
```bash
# CPU-only mode (automatically detected)
export CUDA_VISIBLE_DEVICES=""
python demos/01_getting_started/optimized_basic_demo.py --quick
```

## 🧪 Validation

### Quick Validation (2-3 minutes)
```bash
# Run all demos
PYTHONPATH=../src python demos/run_all_demos.py --quick

# Basic tests
PYTHONPATH=src python -m pytest tests/ --tb=short -x

# Simple benchmark
PYTHONPATH=src python benchmarks/simple_benchmark_test.py
```

### Comprehensive Validation (10-30 minutes)
```bash
# Full test suite
PYTHONPATH=src python -m pytest tests/

# All benchmarks
PYTHONPATH=src python benchmarks/run_comprehensive_benchmark.py

# Performance profiling
python scripts/profile_tests.py
```

## 🔧 Development Environment

### Environment Variables
```bash
# For accurate benchmarking
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# For deterministic results
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
```

### Development Workflow
```bash
# Setup development environment
pip install -r requirements.txt
pre-commit install  # If using pre-commit hooks

# Run development checks
python scripts/run_tests.py integration
python scripts/cleanup_repo.py  # Clean temporary files
```

## ⚠️ Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch sizes in demos/benchmarks
python demos/01_getting_started/optimized_basic_demo.py --quick --batch-size 2

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### Import Errors
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH=/path/to/shahmod/src:$PYTHONPATH

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### Benchmark Failures
```bash
# CPU-only mode for troubleshooting
export CUDA_VISIBLE_DEVICES=""
PYTHONPATH=src python benchmarks/simple_benchmark_test.py

# Check logs for detailed error information
tail -f benchmarks/results/benchmark_*.log
```

## 🎯 Next Steps

After successful installation:
1. **Try basic demo**: `python demos/01_getting_started/optimized_basic_demo.py --quick`
2. **Run benchmarks**: `python benchmarks/simple_benchmark_test.py`
3. **Explore advanced features**: See `demos/` for progressive examples
4. **Review performance**: Check `BENCHMARKS.md` for detailed results

---

**For development guidelines, see the root-level `CONTRIBUTING.md` file.**