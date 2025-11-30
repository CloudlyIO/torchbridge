# 🧪 PyTorch Optimization Tests

**Comprehensive test suite for validating GPU optimizations.**

## ⚡ Quick Start

```bash
# Run all tests (5-10 minutes)
export PYTHONPATH=src:$PYTHONPATH
python3 -m pytest tests/ -v

# Quick validation (2 minutes)
python3 -m pytest tests/test_basic_functionality.py -v
```

## 📋 Test Modules

| Module | Focus | Runtime |
|--------|-------|---------|
| `test_basic_functionality.py` | Core functionality | 2 min |
| `test_hardware_abstraction.py` | Multi-vendor GPUs | 3 min |
| `test_components.py` | Individual components | 2 min |
| `test_integration.py` | End-to-end workflows | 5 min |

## 🎯 Test Categories

- **Unit Tests** - Fast component validation
- **Integration Tests** - End-to-end workflows
- **Hardware Tests** - GPU-specific functionality
- **Performance Tests** - Regression detection

## 🔧 Requirements

```bash
# Basic testing
pip install pytest

# GPU testing (optional)
nvidia-smi  # Verify CUDA

# Coverage analysis
pip install pytest-cov
python3 -m pytest tests/ --cov=src --cov-report=html
```

## 🧪 Hardware Testing

**CPU Only:**
```bash
export CUDA_VISIBLE_DEVICES=""
python3 -m pytest tests/ -v
```

**GPU:**
```bash
export CUDA_VISIBLE_DEVICES=0
python3 -m pytest tests/test_hardware_abstraction.py -v
```

## 🔧 Troubleshooting

**Common Issues:**
- Clear compilation cache: `rm -rf ~/.cache/torch/`
- Set path: `export PYTHONPATH=src:$PYTHONPATH`
- Memory errors: Use `--quick` flags in tests

For detailed testing guide: [Testing Guide](../docs/testing_guide.md)

**🎯 Run tests to validate 2-6x performance improvements!**