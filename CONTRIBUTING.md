# Contributing to TorchBridge

Guide for developers contributing to cross-backend validation and configuration intelligence for PyTorch.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Git

Optional (for GPU backend development):
- CUDA 12.0+ and NVCC (NVIDIA)
- ROCm 5.6+ (AMD)
- PyTorch/XLA (TPU)

## Development Setup

```bash
# Clone and install
git clone https://github.com/CloudlyIO/torchbridge.git  # update URL when repo goes public
cd torchbridge
pip install -r requirements.txt

# Validate
PYTHONPATH=src python3 -c "import torchbridge; print(f'TorchBridge v{torchbridge.__version__} ready')"

# Run tests
PYTHONPATH=src python3 -m pytest tests/ -q
```

### Editable Install (alternative)

```bash
pip install -e .[dev,all]
```

## Building CUDA Extensions

If you're working on NVIDIA-specific backend code that includes custom CUDA kernels:

```bash
# Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}')"
nvcc --version

# Install build dependencies
pip install pybind11 ninja

# Build extensions
python setup.py build_ext --inplace

# Or editable install
pip install -e .
```

### Build troubleshooting

**nvcc not found:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
```

**Compilation errors:**
```bash
python setup.py clean --all
rm -rf build/ dist/ *.egg-info
python setup.py build_ext --inplace -v
```

**GCC version:** Requires GCC 9+ or Clang 10+ with C++17 support.

## Project Architecture

```
src/torchbridge/
├── backends/          # Vendor-specific backend implementations (NVIDIA, AMD, Trainium, TPU + factory)
├── core/              # Hardware detection, config, architecture enums
├── precision/         # Quantization compatibility matrix + torchao dispatch
├── attention/         # Attention kernel compatibility matrix + dispatcher
├── distributed/       # FSDP config advisor, heterogeneous cluster advisor, topology
├── inference/         # Disaggregated fleet, KV handoff, speculative, phase detection
├── models/            # LLM KV cache advisor
├── checkpoint/        # DCP wrapper with cross-backend metadata normalization
├── adapters/          # Adapter compatibility matrix + config (LoRA/QLoRA method selection)
├── benchmarks/        # 5 claim benchmarks with real measured results
├── testing/           # DivergenceTracer, ToleranceDB, MultiStepTracer, @cross_backend, OTel exporter
├── validation/        # UnifiedValidator — model structure, hardware, numerical stability
├── cli/               # 13 CLI entry points
└── utils/             # Utilities
```

### Key Abstractions

- **`BaseBackend`** -- Abstract interface all backends implement
- **`BackendFactory`** -- Automatic hardware detection and backend creation
- **`TorchBridgeConfig`** -- Unified configuration
- **`UnifiedManager`** -- High-level optimization API
- **`UnifiedValidator`** -- Cross-backend validation

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow naming conventions:
- Classes: `PascalCase` (e.g., `NVIDIABackend`)
- Functions: `snake_case` (e.g., `detect_best_backend`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_SEQUENCE_LENGTH`)

### 3. Test

```bash
# Run full suite
PYTHONPATH=src python3 -m pytest tests/ -q

# Run specific module
PYTHONPATH=src python3 -m pytest tests/test_backends.py -v

# Run single test
PYTHONPATH=src python3 -m pytest tests/test_backends.py::TestNVIDIA::test_device_info -v -s

# Lint
ruff check src/ tests/
```

### 4. Submit PR

Before submitting:
- [ ] Tests pass: `PYTHONPATH=src python3 -m pytest tests/ -q`
- [ ] Linting clean: `ruff check src/ tests/`
- [ ] No hardcoded device assumptions (support CPU fallback)
- [ ] Documentation updated if adding new features
- [ ] Imports verified: test your public API imports

## Contribution Areas

### Backend Development
Improving vendor-specific backends -- better hardware detection, optimization strategies, memory management, precision support.

### Cross-Backend Consistency
Ensuring the unified API behaves identically across backends. Adding validation tests that run on multiple hardware types.

### Distributed Training
Extending tensor/pipeline/data parallelism support across backend types.

### Documentation
Improving guides, adding examples, keeping hardware matrix current.

## Code Quality

- **PEP 8** naming conventions
- **Type hints** on all public APIs
- **Docstrings** for public functions and classes
- **Tests** for new functionality
- **Device-agnostic patterns:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

## Common Issues

### Import Errors
```bash
# Always set PYTHONPATH for development
export PYTHONPATH=src
```

### Test Failures
Platform-specific tests are automatically skipped when hardware isn't available. This is expected behavior.

## Getting Help

- **Issues:** Report bugs via GitHub Issues
- **Discussions:** Technical questions and design discussions
- **Code Review:** Submit PRs for collaborative development
