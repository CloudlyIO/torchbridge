# TorchBridge Test Suite

Comprehensive test suite for validating GPU optimizations.

## Quick Start

```bash
# Run all tests (excluding GPU/slow)
python3 -m pytest tests/ -v -m "not gpu and not slow"

# Run specific test category
python3 -m pytest tests/unit/ -v          # Fast unit tests
python3 -m pytest tests/integration/ -v   # Integration tests
python3 -m pytest tests/backends/ -v      # Backend tests
python3 -m pytest tests/features/ -v      # Feature tests
python3 -m pytest tests/e2e/ -v           # End-to-end tests

# Quick validation
python3 -m pytest tests/unit/ -v --maxfail=3
```

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── README.md
│
├── unit/                    # Fast, isolated tests (<1s each)
│   ├── test_configs.py
│   ├── test_kernel_registry.py
│   ├── test_package_installation.py
│   └── test_performance_tracker.py
│
├── integration/             # Multi-component tests (1-30s)
│   ├── test_backend_integration.py
│   ├── test_backend_unification.py
│   ├── test_distributed_integration.py
│   ├── test_integration.py
│   ├── test_kernel_integration.py
│   ├── test_llm_integration.py
│   ├── test_multimodal_integration.py
│   ├── test_small_model_integration.py
│   └── test_vision_model_integration.py
│
├── backends/                # Hardware backend tests
│   ├── test_nvidia_backend.py
│   ├── test_nvidia_config.py
│   ├── test_amd_backend.py
│   ├── test_intel_backend.py
│   ├── test_tpu_backend.py
│   ├── test_tpu_config.py
│   ├── test_hardware_abstraction.py
│   └── test_custom_kernels.py
│
├── features/                # Feature-specific tests
│   ├── test_advanced.py
│   ├── test_advanced_memory.py
│   ├── test_attention_compatibility.py
│   ├── test_auto_optimization.py
│   ├── test_compiler.py
│   ├── test_distributed_scale.py
│   ├── test_dynamic_shapes.py
│   ├── test_flex_attention.py
│   ├── test_fp8_native.py
│   ├── test_fp8_training.py
│   ├── test_moe.py
│   ├── test_neural_operator_fusion.py
│   ├── test_next_gen.py
│   └── test_ultra_precision.py
│
├── e2e/                     # End-to-end tests
│   ├── test_deployment.py
│   ├── test_monitoring.py
│   └── test_serving.py
│
├── benchmarks_tests/        # Benchmark validation tests
│   ├── test_advanced_memory_benchmarks.py
│   ├── test_cli_benchmarks.py
│   └── test_next_gen_benchmarks.py
│
├── cli/                     # CLI command tests
├── cloud_testing/           # Cloud platform tests
├── patterns/                # Optimization pattern tests
└── regression/              # Regression detection tests
```

## Test Markers

Use pytest markers to run specific test categories:

```bash
# By test type
pytest -m unit           # Fast unit tests
pytest -m integration    # Integration tests
pytest -m e2e            # End-to-end tests
pytest -m benchmark      # Performance benchmarks

# By hardware requirement
pytest -m gpu            # Requires CUDA GPU
pytest -m tpu            # Requires TPU
pytest -m amd            # Requires AMD GPU (ROCm)
pytest -m intel          # Requires Intel GPU (XPU)
pytest -m fp8            # Requires FP8 hardware (H100+)

# By duration
pytest -m slow           # Long-running tests
pytest -m "not slow"     # Quick tests only
```

## Hardware-Specific Testing

**CPU Only:**
```bash
pytest tests/unit/ tests/integration/ -v -m "not gpu"
```

**NVIDIA GPU:**
```bash
pytest tests/backends/test_nvidia_backend.py -v
pytest tests/features/test_fp8_training.py -v  # H100+ required
```

**Multi-GPU:**
```bash
export CUDA_VISIBLE_DEVICES=0,1
pytest tests/features/test_distributed_scale.py -v
```

## Coverage

```bash
pytest tests/ --cov=src/torchbridge --cov-report=html
open htmlcov/index.html
```

## Troubleshooting

- Clear cache: `rm -rf ~/.cache/torch/ __pycache__`
- Memory errors: Use `pytest -x` to stop on first failure
- Check skips: `pytest -v -rs` to see skip reasons
