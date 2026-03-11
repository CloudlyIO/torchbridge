# TorchBridge Test Suite

Comprehensive test suite for TorchBridge cross-backend validation and configuration intelligence — 1,950+ tests.

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
python3 -m pytest tests/distributed/ -v   # Distributed tests
python3 -m pytest tests/cli/ -v           # CLI tests

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
│
├── models/                  # Real-model cross-backend tests (@real_model marker)
│   ├── test_qwen3_hal.py
│   ├── test_deepseek_r1_hal.py
│   ├── test_dinov2_hal.py
│   ├── test_minilm_hal.py
│   ├── test_whisper_hal.py
│   └── test_qwen25_vl_hal.py
│
├── integration/             # Multi-component tests (1-30s)
│   ├── test_backend_integration.py
│   ├── test_backend_unification.py
│   ├── test_distributed_integration.py
│   ├── test_full_pipeline.py
│   ├── test_kernel_integration.py
│   └── test_llm_integration.py
│
├── backends/                # Hardware backend tests
│   ├── test_amd_backend.py
│   ├── test_blackwell_detection.py
│   ├── test_cdna4_detection.py
│   ├── test_custom_kernels.py
│   ├── test_hardware_abstraction.py
│   ├── test_nvidia_backend.py
│   ├── test_nvidia_config.py
│   ├── test_tpu_backend.py
│   └── test_tpu_config.py
│
├── features/                # Feature-specific tests
│   ├── test_auto_optimization.py
│   └── test_distributed_scale.py
│
├── e2e/                     # End-to-end tests
│
├── distributed/             # Distributed training tests
│   ├── test_distributed_llama.py
│   └── test_pipeline_parallel.py
│
├── benchmark/               # Benchmark validation tests
│   ├── test_advanced_memory_benchmarks.py
│   ├── test_cli_benchmarks.py
│   └── test_next_gen_benchmarks.py
│
├── cli/                     # CLI command tests
│
├── regression/              # Regression detection tests
│   ├── test_baseline_manager.py
│   ├── test_regression_detector.py
│   └── test_threshold_manager.py
│
└── cloud_testing/           # Cloud platform test harnesses
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
pytest -m fp8            # Requires FP8 hardware (H100+)

# By model type
pytest -m real_model     # Real pretrained model tests

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
pytest tests/backends/test_blackwell_detection.py -v
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
