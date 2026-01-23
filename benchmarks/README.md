# KernelPyTorch Benchmarks

Benchmarking suite for validating GPU optimizations.

## Quick Start

```bash
# Quick validation
python3 benchmarks/simple_benchmark_test.py

# Backend comparison
python3 benchmarks/backend_comparison_benchmark.py --quick

# Unified runner
python3 benchmarks/unified_runner.py --quick
```

## Directory Structure

```
benchmarks/
├── README.md
├── __init__.py
│
├── Core Benchmarks
│   ├── simple_benchmark_test.py     # Quick validation
│   ├── quick_benchmark.py           # Fast benchmark suite
│   ├── unified_runner.py            # All-in-one runner
│   ├── run_comprehensive_benchmark.py
│   └── regression_benchmark.py      # Regression detection
│
├── Backend Benchmarks
│   ├── backend_comparison.py        # Cross-backend comparison
│   ├── backend_comparison_benchmark.py
│   ├── nvidia_config_benchmarks.py  # NVIDIA-specific
│   ├── nvidia_integration_benchmark.py
│   ├── amd_integration_benchmark.py # AMD-specific
│   ├── amd_optimization_benchmark.py
│   ├── intel_benchmark.py           # Intel-specific
│   └── tpu_integration_benchmark.py # TPU-specific
│
├── Feature Benchmarks
│   ├── custom_kernel_benchmark.py
│   ├── hardware_abstraction_benchmark.py
│   ├── dynamic_shapes_benchmark.py
│   └── cli_performance_benchmark.py
│
├── framework/                       # Benchmark infrastructure
│   ├── benchmark_runner.py
│   ├── metrics_collector.py
│   ├── timing_utils.py
│   ├── analysis_engine.py
│   └── baseline_implementations.py
│
├── regression/                      # Regression testing
│   ├── baseline_manager.py
│   ├── regression_detector.py
│   ├── threshold_manager.py
│   └── reporting/
│
├── next_gen/                        # Cutting-edge benchmarks
│   ├── demo_cutting_edge_benchmark.py
│   ├── enhanced_benchmark_runner.py
│   └── cutting_edge_baselines.py
│
├── models/                          # Model-specific benchmarks
│   └── small_model_benchmark.py
│
├── configs/                         # Benchmark configurations
│   └── *.yaml
│
└── results/                         # Benchmark results (gitignored)
    └── *.json
```

## Available Benchmarks

| Benchmark | Focus | Hardware |
|-----------|-------|----------|
| `simple_benchmark_test.py` | Quick validation | Any |
| `nvidia_config_benchmarks.py` | NVIDIA backend | NVIDIA GPU |
| `amd_integration_benchmark.py` | AMD backend | AMD GPU |
| `intel_benchmark.py` | Intel backend | Intel XPU |
| `tpu_integration_benchmark.py` | TPU backend | Cloud TPU |
| `backend_comparison_benchmark.py` | Cross-backend | Any GPU |
| `custom_kernel_benchmark.py` | CUDA kernels | CUDA GPU |
| `hardware_abstraction_benchmark.py` | Hardware layer | Any |
| `dynamic_shapes_benchmark.py` | Dynamic shapes | Any |
| `regression_benchmark.py` | Regression detection | Any |

## Running Benchmarks

```bash
# Quick validation (30 seconds)
python3 benchmarks/simple_benchmark_test.py

# Full benchmark suite
python3 benchmarks/run_comprehensive_benchmark.py

# Unified runner with options
python3 benchmarks/unified_runner.py --quick
python3 benchmarks/unified_runner.py --backends nvidia,amd
python3 benchmarks/unified_runner.py --output results/

# Backend-specific
python3 benchmarks/nvidia_config_benchmarks.py --quick
python3 benchmarks/amd_integration_benchmark.py
python3 benchmarks/tpu_integration_benchmark.py
```

## Output

Results are saved to `benchmarks/results/` (gitignored).

For detailed analysis, see [BENCHMARKS.md](../BENCHMARKS.md).
