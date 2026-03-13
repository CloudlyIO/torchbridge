# TorchBridge Benchmarks

Benchmarking suite for validating cross-backend performance across NVIDIA, AMD, Trainium, and TPU.

## Quick Start

```bash
# Backend comparison
python3 benchmarks/backend_comparison.py --quick

# Unified runner
python3 benchmarks/unified_runner.py --quick
```

## Directory Structure

```
benchmarks/
├── README.md
├── __init__.py
│
├── Backend Benchmarks
│   ├── backend_comparison.py        # Cross-backend comparison
│   ├── nvidia_integration_benchmark.py  # NVIDIA-specific
│   ├── amd_integration_benchmark.py # AMD-specific
│   └── tpu_integration_benchmark.py # TPU-specific
│
├── Feature Benchmarks
│   └── cli_performance_benchmark.py
│
├── framework/                       # Benchmark infrastructure
│   ├── benchmark_runner.py
│   ├── metrics_collector.py
│   ├── timing_utils.py
│   ├── analysis_engine.py
│   ├── cutting_edge_baselines.py
│   └── results_analyzer.py
│
├── regression/                      # Regression testing
│   ├── baseline_manager.py
│   ├── regression_detector.py
│   ├── threshold_manager.py
│   └── reporting/
│
└── results/                         # Benchmark results (gitignored)
    └── *.json
```

## Available Benchmarks

| Benchmark | Focus | Hardware |
|-----------|-------|----------|
| `nvidia_integration_benchmark.py` | NVIDIA backend | NVIDIA GPU |
| `amd_integration_benchmark.py` | AMD backend | AMD GPU |
| `tpu_integration_benchmark.py` | TPU backend | Cloud TPU |
| `backend_comparison.py` | Cross-backend | Any GPU |
| `cli_performance_benchmark.py` | CLI latency | Any |
| `unified_runner.py` | All-in-one runner | Any |

## Running Benchmarks

```bash
# Unified runner with options
python3 benchmarks/unified_runner.py --quick
python3 benchmarks/unified_runner.py --backends nvidia,amd
python3 benchmarks/unified_runner.py --output results/

# Backend-specific
python3 benchmarks/nvidia_integration_benchmark.py --quick
python3 benchmarks/amd_integration_benchmark.py
python3 benchmarks/tpu_integration_benchmark.py
```

## Output

Results are saved to `benchmarks/results/` (gitignored).

For detailed analysis, see [Hardware Matrix](../docs/reference/hardware-matrix.md).
