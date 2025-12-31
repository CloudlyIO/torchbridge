# 🏁 PyTorch Optimization Benchmarks (v0.3.3)

**Benchmarking suite for validating GPU optimizations.**

## ⚡ Quick Start

```bash
# Quick validation (30 seconds)
PYTHONPATH=src python3 benchmarks/simple_benchmark_test.py

# NVIDIA backend benchmarks
PYTHONPATH=src python3 benchmarks/nvidia_config_benchmarks.py --quick

# Backend comparison
PYTHONPATH=src python3 benchmarks/backend_comparison_benchmark.py --quick
```

## 📊 Available Benchmarks

| Benchmark | Focus | Hardware |
|-----------|-------|----------|
| `simple_benchmark_test.py` | Quick validation | Any |
| `nvidia_config_benchmarks.py` | NVIDIA backend performance | NVIDIA GPU |
| `nvidia_integration_benchmark.py` | NVIDIA integration tests | NVIDIA GPU |
| `tpu_integration_benchmark.py` | TPU backend performance | Cloud TPU |
| `backend_comparison_benchmark.py` | Cross-backend comparison | Any GPU |
| `custom_kernel_benchmark.py` | Custom CUDA kernels | CUDA GPU |
| `hardware_abstraction_benchmark.py` | Hardware layer performance | Any |
| `dynamic_shapes_benchmark.py` | Dynamic shape handling | Any |
| `regression_benchmark.py` | Performance regression detection | Any |
| `cli_performance_benchmark.py` | CLI tool performance | Any |

## 🎯 Benchmark Categories

- **Performance** - Latency, throughput, memory usage
- **Integration** - Backend-specific optimization validation
- **Regression** - Detect performance regressions across versions
- **Scalability** - Batch size, sequence length scaling

## 🔧 Configuration

**Hardware Support:**
- NVIDIA GPU: RTX 4090, A100, H100, Blackwell
- Cloud TPU: v4, v5e, v5p, v6e
- CPU: Fallback for non-GPU systems

## 🚀 Running Benchmarks

```bash
# Comprehensive benchmark run
PYTHONPATH=src python3 benchmarks/run_comprehensive_benchmark.py

# Unified runner (all benchmarks)
PYTHONPATH=src python3 benchmarks/unified_runner.py --quick

# Quick benchmark
PYTHONPATH=src python3 benchmarks/quick_benchmark.py
```

## 📈 Output

Benchmarks produce JSON results in `local/benchmark_results/` (gitignored).

For detailed results: [BENCHMARKS.md](../BENCHMARKS.md)

**🎯 Run benchmarks to validate optimization improvements!**