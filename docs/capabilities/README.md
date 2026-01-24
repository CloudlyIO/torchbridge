# Technical Capabilities

> **Version**: v0.4.18 | Deep technical documentation for advanced users and contributors

KernelPyTorch provides comprehensive GPU optimization capabilities designed for production ML workloads.

## Core Architecture

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | Framework design, optimization strategies, module hierarchy |
| [Hardware Abstraction](hardware.md) | Multi-vendor GPU/TPU support with unified APIs |
| [Core Components](core_components.md) | Optimization module reference and internals |

## Performance & Optimization

| Document | Description |
|----------|-------------|
| [Performance Benchmarks](performance.md) | Measured results across NVIDIA, AMD, Intel, TPU |
| [Hardware Kernels](hardware_kernels.md) | Custom CUDA/Triton kernel implementations |
| [Performance Regression Testing](performance_regression_testing.md) | Automated monitoring and regression detection |
| [Hardware Matrix](HARDWARE_MATRIX.md) | Complete hardware support matrix |

## Advanced Features

| Document | Description |
|----------|-------------|
| [Dynamic Shape Bucketing](dynamic_shape_bucketing.md) | Variable input optimization for 3x speedup |
| [CLI Reference](cli_reference.md) | Command-line interface for optimization and benchmarking |

## External Resources

- [References](references.md) - Papers, tools, external links

---

**See Also**: [Getting Started](../getting-started/README.md) | [Backends](../backends/README.md) | [Guides](../guides/README.md)
