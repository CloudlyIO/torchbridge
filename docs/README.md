# KernelPyTorch Documentation

> Production-ready PyTorch GPU optimization framework
> Current Version: v0.4.18

## Quick Navigation

### 🚀 Getting Started
- [Installation](getting-started/installation.md) - Setup & dependencies
- [Quickstart](getting-started/quickstart.md) - 5-minute tutorial
- [Troubleshooting](getting-started/troubleshooting.md) - Common issues

### 📚 User Guides
- [All Guides](guides/README.md) - Deployment, testing, model optimization
- [Small Models](guides/small_model_guide.md) - BERT, GPT-2, DistilBERT
- [Vision Models](guides/vision_model_guide.md) - ResNet, ViT, Stable Diffusion

### ⚙️ Hardware Backends
- [Backend Selection](backends/README.md) - Choose the right backend
- [NVIDIA](backends/nvidia.md) | [AMD](backends/amd.md) | [Intel](backends/intel.md) | [TPU](backends/tpu.md)

### 🔬 Technical Deep-Dives
- [Capabilities Overview](capabilities/README.md)
- [Architecture](capabilities/architecture.md) - Framework design
- [Performance](capabilities/performance.md) - Benchmarks & optimization

### ☁️ Cloud Deployment
- [Cloud Guide](cloud-deployment/README.md) - AWS, GCP, Azure deployment

### 📋 Project Planning
- [Roadmap](planning/README.md) - Development roadmap & future features
- Internal strategic documents

---

## Quick Start Guide

**New Users** (Start here!)
1. [Installation Guide](getting-started/installation.md) → [Quick Start](getting-started/quickstart.md) → [Demo Suite](../demos/) → [Benchmarks](../BENCHMARKS.md)
2. Expected: 5-minute setup, verified demo results

**Developers**
1. [Architecture](capabilities/architecture.md) → [API Reference](../API.md) → [Testing Guide](guides/testing_guide.md)
2. Focus: Framework design, development workflow, comprehensive testing

**Hardware Engineers**
1. [Hardware Guide](capabilities/hardware.md) → [Cloud Deployment](cloud-deployment/README.md) → [Performance Analysis](capabilities/performance.md)
2. Focus: Multi-vendor support, deployment, hardware-specific optimization

**Performance Engineers**
1. [Performance Analysis](capabilities/performance.md) → [Regression Testing](capabilities/performance_regression_testing.md) → [Testing Guide](guides/testing_guide.md)
2. Focus: Performance optimization, automated regression detection

---

## Key Features

- **🚀 Advanced Attention**: Ring, Sparse, Context Parallel - up to 6.1x speedup
- **🔥 FP8 Training**: Production-ready with 2x H100 speedup
- **🖥️ Hardware Abstraction**: NVIDIA/AMD/Intel/TPU/ASIC unified support
- **📊 Comprehensive Benchmarking**: Statistical validation against Flash Attention 3, vLLM, Mamba

---

**Version**: v0.4.18 | **License**: MIT | **Last Updated**: Jan 22, 2026
