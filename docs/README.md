# 📚 Documentation

**Complete guide to PyTorch GPU optimization framework with 2-6x performance improvements.**

## 🚀 Quick Start

| Step | Document | Time | Purpose |
|------|----------|------|---------|
| 1️⃣ | [Installation Guide](guides/installation.md) | 5 min | Install and verify |
| 2️⃣ | [Quick Demo](../demos/README.md) | 2 min | See optimizations working |
| 3️⃣ | [Benchmarks](../BENCHMARKS.md) | 5 min | Validate claims |

## 📖 Documentation Structure

### **📋 Guides** - Setup & Development
- **[Installation Guide](guides/installation.md)** - Complete installation instructions
- **[Quick Start Guide](guides/quickstart.md)** - Get started in 5 minutes
- **[Testing Guide](guides/testing_guide.md)** - Comprehensive testing and benchmarking
- **[CUDA Setup](guides/cuda_setup.md)** - Detailed CUDA/Triton configuration
- **[Docker Guide](guides/docker.md)** - Containerized development and deployment
- **[Cloud Testing](guides/cloud_testing_guide.md)** - AWS/Azure/GCP deployment

### **⚡ Capabilities** - Technical Documentation
- **[Architecture](capabilities/architecture.md)** - Framework design and optimization strategies
- **[Hardware Abstraction](capabilities/hardware.md)** - Multi-vendor GPU support (NVIDIA/AMD/Intel/Custom)
- **[Performance Analysis](capabilities/performance.md)** - Detailed benchmarking and optimization results
- **[Dynamic Shape Bucketing](capabilities/dynamic_shape_bucketing.md)** - Variable input optimization
- **[Core Components](capabilities/core_components.md)** - Core optimization modules
- **[Hardware Kernels](capabilities/hardware_kernels.md)** - Low-level kernel implementations
- **[CLI Reference](capabilities/cli_reference.md)** - Command-line interface documentation
- **[External References](capabilities/references.md)** - Links to papers, documentation, tools

### **🗺️ Roadmaps** - Planning & Development
- **[Main Roadmap](roadmaps/roadmap.md)** - Development roadmap and future plans
- **[Immediate Tasks](roadmaps/immediate_tasks.md)** - Current priority tasks
- **[Performance Testing Plan](roadmaps/performance_regression_testing_plan.md)** - Testing strategy
- **[NVIDIA Optimization Roadmap](roadmaps/nvidia_optimization_roadmap.md)** - NVIDIA-specific optimizations
- **[TPU Integration Roadmap](roadmaps/tpu_integration_roadmap.md)** - Google TPU support planning

## 🎯 By Use Case

### **New Users** (Start here!)
1. [Installation Guide](guides/installation.md) → [Quick Start](guides/quickstart.md) → [Demo Suite](../demos/) → [Benchmarks](../BENCHMARKS.md)
2. Expected: 5-minute setup, verified demo results: 5/5 success in 54.6s

### **Developers**
1. [Architecture](capabilities/architecture.md) → [API Reference](../API.md) → [Testing Guide](guides/testing_guide.md)
2. Focus: Framework design, development workflow, comprehensive testing

### **Hardware Engineers**
1. [Hardware Guide](capabilities/hardware.md) → [Cloud Testing](guides/cloud_testing_guide.md) → [Performance Analysis](capabilities/performance.md)
2. Focus: Multi-vendor support, deployment, hardware-specific optimization

### **Performance Engineers**
1. [Performance Analysis](capabilities/performance.md) → [Benchmarks](../BENCHMARKS.md) → [Testing Guide](guides/testing_guide.md)
2. Focus: Detailed performance analysis, statistical validation, regression testing

## ✨ Key Features

- **🚀 Advanced Attention**: Ring, Sparse, Context Parallel - up to 6.1x speedup
- **🔥 FP8 Training**: Production-ready with 2x H100 speedup
- **🖥️ Hardware Abstraction**: NVIDIA/AMD/Intel/TPU/ASIC unified support
- **📊 Comprehensive Benchmarking**: Statistical validation against Flash Attention 3, vLLM, Mamba

---

**🎯 Ready to achieve 2-6x PyTorch performance improvements!**