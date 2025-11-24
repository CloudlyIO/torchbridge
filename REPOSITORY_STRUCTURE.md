# 📁 Repository Structure Guide

**Clear organization and navigation guide for the PyTorch optimization framework.**

## 🎯 Overview

This repository is organized for **ease of use**, **development efficiency**, and **production deployment**:

- **`src/`** - Core framework implementation
- **`demos/`** - Production-ready examples and tutorials
- **`benchmarks/`** - Performance comparison and validation
- **`tests/`** - Comprehensive testing framework
- **`docs/`** - Technical documentation
- **`scripts/`** - Setup and validation utilities

## 📂 Detailed Structure

```
shahmod/
├── 🚀 Quick Start Files
│   ├── README.md                    # Main project overview
│   ├── BENCHMARK_QUICKSTART.md     # Simple benchmark instructions
│   ├── CUDA_SETUP_GUIDE.md         # GPU/CUDA setup instructions
│   └── requirements.txt            # Python dependencies
│
├── 🔧 Core Framework
│   └── src/kernel_pytorch/          # Main optimization framework
│       ├── compiler_integration/    # FlashLight, PyGraph, TorchInductor
│       ├── next_gen_optimizations/  # 2024-2025 cutting-edge techniques
│       ├── distributed_scale/       # Multi-GPU and cluster optimization
│       ├── testing_framework/       # Hardware simulation and validation
│       └── utils/                   # Utilities and helper functions
│
├── 🎓 Examples & Tutorials
│   └── demos/                       # **START HERE for learning**
│       ├── 01_getting_started/      # Basic optimization fundamentals
│       ├── 02_compiler_optimizations/ # FlashLight and PyGraph demos
│       ├── 03_advanced_attention/   # Advanced attention patterns
│       ├── 04_gpu_integration/      # GPU kernel optimization
│       ├── 05_next_generation/      # 2025+ cutting-edge techniques
│       ├── 06_testing_framework/    # Testing and validation examples
│       └── 07_production_ready/     # Production deployment patterns
│
├── 🏁 Performance & Validation
│   ├── benchmarks/                  # **Performance comparison framework**
│   │   ├── framework/              # Standard benchmark infrastructure
│   │   ├── next_gen/               # Cutting-edge comparison (2024-2025)
│   │   └── README.md               # Benchmark documentation
│   │
│   └── tests/                       # **Comprehensive test suite**
│       ├── test_*.py               # Categorized test modules
│       └── test_configs.py         # Test configuration framework
│
├── 📚 Documentation
│   └── docs/                        # **Technical documentation**
│       ├── implementation_guide.md  # Framework implementation details
│       ├── advanced_optimizations_guide.md # Advanced usage patterns
│       ├── EXTERNAL_REFERENCES.md  # Research papers and resources
│       └── TECHNICAL_OVERVIEW.md   # Architecture overview
│
├── 🛠️ Development Tools
│   ├── scripts/                     # **Development and validation scripts**
│   │   ├── validate_gpu_setup.py   # GPU/CUDA validation tool
│   │   ├── cleanup_repo.py         # Repository maintenance and cleanup
│   │   ├── profile_tests.py        # Performance profiling utility
│   │   └── test_all_changes.py     # Comprehensive validation testing
│   │
│   └── 🔧 Configuration Files
│       ├── pytest.ini              # Test configuration
│       ├── pyproject.toml          # Python project configuration
│       └── setup.py                # Package installation setup
│
└── 📋 Project Management
    ├── OPTIMIZATION_ROADMAP_2025_2026.md # Technology roadmap
    ├── TESTING_STRATEGY.md              # Testing methodology
    ├── CLAUDE.md                        # AI development notes
    └── LICENSE                          # MIT license
```

## 🚀 Getting Started Navigation

### **New Users - Start Here:**
```bash
# 1. Read overview
cat README.md

# 2. Validate setup
python3 scripts/validate_gpu_setup.py

# 3. Quick benchmark test
python3 benchmarks/simple_benchmark_test.py

# 4. Try first demo
python3 demos/01_getting_started/optimized_basic_demo.py --quick
```

### **Developers - Core Framework:**
```bash
# Explore core implementations
ls src/kernel_pytorch/

# Review architecture
cat docs/TECHNICAL_OVERVIEW.md

# Run comprehensive tests
python3 run_tests.py integration
```

### **Researchers - Cutting-Edge Features:**
```bash
# Latest optimization techniques
ls src/kernel_pytorch/next_gen_optimizations/

# Benchmark against state-of-the-art
python3 benchmarks/next_gen/demo_cutting_edge_benchmark.py --quick

# Research roadmap
cat OPTIMIZATION_ROADMAP_2025_2026.md
```

## 📁 Directory Purposes

### **Core Implementation (`src/`)**
- **Purpose**: Production-ready optimization framework
- **Contents**: Compiler integration, advanced optimizations, testing infrastructure
- **Usage**: Import as `from kernel_pytorch.* import ...`

### **Learning & Examples (`demos/`)**
- **Purpose**: Educational examples and production patterns
- **Organization**: Progressive complexity (01 → 07)
- **Usage**: Run individual demos to learn specific techniques

### **Performance Validation (`benchmarks/`)**
- **Purpose**: Compare against industry standards and state-of-the-art
- **Key Files**:
  - `simple_benchmark_test.py` - Quick validation
  - `next_gen/demo_cutting_edge_benchmark.py` - Latest comparisons
- **Usage**: Validate performance claims and regressions

### **Quality Assurance (`tests/`)**
- **Purpose**: Comprehensive testing across all components
- **Organization**: Categorized by functionality and complexity
- **Usage**: `python3 run_tests.py [unit|integration|stress]`

### **Documentation (`docs/`)**
- **Purpose**: Technical documentation and implementation guides
- **Audience**: Framework developers and advanced users
- **Usage**: Reference for implementation details and best practices

### **Development Tools (`scripts/`)**
- **Purpose**: Setup validation, profiling, and development utilities
- **Key Tools**:
  - `scripts/validate_gpu_setup.py` - Setup validation
  - `scripts/cleanup_repo.py` - Repository maintenance
  - `scripts/profile_tests.py` - Performance profiling
  - `scripts/test_all_changes.py` - Comprehensive validation
- **Usage**: Development workflow automation and maintenance

## 🧭 Navigation Tips

### **By Experience Level:**

**🟢 Beginner**
```
README.md → BENCHMARK_QUICKSTART.md → demos/01_getting_started/
```

**🟡 Intermediate**
```
demos/02_compiler_optimizations/ → benchmarks/simple_benchmark_test.py
```

**🟠 Advanced**
```
src/kernel_pytorch/ → docs/advanced_optimizations_guide.md → tests/
```

**🔴 Research/Cutting-Edge**
```
benchmarks/next_gen/ → src/kernel_pytorch/next_gen_optimizations/ → OPTIMIZATION_ROADMAP_2025_2026.md
```

### **By Use Case:**

**📊 Performance Benchmarking**
```bash
# Quick validation
python3 benchmarks/simple_benchmark_test.py

# Industry comparison
python3 benchmarks/next_gen/demo_cutting_edge_benchmark.py --quick

# Custom benchmarking
# See: benchmarks/README.md
```

**🧪 Development & Testing**
```bash
# Setup validation
python3 scripts/validate_gpu_setup.py

# Run tests
python3 run_tests.py integration

# Profile performance
python3 scripts/profile_tests.py
```

**🎓 Learning & Examples**
```bash
# Start with basics
python3 demos/01_getting_started/optimized_basic_demo.py

# Advanced patterns
python3 demos/02_compiler_optimizations/optimized_flashlight_demo.py

# Cutting-edge techniques
python3 demos/05_next_generation/neuromorphic_simulation_demo.py
```

## 📋 File Naming Conventions

### **Demo Files**
- **`optimized_*.py`** - Production-ready examples with benchmarking
- **`demo_*.py`** - Educational examples focusing on concepts
- **`*_demo.py`** - Standard demonstration scripts

### **Test Files**
- **`test_*.py`** - Test modules organized by functionality
- **`test_configs.py`** - Test configuration and data generation

### **Documentation Files**
- **`*.md`** - Markdown documentation
- **`*_guide.md`** - Step-by-step guides
- **`*_strategy.md`** - Methodology and approach documents

### **Script Files**
- **`validate_*.py`** - Validation and setup scripts
- **`profile_*.py`** - Performance profiling utilities
- **`run_*.py`** - Execution and orchestration scripts

## 🔄 Maintenance & Updates

**Automatic Cleanup:**
```bash
# Clean test results (keeps latest 5)
find test_results/ -name "*.json" | sort | head -n -5 | xargs rm -f

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
```

**Documentation Updates:**
- Update this file when adding new directories
- Keep README.md synchronized with major changes
- Update roadmap when completing Priority items

## 🎯 Design Principles

1. **Progressive Discovery** - Start simple, increase complexity gradually
2. **Clear Separation** - Implementation, examples, tests, docs are distinct
3. **Self-Documenting** - Directory and file names indicate purpose
4. **Performance Focus** - Easy to find and run performance validation
5. **Development Friendly** - Tools and scripts support efficient workflows

---

**🚀 This structure supports both learning the framework and contributing to its development efficiently.**