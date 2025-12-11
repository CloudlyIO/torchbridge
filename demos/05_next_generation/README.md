# Next-Generation Optimizations Demos

This directory contains comprehensive demonstrations of the latest PyTorch optimization techniques implemented in the `kernel_pytorch.optimizations.next_gen` module.

## 🚀 Available Demos

### Individual Optimization Demos

1. **`advanced_flex_attention_demo.py`**
   - FlashLight compiler framework
   - GQA (Grouped Query Attention) optimization
   - Paged attention for inference
   - Automatic kernel compilation

2. **`ultra_precision_demo.py`**
   - FP4, NVFP4, MXFP quantization formats
   - Information entropy-based precision allocation
   - Adaptive precision allocation strategies

3. **`structured_sparsity_demo.py`**
   - 2:4 structured sparsity patterns
   - Dynamic sparsity optimization
   - Hardware-accelerated sparse operations

### Unified Demo Runner

- **`run_next_gen_demos.py`**
  - Runs all individual demos
  - Integration testing between optimizations
  - Performance metrics collection
  - Production readiness assessment

## 📋 Usage

### Quick Demo Run
```bash
# From project root
cd demos/05_next_generation
PYTHONPATH=../../src python3 run_next_gen_demos.py --device cpu --quick
```

### Individual Demos
```bash
# Advanced FlexAttention
PYTHONPATH=../../src python3 advanced_flex_attention_demo.py --device cpu --quick

# Ultra-Precision Optimization
PYTHONPATH=../../src python3 ultra_precision_demo.py --device cpu --quick

# Structured Sparsity
PYTHONPATH=../../src python3 structured_sparsity_demo.py --device cpu --quick
```

### Command Options
- `--device auto|cpu|cuda` - Device selection
- `--quick` - Fast execution mode (reduced iterations)
- `--output filename.json` - Save results to file

## 📊 Expected Results

- **Demo Success Rate**: 100% (3/3 demos)
- **Integration Tests**: 100% (2/2 tests)
- **Performance Benefits**: Up to 1.39x speedup, 12.5% memory savings
- **Status**: DEVELOPMENT READY

## 🔗 Related

- **Tests**: `/tests/test_next_gen*.py`
- **Benchmarks**: `/tests/test_next_gen_benchmarks.py`
- **Source Code**: `/src/kernel_pytorch/optimizations/next_gen/`