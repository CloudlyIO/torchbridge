# 🧪 Testing & Benchmarking Guide

**Comprehensive testing strategy and benchmarking framework for PyTorch GPU optimization validation.**

## 🎯 Overview

This guide covers:
1. **Multi-Tiered Testing Strategy** - Unit, integration, and stress tests
2. **Benchmarking Framework** - Compare against state-of-the-art implementations
3. **Performance Validation** - Statistical analysis and regression detection
4. **CI/CD Integration** - Automated testing and quality gates

## 🚀 Quick Start

### Validate Unified Framework (30 seconds)
```bash
# Test unified architecture v0.3.3 with production backends
PYTHONPATH=src python3 -c "
from kernel_pytorch import KernelPyTorchConfig, UnifiedManager
from kernel_pytorch.validation import UnifiedValidator
print('✅ Unified architecture v0.3.3 with production backends imports successful')
"

# Quick unified validation test
PYTHONPATH=src python3 -c "
from kernel_pytorch.validation import UnifiedValidator
validator = UnifiedValidator()
print('✅ Unified validation framework ready')
"
```

### NVIDIA Hardware Integration Test (1 minute)
```bash
# Quick NVIDIA integration validation
python3 scripts/test_nvidia_integration.py --quick

# Comprehensive NVIDIA integration test
python3 scripts/test_nvidia_integration.py
```

### Run Comprehensive Benchmarks (5 minutes)
```bash
# Compare against cutting-edge implementations
cd demos && PYTHONPATH=../src python3 precision/adaptive.py --validate
cd demos && PYTHONPATH=../src python3 attention/fusion.py --validate
cd demos && PYTHONPATH=../src python3 experimental/ultra_precision.py --validate
```

## 🎯 NVIDIA Hardware Testing

### **Test NVIDIA Integration (v0.2.3)**

The NVIDIA integration test suite validates hardware detection, configuration, and optimization:

```bash
# Standalone execution
python3 scripts/test_nvidia_integration.py           # Full test suite
python3 scripts/test_nvidia_integration.py --quick   # Quick validation

# As pytest
PYTHONPATH=src pytest scripts/test_nvidia_integration.py -v

# Individual components
PYTHONPATH=src python3 -m pytest tests/test_nvidia_config.py -v          # Unit tests
PYTHONPATH=src python3 demos/nvidia_configuration_demo.py --quick        # Demo test
PYTHONPATH=src python3 benchmarks/nvidia_config_benchmarks.py --quick    # Benchmark test
```

**What it tests:**
- ✅ Hardware architecture detection (H100, Blackwell, Ampere, Pascal)
- ✅ Configuration system integration
- ✅ FP8 training enablement for supported hardware
- ✅ Performance validation (<1ms config creation)
- ✅ Serialization/deserialization
- ✅ Integration with unified validation framework

**Expected results:**
- **CPU (no GPU)**: Architecture=`pascal`, FP8=`False`
- **H100/Hopper**: Architecture=`hopper`, FP8=`True`, TensorCore=`4`
- **A100/Ampere**: Architecture=`ampere`, FP8=`False`, TensorCore=`3`

## 📊 Test Categories

### **🚀 Unit Tests** (Development Speed)
- **Purpose**: Fast feedback during development
- **Target**: < 1 second per test
- **Data Scale**: Small but functionally representative
- **Usage**: Individual component validation

```bash
# Run unit tests
PYTHONPATH=src python3 -m pytest tests/test_nvidia_config.py -v              # NVIDIA config tests
PYTHONPATH=src python3 -m pytest tests/test_basic_functionality.py -v        # Basic functionality
python3 -m pytest tests/test_components/ -v
```

**Example Unit Test:**
```python
def test_fused_gelu_correctness():
    """Test FusedGELU produces correct results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Small test data
    x = torch.randn(32, 768, device=device, dtype=torch.float16)

    # Reference implementation
    reference = torch.nn.functional.gelu(x)

    # Optimized implementation
    fused_gelu = FusedGELU()
    optimized = fused_gelu(x)

    # Numerical precision check
    assert torch.allclose(reference, optimized, atol=1e-3, rtol=1e-3)
```

### **🔗 Integration Tests** (Realistic Validation)
- **Purpose**: Validate realistic scenarios
- **Target**: 5-30 seconds per test
- **Data Scale**: Production-representative
- **Usage**: End-to-end workflow validation

```bash
# Run integration tests
PYTHONPATH=src python3 -m pytest scripts/test_nvidia_integration.py -v       # NVIDIA integration suite
python3 -m pytest tests/test_integration/ -v                                 # System integration
python3 -m pytest tests/test_distributed_scale.py -v                         # Distributed integration
```

**Example Integration Test:**
```python
def test_multi_layer_optimization():
    """Test complete model optimization pipeline"""
    # Create realistic model
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 3072),
        FusedGELU(),
        torch.nn.Linear(3072, 768),
        OptimizedLayerNorm(768)
    )

    # Production-scale data
    batch_size, seq_len, hidden_size = 8, 512, 768
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Test forward and backward passes
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert output.shape == x.shape
    assert not torch.isnan(output).any()
```

### **💪 Stress Tests** (Performance Validation)
- **Purpose**: Test limits and performance boundaries
- **Target**: 30 seconds - 5 minutes per test
- **Data Scale**: Large-scale, memory-intensive
- **Usage**: Performance regression detection

```bash
# Run stress tests
python3 -m pytest tests/test_stress/ -v --slow
python3 -m pytest tests/test_performance_regression.py -v
```

**Example Stress Test:**
```python
@pytest.mark.slow
def test_large_scale_attention():
    """Test attention with production-scale inputs"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for stress testing")

    # Large-scale configuration
    batch_size, num_heads, seq_len, head_dim = 16, 32, 2048, 64

    # Memory-intensive test
    attention = MultiHeadAttention(
        embed_dim=num_heads * head_dim,
        num_heads=num_heads,
        optimization_level="aggressive"
    ).cuda()

    x = torch.randn(seq_len, batch_size, num_heads * head_dim).cuda()

    # Performance benchmark
    with torch.no_grad():
        start_time = time.perf_counter()
        output = attention(x, x, x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

    # Performance assertions
    assert output.shape == x.shape
    assert elapsed < 2.0  # Should complete within 2 seconds
```

## 🎚️ Data Configuration Tiers

| Config | Dimensions | Memory | Target Time | Purpose |
|--------|------------|---------|-------------|---------|
| `micro` | 1×2×32×16 | 0.0MB | < 0.1s | Algorithm correctness |
| `small` | 1×4×64×32 | 0.1MB | < 0.5s | Basic functionality |
| `medium` | 2×8×128×64 | 1.5MB | < 5s | Integration testing |
| `realistic` | 2×8×512×64 | 6.0MB | < 30s | Production scenarios |
| `large` | 4×16×1024×64 | 48.0MB | < 60s | Performance validation |
| `xlarge` | 8×32×2048×128 | 768.0MB | < 300s | Stress testing |

### Specialized Configurations
- `long_sequence`: 1×8×4096×64 (24MB) - Long context handling
- `high_heads`: 2×64×256×64 (24MB) - Multi-head scaling
- `large_batch`: 32×8×256×64 (48MB) - Batch processing
- `mixed_precision`: Various dtypes - Precision handling

## 🔬 Benchmarking Framework

### State-of-the-Art Comparisons

**Compare against industry leaders:**
```python
# Benchmarking API
from kernel_pytorch.benchmarking import CompetitiveBenchmark

benchmark = CompetitiveBenchmark(
    baselines=['flash_attention_3', 'vllm', 'ring_attention', 'mamba'],
    metrics=['latency', 'throughput', 'memory_usage', 'accuracy']
)

results = benchmark.run_comparison(
    model_config={'batch_size': 8, 'seq_len': 512, 'hidden_size': 768},
    optimization_level='production'
)

print(f"Our framework vs baselines: {results.speedup_summary}")
```

### Performance Targets

| Benchmark | Target Speedup | Memory Reduction | Accuracy Preservation |
|-----------|----------------|------------------|----------------------|
| **Flash Attention 3** | 1.2-1.5x | 10-20% | >99.9% |
| **vLLM Production** | 1.1-1.3x | 15-25% | >99.8% |
| **Ring Attention** | 1.5-2.0x | 30-40% | >99.9% |
| **Mamba (SSM)** | 0.8-1.2x* | 20-30% | >99.5% |

*Note: Mamba comparison depends on sequence length - we excel at shorter sequences*

### Statistical Validation

```python
def benchmark_with_statistics(optimization_fn, baseline_fn, num_trials=20):
    """Run statistically significant performance comparison"""
    from scipy import stats

    optimization_times = []
    baseline_times = []

    for _ in range(num_trials):
        # Warmup
        for _ in range(5):
            optimization_fn()
            baseline_fn()

        # Measure optimized version
        torch.cuda.synchronize()
        start = time.perf_counter()
        result_opt = optimization_fn()
        torch.cuda.synchronize()
        optimization_times.append(time.perf_counter() - start)

        # Measure baseline
        torch.cuda.synchronize()
        start = time.perf_counter()
        result_base = baseline_fn()
        torch.cuda.synchronize()
        baseline_times.append(time.perf_counter() - start)

    # Statistical analysis
    speedup = np.mean(baseline_times) / np.mean(optimization_times)
    t_stat, p_value = stats.ttest_ind(baseline_times, optimization_times)

    return {
        'speedup': speedup,
        'significance': p_value < 0.05,
        'confidence_interval': stats.t.interval(0.95, len(baseline_times)-1),
        'effect_size': (np.mean(baseline_times) - np.mean(optimization_times)) / np.std(baseline_times)
    }
```

## 🎯 Test Organization

### Test Directory Structure
```
tests/
├── test_basic_functionality.py      # Core functionality
├── test_components/                  # Individual components
│   ├── test_fused_gelu.py
│   ├── test_attention_layers.py
│   └── test_optimization_layers.py
├── test_integration/                 # End-to-end tests
│   ├── test_model_optimization.py
│   ├── test_distributed_training.py
│   └── test_inference_pipeline.py
├── test_hardware_abstraction/        # Hardware-specific tests
│   ├── test_nvidia_adapter.py
│   ├── test_amd_adapter.py
│   └── test_custom_hardware.py
├── test_benchmarking/               # Performance tests
│   ├── test_competitive_benchmarks.py
│   ├── test_regression_detection.py
│   └── test_statistical_validation.py
└── test_stress/                     # Stress and edge cases
    ├── test_memory_limits.py
    ├── test_large_scale.py
    └── test_edge_cases.py
```

### Test Configuration
```python
# conftest.py - Shared test configuration
import pytest
import torch

@pytest.fixture(scope="session")
def device():
    """Shared device fixture"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(params=['small', 'medium', 'realistic'])
def data_config(request):
    """Parameterized data configurations"""
    configs = {
        'small': {'batch_size': 2, 'seq_len': 64, 'hidden_size': 256},
        'medium': {'batch_size': 4, 'seq_len': 128, 'hidden_size': 512},
        'realistic': {'batch_size': 8, 'seq_len': 512, 'hidden_size': 768}
    }
    return configs[request.param]

@pytest.fixture
def optimized_model(device, data_config):
    """Create optimized model for testing"""
    from kernel_pytorch.components import create_optimized_model
    return create_optimized_model(
        config=data_config,
        device=device,
        optimization_level='test'
    )
```

## 🚀 Running Tests

### Development Workflow
```bash
# Quick development tests (< 30 seconds)
python3 -m pytest tests/test_basic_functionality.py -v

# Pre-commit tests (< 2 minutes)
python3 -m pytest tests/test_components/ tests/test_integration/ -v

# Full test suite (< 10 minutes)
python3 -m pytest tests/ -v --tb=short

# Stress tests (CI only, < 30 minutes)
python3 -m pytest tests/ -v --slow --stress
```

### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Comprehensive Testing

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run unit tests
        run: pytest tests/test_basic_functionality.py tests/test_components/ -v --cov

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install pytest
      - name: Run integration tests
        run: pytest tests/test_integration/ -v

  gpu-tests:
    runs-on: [self-hosted, gpu]
    needs: integration-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Setup GPU environment
        run: |
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
          pip install -r requirements.txt
          pip install pytest
      - name: Run GPU tests
        run: |
          export CUDA_VISIBLE_DEVICES=0
          pytest tests/test_hardware_abstraction/ tests/test_benchmarking/ -v
      - name: Run stress tests
        run: |
          export CUDA_VISIBLE_DEVICES=0
          pytest tests/test_stress/ -v --slow
```

## 📊 Performance Monitoring

### Regression Detection
```python
# tests/test_performance_regression.py
import pytest
import time
import torch
from kernel_pytorch.benchmarking import PerformanceTracker

class TestPerformanceRegression:
    """Detect performance regressions in optimizations"""

    def setup_method(self):
        self.tracker = PerformanceTracker()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.mark.benchmark
    def test_attention_performance(self):
        """Ensure attention performance doesn't regress"""
        from kernel_pytorch.components import AttentionLayer

        # Standard configuration
        batch_size, seq_len, hidden_size = 8, 512, 768
        attention = AttentionLayer(hidden_size, num_heads=12).to(self.device)
        x = torch.randn(batch_size, seq_len, hidden_size, device=self.device)

        # Benchmark current performance
        with torch.no_grad():
            current_time = self.tracker.measure_execution_time(
                lambda: attention(x), num_trials=20
            )

        # Compare against baseline
        baseline_time = self.tracker.get_baseline('attention_512_768')

        if baseline_time is not None:
            speedup = baseline_time / current_time
            assert speedup >= 0.95, f"Performance regression: {speedup:.2f}x vs baseline"

        # Update baseline if improved
        self.tracker.update_baseline('attention_512_768', current_time)
```

### Memory Profiling
```python
def profile_memory_usage(model_fn, input_data, device):
    """Profile memory usage of model execution"""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure peak memory
        initial_memory = torch.cuda.memory_allocated()
        output = model_fn(input_data)
        peak_memory = torch.cuda.max_memory_allocated()

        memory_overhead = peak_memory - initial_memory
        return {
            'peak_memory_mb': peak_memory / 1024**2,
            'overhead_mb': memory_overhead / 1024**2,
            'output_shape': output.shape
        }
    else:
        # CPU memory profiling
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        output = model_fn(input_data)
        final_memory = process.memory_info().rss

        return {
            'memory_increase_mb': (final_memory - initial_memory) / 1024**2,
            'output_shape': output.shape
        }
```

## 🔧 Debugging and Profiling

### Debug Test Failures
```bash
# Verbose test output with full traceback
python3 -m pytest tests/test_failing.py -vvv --tb=long

# Drop into debugger on failure
python3 -m pytest tests/test_failing.py --pdb

# Run specific test with profiling
python3 -m pytest tests/test_performance.py::test_attention_speed -v --profile
```

### Performance Profiling
```python
# Profile specific components
from kernel_pytorch.utils import ProfiledExecution

with ProfiledExecution() as profiler:
    model = create_optimized_model()
    output = model(test_input)

# Analyze results
profiler.print_summary()
profiler.export_chrome_trace('profile_results.json')
```

## 🎯 Best Practices

### Test Writing Guidelines
1. **Use descriptive test names** - Clearly state what is being tested
2. **Test one thing at a time** - Each test should validate a single behavior
3. **Use appropriate fixtures** - Share setup code via pytest fixtures
4. **Include performance bounds** - Set reasonable time/memory limits
5. **Test edge cases** - Include boundary conditions and error cases

### Performance Testing Guidelines
1. **Warmup before measurement** - Run several iterations before timing
2. **Use statistical significance** - Multiple trials with statistical analysis
3. **Control for variance** - Account for system load and other factors
4. **Test on target hardware** - GPU tests require actual GPU environment
5. **Set realistic baselines** - Compare against appropriate alternatives

### Continuous Integration Guidelines
1. **Fast feedback loop** - Keep basic tests under 2 minutes
2. **Parallel execution** - Run independent tests in parallel
3. **GPU resource management** - Limit GPU tests to avoid conflicts
4. **Artifact collection** - Save performance reports and profiles
5. **Regression alerts** - Notify on performance degradation

## 📋 Test Checklist

### Pre-Commit Checklist
- [ ] Unit tests pass locally
- [ ] Integration tests pass
- [ ] Performance tests show no regression
- [ ] Code coverage maintained
- [ ] Static analysis clean

### Release Testing Checklist
- [ ] Full test suite passes
- [ ] Benchmarks validate performance claims
- [ ] Multi-platform testing complete
- [ ] Memory usage within bounds
- [ ] Documentation examples work

### Performance Validation Checklist
- [ ] Speedup measurements statistically significant
- [ ] Memory usage reduction validated
- [ ] Accuracy preservation confirmed
- [ ] Hardware compatibility verified
- [ ] Regression tests updated

---

## 🚀 Next Steps

1. **Run quick validation** to ensure setup is working
2. **Execute comprehensive benchmarks** to validate performance claims
3. **Set up development workflow** with appropriate test categories
4. **Configure CI/CD pipeline** for automated testing
5. **Implement performance monitoring** for regression detection

**🎯 Ready for robust testing and benchmarking of GPU optimizations!**