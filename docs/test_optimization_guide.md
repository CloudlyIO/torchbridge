# 🚀 Test Performance Optimization Guide

This document provides strategies and tools for optimizing slow tests in the codebase.

## 🎯 Performance Optimization Results

### **test_multiple_pattern_compilation**
- **Before**: 27.077 seconds
- **After**: 2.237 seconds
- **Speedup**: 12.1x faster (91.7% time reduction)

## 📊 Common Test Performance Bottlenecks

### **1. Large Tensor Dimensions**
**Problem**: Tests using production-size tensors (e.g., seq_len=512)
**Solution**: Use smaller test-specific dimensions

```python
# ❌ Slow (production size)
batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64

# ✅ Fast (test size)
batch_size, num_heads, seq_len, head_dim = 1, 4, 64, 32
```

### **2. Expensive Pattern Testing**
**Problem**: Testing all patterns including slow ones (dilated attention)
**Solution**: Split into fast and comprehensive tests

```python
# ❌ Slow - tests all patterns
patterns = ["causal", "sliding_window", "dilated"]

# ✅ Fast - only essential patterns
patterns = ["causal"]

# ✅ Add separate comprehensive test for full coverage
@pytest.mark.slow
def test_comprehensive_patterns():
    patterns = ["causal", "sliding_window", "dilated"]
```

### **3. torch.compile Failures**
**Problem**: Fallback logic is extremely slow
**Solution**: Mock or skip compilation in tests

## 🛠️ Optimization Strategies

### **Strategy 1: Tiered Test Fixtures**

```python
@pytest.fixture
def small_inputs():
    """Fast inputs for basic functionality testing"""
    return create_tensors(batch_size=1, seq_len=32)

@pytest.fixture
def medium_inputs():
    """Medium inputs for integration testing"""
    return create_tensors(batch_size=2, seq_len=128)

@pytest.fixture
def large_inputs():
    """Large inputs for performance/stress testing"""
    return create_tensors(batch_size=4, seq_len=512)
```

### **Strategy 2: Test Categories**

```python
# Fast tests (< 1 second)
def test_basic_functionality(small_inputs):
    pass

# Slow tests (marked for CI/manual runs)
@pytest.mark.slow
def test_comprehensive_functionality(large_inputs):
    pass
```

### **Strategy 3: Compilation Mocking**

```python
@pytest.fixture
def mock_compiler(monkeypatch):
    """Mock slow compilation operations"""
    def fast_compile(*args, **kwargs):
        return MockKernel()

    monkeypatch.setattr('torch.compile', fast_compile)
```

## 🔧 Test Optimization Checklist

### **Before Optimizing:**
1. ✅ Profile the test to identify bottlenecks
2. ✅ Measure baseline performance
3. ✅ Identify the slowest operations

### **Optimization Steps:**
1. ✅ Reduce tensor dimensions
2. ✅ Limit expensive operations
3. ✅ Use mocking for slow external calls
4. ✅ Split comprehensive tests from fast tests

### **After Optimizing:**
1. ✅ Verify functionality still works
2. ✅ Measure performance improvement
3. ✅ Update documentation

## 📈 Performance Targets

| Test Type | Target Time | Optimization Level |
|-----------|-------------|-------------------|
| **Unit Tests** | < 0.1s | Minimal inputs, mocked dependencies |
| **Integration Tests** | < 1.0s | Small inputs, essential patterns |
| **Comprehensive Tests** | < 5.0s | Medium inputs, marked as `@slow` |
| **Stress Tests** | < 30s | Full inputs, CI-only |

## 🚀 Quick Optimization Commands

### **Profile any test:**
```bash
PYTHONPATH=src python3 -c "
import time
start = time.time()
# Run your test logic here
print(f'Test took: {time.time() - start:.3f}s')
"
```

### **Run only fast tests:**
```bash
pytest -m "not slow" tests/
```

### **Run comprehensive tests:**
```bash
pytest -m "slow" tests/
```

## 🎊 Success Metrics

- **Development Tests**: All tests < 1 second each
- **CI Pipeline**: Fast tests complete in < 30 seconds total
- **Coverage**: 100% functionality coverage across fast + comprehensive tests
- **Developer Experience**: Sub-second feedback for common test runs
