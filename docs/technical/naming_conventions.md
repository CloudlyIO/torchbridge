# 🎯 KernelPyTorch Naming Conventions Guide

**Version**: 1.0
**Date**: November 28, 2024
**Status**: Adopted as project standard

## 📋 Overview

This document establishes consistent naming conventions for the KernelPyTorch project, following Python PEP 8 standards and deep learning best practices.

## 🎨 Core Naming Standards

### **1. Classes and Types**
**Convention**: `PascalCase` (also called CapWords)

```python
# ✅ Correct Examples
class RingAttentionLayer(nn.Module):
    pass

class FP8TrainingEngine:
    pass

class DynamicSparseConfig:
    pass

# ❌ Incorrect Examples
class ring_attention_layer(nn.Module):  # snake_case
    pass

class FP8trainingEngine:  # inconsistent casing
    pass
```

### **2. Functions and Methods**
**Convention**: `snake_case`

```python
# ✅ Correct Examples
def create_ring_attention(d_model: int, num_heads: int) -> RingAttentionLayer:
    pass

def estimate_memory_usage(seq_length: int) -> float:
    pass

def validate_ring_attention_setup(config: RingAttentionConfig) -> bool:
    pass

# ❌ Incorrect Examples
def createRingAttention(d_model: int) -> RingAttentionLayer:  # camelCase
    pass

def EstimateMemoryUsage(seq_length: int) -> float:  # PascalCase
    pass
```

### **3. Variables and Parameters**
**Convention**: `snake_case`

```python
# ✅ Correct Examples
attention_mask = torch.ones(batch_size, seq_length)
num_attention_heads = 12
sequence_length = 1024
fp8_config = FP8Config()

# ❌ Incorrect Examples
attentionMask = torch.ones(batch_size, seq_length)  # camelCase
NumAttentionHeads = 12  # PascalCase
sequenceLength = 1024  # camelCase
```

### **4. Constants and Enum Values**
**Convention**: `UPPER_SNAKE_CASE`

```python
# ✅ Correct Examples
class SparsePattern(Enum):
    RANDOM = "random"
    BLOCK_SPARSE = "block_sparse"
    DYNAMIC_THRESHOLD = "dynamic_threshold"

class FP8Format(Enum):
    E4M3 = "e4m3"
    E5M2 = "e5m2"

# Module-level constants
DEFAULT_ATTENTION_HEADS = 8
MAX_SEQUENCE_LENGTH = 1_000_000

# ❌ Incorrect Examples
class SparsePattern(Enum):
    Random = "random"  # PascalCase
    blockSparse = "block_sparse"  # camelCase
```

### **5. Modules and Packages**
**Convention**: `snake_case`

```python
# ✅ Correct Examples
# Files
ring_attention.py
sparse_attention.py
fp8_training_engine.py

# Packages
advanced_attention/
precision/
hardware_abstraction/

# ❌ Incorrect Examples
RingAttention.py  # PascalCase
sparseAttention.py  # camelCase
FP8TrainingEngine.py  # PascalCase
```

### **6. Private Members**
**Convention**: Leading underscore `_snake_case`

```python
# ✅ Correct Examples
class RingAttentionLayer(nn.Module):
    def __init__(self):
        self._internal_state = {}
        self._compute_attention_weights = True

    def _validate_input_shapes(self, x: torch.Tensor) -> bool:
        pass

    def __private_method(self):  # Double underscore for name mangling
        pass

# ❌ Incorrect Examples
def _ValidateInputShapes(self):  # PascalCase with underscore
    pass
```

## 🏗️ Architecture-Specific Conventions

### **Advanced Attention Module**
```python
# Classes (PascalCase)
class RingAttentionLayer(nn.Module): pass
class DynamicSparseAttention(nn.Module): pass
class ContextParallelAttention(nn.Module): pass

# Factory Functions (snake_case)
create_ring_attention()
create_sparse_attention()
create_context_parallel_attention()

# Utility Functions (snake_case)
estimate_memory_usage()
compute_attention_efficiency()
validate_ring_attention_setup()

# Enums (PascalCase class, UPPER_SNAKE_CASE values)
class SparsePattern(Enum):
    RANDOM = "random"
    BLOCK_SPARSE = "block_sparse"
```

### **Precision/FP8 Module**
```python
# Classes (PascalCase)
class FP8TrainingEngine: pass
class FP8LinearLayer(nn.Module): pass
class FP8Config: pass

# Factory Functions (snake_case)
create_fp8_trainer()
convert_model_to_fp8()

# Validation Functions (snake_case)
validate_fp8_setup()

# Enums (PascalCase class, UPPER_SNAKE_CASE values)
class FP8Format(Enum):
    E4M3 = "e4m3"
    E5M2 = "e5m2"
```

### **Hardware Abstraction Layer**
```python
# Classes (PascalCase)
class HardwareAbstractionLayer: pass
class DeviceCoordinator: pass
class GPUOptimizer: pass

# Functions (snake_case)
detect_available_devices()
optimize_for_hardware()
create_device_mesh()
```

## 📚 Type Annotations

### **Function Signatures**
```python
# ✅ Correct Examples
def create_ring_attention(
    d_model: int,
    num_heads: int,
    max_sequence_length: int = 1_000_000,
    device: Optional[torch.device] = None
) -> RingAttentionLayer:
    pass

def process_attention_batch(
    input_tensors: List[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    pass
```

### **Generic Types**
```python
# ✅ Correct Examples
from typing import TypeVar, Generic

AttentionLayerType = TypeVar('AttentionLayerType', bound=nn.Module)
ConfigType = TypeVar('ConfigType')

class BaseAttentionFactory(Generic[AttentionLayerType]):
    pass
```

## 🔧 Import Organization

### **Import Statement Conventions**
```python
# ✅ Correct Examples - Group and order imports properly
# Standard library
import math
import warnings
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports - relative
from .ring_attention import RingAttentionLayer, create_ring_attention
from .sparse_attention import DynamicSparseAttention, SparsePattern
from ..precision import FP8TrainingEngine, FP8Format

# Local imports - absolute (when needed)
from kernel_pytorch.hardware_abstraction import HardwareAbstractionLayer
```

## 🧪 Testing Conventions

### **Test Classes and Methods**
```python
# ✅ Correct Examples
class TestRingAttention:
    def test_linear_memory_complexity(self): pass
    def test_distributed_processing(self): pass
    def test_million_token_sequences(self): pass

class TestFP8Training:
    def test_e4m3_format_training(self): pass
    def test_dynamic_scaling(self): pass
    def test_model_conversion(self): pass

# Test fixtures (snake_case)
@pytest.fixture
def sample_attention_config():
    pass

@pytest.fixture
def fp8_training_engine():
    pass
```

## 📖 Documentation Conventions

### **Docstring Standards**
```python
# ✅ Correct Examples
class RingAttentionLayer(nn.Module):
    """
    Ring Attention implementation for linear memory complexity.

    Enables processing of 1M+ token sequences with O(N) memory usage
    instead of the standard O(N²) complexity.

    Args:
        d_model: Model dimension size
        num_heads: Number of attention heads
        max_sequence_length: Maximum supported sequence length

    Returns:
        Configured RingAttentionLayer instance

    Example:
        >>> attention = RingAttentionLayer(d_model=512, num_heads=8)
        >>> output = attention(input_tensor)
    """
```

## ⚡ Performance and Optimization Naming

### **Benchmark and Performance Functions**
```python
# ✅ Correct Examples
def benchmark_attention_performance(): pass
def measure_memory_efficiency(): pass
def profile_training_speed(): pass
def optimize_for_h100(): pass

# Performance test classes
class TestAttentionPerformance: pass
class TestMemoryEfficiency: pass
```

## 📁 File and Directory Structure

### **Directory Organization**
```
src/kernel_pytorch/
├── advanced_attention/          # snake_case package names
│   ├── __init__.py
│   ├── ring_attention.py       # snake_case file names
│   ├── sparse_attention.py
│   └── context_parallel.py
├── precision/
│   ├── __init__.py
│   ├── fp8_training_engine.py
│   └── fp8_optimizations.py
└── hardware_abstraction/
    ├── __init__.py
    ├── device_coordinator.py
    └── gpu_optimizer.py
```

## 🚨 Common Mistakes to Avoid

### **❌ Inconsistent Casing**
```python
# Don't mix naming conventions within the same context
class ringAttentionLayer(nn.Module):  # Wrong: camelCase for class
    def CreateRingAttention(self):     # Wrong: PascalCase for method
        pass

# Don't use inconsistent abbreviations
class FP8Engine: pass      # Inconsistent with...
class FloatingPoint8TrainingEngine: pass  # ...this full form
```

### **❌ Unclear Abbreviations**
```python
# Don't use unclear abbreviations
def calc_attn_eff():  # Unclear: calculate_attention_efficiency()
    pass

def proc_btch():      # Unclear: process_batch()
    pass
```

### **❌ Non-Descriptive Names**
```python
# Don't use non-descriptive variable names
def create_attention(x, y, z):  # Unclear parameters
    pass

# Do use descriptive names
def create_attention(d_model, num_heads, max_seq_len):  # Clear parameters
    pass
```

## ✅ Validation Checklist

Before submitting code, verify:

- [ ] **Classes**: Use `PascalCase` (e.g., `RingAttentionLayer`)
- [ ] **Functions**: Use `snake_case` (e.g., `create_ring_attention`)
- [ ] **Variables**: Use `snake_case` (e.g., `attention_mask`)
- [ ] **Constants**: Use `UPPER_SNAKE_CASE` (e.g., `MAX_SEQUENCE_LENGTH`)
- [ ] **Files/Modules**: Use `snake_case` (e.g., `ring_attention.py`)
- [ ] **Private members**: Use `_snake_case` prefix
- [ ] **Type annotations**: Consistent with naming conventions
- [ ] **Imports**: Properly organized and grouped
- [ ] **Documentation**: Clear and consistent

## 📝 Implementation Status

### **✅ Current Compliance**
The existing codebase (as of November 28, 2024) already follows these conventions:

- **Advanced Attention**: All classes, functions, and variables properly named
- **FP8 Precision**: Consistent naming throughout the module
- **Hardware Abstraction**: Follows established patterns
- **Test Suite**: Proper test naming conventions

### **🎯 Future Development**
All new code must follow these conventions. Code reviews will verify compliance with this naming standard.

---

## 📚 References

- **[PEP 8](https://peps.python.org/pep-0008/)**: Style Guide for Python Code
- **[PEP 257](https://peps.python.org/pep-0257/)**: Docstring Conventions
- **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**: Additional Python conventions
- **[PyTorch Contributing Guide](https://pytorch.org/docs/stable/community/contribution_guide.html)**: PyTorch-specific conventions

---

**Naming Conventions Established** ✅
*Consistent, readable, and maintainable code naming standards for KernelPyTorch*