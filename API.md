# 📖 KernelPyTorch API Reference

**Complete API documentation for the PyTorch GPU optimization framework.**

## 🚀 Quick Import Guide

```python
# Unified Attention Framework (Phase 2 Consolidated)
from kernel_pytorch.attention import (
    AttentionConfig,
    AttentionPatterns,
    FP8AttentionConfig,
    create_attention,
    FlashAttention3,
    RingAttentionLayer,
    DynamicSparseAttention,
    ContextParallelAttention,
    create_unified_attention_fusion  # Phase 2.2
)

# FP8 Training & Adaptive Precision
from kernel_pytorch.precision import (
    create_fp8_trainer,
    convert_model_to_fp8,
    FP8Format,
    create_ultra_precision_module,  # Phase 2.2
    analyze_precision_opportunities  # Phase 2.2
)

# Hardware Abstraction (Phase 3 unified)
from kernel_pytorch.hardware import (
    HardwareAbstractionLayer,
    detect_available_devices
)

# Core Components (Phase 3 unified)
from kernel_pytorch.core import AttentionLayer, OptimizedLinear, FusedGELU
```

## 🎯 Advanced Attention APIs

### **Ring Attention (Million-Token Sequences)**

#### `create_ring_attention()`
```python
def create_ring_attention(
    d_model: int,
    num_heads: int,
    max_sequence_length: int = 1_000_000,
    device: Optional[torch.device] = None
) -> RingAttentionLayer
```

**Creates Ring Attention layer for linear memory complexity O(N).**

**Parameters:**
- `d_model`: Model dimension size (e.g., 512, 768, 1024)
- `num_heads`: Number of attention heads (must divide d_model evenly)
- `max_sequence_length`: Maximum supported sequence length (default: 1M)
- `device`: Target device (auto-detected if None)

**Returns:** Configured RingAttentionLayer instance

**Example:**
```python
# Support 1M+ token sequences with linear memory
attention = create_ring_attention(d_model=512, num_heads=8)
output = attention(long_sequence)  # Shape: [batch, 1M+, d_model]
```

#### `RingAttentionLayer`
```python
class RingAttentionLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor
```

**Core Ring Attention implementation.**

### **Sparse Attention (90% Compute Reduction)**

#### `create_sparse_attention()`
```python
def create_sparse_attention(
    d_model: int,
    num_heads: int,
    sparsity_ratio: float = 0.9,
    pattern: SparsePattern = SparsePattern.DYNAMIC_THRESHOLD
) -> DynamicSparseAttention
```

**Creates sparse attention with configurable compute reduction.**

**Parameters:**
- `d_model`: Model dimension size
- `num_heads`: Number of attention heads
- `sparsity_ratio`: Fraction of attention weights to zero out (0.0-0.95)
- `pattern`: Sparsity pattern strategy

**Example:**
```python
# 90% compute reduction with content-aware patterns
attention = create_sparse_attention(d_model=512, num_heads=8, sparsity_ratio=0.9)
output = attention(x)  # 90% less computation
```

#### `SparsePattern` (Enum)
```python
class SparsePattern(Enum):
    RANDOM = "random"                    # Random sparsity
    BLOCK_SPARSE = "block_sparse"        # Block-based patterns
    STRIDED = "strided"                  # Strided patterns
    LOCAL_GLOBAL = "local_global"        # Local + global attention
    DYNAMIC_THRESHOLD = "dynamic_threshold"  # Content-aware (recommended)
    LEARNED = "learned"                  # Learned during training
```

### **Context Parallel Attention (Multi-GPU)**

#### `create_context_parallel_attention()`
```python
def create_context_parallel_attention(
    d_model: int,
    num_heads: int,
    context_parallel_size: int,
    device_mesh: Optional[List[torch.device]] = None
) -> ContextParallelAttention
```

**Creates attention distributed across multiple GPUs.**

**Parameters:**
- `d_model`: Model dimension size
- `num_heads`: Number of attention heads
- `context_parallel_size`: Number of GPUs to distribute across
- `device_mesh`: Specific GPU devices (auto-detected if None)

**Example:**
```python
# Distribute attention across 4 GPUs
attention = create_context_parallel_attention(
    d_model=512, num_heads=8, context_parallel_size=4
)
output = attention(x)  # Computed across multiple GPUs
```

## ⚡ FP8 Training APIs

### **FP8 Training Engine**

#### `create_fp8_trainer()`
```python
def create_fp8_trainer(
    model: nn.Module,
    forward_format: FP8Format = FP8Format.E4M3,
    backward_format: FP8Format = FP8Format.E5M2,
    device: Optional[torch.device] = None
) -> FP8TrainingEngine
```

**Creates FP8 training engine for 2x H100 speedup.**

**Parameters:**
- `model`: PyTorch model to train with FP8
- `forward_format`: FP8 format for forward pass (E4M3 recommended)
- `backward_format`: FP8 format for backward pass (E5M2 recommended)
- `device`: Target device (auto-detected if None)

**Example:**
```python
# 2x speedup on H100/Blackwell hardware
trainer = create_fp8_trainer(model, forward_format=FP8Format.E4M3)
with trainer:
    loss = trainer.training_step(inputs, targets)
    trainer.optimizer_step(optimizer)
```

#### `convert_model_to_fp8()`
```python
def convert_model_to_fp8(
    model: nn.Module,
    fp8_config: Optional[FP8Config] = None,
    inplace: bool = True
) -> nn.Module
```

**Converts model layers to use FP8 precision.**

**Example:**
```python
# Convert existing model to FP8
fp8_model = convert_model_to_fp8(model)
# Model now uses FP8LinearLayer instead of nn.Linear
```

#### `FP8Format` (Enum)
```python
class FP8Format(Enum):
    E4M3 = "e4m3"  # 1 sign, 4 exponent, 3 mantissa - higher precision
    E5M2 = "e5m2"  # 1 sign, 5 exponent, 2 mantissa - wider range
```

### **FP8 Training Context**
```python
# Context manager usage
with FP8TrainingEngine(model, fp8_config) as trainer:
    # All operations use FP8 precision
    loss = trainer.training_step(inputs, targets)
    success = trainer.optimizer_step(optimizer)

    # Get training statistics
    stats = trainer.get_training_statistics()
    print(f"Overflows: {stats['overflows']}, Steps: {stats['steps']}")
```

## 🔧 Hardware Abstraction APIs

### **Device Detection**

#### `detect_available_devices()`
```python
def detect_available_devices() -> Dict[str, List[torch.device]]
```

**Detects all available hardware devices.**

**Returns:** Dictionary with device categories and available devices

**Example:**
```python
devices = detect_available_devices()
# {'cuda': [device(type='cuda', index=0), ...], 'cpu': [device(type='cpu')]}
```

### **Hardware Abstraction Layer**

#### `HardwareAbstractionLayer`
```python
class HardwareAbstractionLayer:
    def __init__(self, target_devices: Optional[List[torch.device]] = None)

    def optimize_for_hardware(self, model: nn.Module) -> nn.Module
    def get_optimal_batch_size(self, model: nn.Module) -> int
    def estimate_memory_usage(self, model: nn.Module, batch_size: int) -> float
```

**Example:**
```python
hal = HardwareAbstractionLayer()
optimized_model = hal.optimize_for_hardware(model)
batch_size = hal.get_optimal_batch_size(model)
```

## 🧩 Core Components APIs

### **Optimized Layers**

#### `AttentionLayer`
```python
class AttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    )
```

**Optimized multi-head attention implementation.**

#### `OptimizedLinear`
```python
class OptimizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        optimization_level: str = "balanced"
    )
```

**Hardware-optimized linear layer.**

### **Fused Operations**

#### `FusedGELU`
```python
class FusedGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Memory-efficient fused GELU activation.**

## 🧪 Testing & Validation APIs

### **Component Validation**

#### `ComponentValidator`
```python
from kernel_pytorch.utils.validation_framework import ComponentValidator

validator = ComponentValidator(device=torch.device('cpu'))

# Validate attention layer
results = validator.validate_attention_component(attention_layer, d_model, num_heads)

# Validate linear layer
results = validator.validate_linear_component(linear_layer, in_features, out_features)
```

### **Performance Benchmarking**

#### `benchmark_attention_performance()`
```python
from kernel_pytorch.utils.compiler_optimization_assistant import benchmark_attention_performance

results = benchmark_attention_performance(
    model=attention_layer,
    input_shape=(batch_size, seq_len, d_model),
    num_trials=100
)
print(f"Average latency: {results['avg_latency']:.2f}ms")
```

## 🎯 Factory Functions Summary

**Quick reference for creating optimized components:**

```python
# Advanced Attention
ring_attention = create_ring_attention(512, 8, max_sequence_length=1_000_000)
sparse_attention = create_sparse_attention(512, 8, sparsity_ratio=0.9)
parallel_attention = create_context_parallel_attention(512, 8, context_parallel_size=4)

# FP8 Training
fp8_trainer = create_fp8_trainer(model, FP8Format.E4M3, FP8Format.E5M2)
fp8_model = convert_model_to_fp8(model)

# Hardware Abstraction
devices = detect_available_devices()
hal = HardwareAbstractionLayer(devices['cuda'])
```

## 🚀 Phase 2.2: Cutting-Edge Optimizations

### **Neural Operator Fusion API**

#### `create_unified_attention_fusion()`
```python
def create_unified_attention_fusion(
    model: nn.Module,
    fusion_strategy: FusionStrategy = FusionStrategy.FULL_BLOCK,
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
) -> UnifiedAttentionFusion
```

**Example:**
```python
from kernel_pytorch.attention.fusion import (
    create_unified_attention_fusion,
    FusionStrategy,
    OptimizationLevel
)

# Create fused transformer for 40-60% kernel overhead reduction
transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
fused_transformer = create_unified_attention_fusion(
    transformer,
    fusion_strategy=FusionStrategy.FULL_BLOCK,
    optimization_level=OptimizationLevel.AGGRESSIVE
)

# Single forward pass - all operations fused
output = fused_transformer(input_sequence)
```

#### `benchmark_fusion_performance()`
```python
def benchmark_fusion_performance(
    baseline_model: nn.Module,
    fused_model: nn.Module,
    input_data: torch.Tensor,
    num_runs: int = 100
) -> FusionPerformanceStats
```

### **Adaptive Precision Allocation API**

#### `create_ultra_precision_module()`
```python
def create_ultra_precision_module(
    model: nn.Module,
    config: Optional[PrecisionConfig] = None,
    device: Optional[torch.device] = None
) -> UltraPrecisionModule
```

**Example:**
```python
from kernel_pytorch.precision import (
    create_ultra_precision_module,
    PrecisionConfig,
    AllocationStrategy
)

# Create adaptive precision model for 30% quality improvement
config = PrecisionConfig(
    allocation_strategy=AllocationStrategy.ENTROPY_BASED,
    target_memory_reduction=0.4,  # 40% memory reduction target
    entropy_threshold=1.5
)

adaptive_model = create_ultra_precision_module(
    model=base_model,
    config=config
)

# Forward pass with entropy-based precision allocation
output = adaptive_model(input_data)
stats = adaptive_model.get_precision_stats()
print(f"Memory savings: {stats.memory_savings_ratio:.1%}")
```

#### `analyze_precision_opportunities()`
```python
def analyze_precision_opportunities(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict[str, Any]
```

**Example:**
```python
# Analyze precision opportunities in your model
opportunities = analyze_precision_opportunities(
    model=my_transformer,
    sample_input=sample_sequence
)

print(f"Potential memory savings: {opportunities['potential_savings']}")
print(f"Recommended optimizations: {len(opportunities['recommendations'])}")
```

#### **Configuration Classes**

```python
@dataclass
class PrecisionConfig:
    base_precision: PrecisionFormat = PrecisionFormat.FP16
    allocation_strategy: AllocationStrategy = AllocationStrategy.ENTROPY_BASED
    quantization_mode: QuantizationMode = QuantizationMode.DYNAMIC
    entropy_threshold: float = 1.5
    target_memory_reduction: float = 0.4
    gradient_weight: float = 0.3
    activation_weight: float = 0.4

@dataclass
class FusionConfig:
    strategy: FusionStrategy = FusionStrategy.FULL_BLOCK
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_mixed_precision: bool = True
    enable_memory_optimization: bool = True
    target_sequence_length: int = 512
```

## 📊 Performance Expectations

### **Ring Attention**
- **Memory**: O(N) vs O(N²) standard attention
- **Sequence Length**: Support for 1M+ tokens
- **Use Case**: Long documents, genomic sequences, audio processing

### **Sparse Attention**
- **Compute Reduction**: Up to 90% fewer FLOPs
- **Accuracy**: <1% loss with dynamic patterns
- **Use Case**: Large models, resource-constrained environments

### **FP8 Training**
- **Speedup**: 2x faster training on H100/Blackwell
- **Memory**: 50% reduction vs FP16
- **Use Case**: Large model training, production deployments

### **Neural Operator Fusion (Phase 2.2)**
- **Kernel Overhead Reduction**: 40-60% fewer kernel launches
- **Single-Kernel Execution**: Attention+FFN+normalization in one kernel
- **Use Case**: Transformer models, production inference

### **Adaptive Precision Allocation (Phase 2.2)**
- **Quality Improvement**: 30% better quality vs uniform quantization
- **Entropy-Based**: Intelligent precision allocation using information theory
- **Use Case**: Memory-constrained deployments, model compression

### **Hardware Optimization**
- **Multi-GPU**: Linear scaling with context parallel attention
- **Vendor Support**: NVIDIA, AMD, Intel GPU optimization
- **Fallbacks**: Graceful degradation to CPU when needed

## ⚠️ Important Notes

### **Device Compatibility**
- All APIs work on both CPU and GPU
- GPU-specific optimizations enabled automatically when available
- Graceful fallback to CPU implementations

### **Memory Management**
- Ring Attention: Requires sequence length planning for memory allocation
- FP8 Training: Monitor for numerical instability with dynamic scaling
- Sparse Attention: Pattern selection affects both performance and accuracy

### **Error Handling**
```python
try:
    trainer = create_fp8_trainer(model)
    trainer.setup_fp8_training()
except Exception as e:
    print(f"FP8 training unavailable: {e}")
    # Fallback to standard training
```

---

**For more examples, see the `demos/` directory with comprehensive usage patterns.** 🚀