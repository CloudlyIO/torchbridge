# Mixture of Experts (MoE) Implementation

> **Status**: ✅ Implemented | **Version**: v0.4.x

**Expert routing and sparse attention optimization - IMPLEMENTED**

## Overview

Mixture of Experts (MoE) is a neural network architecture where different subnetworks (experts) specialize in different parts of the input space. A gating network routes each input to a subset of experts, enabling models to scale to trillions of parameters while keeping compute manageable.

### Why MoE Matters

- **Scaling Efficiency**: 8x parameters with 2x compute (compared to dense models)
- **Industry Adoption**: DeepSeek-V3, Mixtral 8x7B, GPT-4, Gemini all use MoE
- **Memory Challenges**: Expert weights can dominate memory usage
- **Load Balancing**: Uneven expert utilization wastes compute

---

## Current Status: IMPLEMENTED

MoE is fully implemented in TorchBridge v0.4.x:

```python
from torchbridge.mixture_of_experts import (
    MoELayer,
    SparseMoELayer,
    SwitchTransformerMoE,
    GLaMStyleMoE,
    create_moe_layer,
    MoEConfig,
    TopKRouter,
    SwitchRouter,
    LoadBalancer,
    FeedForwardExpert,
)
```

### Available Features

#### MoE Layers
- `MoELayer` - Standard MoE layer with configurable routing
- `SparseMoELayer` - Memory-efficient sparse MoE
- `SwitchTransformerMoE` - Google Switch Transformer style
- `GLaMStyleMoE` - Google GLaM architecture style

#### Routing Strategies
- `TopKRouter` - Standard top-k expert selection
- `SwitchRouter` - Single-expert routing (Switch Transformer)
- `HashRouter` - Deterministic hash-based routing
- `LearnedRouter` - Learned routing weights
- `DynamicCapacityRouter` - Adaptive capacity routing

#### Expert Networks
- `FeedForwardExpert` - Standard FFN expert
- `ConvolutionalExpert` - Conv-based expert
- `AttentionExpert` - Attention-based expert
- `ParameterEfficientExpert` - LoRA-style efficient expert

#### Optimization Utilities
- `ExpertParallelism` - Multi-GPU expert distribution
- `LoadBalancer` - Prevent expert collapse
- `ExpertScheduler` - Training scheduling
- `MemoryEfficientSwitching` - Reduce memory during switching

---

## Quick Start

```python
import torch
from torchbridge import create_moe

# Create MoE layer
moe = create_moe(
    hidden_size=512,
    num_experts=8,
    top_k=2,
    moe_type="standard"  # or "sparse", "switch", "glam"
)

# Forward pass
x = torch.randn(32, 128, 512)  # [batch, seq, hidden]
output = moe(x)
```

### Configuration Options

```python
from torchbridge.mixture_of_experts import MoEConfig, create_moe_layer

config = MoEConfig(
    num_experts=8,
    expert_capacity=128,
    top_k=2,
    hidden_size=512,
    intermediate_size=2048,
    load_balancing_loss_weight=0.01,
    jitter_noise=0.1,
)

moe = create_moe_layer(moe_type="standard", config=config)
```

---

## Future Considerations

Potential enhancements for future releases:

1. **Expert Routing Kernels** - Custom Triton kernels for routing
2. **Heterogeneous Experts** - Mix different expert architectures
3. **Expert Caching** - Intelligent expert weight caching
4. **Cross-Node Expert Parallelism** - Multi-node expert distribution

---

**Document Version**: 1.0 (Updated to reflect implemented status)
