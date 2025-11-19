"""
Mixture of Experts (MoE) Implementation (2024-2025)

State-of-the-art MoE implementations based on latest research:
- Sparse expert routing with top-k gating
- Expert parallelism and load balancing
- Dynamic capacity factor optimization
- Memory-efficient expert switching
- Advanced routing strategies (Switch, GLaM, PaLM-style)

Key Features:
- Up to 1000x model capacity increase with minimal compute overhead
- Expert parallelism for distributed training
- Load balancing to prevent expert collapse
- Memory-efficient routing and switching
"""

from .moe_layers import (
    MoELayer,
    SparseMoELayer,
    SwitchTransformerMoE,
    GLaMStyleMoE,
    create_moe_layer,
    MoEConfig
)

from .expert_networks import (
    FeedForwardExpert,
    ConvolutionalExpert,
    AttentionExpert,
    ParameterEfficientExpert
)

from .routing import (
    TopKRouter,
    SwitchRouter,
    HashRouter,
    LearnedRouter,
    DynamicCapacityRouter
)

from .optimization import (
    ExpertParallelism,
    LoadBalancer,
    ExpertScheduler,
    MemoryEfficientSwitching
)

__all__ = [
    # Core MoE layers
    'MoELayer',
    'SparseMoELayer',
    'SwitchTransformerMoE',
    'GLaMStyleMoE',
    'create_moe_layer',
    'MoEConfig',

    # Expert implementations
    'FeedForwardExpert',
    'ConvolutionalExpert',
    'AttentionExpert',
    'ParameterEfficientExpert',

    # Routing strategies
    'TopKRouter',
    'SwitchRouter',
    'HashRouter',
    'LearnedRouter',
    'DynamicCapacityRouter',

    # Optimization utilities
    'ExpertParallelism',
    'LoadBalancer',
    'ExpertScheduler',
    'MemoryEfficientSwitching'
]