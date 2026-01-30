"""
Advanced Memory Optimization Techniques (2025-2026)

Implementation of cutting-edge memory optimization techniques based on latest research:
- Deep Optimizer States with 2.5x speedup
- ZenFlow stall-free offloading
- MLP-Offload multi-path techniques
- Arctic long sequence training support
- Advanced gradient checkpointing strategies
- Memory-efficient attention variants

Key Features:
- CPU-GPU hybrid optimization with interleaved offloading
- Multi-level memory hierarchy optimization
- Dynamic memory allocation strategies
- Gradient accumulation with compression
- Memory pool management for large models
"""

from .advanced_checkpointing import (
    AdaptiveCheckpointing,
    DynamicActivationOffloading,
    MemoryEfficientBackprop,
    SelectiveGradientCheckpointing,
)
from .deep_optimizer_states import (
    CPUGPUHybridOptimizer,
    DeepOptimizerStates,
    InterleaveOffloadingOptimizer,
    MemoryConfig,
)
from .gradient_compression import (
    AdaptiveCompressionOptimizer,
    GradientCompressor,
    LossyGradientCompression,
    QuantizedGradientAccumulation,
)
from .long_sequence_optimization import (
    IncrementalSequenceCache,
    LongSequenceOptimizer,
    SegmentedAttentionMemory,
    StreamingSequenceProcessor,
)
from .memory_pool_management import (
    DynamicMemoryPool,
    MemoryFragmentationOptimizer,
    MemoryPoolManager,
    SmartMemoryAllocator,
)

__all__ = [
    # Deep optimizer states
    'DeepOptimizerStates',
    'InterleaveOffloadingOptimizer',
    'CPUGPUHybridOptimizer',
    'MemoryConfig',

    # Advanced checkpointing
    'SelectiveGradientCheckpointing',
    'AdaptiveCheckpointing',
    'MemoryEfficientBackprop',
    'DynamicActivationOffloading',

    # Memory pool management
    'DynamicMemoryPool',
    'MemoryPoolManager',
    'SmartMemoryAllocator',
    'MemoryFragmentationOptimizer',

    # Gradient compression
    'GradientCompressor',
    'LossyGradientCompression',
    'AdaptiveCompressionOptimizer',
    'QuantizedGradientAccumulation',

    # Long sequence optimization
    'LongSequenceOptimizer',
    'SegmentedAttentionMemory',
    'StreamingSequenceProcessor',
    'IncrementalSequenceCache'
]
