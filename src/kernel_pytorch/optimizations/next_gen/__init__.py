"""
Next-Generation Optimizations (2025)

Implementation of the absolute latest optimization techniques discovered through research:
- Advanced FlexAttention with FlashLight compiler framework
- PyGraph CUDA Graph automation
- Ultra-precision techniques (FP4, MXFP variants)
- FSDP2 with DTensor integration
- Advanced sparsity patterns (2:4 structured)
- Information entropy-based precision allocation

These implementations represent the cutting edge of 2025 optimization research.
"""

# Ultra-precision imports from consolidated module
from kernel_pytorch.precision.ultra_precision import (
    FP4Quantizer,
    InformationEntropyPrecision,
    MXFPOptimizer,
    PrecisionFormat,  # noqa: F401
)
from kernel_pytorch.precision.ultra_precision import (
    ModelPrecisionOptimizer as AdaptivePrecisionAllocator,  # Alias for backward compatibility
)

from .advanced_flex_attention import (
    AdvancedFlexAttention,
    FlashLightCompiler,
    GQAOptimizedAttention,
    PagedAttentionDecoder,
    create_advanced_flex_attention,
)
from .fsdp2_integration import (
    AdvancedPrefetching,
    DTensorSharding,
    FSDP2Config,
    FSDP2Manager,
    HybridShardingOptimizer,
    create_fsdp2_manager,
)
from .pygraph_optimizer import (
    AutoGraphCapture,
    CUDAGraphManager,
    PyGraphOptimizer,
    SelectiveCUDAGraphs,
    create_pygraph_optimizer,
)
from .structured_sparsity import (
    AcceleratedSparseOps,
    DynamicSparsityOptimizer,
    SparsityPatternGenerator,
    StructuredSparsity24,
    create_structured_sparsity_optimizer,
)

__all__ = [
    # Advanced FlexAttention
    'FlashLightCompiler',
    'AdvancedFlexAttention',
    'GQAOptimizedAttention',
    'PagedAttentionDecoder',
    'create_advanced_flex_attention',

    # PyGraph optimization
    'PyGraphOptimizer',
    'CUDAGraphManager',
    'AutoGraphCapture',
    'SelectiveCUDAGraphs',
    'create_pygraph_optimizer',

    # Ultra-precision techniques
    'FP4Quantizer',
    'MXFPOptimizer',
    'InformationEntropyPrecision',
    'AdaptivePrecisionAllocator',

    # FSDP2 integration
    'FSDP2Manager',
    'DTensorSharding',
    'AdvancedPrefetching',
    'HybridShardingOptimizer',
    'FSDP2Config',
    'create_fsdp2_manager',

    # Structured sparsity
    'StructuredSparsity24',
    'DynamicSparsityOptimizer',
    'SparsityPatternGenerator',
    'AcceleratedSparseOps',
    'create_structured_sparsity_optimizer'
]
