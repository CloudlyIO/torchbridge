"""
GPU Optimization Patterns for PyTorch

This module provides educational guides and practical implementations of
optimization patterns for building GPU-efficient PyTorch neural networks.

🎓 EDUCATIONAL PURPOSE:
These modules teach you how to recognize and implement optimization patterns
that lead to significant GPU performance improvements. Each pattern includes:
- Theoretical background and mathematical foundations
- Before/after implementation comparisons
- Performance measurement methodologies
- Production-ready implementation examples

🔧 OPTIMIZATION PATTERN CATEGORIES:
- fusion_strategies: When and how to fuse operations for GPU efficiency
- memory_efficiency: Patterns for optimal GPU memory utilization
- compute_intensity: Techniques for maximizing arithmetic intensity
- compiler_friendly: Writing code that optimizes well with torch.compile

💡 PRACTICAL VALUE:
These patterns provide reusable optimization strategies that can be applied
to any PyTorch model, with demonstrated performance improvements and
detailed explanations of why each pattern works.
"""

from .fusion_strategies import (
    identify_fusion_opportunities,
    apply_operation_fusion,
    validate_fusion_correctness,
    FusionPattern,
    COMMON_FUSION_PATTERNS
)

from .memory_efficiency import (
    analyze_memory_access_patterns,
    optimize_tensor_layouts,
    minimize_memory_allocations,
    MemoryOptimizationStrategy,
    MEMORY_OPTIMIZATION_GUIDE,
    MemoryEfficientSequential,
    AdaptiveMemoryManager
)

from .compute_intensity import (
    calculate_arithmetic_intensity,
    optimize_flop_to_byte_ratio,
    identify_compute_bottlenecks,
    ComputeOptimizationPattern,
    COMPUTE_INTENSITY_TARGETS,
    ComputeIntensityProfiler
)

from .compiler_friendly import (
    check_compilation_compatibility,
    optimize_for_torch_compile,
    avoid_compilation_pitfalls,
    CompilationPattern,
    COMPILER_BEST_PRACTICES,
    CompilerOptimizedModule,
    OptimizedLinearGELU,
    OptimizedTransformerBlock
)

__all__ = [
    # Fusion strategies
    "identify_fusion_opportunities",
    "apply_operation_fusion",
    "validate_fusion_correctness",
    "FusionPattern",
    "COMMON_FUSION_PATTERNS",

    # Memory efficiency
    "analyze_memory_access_patterns",
    "optimize_tensor_layouts",
    "minimize_memory_allocations",
    "MemoryOptimizationStrategy",
    "MEMORY_OPTIMIZATION_GUIDE",
    "MemoryEfficientSequential",
    "AdaptiveMemoryManager",

    # Compute intensity
    "calculate_arithmetic_intensity",
    "optimize_flop_to_byte_ratio",
    "identify_compute_bottlenecks",
    "ComputeOptimizationPattern",
    "COMPUTE_INTENSITY_TARGETS",
    "ComputeIntensityProfiler",

    # Compiler-friendly patterns
    "check_compilation_compatibility",
    "optimize_for_torch_compile",
    "avoid_compilation_pitfalls",
    "CompilationPattern",
    "COMPILER_BEST_PRACTICES",
    "CompilerOptimizedModule",
    "OptimizedLinearGELU",
    "OptimizedTransformerBlock",
]