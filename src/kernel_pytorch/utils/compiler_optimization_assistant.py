"""
Compiler Optimization Assistant (2025) - Refactored

This module now serves as a compatibility layer for the refactored compiler optimization system.
The functionality has been split into focused modules:

- model_analyzer.py: Model analysis and architecture inspection
- optimization_recommendations.py: Recommendation engine and implementation
- compiler_assistant.py: Main orchestration and unified interface

This maintains backward compatibility while providing better code organization.
"""

import warnings

# Import all functionality from split modules
from .model_analyzer import (
    CodeAnalyzer,
    ModelAnalysisResult,
    PerformanceBottleneckDetector
)

from .optimization_recommendations import (
    OptimizationRecommendation,
    OptimizationRecommendationEngine,
    OptimizationImplementer
)

from .compiler_assistant import (
    CompilerOptimizationAssistant,
    demonstrate_optimization_assistant
)

# Backward compatibility warning
warnings.warn(
    "compiler_optimization_assistant.py has been refactored into multiple focused modules. "
    "Consider importing from the specific modules directly: "
    "model_analyzer, optimization_recommendations, compiler_assistant",
    FutureWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
__all__ = [
    # Core data classes
    'OptimizationRecommendation',
    'ModelAnalysisResult',

    # Analysis classes
    'CodeAnalyzer',
    'PerformanceBottleneckDetector',

    # Recommendation classes
    'OptimizationRecommendationEngine',
    'OptimizationImplementer',

    # Main interface
    'CompilerOptimizationAssistant',
    'demonstrate_optimization_assistant'
]