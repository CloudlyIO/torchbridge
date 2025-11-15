"""
Utility modules for profiling, benchmarking, and analysis.
"""

from .profiling import (
    KernelProfiler,
    ComparisonSuite,
    quick_benchmark,
    compare_functions,
    profile_model_inference
)

__all__ = [
    'KernelProfiler',
    'ComparisonSuite',
    'quick_benchmark',
    'compare_functions',
    'profile_model_inference'
]