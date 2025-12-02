"""
Utility modules for profiling, benchmarking, analysis, and infrastructure.
"""

from .profiling import (
    KernelProfiler,
    ComparisonSuite,
    quick_benchmark,
    compare_functions,
    profile_model_inference
)

# Optional imports for advanced features
try:
    from .ab_testing import (
        HardwareABTestingFramework,
        ABExperiment,
        TrafficSplitter,
        StatisticalSignificance
    )
    _ab_testing_available = True
except ImportError:
    _ab_testing_available = False

try:
    from .universal_inference_engine import (
        UniversalInferenceEngine,
        InferenceRequest,
        InferenceResponse,
        RequestProfile
    )
    _inference_engine_available = True
except ImportError:
    _inference_engine_available = False

# Build __all__ dynamically based on available imports
__all__ = [
    # Profiling and benchmarking (always available)
    'KernelProfiler',
    'ComparisonSuite',
    'quick_benchmark',
    'compare_functions',
    'profile_model_inference'
]

# Add optional exports if available
if _ab_testing_available:
    __all__.extend([
        'HardwareABTestingFramework',
        'ABExperiment',
        'TrafficSplitter',
        'StatisticalSignificance'
    ])

if _inference_engine_available:
    __all__.extend([
        'UniversalInferenceEngine',
        'InferenceRequest',
        'InferenceResponse',
        'RequestProfile'
    ])