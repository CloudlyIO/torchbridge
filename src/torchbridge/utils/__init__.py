"""
Utility modules for profiling, benchmarking, analysis, and infrastructure.
"""

from .cache import LRUCache, TTLCache
from .profiling import (
    ComparisonSuite,
    KernelProfiler,
    compare_functions,
    profile_model_inference,
    quick_benchmark,
)

# Optional imports for advanced features
try:
    from .ab_testing import (
        ABExperiment,  # noqa: F401
        HardwareABTestingFramework,  # noqa: F401
        StatisticalSignificance,  # noqa: F401
        TrafficSplitter,  # noqa: F401
    )
    _ab_testing_available = True
except ImportError:
    _ab_testing_available = False

try:
    from .universal_inference_engine import (
        InferenceRequest,  # noqa: F401
        InferenceResponse,  # noqa: F401
        RequestProfile,  # noqa: F401
        UniversalInferenceEngine,  # noqa: F401
    )
    _inference_engine_available = True
except ImportError:
    _inference_engine_available = False

# Build __all__ dynamically based on available imports
__all__ = [
    # Cache utilities (always available)
    'LRUCache',
    'TTLCache',
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
