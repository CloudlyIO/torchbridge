"""
PyTorch Optimization Benchmark Framework

Comprehensive benchmarking suite for comparing optimization techniques
against state-of-the-art implementations.
"""

from .benchmark_runner import BenchmarkRunner, BenchmarkConfig
from .baseline_implementations import (
    PyTorchNativeBaseline,
    FlashAttentionBaseline,
    HuggingFaceBaseline
)
from .metrics_collector import MetricsCollector, PerformanceMetrics

# Optional import for advanced analysis
try:
    from .analysis_engine import AnalysisEngine, StatisticalAnalysis
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    AnalysisEngine = None
    StatisticalAnalysis = None
    ADVANCED_ANALYSIS_AVAILABLE = False

__version__ = "1.0.0"
__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "PyTorchNativeBaseline",
    "FlashAttentionBaseline",
    "HuggingFaceBaseline",
    "MetricsCollector",
    "PerformanceMetrics",
    "ADVANCED_ANALYSIS_AVAILABLE"
]

if ADVANCED_ANALYSIS_AVAILABLE:
    __all__.extend(["AnalysisEngine", "StatisticalAnalysis"])