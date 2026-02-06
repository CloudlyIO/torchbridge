"""
PyTorch Optimization Benchmark Framework

Comprehensive benchmarking suite for comparing optimization techniques
against state-of-the-art implementations.

Version: 0.5.0
"""

from .baseline_implementations import (
    FlashAttentionBaseline,
    HuggingFaceBaseline,
    PyTorchNativeBaseline,
)
from .benchmark_runner import BenchmarkConfig, BenchmarkRunner
from .metrics_collector import MetricsCollector, PerformanceMetrics

# Timing utilities
from .timing_utils import (
    TimingResult,
    benchmark_function,
    calculate_throughput,
    calculate_tokens_per_second,
    compare_results,
    create_summary_table,
    get_cuda_memory_snapshot,
    print_result,
    print_section_header,
    run_timed_iterations,
    timed,
    timer,
    track_memory,
)

# Optional import for advanced analysis
try:
    from .analysis_engine import AnalysisEngine, StatisticalAnalysis
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    AnalysisEngine = None
    StatisticalAnalysis = None
    ADVANCED_ANALYSIS_AVAILABLE = False

__version__ = "0.5.0"
__all__ = [
    # Runner and config
    "BenchmarkRunner",
    "BenchmarkConfig",
    # Baselines
    "PyTorchNativeBaseline",
    "FlashAttentionBaseline",
    "HuggingFaceBaseline",
    # Metrics
    "MetricsCollector",
    "PerformanceMetrics",
    # Timing utilities
    "TimingResult",
    "run_timed_iterations",
    "benchmark_function",
    "timer",
    "track_memory",
    "timed",
    "get_cuda_memory_snapshot",
    "calculate_throughput",
    "calculate_tokens_per_second",
    "compare_results",
    "print_section_header",
    "print_result",
    "create_summary_table",
    # Flags
    "ADVANCED_ANALYSIS_AVAILABLE",
]

if ADVANCED_ANALYSIS_AVAILABLE:
    __all__.extend(["AnalysisEngine", "StatisticalAnalysis"])
