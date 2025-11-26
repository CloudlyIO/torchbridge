"""
Scalable Evaluation and A/B Testing Framework

Comprehensive framework for large-scale model evaluation and A/B testing across
heterogeneous hardware configurations with statistical rigor and real-time monitoring.
"""

from .distributed_evaluation import (
    ScalableEvaluationFramework,
    EvaluationCluster,
    EvaluationTask,
    EvaluationResults
)

from .ab_testing import (
    HardwareABTestingFramework,
    ABExperiment,
    TrafficSplitter,
    StatisticalSignificance
)

from .performance_comparison import (
    HardwarePerformanceComparator,
    BenchmarkSuite,
    PerformanceProfile,
    CrossHardwareMetrics
)

from .metrics_collection import (
    MetricsBackend,
    RealTimeMetricsCollector,
    PerformanceMetrics,
    HardwareMetrics
)

from .evaluation_orchestrator import (
    EvaluationOrchestrator,
    EvaluationPipeline,
    AutomatedBenchmarking
)

__all__ = [
    # Distributed Evaluation
    'ScalableEvaluationFramework',
    'EvaluationCluster',
    'EvaluationTask',
    'EvaluationResults',

    # A/B Testing
    'HardwareABTestingFramework',
    'ABExperiment',
    'TrafficSplitter',
    'StatisticalSignificance',

    # Performance Comparison
    'HardwarePerformanceComparator',
    'BenchmarkSuite',
    'PerformanceProfile',
    'CrossHardwareMetrics',

    # Metrics Collection
    'MetricsBackend',
    'RealTimeMetricsCollector',
    'PerformanceMetrics',
    'HardwareMetrics',

    # Orchestration
    'EvaluationOrchestrator',
    'EvaluationPipeline',
    'AutomatedBenchmarking'
]