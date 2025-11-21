"""
Comprehensive GPU Optimization Testing and Validation Framework (2025)

Advanced testing infrastructure for validating compiler optimizations, kernel performance,
and GPU utilization across simulated and real hardware environments.

Features:
- Hardware simulation and emulation framework
- Performance benchmarking and regression testing
- Memory and compute validation tools
- Automated CI/CD pipeline integration
- Cross-platform GPU testing (NVIDIA, AMD, Intel)
"""

from .hardware_simulator import (
    GPUSimulator,
    MemorySimulator,
    ComputeSimulator,
    create_hardware_simulator
)

from .performance_benchmarks import (
    PerformanceBenchmarkSuite,
    KernelBenchmark,
    CompilerBenchmark,
    create_benchmark_suite
)

from .validation_tools import (
    OptimizationValidator,
    MemoryValidator,
    PerformanceProfiler,
    create_validation_suite
)

from .integration_tests import (
    IntegrationTestRunner,
    HardwareTestSuite,
    CompilerTestSuite,
    create_integration_test_runner
)

from .ci_pipeline import (
    CIPipelineManager,
    TestEnvironmentManager,
    ResultsAggregator,
    create_ci_pipeline
)

__all__ = [
    # Hardware simulation
    'GPUSimulator',
    'MemorySimulator',
    'ComputeSimulator',
    'create_hardware_simulator',

    # Performance benchmarking
    'PerformanceBenchmarkSuite',
    'KernelBenchmark',
    'CompilerBenchmark',
    'create_benchmark_suite',

    # Validation tools
    'OptimizationValidator',
    'MemoryValidator',
    'PerformanceProfiler',
    'create_validation_suite',

    # Integration testing
    'IntegrationTestRunner',
    'HardwareTestSuite',
    'CompilerTestSuite',
    'create_integration_test_runner',

    # CI/CD pipeline
    'CIPipelineManager',
    'TestEnvironmentManager',
    'ResultsAggregator',
    'create_ci_pipeline'
]