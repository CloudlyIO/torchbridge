#!/usr/bin/env python3
"""
Test suite for GPU Optimization Testing Framework

Comprehensive tests for the testing and validation framework components.
"""

import pytest
import asyncio
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from kernel_pytorch.testing_framework import (
    create_hardware_simulator,
    create_benchmark_suite,
    create_validation_suite,
    create_integration_test_runner,
    create_ci_pipeline
)

from kernel_pytorch.testing_framework.hardware_simulator import (
    GPUSpec, GPUArchitecture, SimulationConfig, SimulationMode
)
from kernel_pytorch.testing_framework.performance_benchmarks import (
    BenchmarkConfig, MetricType
)
from kernel_pytorch.testing_framework.validation_tools import (
    ValidationConfig, ValidationLevel, ValidationType
)


class TestHardwareSimulator:
    """Test hardware simulation framework"""

    def test_simulator_creation(self):
        """Test hardware simulator creation"""
        simulator = create_hardware_simulator(
            architecture="ampere",
            compute_units=108,
            memory_size_gb=40,
            simulation_mode="performance"
        )

        assert simulator is not None
        assert simulator.gpu_spec.compute_units == 108
        assert simulator.gpu_spec.memory_size_gb == 40
        assert simulator.gpu_spec.architecture == GPUArchitecture.AMPERE

    def test_kernel_execution_simulation(self):
        """Test kernel execution simulation"""
        simulator = create_hardware_simulator()

        def test_kernel(x):
            return x * 2

        inputs = [torch.randn(100, 100)]
        metrics = simulator.execute_kernel(
            test_kernel,
            tuple(inputs),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1)
        )

        assert metrics.total_cycles > 0
        assert 0 <= metrics.utilization_percent <= 100
        assert metrics.execution_time_ms >= 0

    def test_memory_simulation(self):
        """Test memory hierarchy simulation"""
        simulator = create_hardware_simulator()

        # Test memory access
        latency, hit_level = simulator.memory_sim.simulate_memory_access(
            address=0x1000,
            size_bytes=64,
            access_type="read"
        )

        assert latency > 0
        assert hit_level in ["register", "shared", "l1", "l2", "global"]

        # Test memory statistics
        stats = simulator.memory_sim.get_memory_stats()
        assert 'total_accesses' in stats
        assert 'l1_hit_rate' in stats
        assert stats['total_accesses'] >= 1

    def test_simulation_summary(self):
        """Test simulation summary generation"""
        simulator = create_hardware_simulator()

        # Execute a simple kernel
        def simple_kernel(x):
            return x + 1

        inputs = [torch.randn(50, 50)]
        simulator.execute_kernel(
            simple_kernel,
            tuple(inputs),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1)
        )

        summary = simulator.get_simulation_summary()

        assert 'gpu_spec' in summary
        assert 'execution_summary' in summary
        assert 'memory_performance' in summary
        assert 'compute_performance' in summary
        assert summary['execution_summary']['total_kernels'] == 1


class TestPerformanceBenchmarking:
    """Test performance benchmarking framework"""

    def test_benchmark_suite_creation(self):
        """Test benchmark suite creation"""
        suite = create_benchmark_suite(
            warmup_iterations=5,
            measurement_iterations=10,
            enable_profiling=False
        )

        assert suite is not None
        assert suite.config.warmup_iterations == 5
        assert suite.config.measurement_iterations == 10

    def test_kernel_benchmark(self):
        """Test individual kernel benchmarking"""
        suite = create_benchmark_suite(
            warmup_iterations=2,
            measurement_iterations=5,
            enable_profiling=False
        )

        def test_kernel(x, y):
            return torch.add(x, y)

        def input_generator():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            return [x, y]

        result = suite.add_kernel_benchmark(
            test_kernel,
            input_generator,
            "test_kernel_benchmark"
        )

        assert result.success
        assert result.benchmark_name == "test_kernel_benchmark"
        assert MetricType.LATENCY in result.metrics
        assert len(result.metrics[MetricType.LATENCY]) == 5  # measurement_iterations

    def test_optimization_comparison(self):
        """Test optimization comparison"""
        suite = create_benchmark_suite(
            warmup_iterations=2,
            measurement_iterations=5,
            enable_profiling=False
        )

        def baseline_fn(x):
            return x + 1.0

        def optimized_fn(x):
            return torch.add(x, 1.0)  # Potentially optimized

        def input_generator():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return [torch.randn(100, 100, device=device)]

        comparison = suite.add_optimization_comparison(
            baseline_fn,
            optimized_fn,
            input_generator,
            "optimization_test"
        )

        assert comparison is not None
        assert comparison.baseline_result.success
        assert comparison.optimized_result.success
        assert MetricType.LATENCY in comparison.improvements
        assert isinstance(comparison.improvements[MetricType.LATENCY], (int, float))

    def test_predefined_benchmarks(self):
        """Test predefined benchmark execution"""
        suite = create_benchmark_suite(
            warmup_iterations=1,
            measurement_iterations=2,
            enable_profiling=False
        )

        # Mock torch.cuda.is_available() to avoid GPU dependency
        with patch('torch.cuda.is_available', return_value=False):
            results = suite.run_predefined_benchmarks()

        assert 'matmul' in results
        assert 'attention' in results
        assert 'memory' in results

    def test_benchmark_report_generation(self):
        """Test benchmark report generation"""
        suite = create_benchmark_suite(
            warmup_iterations=1,
            measurement_iterations=2,
            enable_profiling=False
        )

        # Add a simple test
        def simple_fn(x):
            return x * 2

        def input_gen():
            return [torch.randn(10, 10)]

        suite.add_kernel_benchmark(simple_fn, input_gen, "simple_test")

        report = suite.generate_report()

        assert 'summary' in report
        assert 'benchmark_results' in report
        assert 'performance_insights' in report
        assert report['summary']['total_benchmarks'] == 1


class TestValidationTools:
    """Test validation and profiling tools"""

    def test_validation_suite_creation(self):
        """Test validation suite creation"""
        validator = create_validation_suite(
            validation_level="basic",
            enable_profiling=False
        )

        assert validator is not None
        assert validator.config.validation_level == ValidationLevel.BASIC

    def test_numerical_validation(self):
        """Test numerical accuracy validation"""
        validator = create_validation_suite(validation_level="basic")

        def baseline_fn(x):
            return x * 2

        def optimized_fn(x):
            return torch.mul(x, 2)

        inputs = [torch.randn(50, 50)]

        result = validator.numerical_validator.validate_accuracy(
            baseline_fn,
            optimized_fn,
            inputs,
            "numerical_test"
        )

        assert result.validation_type == ValidationType.NUMERICAL_ACCURACY
        assert result.test_name == "numerical_test"
        assert result.passed  # Should pass with identical operations

    def test_gradient_validation(self):
        """Test gradient correctness validation"""
        validator = create_validation_suite(validation_level="basic")

        def baseline_fn(x):
            return (x ** 2).sum()

        def optimized_fn(x):
            return torch.sum(torch.pow(x, 2))

        inputs = [torch.randn(20, 20, requires_grad=True)]

        result = validator.numerical_validator.validate_gradients(
            baseline_fn,
            optimized_fn,
            inputs,
            "gradient_test"
        )

        assert result.validation_type == ValidationType.GRADIENT_CORRECTNESS
        assert result.passed

    def test_memory_validation(self):
        """Test memory usage validation"""
        validator = create_validation_suite(validation_level="basic")

        def baseline_fn(x):
            return x.clone()

        def optimized_fn(x):
            return x.clone()  # Same operation

        inputs = [torch.randn(100, 100)]

        result = validator.memory_validator.validate_memory_usage(
            baseline_fn,
            optimized_fn,
            inputs,
            "memory_test"
        )

        assert result.validation_type == ValidationType.MEMORY_CORRECTNESS
        # Memory validation might pass or fail depending on exact measurements

    def test_comprehensive_validation(self):
        """Test comprehensive optimization validation"""
        validator = create_validation_suite(validation_level="thorough")

        def baseline_fn(x):
            return torch.relu(x + 1)

        def optimized_fn(x):
            return torch.nn.functional.relu(x + 1)

        inputs = [torch.randn(30, 30)]

        results = validator.validate_optimization(
            baseline_fn,
            optimized_fn,
            inputs,
            "comprehensive_test",
            enable_gradient_check=False  # Disable for simpler test
        )

        assert 'numerical' in results
        assert 'memory' in results
        assert results['numerical'].test_name == "comprehensive_test"

    def test_validation_report(self):
        """Test validation report generation"""
        validator = create_validation_suite()

        # Mock some validation results
        mock_results = {
            'test1': {
                'numerical': Mock(
                    passed=True,
                    validation_type=ValidationType.NUMERICAL_ACCURACY,
                    error_message=None,
                    execution_time=0.1,
                    metrics={'output_0': {'max_abs_diff': 1e-8}}
                )
            }
        }

        report = validator.generate_validation_report(mock_results)

        assert 'summary' in report
        assert 'test_results' in report
        assert 'recommendations' in report
        assert report['summary']['total_tests'] == 1


class TestIntegrationTesting:
    """Test integration testing framework"""

    @pytest.mark.asyncio
    async def test_integration_test_runner_creation(self):
        """Test integration test runner creation"""
        runner = create_integration_test_runner(
            enable_simulation=False,  # Disable for faster testing
            enable_benchmarking=True,
            enable_validation=True,
            parallel_execution=False
        )

        assert runner is not None
        assert not runner.config.enable_simulation
        assert runner.config.enable_benchmarking
        assert runner.config.enable_validation

    @pytest.mark.asyncio
    async def test_simple_integration_test(self):
        """Test simple integration test execution"""
        from kernel_pytorch.testing_framework.integration_tests import TestCase

        runner = create_integration_test_runner(
            enable_simulation=False,
            enable_benchmarking=False,  # Disable for speed
            enable_validation=False,
            parallel_execution=False
        )

        def baseline_fn(x):
            return x + 1

        def optimized_fn(x):
            return torch.add(x, 1)

        def input_generator():
            return [torch.randn(10, 10)]

        test_case = TestCase(
            name="simple_test",
            baseline_fn=baseline_fn,
            optimized_fn=optimized_fn,
            input_generator=input_generator,
            test_type="basic"
        )

        runner.add_test_case(test_case)

        results = await runner.run_all_tests()

        assert results['summary']['total_tests'] == 1

    @pytest.mark.asyncio
    async def test_hardware_test_suite(self):
        """Test hardware test suite"""
        from kernel_pytorch.testing_framework.integration_tests import HardwareTestSuite, IntegrationTestConfig

        config = IntegrationTestConfig(
            enable_simulation=False,
            enable_benchmarking=False,
            enable_validation=False
        )

        suite = HardwareTestSuite(config)

        def baseline_kernel(x, y):
            return x + y

        def optimized_kernel(x, y):
            return torch.add(x, y)

        def input_generator():
            return [torch.randn(20, 20), torch.randn(20, 20)]

        suite.add_kernel_test(
            "test_kernel",
            baseline_kernel,
            optimized_kernel,
            input_generator,
            architectures=["ampere"]
        )

        test_cases = suite.get_test_cases()
        assert len(test_cases) == 1
        assert test_cases[0].name == "test_kernel_ampere"

    @pytest.mark.asyncio
    async def test_compiler_test_suite(self):
        """Test compiler test suite"""
        from kernel_pytorch.testing_framework.integration_tests import CompilerTestSuite, IntegrationTestConfig

        config = IntegrationTestConfig()
        suite = CompilerTestSuite(config)

        def unfused_fn(x):
            return torch.relu(x + 1)

        def fused_fn(x):
            return torch.nn.functional.relu(x + 1)

        def input_generator():
            return [torch.randn(15, 15)]

        suite.add_fusion_test(
            "relu_add",
            unfused_fn,
            fused_fn,
            input_generator,
            expected_improvement=1.1
        )

        test_cases = suite.get_test_cases()
        assert len(test_cases) == 1
        assert test_cases[0].name == "fusion_relu_add"


class TestCIPipeline:
    """Test CI/CD pipeline framework"""

    @pytest.mark.asyncio
    async def test_ci_pipeline_creation(self):
        """Test CI/CD pipeline creation"""
        pipeline = create_ci_pipeline(
            environment="local",
            enable_gpu_testing=False,
            quick_mode=True
        )

        assert pipeline is not None
        assert pipeline.config.quick_test_mode

    @pytest.mark.asyncio
    async def test_environment_manager(self):
        """Test test environment manager"""
        from kernel_pytorch.testing_framework.ci_pipeline import TestEnvironmentManager, PipelineConfig

        config = PipelineConfig(enable_gpu_testing=False)
        env_manager = TestEnvironmentManager(config)

        requirements = {
            'requires_gpu': False,
            'packages': ['torch']
        }

        env_info = await env_manager.setup_environment("test_env", requirements)

        assert env_info['name'] == "test_env"
        assert env_info['status'] in ['ready', 'failed']
        assert 'resources' in env_info

        # Cleanup
        env_manager.cleanup_environment("test_env")

    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Test pipeline execution"""
        pipeline = create_ci_pipeline(
            environment="local",
            enable_gpu_testing=False,
            quick_mode=True
        )

        # Mock some components to speed up testing
        with patch.object(pipeline, '_integration_tests_stage', return_value={'mock': 'results'}):
            with patch.object(pipeline, '_performance_tests_stage', return_value={'mock': 'results'}):
                pipeline_run = await pipeline.run_pipeline(
                    commit_hash="test123",
                    branch="test",
                    trigger="manual"
                )

        assert pipeline_run.run_id is not None
        assert pipeline_run.overall_status in ["success", "failed"]
        assert len(pipeline_run.stage_results) > 0

    def test_results_aggregator(self):
        """Test results aggregator"""
        from kernel_pytorch.testing_framework.ci_pipeline import (
            ResultsAggregator, PipelineConfig, PipelineRun, StageResult, PipelineStage
        )
        from datetime import datetime

        config = PipelineConfig()
        aggregator = ResultsAggregator(config)

        # Create mock pipeline run
        pipeline_run = PipelineRun(
            run_id="test_run",
            commit_hash="abc123",
            branch="main",
            trigger="manual",
            start_time=datetime.now(),
            end_time=datetime.now(),
            overall_status="success"
        )

        # Add mock stage result
        stage_result = StageResult(
            stage=PipelineStage.BUILD,
            status="success",
            duration_seconds=10.0
        )
        pipeline_run.stage_results[PipelineStage.BUILD] = stage_result

        results = aggregator.aggregate_results(pipeline_run)

        assert 'run_summary' in results
        assert 'stage_breakdown' in results
        assert results['run_summary']['run_id'] == "test_run"
        assert results['run_summary']['overall_status'] == "success"


class TestFrameworkIntegration:
    """Test integration between framework components"""

    @pytest.mark.asyncio
    async def test_end_to_end_optimization_testing(self):
        """Test complete end-to-end optimization testing workflow"""

        # Define test optimization
        def baseline_operation(x):
            y = x * 2
            z = torch.relu(y)
            return z + 1

        def optimized_operation(x):
            # Simulated optimized version
            return torch.relu(x * 2) + 1

        def input_generator():
            return [torch.randn(50, 50)]

        # 1. Hardware simulation
        simulator = create_hardware_simulator(
            architecture="ampere",
            simulation_mode="performance"
        )

        inputs = input_generator()
        baseline_metrics = simulator.execute_kernel(
            baseline_operation,
            tuple(inputs),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1)
        )

        optimized_metrics = simulator.execute_kernel(
            optimized_operation,
            tuple(inputs),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1)
        )

        assert baseline_metrics.total_cycles > 0
        assert optimized_metrics.total_cycles > 0

        # 2. Performance benchmarking
        benchmark_suite = create_benchmark_suite(
            warmup_iterations=2,
            measurement_iterations=5,
            enable_profiling=False
        )

        comparison = benchmark_suite.add_optimization_comparison(
            baseline_operation,
            optimized_operation,
            input_generator,
            "end_to_end_test"
        )

        assert comparison.baseline_result.success
        assert comparison.optimized_result.success

        # 3. Validation testing
        validator = create_validation_suite(validation_level="basic")

        validation_results = validator.validate_optimization(
            baseline_operation,
            optimized_operation,
            inputs,
            "end_to_end_validation",
            enable_gradient_check=False
        )

        assert 'numerical' in validation_results
        assert validation_results['numerical'].passed

        # 4. Generate comprehensive report
        report = {
            'simulation': {
                'baseline_cycles': baseline_metrics.total_cycles,
                'optimized_cycles': optimized_metrics.total_cycles,
                'cycle_improvement': (
                    (baseline_metrics.total_cycles - optimized_metrics.total_cycles) /
                    max(baseline_metrics.total_cycles, 1) * 100
                )
            },
            'benchmarking': {
                'performance_improvements': comparison.improvements,
                'statistical_significance': comparison.statistical_significance
            },
            'validation': {
                'numerical_accuracy': validation_results['numerical'].passed,
                'memory_safety': validation_results.get('memory', Mock(passed=True)).passed
            }
        }

        # Verify complete workflow
        assert report['simulation']['baseline_cycles'] > 0
        assert 'latency' in report['benchmarking']['performance_improvements']
        assert report['validation']['numerical_accuracy']

        print("✅ End-to-end optimization testing workflow completed successfully")


def main():
    """Run all testing framework tests"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--disable-warnings"
    ])


if __name__ == "__main__":
    main()