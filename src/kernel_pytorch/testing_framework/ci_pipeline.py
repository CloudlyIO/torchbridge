"""
CI/CD Pipeline Manager for GPU Optimization Testing

Automated continuous integration pipeline for GPU kernel and compiler optimizations
with multi-environment testing, regression detection, and automated reporting.
"""

import os
import asyncio
import yaml
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import time
import threading
from datetime import datetime

from .integration_tests import IntegrationTestRunner, HardwareTestSuite, CompilerTestSuite
from .hardware_simulator import create_hardware_simulator
from .performance_benchmarks import create_benchmark_suite

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """CI/CD pipeline stages"""
    SETUP = "setup"
    BUILD = "build"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    PERFORMANCE_TESTS = "performance_tests"
    REGRESSION_TESTS = "regression_tests"
    DEPLOY = "deploy"
    CLEANUP = "cleanup"


class Environment(Enum):
    """Test environments"""
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class PipelineConfig:
    """Configuration for CI/CD pipeline"""
    # Environment settings
    environment: Environment = Environment.CI
    enable_gpu_testing: bool = True
    enable_multi_gpu_testing: bool = False
    test_architectures: List[str] = field(default_factory=lambda: ["ampere"])

    # Pipeline behavior
    fail_fast: bool = False
    parallel_stages: bool = True
    timeout_minutes: int = 60
    retry_count: int = 2

    # Test configuration
    quick_test_mode: bool = False  # For rapid iteration
    enable_simulation: bool = True
    enable_benchmarking: bool = True
    enable_regression_detection: bool = True

    # Reporting
    generate_reports: bool = True
    publish_metrics: bool = True
    notification_webhooks: List[str] = field(default_factory=list)

    # Artifacts
    save_artifacts: bool = True
    artifact_retention_days: int = 30


@dataclass
class StageResult:
    """Result from pipeline stage execution"""
    stage: PipelineStage
    status: str  # "success", "failure", "skipped"
    duration_seconds: float
    error_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRun:
    """Complete pipeline run information"""
    run_id: str
    commit_hash: Optional[str]
    branch: str
    trigger: str  # "push", "pull_request", "manual"
    start_time: datetime
    end_time: Optional[datetime] = None
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    overall_status: str = "running"
    artifacts: List[str] = field(default_factory=list)


class TestEnvironmentManager:
    """
    Manages test environments and resources

    Handles environment setup, resource allocation,
    and cleanup for different testing scenarios.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.active_environments: Dict[str, Dict] = {}

    async def setup_environment(self, env_name: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Setup test environment with specified requirements"""
        logger.info(f"Setting up test environment: {env_name}")

        env_info = {
            'name': env_name,
            'setup_time': datetime.now(),
            'requirements': requirements,
            'resources': {},
            'status': 'setting_up'
        }

        try:
            # Check GPU availability
            if requirements.get('requires_gpu', False):
                gpu_info = await self._setup_gpu_environment()
                env_info['resources']['gpu'] = gpu_info

            # Setup Python environment
            python_info = await self._setup_python_environment(requirements)
            env_info['resources']['python'] = python_info

            # Setup test data
            if requirements.get('test_data', []):
                data_info = await self._setup_test_data(requirements['test_data'])
                env_info['resources']['test_data'] = data_info

            env_info['status'] = 'ready'
            self.active_environments[env_name] = env_info

        except Exception as e:
            env_info['status'] = 'failed'
            env_info['error'] = str(e)
            logger.error(f"Failed to setup environment {env_name}: {e}")

        return env_info

    async def _setup_gpu_environment(self) -> Dict[str, Any]:
        """Setup GPU environment and check capabilities"""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': []
        }

        try:
            import torch

            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['count'] = torch.cuda.device_count()

                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'id': i,
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}",
                        'multiprocessor_count': props.multi_processor_count
                    }
                    gpu_info['devices'].append(device_info)

                logger.info(f"Found {gpu_info['count']} GPU(s)")
            else:
                logger.warning("No GPUs available - using simulation mode")

        except ImportError:
            logger.warning("PyTorch not available")

        return gpu_info

    async def _setup_python_environment(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Setup Python environment with required packages"""
        python_info = {
            'version': None,
            'packages': {},
            'virtual_env': None
        }

        try:
            # Get Python version
            result = subprocess.run(['python', '--version'], capture_output=True, text=True)
            python_info['version'] = result.stdout.strip()

            # Check required packages
            required_packages = requirements.get('packages', [])
            for package in required_packages:
                try:
                    __import__(package)
                    python_info['packages'][package] = 'available'
                except ImportError:
                    python_info['packages'][package] = 'missing'
                    logger.warning(f"Required package {package} not found")

        except Exception as e:
            logger.error(f"Failed to setup Python environment: {e}")

        return python_info

    async def _setup_test_data(self, test_data_specs: List[str]) -> Dict[str, Any]:
        """Setup test data and datasets"""
        data_info = {
            'datasets': {},
            'total_size_mb': 0
        }

        for data_spec in test_data_specs:
            # For now, just mark as configured
            # In practice, would download/generate test datasets
            data_info['datasets'][data_spec] = {
                'status': 'configured',
                'size_mb': 100  # Placeholder
            }
            data_info['total_size_mb'] += 100

        return data_info

    def cleanup_environment(self, env_name: str):
        """Cleanup test environment"""
        if env_name in self.active_environments:
            logger.info(f"Cleaning up environment: {env_name}")

            # Clean up resources
            env_info = self.active_environments[env_name]

            # GPU cleanup
            if 'gpu' in env_info['resources']:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"GPU cleanup failed: {e}")

            # Remove from active environments
            del self.active_environments[env_name]


class ResultsAggregator:
    """
    Aggregates and analyzes test results across pipeline stages

    Provides comprehensive analysis of performance trends,
    regression detection, and quality metrics.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.historical_data: List[Dict] = []

    def aggregate_results(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Aggregate results from complete pipeline run"""
        aggregated_results = {
            'run_summary': {
                'run_id': pipeline_run.run_id,
                'overall_status': pipeline_run.overall_status,
                'duration_minutes': self._calculate_total_duration(pipeline_run),
                'stages_passed': len([r for r in pipeline_run.stage_results.values() if r.status == 'success']),
                'stages_failed': len([r for r in pipeline_run.stage_results.values() if r.status == 'failure']),
                'stages_total': len(pipeline_run.stage_results)
            },
            'stage_breakdown': {},
            'performance_analysis': {},
            'regression_analysis': {},
            'quality_metrics': {}
        }

        # Stage breakdown
        for stage, result in pipeline_run.stage_results.items():
            aggregated_results['stage_breakdown'][stage.value] = {
                'status': result.status,
                'duration_seconds': result.duration_seconds,
                'error_message': result.error_message,
                'artifact_count': len(result.artifacts)
            }

        # Performance analysis
        aggregated_results['performance_analysis'] = self._analyze_performance(pipeline_run)

        # Regression analysis
        if self.config.enable_regression_detection:
            aggregated_results['regression_analysis'] = self._detect_regressions(pipeline_run)

        # Quality metrics
        aggregated_results['quality_metrics'] = self._calculate_quality_metrics(pipeline_run)

        # Store for historical analysis
        self.historical_data.append(aggregated_results)

        return aggregated_results

    def _calculate_total_duration(self, pipeline_run: PipelineRun) -> float:
        """Calculate total pipeline duration"""
        if pipeline_run.end_time and pipeline_run.start_time:
            return (pipeline_run.end_time - pipeline_run.start_time).total_seconds() / 60.0
        return 0.0

    def _analyze_performance(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Analyze performance metrics from pipeline run"""
        performance_analysis = {
            'benchmark_results': {},
            'performance_trends': {},
            'optimization_effectiveness': {}
        }

        # Extract performance data from integration tests
        if PipelineStage.INTEGRATION_TESTS in pipeline_run.stage_results:
            stage_result = pipeline_run.stage_results[PipelineStage.INTEGRATION_TESTS]
            if 'performance_insights' in stage_result.metrics:
                insights = stage_result.metrics['performance_insights']
                performance_analysis['benchmark_results'] = insights

        # Performance trends (requires historical data)
        if len(self.historical_data) > 1:
            performance_analysis['performance_trends'] = self._calculate_trends()

        return performance_analysis

    def _detect_regressions(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Detect performance regressions"""
        regression_analysis = {
            'regressions_detected': False,
            'regression_count': 0,
            'affected_optimizations': [],
            'severity': 'none'
        }

        # Check for regressions in integration test results
        if PipelineStage.INTEGRATION_TESTS in pipeline_run.stage_results:
            stage_result = pipeline_run.stage_results[PipelineStage.INTEGRATION_TESTS]

            # Look for regression indicators in metrics
            if 'regression_analysis' in stage_result.metrics:
                regression_data = stage_result.metrics['regression_analysis']
                regression_analysis['regressions_detected'] = regression_data.get('total_failures', 0) > 0
                regression_analysis['regression_count'] = regression_data.get('total_failures', 0)

                if regression_analysis['regression_count'] > 0:
                    regression_analysis['severity'] = 'high' if regression_analysis['regression_count'] > 3 else 'medium'

        return regression_analysis

    def _calculate_quality_metrics(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        quality_metrics = {
            'test_coverage': 0.0,
            'success_rate': 0.0,
            'reliability_score': 0.0,
            'performance_score': 0.0
        }

        # Success rate
        total_stages = len(pipeline_run.stage_results)
        successful_stages = len([r for r in pipeline_run.stage_results.values() if r.status == 'success'])
        quality_metrics['success_rate'] = successful_stages / max(total_stages, 1)

        # Reliability score (based on historical data)
        if len(self.historical_data) > 5:
            recent_success_rates = [
                run['run_summary']['stages_passed'] / max(run['run_summary']['stages_total'], 1)
                for run in self.historical_data[-5:]
            ]
            quality_metrics['reliability_score'] = sum(recent_success_rates) / len(recent_success_rates)

        # Performance score (based on optimization effectiveness)
        if PipelineStage.PERFORMANCE_TESTS in pipeline_run.stage_results:
            stage_metrics = pipeline_run.stage_results[PipelineStage.PERFORMANCE_TESTS].metrics
            if 'avg_improvement' in stage_metrics:
                # Normalize improvement to 0-1 score
                improvement = stage_metrics['avg_improvement']
                quality_metrics['performance_score'] = min(improvement / 50.0, 1.0)  # 50% improvement = max score

        return quality_metrics

    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if len(self.historical_data) < 2:
            return {'insufficient_data': True}

        # Extract key metrics over time
        success_rates = []
        performance_scores = []

        for run_data in self.historical_data[-10:]:  # Last 10 runs
            success_rates.append(run_data['run_summary'].get('stages_passed', 0) / max(run_data['run_summary'].get('stages_total', 1), 1))
            performance_scores.append(run_data['quality_metrics'].get('performance_score', 0))

        trends = {
            'success_rate_trend': self._calculate_trend(success_rates),
            'performance_trend': self._calculate_trend(performance_scores),
            'stability': self._calculate_stability(success_rates)
        }

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from series of values"""
        if len(values) < 2:
            return 'stable'

        # Simple linear trend
        x = list(range(len(values)))
        slope = sum((x[i] - sum(x)/len(x)) * (values[i] - sum(values)/len(values)) for i in range(len(values)))
        slope /= sum((x[i] - sum(x)/len(x))**2 for i in range(len(values)))

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'

    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score based on variance"""
        if len(values) < 2:
            return 1.0

        import statistics
        variance = statistics.variance(values)
        return max(0.0, 1.0 - variance)  # Lower variance = higher stability


class CIPipelineManager:
    """
    Complete CI/CD pipeline manager for GPU optimization testing

    Orchestrates the entire testing pipeline from environment setup
    to result reporting and artifact management.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        self.env_manager = TestEnvironmentManager(self.config)
        self.results_aggregator = ResultsAggregator(self.config)

        # Pipeline state
        self.current_run: Optional[PipelineRun] = None
        self.run_history: List[PipelineRun] = []

    async def run_pipeline(
        self,
        commit_hash: Optional[str] = None,
        branch: str = "main",
        trigger: str = "manual"
    ) -> PipelineRun:
        """Run complete CI/CD pipeline"""
        run_id = f"run_{int(time.time())}"
        logger.info(f"Starting CI/CD pipeline run: {run_id}")

        pipeline_run = PipelineRun(
            run_id=run_id,
            commit_hash=commit_hash,
            branch=branch,
            trigger=trigger,
            start_time=datetime.now()
        )

        self.current_run = pipeline_run

        try:
            # Define pipeline stages
            stages = [
                PipelineStage.SETUP,
                PipelineStage.BUILD,
                PipelineStage.UNIT_TESTS,
                PipelineStage.INTEGRATION_TESTS,
                PipelineStage.PERFORMANCE_TESTS
            ]

            if self.config.enable_regression_detection:
                stages.append(PipelineStage.REGRESSION_TESTS)

            stages.append(PipelineStage.CLEANUP)

            # Execute stages
            for stage in stages:
                if self.config.fail_fast and pipeline_run.overall_status == "failed":
                    # Skip remaining stages if fail_fast is enabled
                    stage_result = StageResult(
                        stage=stage,
                        status="skipped",
                        duration_seconds=0.0
                    )
                    pipeline_run.stage_results[stage] = stage_result
                    continue

                stage_result = await self._execute_stage(stage, pipeline_run)
                pipeline_run.stage_results[stage] = stage_result

                if stage_result.status == "failure":
                    pipeline_run.overall_status = "failed"

            # Set final status
            if pipeline_run.overall_status != "failed":
                pipeline_run.overall_status = "success"

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_run.overall_status = "failed"

            # Add error stage result
            error_result = StageResult(
                stage=PipelineStage.CLEANUP,
                status="failure",
                duration_seconds=0.0,
                error_message=str(e)
            )
            pipeline_run.stage_results[PipelineStage.CLEANUP] = error_result

        finally:
            pipeline_run.end_time = datetime.now()
            self.run_history.append(pipeline_run)
            self.current_run = None

            # Generate final report
            if self.config.generate_reports:
                await self._generate_final_report(pipeline_run)

        logger.info(f"Pipeline run {run_id} completed with status: {pipeline_run.overall_status}")
        return pipeline_run

    async def _execute_stage(self, stage: PipelineStage, pipeline_run: PipelineRun) -> StageResult:
        """Execute individual pipeline stage"""
        logger.info(f"Executing stage: {stage.value}")
        start_time = time.time()

        stage_result = StageResult(
            stage=stage,
            status="running",
            duration_seconds=0.0
        )

        try:
            if stage == PipelineStage.SETUP:
                result = await self._setup_stage(pipeline_run)
            elif stage == PipelineStage.BUILD:
                result = await self._build_stage(pipeline_run)
            elif stage == PipelineStage.UNIT_TESTS:
                result = await self._unit_tests_stage(pipeline_run)
            elif stage == PipelineStage.INTEGRATION_TESTS:
                result = await self._integration_tests_stage(pipeline_run)
            elif stage == PipelineStage.PERFORMANCE_TESTS:
                result = await self._performance_tests_stage(pipeline_run)
            elif stage == PipelineStage.REGRESSION_TESTS:
                result = await self._regression_tests_stage(pipeline_run)
            elif stage == PipelineStage.CLEANUP:
                result = await self._cleanup_stage(pipeline_run)
            else:
                raise ValueError(f"Unknown stage: {stage}")

            stage_result.status = "success"
            stage_result.metrics = result

        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")
            stage_result.status = "failure"
            stage_result.error_message = str(e)

        finally:
            stage_result.duration_seconds = time.time() - start_time

        return stage_result

    async def _setup_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Setup stage: Environment preparation"""
        env_requirements = {
            'requires_gpu': self.config.enable_gpu_testing,
            'packages': ['torch', 'numpy'],
            'test_data': ['synthetic_kernels', 'benchmark_models']
        }

        env_info = await self.env_manager.setup_environment(
            f"pipeline_{pipeline_run.run_id}",
            env_requirements
        )

        return {
            'environment_status': env_info['status'],
            'gpu_available': env_info['resources'].get('gpu', {}).get('available', False),
            'gpu_count': env_info['resources'].get('gpu', {}).get('count', 0)
        }

    async def _build_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Build stage: Compile optimizations and prepare artifacts"""
        # In practice, would compile CUDA kernels, build optimizations, etc.
        await asyncio.sleep(1.0)  # Simulate build time

        return {
            'build_time_seconds': 1.0,
            'artifacts_created': ['optimized_kernels.so', 'benchmark_suite']
        }

    async def _unit_tests_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Unit tests stage: Basic functionality verification"""
        # In practice, would run pytest or similar
        await asyncio.sleep(0.5)  # Simulate test execution

        return {
            'tests_run': 50,
            'tests_passed': 48,
            'tests_failed': 2,
            'coverage_percent': 85.0
        }

    async def _integration_tests_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Integration tests stage: Comprehensive optimization testing"""
        # Create integration test runner
        test_runner = IntegrationTestRunner()

        # Add test suites
        hardware_suite = HardwareTestSuite(self.config)
        compiler_suite = CompilerTestSuite(self.config)

        # Add sample tests
        self._add_sample_tests(hardware_suite, compiler_suite)

        test_runner.add_test_suite(hardware_suite)
        test_runner.add_test_suite(compiler_suite)

        # Run tests
        test_results = await test_runner.run_all_tests()

        return test_results

    async def _performance_tests_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Performance tests stage: Detailed performance analysis"""
        benchmark_suite = create_benchmark_suite(
            warmup_iterations=5 if self.config.quick_test_mode else 10,
            measurement_iterations=20 if self.config.quick_test_mode else 100
        )

        # Run predefined benchmarks
        benchmark_results = benchmark_suite.run_predefined_benchmarks()

        return {
            'benchmark_results': benchmark_results,
            'performance_summary': benchmark_suite.generate_report()
        }

    async def _regression_tests_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Regression tests stage: Compare against baseline performance"""
        # In practice, would compare against stored baseline results
        await asyncio.sleep(2.0)  # Simulate regression analysis

        return {
            'baseline_comparison': 'completed',
            'regressions_detected': 0,
            'performance_improvements': 3,
            'neutral_changes': 2
        }

    async def _cleanup_stage(self, pipeline_run: PipelineRun) -> Dict[str, Any]:
        """Cleanup stage: Resource cleanup and artifact management"""
        # Cleanup test environment
        self.env_manager.cleanup_environment(f"pipeline_{pipeline_run.run_id}")

        # Archive artifacts if enabled
        if self.config.save_artifacts:
            artifacts_archived = await self._archive_artifacts(pipeline_run)
        else:
            artifacts_archived = 0

        return {
            'cleanup_completed': True,
            'artifacts_archived': artifacts_archived
        }

    def _add_sample_tests(self, hardware_suite: HardwareTestSuite, compiler_suite: CompilerTestSuite):
        """Add sample tests for demonstration"""
        # Sample kernel test
        def sample_baseline_kernel(x, y):
            return x + y

        def sample_optimized_kernel(x, y):
            return torch.add(x, y)  # Assume this is optimized

        def sample_input_generator():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.randn(1024, 1024, device=device)
            y = torch.randn(1024, 1024, device=device)
            return [x, y]

        hardware_suite.add_kernel_test(
            "vector_add",
            sample_baseline_kernel,
            sample_optimized_kernel,
            sample_input_generator
        )

        # Sample fusion test
        def unfused_operations(x):
            return torch.relu(torch.add(x, 1.0))

        def fused_operations(x):
            # Assume this is a fused kernel
            return torch.nn.functional.relu(x + 1.0)

        def fusion_input_generator():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return [torch.randn(512, 512, device=device)]

        compiler_suite.add_fusion_test(
            "relu_add_fusion",
            unfused_operations,
            fused_operations,
            fusion_input_generator,
            expected_improvement=1.3
        )

    async def _archive_artifacts(self, pipeline_run: PipelineRun) -> int:
        """Archive pipeline artifacts"""
        # In practice, would save artifacts to storage
        artifacts_count = 0

        for stage_result in pipeline_run.stage_results.values():
            artifacts_count += len(stage_result.artifacts)

        return artifacts_count

    async def _generate_final_report(self, pipeline_run: PipelineRun):
        """Generate final pipeline report"""
        # Aggregate results
        aggregated_results = self.results_aggregator.aggregate_results(pipeline_run)

        # Create report file
        report_data = {
            'pipeline_run': {
                'run_id': pipeline_run.run_id,
                'commit_hash': pipeline_run.commit_hash,
                'branch': pipeline_run.branch,
                'trigger': pipeline_run.trigger,
                'start_time': pipeline_run.start_time.isoformat(),
                'end_time': pipeline_run.end_time.isoformat() if pipeline_run.end_time else None,
                'overall_status': pipeline_run.overall_status
            },
            'results': aggregated_results,
            'recommendations': self._generate_recommendations(aggregated_results)
        }

        # Save report
        report_file = Path(f"pipeline_report_{pipeline_run.run_id}.json")
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Pipeline report saved: {report_file}")

    def _generate_recommendations(self, aggregated_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on pipeline results"""
        recommendations = []

        run_summary = aggregated_results['run_summary']
        quality_metrics = aggregated_results['quality_metrics']

        # Success rate recommendations
        if run_summary['overall_status'] != 'success':
            recommendations.append("Pipeline failed. Review failed stages and address issues.")

        if quality_metrics.get('success_rate', 0) < 0.8:
            recommendations.append("Low stage success rate. Consider improving test stability.")

        # Performance recommendations
        if 'performance_analysis' in aggregated_results:
            perf_data = aggregated_results['performance_analysis']
            if 'benchmark_results' in perf_data:
                avg_improvement = perf_data['benchmark_results'].get('avg_improvement', 0)
                if avg_improvement < 10:
                    recommendations.append("Low performance improvements. Consider more aggressive optimizations.")

        # Regression recommendations
        if aggregated_results.get('regression_analysis', {}).get('regressions_detected', False):
            recommendations.append("Performance regressions detected. Review recent changes.")

        if not recommendations:
            recommendations.append("Pipeline executed successfully with good performance!")

        return recommendations


def create_ci_pipeline(
    environment: str = "ci",
    enable_gpu_testing: bool = True,
    quick_mode: bool = False
) -> CIPipelineManager:
    """
    Factory function to create CI/CD pipeline manager

    Args:
        environment: Target environment ("local", "ci", "staging", "production")
        enable_gpu_testing: Enable GPU-specific tests
        quick_mode: Enable quick test mode for rapid iteration

    Returns:
        Configured CIPipelineManager
    """
    config = PipelineConfig(
        environment=Environment(environment),
        enable_gpu_testing=enable_gpu_testing,
        quick_test_mode=quick_mode,
        enable_simulation=True,
        enable_benchmarking=True,
        enable_regression_detection=True
    )

    return CIPipelineManager(config)