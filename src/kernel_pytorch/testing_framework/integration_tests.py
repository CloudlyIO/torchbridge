"""
Integration Test Runner for GPU Optimizations

Comprehensive integration testing framework that orchestrates hardware simulation,
performance benchmarking, and validation testing for complete optimization validation.
"""

import asyncio
import time
import logging
import json
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import torch
import torch.nn as nn

from .hardware_simulator import GPUSimulator, create_hardware_simulator
from .performance_benchmarks import PerformanceBenchmarkSuite, create_benchmark_suite
from .validation_tools import OptimizationValidator, create_validation_suite

logger = logging.getLogger(__name__)


class TestPhase(Enum):
    """Test execution phases"""
    SETUP = "setup"
    SIMULATION = "simulation"
    BENCHMARKING = "benchmarking"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    CLEANUP = "cleanup"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IntegrationTestConfig:
    """Configuration for integration testing"""
    # Hardware simulation
    enable_simulation: bool = True
    simulation_architecture: str = "ampere"
    simulation_mode: str = "performance"

    # Performance benchmarking
    enable_benchmarking: bool = True
    benchmark_warmup_iterations: int = 10
    benchmark_measurement_iterations: int = 100

    # Validation testing
    enable_validation: bool = True
    validation_level: str = "thorough"
    enable_gradient_checks: bool = True

    # Test execution
    parallel_execution: bool = False
    test_timeout_seconds: int = 300
    enable_detailed_logging: bool = True

    # Result handling
    export_results: bool = True
    results_directory: str = None  # Will use temporary directory
    save_simulation_traces: bool = True


@dataclass
class TestCase:
    """Individual test case specification"""
    name: str
    baseline_fn: Callable
    optimized_fn: Callable
    input_generator: Callable[[], List[torch.Tensor]]
    test_type: str = "optimization"
    expected_improvement: Optional[float] = None  # Expected performance improvement
    custom_config: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Results from integration test execution"""
    test_case: TestCase
    status: TestStatus
    execution_time: float
    phase_results: Dict[TestPhase, Dict[str, Any]] = field(default_factory=dict)
    error_message: Optional[str] = None
    performance_improvement: Optional[float] = None
    validation_passed: bool = False


class HardwareTestSuite:
    """
    Test suite for hardware-specific optimizations

    Tests optimizations across different GPU architectures
    and hardware configurations using simulation.
    """

    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.test_cases: List[TestCase] = []

    def add_kernel_test(
        self,
        test_name: str,
        baseline_kernel: Callable,
        optimized_kernel: Callable,
        input_generator: Callable,
        architectures: List[str] = None
    ):
        """Add kernel test for multiple architectures"""
        architectures = architectures or ["ampere", "hopper"]

        for arch in architectures:
            test_case = TestCase(
                name=f"{test_name}_{arch}",
                baseline_fn=baseline_kernel,
                optimized_fn=optimized_kernel,
                input_generator=input_generator,
                test_type="kernel",
                custom_config={"architecture": arch},
                tags=["hardware", "kernel", arch]
            )
            self.test_cases.append(test_case)

    def add_memory_optimization_test(
        self,
        test_name: str,
        baseline_fn: Callable,
        optimized_fn: Callable,
        input_generator: Callable,
        memory_configs: List[Dict] = None
    ):
        """Add memory optimization test with different configurations"""
        memory_configs = memory_configs or [
            {"memory_size_gb": 16, "bandwidth_gb_s": 800},
            {"memory_size_gb": 40, "bandwidth_gb_s": 1555},
            {"memory_size_gb": 80, "bandwidth_gb_s": 3000}
        ]

        for i, mem_config in enumerate(memory_configs):
            test_case = TestCase(
                name=f"{test_name}_mem_config_{i}",
                baseline_fn=baseline_fn,
                optimized_fn=optimized_fn,
                input_generator=input_generator,
                test_type="memory",
                custom_config={"memory_config": mem_config},
                tags=["hardware", "memory"]
            )
            self.test_cases.append(test_case)

    def get_test_cases(self) -> List[TestCase]:
        """Get all hardware test cases"""
        return self.test_cases


class CompilerTestSuite:
    """
    Test suite for compiler optimizations

    Tests compiler transformations including kernel fusion,
    memory coalescing, and code generation optimizations.
    """

    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.test_cases: List[TestCase] = []

    def add_fusion_test(
        self,
        test_name: str,
        unfused_fn: Callable,
        fused_fn: Callable,
        input_generator: Callable,
        expected_improvement: float = 1.2
    ):
        """Add kernel fusion test"""
        test_case = TestCase(
            name=f"fusion_{test_name}",
            baseline_fn=unfused_fn,
            optimized_fn=fused_fn,
            input_generator=input_generator,
            test_type="fusion",
            expected_improvement=expected_improvement,
            tags=["compiler", "fusion"]
        )
        self.test_cases.append(test_case)

    def add_memory_coalescing_test(
        self,
        test_name: str,
        uncoalesced_fn: Callable,
        coalesced_fn: Callable,
        input_generator: Callable,
        expected_improvement: float = 1.5
    ):
        """Add memory coalescing test"""
        test_case = TestCase(
            name=f"coalescing_{test_name}",
            baseline_fn=uncoalesced_fn,
            optimized_fn=coalesced_fn,
            input_generator=input_generator,
            test_type="coalescing",
            expected_improvement=expected_improvement,
            tags=["compiler", "memory", "coalescing"]
        )
        self.test_cases.append(test_case)

    def add_code_generation_test(
        self,
        test_name: str,
        baseline_fn: Callable,
        optimized_fn: Callable,
        input_generator: Callable,
        optimization_type: str = "general"
    ):
        """Add code generation optimization test"""
        test_case = TestCase(
            name=f"codegen_{test_name}",
            baseline_fn=baseline_fn,
            optimized_fn=optimized_fn,
            input_generator=input_generator,
            test_type="codegen",
            tags=["compiler", "codegen", optimization_type]
        )
        self.test_cases.append(test_case)

    def get_test_cases(self) -> List[TestCase]:
        """Get all compiler test cases"""
        return self.test_cases


class IntegrationTestRunner:
    """
    Comprehensive integration test runner

    Orchestrates hardware simulation, performance benchmarking,
    and validation testing for complete optimization validation.
    """

    def __init__(self, config: IntegrationTestConfig = None):
        self.config = config or IntegrationTestConfig()

        # Test components
        self.hardware_simulator = None
        self.benchmark_suite = None
        self.validation_suite = None

        # Test management
        self.test_cases: List[TestCase] = []
        self.test_results: List[TestResult] = []

        # Setup components
        self._setup_test_components()

    def _setup_test_components(self):
        """Setup testing components based on configuration"""
        if self.config.enable_simulation:
            self.hardware_simulator = create_hardware_simulator(
                architecture=self.config.simulation_architecture,
                simulation_mode=self.config.simulation_mode
            )

        if self.config.enable_benchmarking:
            self.benchmark_suite = create_benchmark_suite(
                warmup_iterations=self.config.benchmark_warmup_iterations,
                measurement_iterations=self.config.benchmark_measurement_iterations
            )

        if self.config.enable_validation:
            self.validation_suite = create_validation_suite(
                validation_level=self.config.validation_level
            )

    def add_test_suite(self, test_suite: Union[HardwareTestSuite, CompilerTestSuite]):
        """Add test suite to runner"""
        self.test_cases.extend(test_suite.get_test_cases())
        logger.info(f"Added {len(test_suite.get_test_cases())} test cases")

    def add_test_case(self, test_case: TestCase):
        """Add individual test case"""
        self.test_cases.append(test_case)

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and return comprehensive results"""
        logger.info(f"Starting integration test run with {len(self.test_cases)} test cases")

        start_time = time.time()
        self.test_results.clear()

        if self.config.parallel_execution:
            # Run tests in parallel
            tasks = [self._run_single_test(test_case) for test_case in self.test_cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle test execution exception
                    error_result = TestResult(
                        test_case=self.test_cases[i],
                        status=TestStatus.FAILED,
                        execution_time=0.0,
                        error_message=str(result)
                    )
                    self.test_results.append(error_result)
                else:
                    self.test_results.append(result)
        else:
            # Run tests sequentially
            for test_case in self.test_cases:
                result = await self._run_single_test(test_case)
                self.test_results.append(result)

        total_time = time.time() - start_time

        # Generate comprehensive report
        report = self._generate_test_report(total_time)

        # Export results if enabled
        if self.config.export_results:
            await self._export_results(report)

        logger.info(f"Integration test run completed in {total_time:.2f}s")
        return report

    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single integration test case"""
        logger.info(f"Running test case: {test_case.name}")

        result = TestResult(
            test_case=test_case,
            status=TestStatus.RUNNING,
            execution_time=0.0
        )

        start_time = time.time()

        try:
            # Setup phase
            result.phase_results[TestPhase.SETUP] = await self._setup_test(test_case)

            # Generate inputs
            inputs = test_case.input_generator()

            # Simulation phase
            if self.config.enable_simulation:
                result.phase_results[TestPhase.SIMULATION] = await self._run_simulation_phase(
                    test_case, inputs
                )

            # Benchmarking phase
            if self.config.enable_benchmarking:
                result.phase_results[TestPhase.BENCHMARKING] = await self._run_benchmark_phase(
                    test_case, inputs
                )

            # Validation phase
            if self.config.enable_validation:
                result.phase_results[TestPhase.VALIDATION] = await self._run_validation_phase(
                    test_case, inputs
                )

            # Analysis phase
            result.phase_results[TestPhase.ANALYSIS] = self._analyze_test_results(result)

            # Determine overall test status
            result.status = self._determine_test_status(result)

        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.error_message = f"Test timed out after {self.config.test_timeout_seconds}s"

        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Test case {test_case.name} failed: {e}")

        finally:
            # Cleanup phase
            try:
                result.phase_results[TestPhase.CLEANUP] = await self._cleanup_test(test_case)
            except Exception as e:
                logger.warning(f"Cleanup failed for {test_case.name}: {e}")

        result.execution_time = time.time() - start_time
        logger.info(f"Test case {test_case.name} completed: {result.status.value}")

        return result

    async def _setup_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Setup test environment"""
        setup_info = {
            'test_name': test_case.name,
            'test_type': test_case.test_type,
            'tags': test_case.tags,
            'requirements_met': self._check_requirements(test_case),
            'setup_time': time.time()
        }

        # Apply custom configuration if provided
        if test_case.custom_config:
            if 'architecture' in test_case.custom_config:
                # Reconfigure simulator for different architecture
                if self.hardware_simulator:
                    arch = test_case.custom_config['architecture']
                    self.hardware_simulator = create_hardware_simulator(
                        architecture=arch,
                        simulation_mode=self.config.simulation_mode
                    )

        return setup_info

    async def _run_simulation_phase(
        self,
        test_case: TestCase,
        inputs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Run hardware simulation phase"""
        if not self.hardware_simulator:
            return {'skipped': 'Simulation disabled'}

        simulation_results = {}

        try:
            # Simulate baseline execution
            baseline_metrics = self.hardware_simulator.execute_kernel(
                test_case.baseline_fn,
                tuple(inputs),
                grid_dim=(1, 1, 1),
                block_dim=(256, 1, 1)
            )

            # Simulate optimized execution
            optimized_metrics = self.hardware_simulator.execute_kernel(
                test_case.optimized_fn,
                tuple(inputs),
                grid_dim=(1, 1, 1),
                block_dim=(256, 1, 1)
            )

            simulation_results = {
                'baseline_metrics': {
                    'total_cycles': baseline_metrics.total_cycles,
                    'utilization_percent': baseline_metrics.utilization_percent,
                    'power_consumption_w': baseline_metrics.power_consumption_w
                },
                'optimized_metrics': {
                    'total_cycles': optimized_metrics.total_cycles,
                    'utilization_percent': optimized_metrics.utilization_percent,
                    'power_consumption_w': optimized_metrics.power_consumption_w
                },
                'simulation_improvement': {
                    'cycle_reduction': (
                        (baseline_metrics.total_cycles - optimized_metrics.total_cycles) /
                        max(baseline_metrics.total_cycles, 1) * 100
                    ),
                    'utilization_improvement': (
                        optimized_metrics.utilization_percent - baseline_metrics.utilization_percent
                    )
                }
            }

        except Exception as e:
            simulation_results = {'error': str(e)}

        return simulation_results

    async def _run_benchmark_phase(
        self,
        test_case: TestCase,
        inputs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Run performance benchmarking phase"""
        if not self.benchmark_suite:
            return {'skipped': 'Benchmarking disabled'}

        try:
            # Run optimization comparison
            comparison = self.benchmark_suite.add_optimization_comparison(
                test_case.baseline_fn,
                test_case.optimized_fn,
                lambda: inputs,
                test_case.name
            )

            benchmark_results = {
                'performance_improvements': comparison.improvements,
                'statistical_significance': comparison.statistical_significance,
                'regression_detected': comparison.regression_detected,
                'baseline_stats': comparison.baseline_result.statistics,
                'optimized_stats': comparison.optimized_result.statistics
            }

            return benchmark_results

        except Exception as e:
            return {'error': str(e)}

    async def _run_validation_phase(
        self,
        test_case: TestCase,
        inputs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Run validation testing phase"""
        if not self.validation_suite:
            return {'skipped': 'Validation disabled'}

        try:
            # Run comprehensive validation
            validation_results = self.validation_suite.validate_optimization(
                test_case.baseline_fn,
                test_case.optimized_fn,
                inputs,
                test_case.name,
                enable_gradient_check=self.config.enable_gradient_checks
            )

            # Convert results to serializable format
            validation_summary = {}
            for validation_type, result in validation_results.items():
                validation_summary[validation_type] = {
                    'passed': result.passed,
                    'error_message': result.error_message,
                    'execution_time': result.execution_time,
                    'key_metrics': result.metrics if isinstance(result.metrics, dict) else {}
                }

            return validation_summary

        except Exception as e:
            return {'error': str(e)}

    def _analyze_test_results(self, result: TestResult) -> Dict[str, Any]:
        """Analyze test results and extract insights"""
        analysis = {
            'overall_success': result.status in [TestStatus.PASSED],
            'phase_success': {},
            'performance_analysis': {},
            'validation_analysis': {}
        }

        # Analyze phase success
        for phase, phase_result in result.phase_results.items():
            analysis['phase_success'][phase.value] = 'error' not in phase_result

        # Performance analysis
        if TestPhase.BENCHMARKING in result.phase_results:
            benchmark_data = result.phase_results[TestPhase.BENCHMARKING]
            if 'performance_improvements' in benchmark_data:
                improvements = benchmark_data['performance_improvements']
                analysis['performance_analysis'] = {
                    'latency_improvement': improvements.get('latency', 0),
                    'throughput_improvement': improvements.get('throughput', 0),
                    'memory_improvement': improvements.get('memory_usage', 0),
                    'meets_expectation': self._check_performance_expectation(
                        result.test_case, improvements
                    )
                }

        # Validation analysis
        if TestPhase.VALIDATION in result.phase_results:
            validation_data = result.phase_results[TestPhase.VALIDATION]
            passed_validations = sum(
                1 for val_result in validation_data.values()
                if isinstance(val_result, dict) and val_result.get('passed', False)
            )
            total_validations = len([
                v for v in validation_data.values()
                if isinstance(v, dict) and 'passed' in v
            ])

            analysis['validation_analysis'] = {
                'passed_validations': passed_validations,
                'total_validations': total_validations,
                'validation_success_rate': (
                    passed_validations / max(total_validations, 1)
                )
            }

        return analysis

    def _determine_test_status(self, result: TestResult) -> TestStatus:
        """Determine overall test status based on phase results"""
        # Check for errors in any phase
        for phase_result in result.phase_results.values():
            if 'error' in phase_result:
                return TestStatus.FAILED

        # Check validation success
        if TestPhase.VALIDATION in result.phase_results:
            validation_data = result.phase_results[TestPhase.VALIDATION]
            if any(
                not val_result.get('passed', True)
                for val_result in validation_data.values()
                if isinstance(val_result, dict) and 'passed' in val_result
            ):
                return TestStatus.FAILED

        # Check for performance regression
        if TestPhase.BENCHMARKING in result.phase_results:
            benchmark_data = result.phase_results[TestPhase.BENCHMARKING]
            if benchmark_data.get('regression_detected', False):
                return TestStatus.FAILED

        return TestStatus.PASSED

    def _check_requirements(self, test_case: TestCase) -> bool:
        """Check if test case requirements are met"""
        for requirement in test_case.requirements:
            if requirement == "cuda" and not torch.cuda.is_available():
                return False
            elif requirement.startswith("min_memory_") and torch.cuda.is_available():
                required_gb = float(requirement.split('_')[-1])
                available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if available_gb < required_gb:
                    return False

        return True

    def _check_performance_expectation(
        self,
        test_case: TestCase,
        improvements: Dict[str, float]
    ) -> bool:
        """Check if performance meets expectations"""
        if test_case.expected_improvement is None:
            return True

        # Use latency improvement as primary metric
        latency_improvement = improvements.get('latency', 0)
        return latency_improvement >= (test_case.expected_improvement - 1.0) * 100

    async def _cleanup_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Cleanup test resources"""
        cleanup_info = {'cleanup_time': time.time()}

        try:
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cleanup_info['success'] = True

        except Exception as e:
            cleanup_info['error'] = str(e)
            cleanup_info['success'] = False

        return cleanup_info

    def _generate_test_report(self, total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed_tests = [r for r in self.test_results if r.status == TestStatus.PASSED]
        failed_tests = [r for r in self.test_results if r.status == TestStatus.FAILED]

        report = {
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(passed_tests) / max(len(self.test_results), 1),
                'total_execution_time': total_execution_time
            },
            'test_results': [],
            'performance_insights': self._analyze_performance_trends(),
            'failure_analysis': self._analyze_failures(),
            'recommendations': []
        }

        # Add detailed test results
        for result in self.test_results:
            test_summary = {
                'name': result.test_case.name,
                'status': result.status.value,
                'execution_time': result.execution_time,
                'test_type': result.test_case.test_type,
                'tags': result.test_case.tags,
                'error_message': result.error_message,
                'phase_results': result.phase_results,
                'analysis': result.phase_results.get(TestPhase.ANALYSIS, {})
            }
            report['test_results'].append(test_summary)

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across tests"""
        performance_data = []

        for result in self.test_results:
            if (TestPhase.BENCHMARKING in result.phase_results and
                'performance_improvements' in result.phase_results[TestPhase.BENCHMARKING]):

                improvements = result.phase_results[TestPhase.BENCHMARKING]['performance_improvements']
                performance_data.append({
                    'test_name': result.test_case.name,
                    'test_type': result.test_case.test_type,
                    'latency_improvement': improvements.get('latency', 0),
                    'throughput_improvement': improvements.get('throughput', 0),
                    'memory_improvement': improvements.get('memory_usage', 0)
                })

        if not performance_data:
            return {'no_data': True}

        # Calculate aggregate statistics
        latency_improvements = [d['latency_improvement'] for d in performance_data]
        throughput_improvements = [d['throughput_improvement'] for d in performance_data]

        return {
            'avg_latency_improvement': np.mean(latency_improvements),
            'max_latency_improvement': np.max(latency_improvements),
            'avg_throughput_improvement': np.mean(throughput_improvements),
            'max_throughput_improvement': np.max(throughput_improvements),
            'best_performing_tests': sorted(
                performance_data,
                key=lambda x: x['latency_improvement'],
                reverse=True
            )[:5],
            'performance_by_type': self._group_performance_by_type(performance_data)
        }

    def _group_performance_by_type(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Group performance data by test type"""
        type_groups = {}

        for data in performance_data:
            test_type = data['test_type']
            if test_type not in type_groups:
                type_groups[test_type] = []
            type_groups[test_type].append(data)

        # Calculate statistics for each type
        type_stats = {}
        for test_type, group_data in type_groups.items():
            latency_improvements = [d['latency_improvement'] for d in group_data]
            type_stats[test_type] = {
                'count': len(group_data),
                'avg_latency_improvement': np.mean(latency_improvements),
                'max_latency_improvement': np.max(latency_improvements),
                'min_latency_improvement': np.min(latency_improvements)
            }

        return type_stats

    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze failure patterns"""
        failed_tests = [r for r in self.test_results if r.status == TestStatus.FAILED]

        failure_analysis = {
            'total_failures': len(failed_tests),
            'failure_by_type': {},
            'failure_by_phase': {},
            'common_errors': {}
        }

        for result in failed_tests:
            # Group by test type
            test_type = result.test_case.test_type
            if test_type not in failure_analysis['failure_by_type']:
                failure_analysis['failure_by_type'][test_type] = 0
            failure_analysis['failure_by_type'][test_type] += 1

            # Group by failure phase
            for phase, phase_result in result.phase_results.items():
                if 'error' in phase_result:
                    phase_name = phase.value
                    if phase_name not in failure_analysis['failure_by_phase']:
                        failure_analysis['failure_by_phase'][phase_name] = 0
                    failure_analysis['failure_by_phase'][phase_name] += 1

            # Collect error messages
            if result.error_message:
                error_key = result.error_message[:100]  # Truncate long messages
                if error_key not in failure_analysis['common_errors']:
                    failure_analysis['common_errors'][error_key] = 0
                failure_analysis['common_errors'][error_key] += 1

        return failure_analysis

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        success_rate = report['summary']['success_rate']
        if success_rate < 0.8:
            recommendations.append(
                f"Low success rate ({success_rate:.1%}). Review failing tests and optimize implementations."
            )

        # Performance recommendations
        if 'performance_insights' in report and 'avg_latency_improvement' in report['performance_insights']:
            avg_improvement = report['performance_insights']['avg_latency_improvement']
            if avg_improvement < 10:
                recommendations.append(
                    f"Low average performance improvement ({avg_improvement:.1f}%). "
                    "Consider more aggressive optimizations."
                )

        # Failure analysis recommendations
        if 'failure_analysis' in report:
            failures = report['failure_analysis']
            if failures['total_failures'] > 0:
                most_common_phase = max(
                    failures.get('failure_by_phase', {}),
                    key=failures['failure_by_phase'].get,
                    default=None
                )
                if most_common_phase:
                    recommendations.append(
                        f"Most failures occur in {most_common_phase} phase. "
                        "Focus debugging efforts on this area."
                    )

        if not recommendations:
            recommendations.append("All tests passed successfully! Consider adding more challenging test cases.")

        return recommendations

    def _convert_enum_keys_to_strings(self, obj):
        """Recursively convert enum keys to strings for JSON serialization"""
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                # Convert enum keys to their string value
                if hasattr(key, 'value'):  # Check if it's an enum
                    new_key = key.value
                else:
                    new_key = key
                new_dict[new_key] = self._convert_enum_keys_to_strings(value)
            return new_dict
        elif isinstance(obj, list):
            return [self._convert_enum_keys_to_strings(item) for item in obj]
        else:
            return obj

    async def _export_results(self, report: Dict[str, Any]):
        """Export test results to files"""
        if self.config.results_directory:
            results_dir = Path(self.config.results_directory)
            results_dir.mkdir(exist_ok=True)
        else:
            # Use temporary directory
            results_dir = Path(tempfile.mkdtemp(prefix="integration_test_results_"))

        timestamp = int(time.time())

        # Convert enum keys to strings before JSON serialization
        serializable_report = self._convert_enum_keys_to_strings(report)

        # Export main report
        report_file = results_dir / f"integration_test_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)

        # Export simulation traces if enabled
        if self.config.save_simulation_traces and self.hardware_simulator:
            trace_file = results_dir / f"simulation_trace_{timestamp}.json"
            self.hardware_simulator.export_trace(str(trace_file))

        logger.info(f"Test results exported to {results_dir}")


def create_integration_test_runner(
    enable_simulation: bool = True,
    enable_benchmarking: bool = True,
    enable_validation: bool = True,
    parallel_execution: bool = False
) -> IntegrationTestRunner:
    """
    Factory function to create integration test runner

    Args:
        enable_simulation: Enable hardware simulation
        enable_benchmarking: Enable performance benchmarking
        enable_validation: Enable validation testing
        parallel_execution: Enable parallel test execution

    Returns:
        Configured IntegrationTestRunner
    """
    config = IntegrationTestConfig(
        enable_simulation=enable_simulation,
        enable_benchmarking=enable_benchmarking,
        enable_validation=enable_validation,
        parallel_execution=parallel_execution
    )

    return IntegrationTestRunner(config)