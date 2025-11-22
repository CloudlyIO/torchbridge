#!/usr/bin/env python3
"""
GPU Optimization Testing Framework Demo

Demonstrates the comprehensive testing and validation framework for GPU kernel
and compiler optimizations, including hardware simulation, performance benchmarking,
validation testing, and CI/CD pipeline integration.
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np
from typing import List

from kernel_pytorch.testing_framework import (
    # Hardware simulation
    create_hardware_simulator,
    GPUSimulator,

    # Performance benchmarking
    create_benchmark_suite,
    PerformanceBenchmarkSuite,

    # Validation tools
    create_validation_suite,
    OptimizationValidator,

    # Integration testing
    create_integration_test_runner,
    HardwareTestSuite,
    CompilerTestSuite,

    # CI/CD pipeline
    create_ci_pipeline,
    CIPipelineManager
)


def create_sample_optimizations():
    """Create sample baseline and optimized implementations for testing"""

    # Sample 1: Matrix multiplication optimization
    def baseline_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Baseline matrix multiplication"""
        return torch.matmul(a, b)

    def optimized_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Optimized matrix multiplication using torch.mm"""
        return torch.mm(a, b)  # Potentially optimized version

    # Sample 2: GELU activation optimization
    def baseline_gelu(x: torch.Tensor) -> torch.Tensor:
        """Baseline GELU implementation"""
        return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def optimized_gelu(x: torch.Tensor) -> torch.Tensor:
        """Optimized GELU using PyTorch's fused implementation"""
        return torch.nn.functional.gelu(x)

    # Sample 3: Element-wise operations fusion
    def baseline_fused_ops(x: torch.Tensor) -> torch.Tensor:
        """Baseline unfused operations"""
        y = x + 1.0
        z = torch.relu(y)
        return z * 2.0

    def optimized_fused_ops(x: torch.Tensor) -> torch.Tensor:
        """Optimized fused operations"""
        # In practice, this would be a single fused kernel
        return torch.relu(x + 1.0) * 2.0

    return {
        'matmul': (baseline_matmul, optimized_matmul),
        'gelu': (baseline_gelu, optimized_gelu),
        'fused_ops': (baseline_fused_ops, optimized_fused_ops)
    }


def create_input_generators():
    """Create input generators for different test scenarios"""

    def matmul_inputs():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        return [a, b]

    def gelu_inputs():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(1024, 1024, device=device)
        return [x]

    def fused_ops_inputs():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(2048, 1024, device=device)
        return [x]

    return {
        'matmul': matmul_inputs,
        'gelu': gelu_inputs,
        'fused_ops': fused_ops_inputs
    }


async def demo_hardware_simulation():
    """Demonstrate hardware simulation capabilities"""
    print("🔧 Hardware Simulation Demo")
    print("=" * 50)

    # Create hardware simulator
    simulator = create_hardware_simulator(
        architecture="ampere",
        compute_units=108,
        memory_size_gb=40,
        simulation_mode="performance"
    )

    optimizations = create_sample_optimizations()
    input_generators = create_input_generators()

    # Simulate GELU optimization
    baseline_gelu, optimized_gelu = optimizations['gelu']
    inputs = input_generators['gelu']()

    print("Simulating baseline GELU...")
    baseline_metrics = simulator.execute_kernel(
        baseline_gelu,
        tuple(inputs),
        grid_dim=(64, 1, 1),
        block_dim=(256, 1, 1)
    )

    print("Simulating optimized GELU...")
    optimized_metrics = simulator.execute_kernel(
        optimized_gelu,
        tuple(inputs),
        grid_dim=(64, 1, 1),
        block_dim=(256, 1, 1)
    )

    # Compare results
    cycle_improvement = ((baseline_metrics.total_cycles - optimized_metrics.total_cycles) /
                        max(baseline_metrics.total_cycles, 1)) * 100

    print(f"Baseline cycles: {baseline_metrics.total_cycles:,}")
    print(f"Optimized cycles: {optimized_metrics.total_cycles:,}")
    print(f"Cycle reduction: {cycle_improvement:.1f}%")
    print(f"Utilization improvement: {optimized_metrics.utilization_percent - baseline_metrics.utilization_percent:.1f}%")

    # Get simulation summary
    summary = simulator.get_simulation_summary()
    print(f"\nSimulation Summary:")
    print(f"Total kernels executed: {summary['execution_summary']['total_kernels']}")
    print(f"Average utilization: {summary['execution_summary'].get('avg_temperature_c', 'N/A')}°C")


async def demo_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities"""
    print("\n📊 Performance Benchmarking Demo")
    print("=" * 50)

    # Create benchmark suite
    benchmark_suite = create_benchmark_suite(
        warmup_iterations=5,
        measurement_iterations=20,
        enable_profiling=True
    )

    optimizations = create_sample_optimizations()
    input_generators = create_input_generators()

    # Benchmark matrix multiplication optimization
    baseline_matmul, optimized_matmul = optimizations['matmul']

    print("Benchmarking matrix multiplication optimization...")
    comparison = benchmark_suite.add_optimization_comparison(
        baseline_matmul,
        optimized_matmul,
        input_generators['matmul'],
        "matmul_optimization"
    )

    print(f"Performance improvements:")
    for metric_type, improvement in comparison.improvements.items():
        significance = comparison.statistical_significance.get(metric_type, False)
        significance_str = "✓" if significance else "✗"
        print(f"  {metric_type.value}: {improvement:+.1f}% {significance_str}")

    print(f"Regression detected: {'Yes' if comparison.regression_detected else 'No'}")

    # Run predefined benchmarks
    print("\nRunning predefined benchmark suite...")
    predefined_results = benchmark_suite.run_predefined_benchmarks()

    # Display matrix multiplication results
    if 'matmul' in predefined_results:
        matmul_results = predefined_results['matmul']['matrix_multiplication']
        print("\nMatrix multiplication performance:")
        for result in matmul_results[:3]:  # Show first 3 sizes
            print(f"  {result['size']}: {result['latency_ms']:.2f}ms, "
                  f"{result['throughput_ops']:.1f} ops/sec")


async def demo_validation_testing():
    """Demonstrate validation testing capabilities"""
    print("\n✅ Validation Testing Demo")
    print("=" * 50)

    # Create validation suite
    validator = create_validation_suite(
        validation_level="thorough",
        enable_profiling=True
    )

    optimizations = create_sample_optimizations()
    input_generators = create_input_generators()

    # Validate GELU optimization
    baseline_gelu, optimized_gelu = optimizations['gelu']
    inputs = input_generators['gelu']()

    print("Validating GELU optimization...")
    validation_results = validator.validate_optimization(
        baseline_gelu,
        optimized_gelu,
        inputs,
        "gelu_validation",
        enable_gradient_check=True
    )

    print("Validation Results:")
    for validation_type, result in validation_results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {validation_type}: {status}")

        if result.error_message:
            print(f"    Error: {result.error_message}")

        if hasattr(result, 'tolerance_used') and result.tolerance_used:
            print(f"    Tolerance: rtol={result.tolerance_used.get('rtol', 'N/A')}, "
                  f"atol={result.tolerance_used.get('atol', 'N/A')}")

    # Generate validation report
    report = validator.generate_validation_report({'gelu_test': validation_results})
    print(f"\nValidation Summary:")
    print(f"  Tests passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
    print(f"  Success rate: {report['summary']['passed_tests']/max(report['summary']['total_tests'],1)*100:.1f}%")


async def demo_integration_testing():
    """Demonstrate integration testing capabilities"""
    print("\n🔗 Integration Testing Demo")
    print("=" * 50)

    # Create integration test runner
    test_runner = create_integration_test_runner(
        enable_simulation=True,
        enable_benchmarking=True,
        enable_validation=True,
        parallel_execution=False  # Sequential for demo
    )

    optimizations = create_sample_optimizations()
    input_generators = create_input_generators()

    # Create hardware test suite
    hardware_suite = HardwareTestSuite(test_runner.config)

    # Add matrix multiplication test for multiple architectures
    baseline_matmul, optimized_matmul = optimizations['matmul']
    hardware_suite.add_kernel_test(
        "matmul_kernel",
        baseline_matmul,
        optimized_matmul,
        input_generators['matmul'],
        architectures=["ampere"]  # Just one for demo
    )

    # Create compiler test suite
    compiler_suite = CompilerTestSuite(test_runner.config)

    # Add fusion test
    baseline_fused, optimized_fused = optimizations['fused_ops']
    compiler_suite.add_fusion_test(
        "element_wise_fusion",
        baseline_fused,
        optimized_fused,
        input_generators['fused_ops'],
        expected_improvement=1.2
    )

    # Add test suites to runner
    test_runner.add_test_suite(hardware_suite)
    test_runner.add_test_suite(compiler_suite)

    print(f"Running {len(test_runner.test_cases)} integration tests...")

    # Run all tests
    results = await test_runner.run_all_tests()

    print("\nIntegration Test Results:")
    print(f"  Total tests: {results['summary']['total_tests']}")
    print(f"  Successful: {results['summary']['successful_benchmarks']}")
    print(f"  Failed: {results['summary']['total_tests'] - results['summary']['successful_benchmarks']}")
    print(f"  Regressions: {results['summary']['regressions_detected']}")

    # Show performance insights
    if 'performance_insights' in results:
        insights = results['performance_insights']
        if 'optimization_effectiveness' in insights:
            effectiveness = insights['optimization_effectiveness']
            print(f"\nOptimization Effectiveness:")
            print(f"  Average improvement: {effectiveness.get('average_improvement_percent', 0):.1f}%")
            print(f"  Best improvement: {effectiveness.get('best_improvement_percent', 0):.1f}%")


async def demo_ci_pipeline():
    """Demonstrate CI/CD pipeline capabilities"""
    print("\n🚀 CI/CD Pipeline Demo")
    print("=" * 50)

    # Create CI/CD pipeline
    pipeline = create_ci_pipeline(
        environment="local",
        enable_gpu_testing=torch.cuda.is_available(),
        quick_mode=True  # Fast mode for demo
    )

    print("Running CI/CD pipeline...")

    # Run complete pipeline
    pipeline_run = await pipeline.run_pipeline(
        commit_hash="demo123",
        branch="main",
        trigger="manual"
    )

    print(f"\nPipeline Results:")
    print(f"  Run ID: {pipeline_run.run_id}")
    print(f"  Status: {pipeline_run.overall_status}")
    print(f"  Duration: {pipeline_run.execution_time:.1f}s")

    print(f"\nStage Results:")
    for stage, result in pipeline_run.stage_results.items():
        status_emoji = "✅" if result.status == "success" else "❌" if result.status == "failure" else "⏭️"
        print(f"  {stage.value}: {status_emoji} {result.status} ({result.duration_seconds:.1f}s)")

        if result.error_message:
            print(f"    Error: {result.error_message}")


async def main():
    """Run comprehensive GPU optimization testing demo"""
    print("🎯 GPU Optimization Testing Framework Demo")
    print("=" * 60)
    print()

    device_info = f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"
    print(device_info)
    print()

    try:
        # Run all demos
        await demo_hardware_simulation()
        await demo_performance_benchmarking()
        await demo_validation_testing()
        await demo_integration_testing()
        await demo_ci_pipeline()

        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Hardware simulation for different GPU architectures")
        print("  ✅ Comprehensive performance benchmarking")
        print("  ✅ Numerical accuracy and gradient validation")
        print("  ✅ Integration testing with multiple optimization types")
        print("  ✅ Complete CI/CD pipeline for automated testing")

        print("\nNext Steps:")
        print("  1. Integrate with your existing optimization implementations")
        print("  2. Configure CI/CD pipeline for your repository")
        print("  3. Add custom test cases for your specific optimizations")
        print("  4. Set up regression testing against performance baselines")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)