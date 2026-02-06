#!/usr/bin/env python3
"""
TPU Integration Comprehensive Benchmark

Benchmarks all TPU functionality including configuration, backend operations,
optimization, compilation, memory management, and validation.

Usage:
    python3 tpu_integration_benchmark.py [--quick] [--device cpu|auto]
    python3 tpu_integration_benchmark.py --help
"""

import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from torchbridge.backends.tpu import (
    TPUBackend,
    TPUMemoryManager,
    TPUOptimizer,
    XLACompiler,
    XLADeviceManager,
    XLADistributedTraining,
    XLAOptimizations,
    XLAUtilities,
    create_xla_integration,
)
from torchbridge.core.config import TorchBridgeConfig
from torchbridge.validation.unified_validator import (
    validate_tpu_configuration,
    validate_tpu_model,
)


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    name: str
    duration: float
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def successful_tests(self) -> int:
        """Count successful tests."""
        return sum(1 for r in self.results if r.success)


class TPUBenchmarkRunner:
    """Comprehensive TPU benchmark runner."""

    def __init__(self, device: str = "auto", quick_mode: bool = False):
        """
        Initialize benchmark runner.

        Args:
            device: Device to use ('cpu', 'auto')
            quick_mode: Run quick subset of benchmarks
        """
        self.device = device
        self.quick_mode = quick_mode
        self.config = TorchBridgeConfig()

        print("🚀 TPU Integration Benchmark Suite")
        print(f"   Device: {device}")
        print(f"   Quick mode: {quick_mode}")
        print(f"   TPU Version: {self.config.hardware.tpu.version.value}")
        print(f"   TPU Topology: {self.config.hardware.tpu.topology.value}")
        print("="*70)

    def benchmark_tpu_configuration(self) -> BenchmarkResult:
        """Benchmark TPU configuration performance."""
        start_time = time.perf_counter()

        try:
            # Test multiple configuration creations
            iterations = 100 if not self.quick_mode else 20
            configs = []

            config_start = time.perf_counter()
            for i in range(iterations):
                config = TorchBridgeConfig()
                configs.append(config)
            config_time = time.perf_counter() - config_start

            # Test configuration modes
            mode_configs = {
                'inference': TorchBridgeConfig.for_inference(),
                'training': TorchBridgeConfig.for_training(),
                'development': TorchBridgeConfig.for_development()
            }

            # Test serialization performance
            serialization_start = time.perf_counter()
            for config in configs[:10]:  # Test subset
                config_dict = config.to_dict()
            serialization_time = time.perf_counter() - serialization_start

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name="TPU Configuration",
                duration=duration,
                success=True,
                metrics={
                    'config_creation_time_per_iteration': config_time / iterations * 1000,  # ms
                    'total_iterations': iterations,
                    'serialization_time': serialization_time * 1000,  # ms
                    'config_modes_created': len(mode_configs),
                    'avg_config_size': len(str(configs[0].to_dict())) if configs else 0
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="TPU Configuration",
                duration=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )

    def benchmark_tpu_backend(self) -> BenchmarkResult:
        """Benchmark TPU backend operations."""
        start_time = time.perf_counter()

        try:
            backend = TPUBackend(self.config)

            # Test model preparation
            models = [
                nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
                nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32)),
                nn.Sequential(nn.Conv1d(16, 32, 3), nn.ReLU(), nn.AdaptiveAvgPool1d(16))
            ]

            preparation_times = []
            for model in models:
                prep_start = time.perf_counter()
                prepared_model = backend.prepare_model(model)
                prep_time = time.perf_counter() - prep_start
                preparation_times.append(prep_time * 1000)  # ms

            # Test data preparation
            data_items = [
                torch.randn(8, 64),
                torch.randn(16, 128),
                {'input': torch.randn(4, 32), 'target': torch.randn(4, 10)}
            ]

            data_prep_times = []
            for data in data_items:
                data_start = time.perf_counter()
                prepared_data = backend.prepare_data(data)
                data_time = time.perf_counter() - data_start
                data_prep_times.append(data_time * 1000)  # ms

            # Test synchronization
            sync_start = time.perf_counter()
            backend.synchronize()
            sync_time = time.perf_counter() - sync_start

            # Test memory stats
            stats_start = time.perf_counter()
            memory_stats = backend.get_memory_stats()
            stats_time = time.perf_counter() - stats_start

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name="TPU Backend",
                duration=duration,
                success=True,
                metrics={
                    'model_preparation_avg_time': sum(preparation_times) / len(preparation_times),
                    'data_preparation_avg_time': sum(data_prep_times) / len(data_prep_times),
                    'synchronization_time': sync_time * 1000,
                    'memory_stats_time': stats_time * 1000,
                    'models_prepared': len(models),
                    'data_items_prepared': len(data_items),
                    'backend_device': str(backend.device),
                    'world_size': backend.world_size
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="TPU Backend",
                duration=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )

    def benchmark_tpu_optimizer(self) -> BenchmarkResult:
        """Benchmark TPU optimizer performance."""
        start_time = time.perf_counter()

        try:
            optimizer = TPUOptimizer(self.config)

            # Test models of different complexities
            test_cases = [
                {
                    'name': 'Simple MLP',
                    'model': nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)),
                    'input': torch.randn(8, 64)
                },
                {
                    'name': 'Medium CNN',
                    'model': nn.Sequential(
                        nn.Conv1d(16, 32, 3), nn.ReLU(),
                        nn.Conv1d(32, 64, 3), nn.ReLU(),
                        nn.AdaptiveAvgPool1d(8),
                        nn.Flatten(),
                        nn.Linear(64*8, 10)
                    ),
                    'input': torch.randn(4, 16, 32)
                }
            ]

            optimization_results = {}

            for test_case in test_cases:
                case_name = test_case['name']
                model = test_case['model']
                sample_input = test_case['input']

                # Test different optimization levels
                for level in ['conservative', 'balanced', 'aggressive']:
                    opt_start = time.perf_counter()
                    result = optimizer.optimize(model, sample_input, optimization_level=level)
                    opt_time = time.perf_counter() - opt_start

                    optimization_results[f"{case_name}_{level}"] = {
                        'optimization_time': opt_time * 1000,  # ms
                        'result_available': result is not None,
                        'optimized_model_available': result.optimized_model is not None if result else False
                    }

            # Test specialized optimizations
            model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
            sample_input = torch.randn(8, 64)

            inference_start = time.perf_counter()
            inference_result = optimizer.optimize_for_inference(model, sample_input)
            inference_time = time.perf_counter() - inference_start

            training_start = time.perf_counter()
            training_result = optimizer.optimize_for_training(model, sample_input)
            training_time = time.perf_counter() - training_start

            # Get optimizer statistics
            stats = optimizer.get_optimization_stats()

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name="TPU Optimizer",
                duration=duration,
                success=True,
                metrics={
                    'optimization_results': optimization_results,
                    'inference_optimization_time': inference_time * 1000,
                    'training_optimization_time': training_time * 1000,
                    'total_optimizations': stats.get('total_optimizations', 0),
                    'test_cases_processed': len(test_cases)
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="TPU Optimizer",
                duration=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )

    def benchmark_xla_compiler(self) -> BenchmarkResult:
        """Benchmark XLA compiler performance."""
        start_time = time.perf_counter()

        try:
            compiler = XLACompiler(self.config.hardware.tpu)

            # Test models for compilation
            models = [
                nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)),
                nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.Dropout(0.1), nn.Linear(32, 16))
            ]

            sample_inputs = [
                torch.randn(8, 64),
                torch.randn(16, 128)
            ]

            compilation_times = []
            for i, (model, sample_input) in enumerate(zip(models, sample_inputs)):
                comp_start = time.perf_counter()
                compiled_model = compiler.compile_model(model, sample_input, use_cache=False)
                comp_time = time.perf_counter() - comp_start
                compilation_times.append(comp_time * 1000)  # ms

            # Test cache performance
            cache_start = time.perf_counter()
            cached_model = compiler.compile_model(models[0], sample_inputs[0], use_cache=True)
            cache_time = time.perf_counter() - cache_start

            # Test optimization functions
            opt_start = time.perf_counter()
            inference_optimized = compiler.optimize_for_inference(models[0], sample_inputs[0])
            training_optimized = compiler.optimize_for_training(models[0], sample_inputs[0])
            opt_time = time.perf_counter() - opt_start

            # Test benchmarking capability
            if not self.quick_mode:
                benchmark_results = compiler.benchmark_compilation(
                    models[0], sample_inputs[0], num_runs=3
                )
            else:
                benchmark_results = {'runs': 0, 'avg_time': 0}

            # Get compilation statistics
            stats = compiler.get_compilation_stats()

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name="XLA Compiler",
                duration=duration,
                success=True,
                metrics={
                    'avg_compilation_time': sum(compilation_times) / len(compilation_times),
                    'cache_access_time': cache_time * 1000,
                    'optimization_time': opt_time * 1000,
                    'models_compiled': len(models),
                    'compilation_stats': stats,
                    'benchmark_runs': benchmark_results.get('runs', 0),
                    'benchmark_avg_time': benchmark_results.get('avg_time', 0) * 1000 if benchmark_results.get('avg_time') else 0
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="XLA Compiler",
                duration=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )

    def benchmark_memory_manager(self) -> BenchmarkResult:
        """Benchmark TPU memory manager performance."""
        start_time = time.perf_counter()

        try:
            memory_manager = TPUMemoryManager(self.config.hardware.tpu)

            # Test tensor allocation
            allocation_times = []
            tensor_shapes = [(64, 64), (128, 128), (256, 256)] if not self.quick_mode else [(32, 32), (64, 64)]

            allocated_tensors = []
            for shape in tensor_shapes:
                alloc_start = time.perf_counter()
                tensor = memory_manager.allocate_tensor(shape, dtype=torch.float32)
                alloc_time = time.perf_counter() - alloc_start
                allocation_times.append(alloc_time * 1000)  # ms
                allocated_tensors.append(tensor)

            # Test layout optimization
            optimization_times = []
            for tensor in allocated_tensors:
                opt_start = time.perf_counter()
                optimized_tensor = memory_manager.optimize_tensor_layout(tensor)
                opt_time = time.perf_counter() - opt_start
                optimization_times.append(opt_time * 1000)  # ms

            # Test memory pool operations
            pool_start = time.perf_counter()
            pool_id = memory_manager.create_memory_pool(5, (32, 32))
            pool_creation_time = time.perf_counter() - pool_start

            # Test pool operations
            pool_ops_start = time.perf_counter()
            pool_tensor = memory_manager.get_tensor_from_pool(pool_id)
            returned = memory_manager.return_tensor_to_pool(pool_id, pool_tensor) if pool_tensor is not None else False
            pool_ops_time = time.perf_counter() - pool_ops_start

            # Test memory statistics
            stats_start = time.perf_counter()
            memory_stats = memory_manager.get_memory_stats()
            pool_stats = memory_manager.get_pool_stats()
            stats_time = time.perf_counter() - stats_start

            # Test memory optimization
            opt_memory_start = time.perf_counter()
            memory_manager.optimize_memory_usage()
            memory_opt_time = time.perf_counter() - opt_memory_start

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name="Memory Manager",
                duration=duration,
                success=True,
                metrics={
                    'avg_allocation_time': sum(allocation_times) / len(allocation_times),
                    'avg_optimization_time': sum(optimization_times) / len(optimization_times),
                    'pool_creation_time': pool_creation_time * 1000,
                    'pool_operations_time': pool_ops_time * 1000,
                    'stats_retrieval_time': stats_time * 1000,
                    'memory_optimization_time': memory_opt_time * 1000,
                    'tensors_allocated': len(allocated_tensors),
                    'memory_stats': {
                        'allocated_memory': memory_stats.allocated_memory,
                        'memory_fraction': memory_stats.memory_fraction,
                        'active_tensors': memory_stats.active_tensors
                    },
                    'pool_stats': pool_stats
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="Memory Manager",
                duration=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )

    def benchmark_xla_integration(self) -> BenchmarkResult:
        """Benchmark XLA integration components."""
        start_time = time.perf_counter()

        try:
            # Test individual components
            device_start = time.perf_counter()
            device_manager = XLADeviceManager(self.config.hardware.tpu)
            device_time = time.perf_counter() - device_start

            dist_start = time.perf_counter()
            distributed = XLADistributedTraining(device_manager)
            dist_time = time.perf_counter() - dist_start

            opt_start = time.perf_counter()
            optimizations = XLAOptimizations(self.config.hardware.tpu)
            opt_time = time.perf_counter() - opt_start

            # Test integration factory
            factory_start = time.perf_counter()
            dev_mgr, dist_training, opts = create_xla_integration(self.config.hardware.tpu)
            factory_time = time.perf_counter() - factory_start

            # Test device operations
            device_ops_start = time.perf_counter()
            device_stats = device_manager.get_device_stats()
            device_manager.sync_all_devices()
            device_ops_time = time.perf_counter() - device_ops_start

            # Test optimizations
            model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

            model_opt_start = time.perf_counter()
            optimized_model = optimizations.optimize_model_for_xla(model)
            hinted_model = optimizations.add_compilation_hints(optimized_model)
            model_opt_time = time.perf_counter() - model_opt_start

            # Test utilities
            util_start = time.perf_counter()
            env_info = XLAUtilities.get_xla_env_info()
            flags = XLAUtilities.optimize_xla_flags(self.config.hardware.tpu.version)
            util_time = time.perf_counter() - util_start

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name="XLA Integration",
                duration=duration,
                success=True,
                metrics={
                    'device_manager_init_time': device_time * 1000,
                    'distributed_init_time': dist_time * 1000,
                    'optimizations_init_time': opt_time * 1000,
                    'factory_creation_time': factory_time * 1000,
                    'device_operations_time': device_ops_time * 1000,
                    'model_optimization_time': model_opt_time * 1000,
                    'utilities_time': util_time * 1000,
                    'device_stats': device_stats,
                    'env_info_available': env_info.get('xla_available', False),
                    'flags_optimized': len(flags)
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="XLA Integration",
                duration=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )

    def benchmark_tpu_validation(self) -> BenchmarkResult:
        """Benchmark TPU validation performance."""
        start_time = time.perf_counter()

        try:
            # Test configuration validation
            config_start = time.perf_counter()
            config_results = validate_tpu_configuration(self.config)
            config_time = time.perf_counter() - config_start

            # Test model validation with different models
            models = [
                nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)),
                nn.Sequential(nn.Linear(63, 31), nn.ReLU(), nn.Linear(31, 7))  # Non-optimal dimensions
            ]

            model_validation_times = []
            validation_results = []

            for i, model in enumerate(models):
                sample_input = torch.randn(8, 64 if i == 0 else 63)

                model_start = time.perf_counter()
                model_results = validate_tpu_model(model, self.config.hardware.tpu, sample_input)
                model_time = time.perf_counter() - model_start

                model_validation_times.append(model_time * 1000)  # ms
                validation_results.append({
                    'passed': model_results.passed,
                    'total': model_results.total_tests,
                    'warnings': model_results.warnings,
                    'success_rate': model_results.success_rate
                })

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name="TPU Validation",
                duration=duration,
                success=True,
                metrics={
                    'config_validation_time': config_time * 1000,
                    'avg_model_validation_time': sum(model_validation_times) / len(model_validation_times),
                    'config_validation_results': {
                        'passed': config_results.passed,
                        'total': config_results.total_tests,
                        'success_rate': config_results.success_rate
                    },
                    'model_validation_results': validation_results,
                    'models_validated': len(models)
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="TPU Validation",
                duration=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )

    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run the complete TPU benchmark suite."""
        suite_start = time.perf_counter()
        suite = BenchmarkSuite()

        benchmarks = [
            self.benchmark_tpu_configuration,
            self.benchmark_tpu_backend,
            self.benchmark_tpu_optimizer,
            self.benchmark_xla_compiler,
            self.benchmark_memory_manager,
            self.benchmark_xla_integration,
            self.benchmark_tpu_validation
        ]

        for i, benchmark_func in enumerate(benchmarks, 1):
            print(f"\n📊 Running benchmark {i}/{len(benchmarks)}: {benchmark_func.__name__}")

            try:
                result = benchmark_func()
                suite.results.append(result)

                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"   {status} {result.name}: {result.duration:.3f}s")

                if not result.success and result.error:
                    print(f"   Error: {result.error}")

            except Exception as e:
                error_result = BenchmarkResult(
                    name=benchmark_func.__name__,
                    duration=0.0,
                    success=False,
                    error=f"Benchmark crashed: {str(e)}"
                )
                suite.results.append(error_result)
                print(f"   ❌ CRASH {benchmark_func.__name__}: {str(e)}")

            # Force garbage collection between benchmarks
            gc.collect()

        suite.total_duration = time.perf_counter() - suite_start
        return suite

    def print_detailed_results(self, suite: BenchmarkSuite) -> None:
        """Print detailed benchmark results."""
        print("\n" + "="*70)
        print("🎯 TPU Integration Benchmark Results")
        print("="*70)

        print("📈 Overall Statistics:")
        print(f"   Total time: {suite.total_duration:.3f}s")
        print(f"   Success rate: {suite.success_rate:.1%}")
        print(f"   Successful tests: {suite.successful_tests}/{len(suite.results)}")

        print("\n📋 Individual Results:")
        for result in suite.results:
            status = "✅" if result.success else "❌"
            print(f"   {status} {result.name}: {result.duration:.3f}s")

            if result.success and result.metrics:
                # Print key metrics
                key_metrics = []
                for key, value in result.metrics.items():
                    if isinstance(value, (int, float)) and 'time' in key.lower():
                        key_metrics.append(f"{key}: {value:.2f}ms")
                    elif isinstance(value, (int, float)) and key in ['total_optimizations', 'models_compiled', 'tensors_allocated']:
                        key_metrics.append(f"{key}: {value}")

                if key_metrics:
                    print(f"      {', '.join(key_metrics[:3])}")  # Show first 3 metrics

            elif not result.success:
                print(f"      Error: {result.error}")

        print("\n🎉 Benchmark completed!")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='TPU Integration Comprehensive Benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of benchmarks')
    parser.add_argument('--device', default='auto', choices=['cpu', 'auto'],
                       help='Device to use for benchmarking')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    try:
        # Set up environment
        os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent / "src")

        # Create benchmark runner
        runner = TPUBenchmarkRunner(device=args.device, quick_mode=args.quick)

        # Run benchmarks
        suite = runner.run_comprehensive_benchmark()

        # Print results
        runner.print_detailed_results(suite)

        # Exit with appropriate code
        exit_code = 0 if suite.success_rate >= 0.8 else 1
        sys.exit(exit_code)

    except Exception as e:
        print(f"\n❌ Benchmark suite crashed: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
