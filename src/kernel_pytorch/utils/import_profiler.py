"""
Import Performance Profiler (2025)

Measures and benchmarks import performance improvements from lazy loading
and other optimization techniques.
"""

import importlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class ImportMetrics:
    """Metrics for a single import operation."""
    module_name: str
    import_time: float
    memory_before: int
    memory_after: int
    memory_delta: int
    success: bool
    error: str | None = None


class ImportProfiler:
    """
    Profiler for measuring import performance and memory usage.

    Provides before/after benchmarks to measure lazy loading benefits.
    """

    def __init__(self):
        self.results: list[ImportMetrics] = []

    @contextmanager
    def profile_import(self, module_name: str):
        """Context manager for profiling a single import."""
        # Clear module cache to ensure clean measurement
        if module_name in sys.modules:
            del sys.modules[module_name]

        memory_before = self._get_memory_usage()
        start_time = time.perf_counter()
        error = None
        success = True

        try:
            yield
        except Exception as e:
            error = str(e)
            success = False
        finally:
            end_time = time.perf_counter()
            memory_after = self._get_memory_usage()

            metrics = ImportMetrics(
                module_name=module_name,
                import_time=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_after - memory_before,
                success=success,
                error=error
            )
            self.results.append(metrics)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import os

            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback - return 0 if psutil not available
            return 0

    def benchmark_imports(self, modules: list[str]) -> dict[str, ImportMetrics]:
        """
        Benchmark import performance for a list of modules.

        Args:
            modules: List of module names to benchmark

        Returns:
            Dictionary mapping module names to their import metrics
        """
        results = {}

        for module_name in modules:
            with self.profile_import(module_name):
                try:
                    importlib.import_module(module_name)
                except Exception:
                    pass  # Error handling in context manager

            if self.results:
                results[module_name] = self.results[-1]

        return results

    def compare_import_strategies(
        self,
        eager_modules: list[str],
        lazy_modules: list[str]
    ) -> dict[str, Any]:
        """
        Compare eager vs lazy import performance.

        Args:
            eager_modules: Modules imported eagerly
            lazy_modules: Modules imported lazily

        Returns:
            Comparison results with timing and memory data
        """
        print(" Benchmarking eager imports...")
        eager_results = self.benchmark_imports(eager_modules)

        print(" Benchmarking lazy imports...")
        lazy_results = self.benchmark_imports(lazy_modules)

        # Calculate aggregate metrics
        eager_total_time = sum(m.import_time for m in eager_results.values() if m.success)
        lazy_total_time = sum(m.import_time for m in lazy_results.values() if m.success)

        eager_total_memory = sum(m.memory_delta for m in eager_results.values() if m.success)
        lazy_total_memory = sum(m.memory_delta for m in lazy_results.values() if m.success)

        speedup = eager_total_time / lazy_total_time if lazy_total_time > 0 else 0
        memory_reduction = (eager_total_memory - lazy_total_memory) / eager_total_memory * 100 if eager_total_memory > 0 else 0

        return {
            'eager_results': eager_results,
            'lazy_results': lazy_results,
            'summary': {
                'eager_total_time': eager_total_time,
                'lazy_total_time': lazy_total_time,
                'speedup_factor': speedup,
                'memory_reduction_percent': memory_reduction,
                'eager_total_memory': eager_total_memory,
                'lazy_total_memory': lazy_total_memory
            }
        }

    def generate_report(self) -> str:
        """Generate a formatted report of import performance."""
        if not self.results:
            return "No import metrics available."

        report = "# Import Performance Report\n\n"
        report += f"**Total imports measured**: {len(self.results)}\n"

        successful_imports = [r for r in self.results if r.success]
        failed_imports = [r for r in self.results if not r.success]

        if successful_imports:
            total_time = sum(r.import_time for r in successful_imports)
            avg_time = total_time / len(successful_imports)
            total_memory = sum(r.memory_delta for r in successful_imports)

            report += f"**Successful imports**: {len(successful_imports)}\n"
            report += f"**Total import time**: {total_time:.4f} seconds\n"
            report += f"**Average import time**: {avg_time:.4f} seconds\n"
            report += f"**Total memory delta**: {total_memory / (1024*1024):.2f} MB\n\n"

            # Individual results table
            report += "## Individual Import Metrics\n\n"
            report += "| Module | Time (s) | Memory (MB) | Status |\n"
            report += "|--------|----------|-------------|--------|\n"

            for result in successful_imports:
                memory_mb = result.memory_delta / (1024*1024)
                report += f"| {result.module_name} | {result.import_time:.4f} | {memory_mb:.2f} |  |\n"

        if failed_imports:
            report += "\n## Failed Imports\n\n"
            for result in failed_imports:
                report += f"- **{result.module_name}**: {result.error}\n"

        return report


def benchmark_lazy_loading_improvements() -> dict[str, Any]:
    """
    Benchmark the lazy loading improvements.

    Returns comprehensive before/after performance comparison.
    """
    profiler = ImportProfiler()

    # Test modules that should benefit from lazy loading
    core_modules = [
        'kernel_pytorch',
        'kernel_pytorch.components',
        'kernel_pytorch.utils'
    ]

    heavy_modules = [
        'kernel_pytorch.distributed_scale',
        'kernel_pytorch.testing_framework',
        'kernel_pytorch.next_gen_optimizations'
    ]

    print(" Benchmarking lazy loading performance...")

    # Benchmark core modules (should be fast)
    core_results = profiler.benchmark_imports(core_modules)

    # Benchmark heavy modules (should show improvement)
    heavy_results = profiler.benchmark_imports(heavy_modules)

    # Generate comprehensive results
    all_results = {**core_results, **heavy_results}

    return {
        'core_modules': core_results,
        'heavy_modules': heavy_results,
        'all_results': all_results,
        'summary': {
            'total_modules': len(all_results),
            'successful_imports': len([r for r in all_results.values() if r.success]),
            'total_import_time': sum(r.import_time for r in all_results.values() if r.success),
            'average_import_time': sum(r.import_time for r in all_results.values() if r.success) / len(all_results) if all_results else 0
        },
        'report': profiler.generate_report()
    }


def measure_cold_startup_time() -> tuple[float, str]:
    """
    Measure cold startup time for the main package.

    Returns:
        Tuple of (startup_time, status_message)
    """
    # Clear all kernel_pytorch modules from cache
    to_remove = [name for name in sys.modules.keys() if name.startswith('kernel_pytorch')]
    for name in to_remove:
        del sys.modules[name]

    start_time = time.perf_counter()
    try:
        end_time = time.perf_counter()
        startup_time = end_time - start_time
        return startup_time, f" Cold startup: {startup_time:.4f}s"
    except Exception as e:
        end_time = time.perf_counter()
        startup_time = end_time - start_time
        return startup_time, f" Startup failed after {startup_time:.4f}s: {e}"


if __name__ == "__main__":
    print(" Running import performance benchmarks...")

    # Test cold startup
    startup_time, startup_message = measure_cold_startup_time()
    print(startup_message)

    # Run comprehensive benchmarks
    results = benchmark_lazy_loading_improvements()
    print("\n Benchmark Summary:")
    print(f"  Total modules tested: {results['summary']['total_modules']}")
    print(f"  Successful imports: {results['summary']['successful_imports']}")
    print(f"  Total import time: {results['summary']['total_import_time']:.4f}s")
    print(f"  Average per module: {results['summary']['average_import_time']:.4f}s")

    print("\n" + results['report'])
