#!/usr/bin/env python3
"""
Comprehensive benchmark script for optimization patterns

This script provides systematic benchmarking of all three optimization patterns:
- Memory Efficiency Patterns
- Compute Intensity Patterns
- Compiler-Friendly Patterns

🎯 BENCHMARKING GOALS:
- Measure performance improvements
- Validate optimization effectiveness
- Compare baseline vs optimized implementations
- Generate performance reports
"""

import torch
import torch.nn as nn
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse

# Import optimization patterns
from kernel_pytorch.optimizations.patterns.memory_efficiency import (
    MemoryEfficientSequential,
    benchmark_memory_optimizations,
    analyze_memory_access_patterns
)
from kernel_pytorch.optimizations.patterns.compute_intensity import (
    analyze_compute_intensity_profile,
    calculate_arithmetic_intensity
)
from kernel_pytorch.optimizations.patterns.compiler_friendly import (
    OptimizedTransformerBlock,
    OptimizedLinearGELU,
    benchmark_compilation_impact,
    check_compilation_compatibility,
    optimize_for_torch_compile
)


class OptimizationPatternBenchmark:
    """Comprehensive benchmark suite for optimization patterns."""

    def __init__(self, device: str = "auto", quick: bool = False):
        """
        Initialize benchmark suite.

        Args:
            device: Device to run benchmarks on ("auto", "cpu", "cuda")
            quick: Whether to run quick benchmarks (fewer iterations)
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.quick = quick
        self.num_runs = 10 if quick else 50
        self.warmup_runs = 3 if quick else 10

        self.results = {}

    def time_execution(self, func, *args, num_runs: int = None) -> Dict[str, float]:
        """Time function execution with proper warmup."""
        num_runs = num_runs or self.num_runs

        # Warmup
        for _ in range(self.warmup_runs):
            func(*args)

        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Time execution
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = func(*args)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            "mean_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }

    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency patterns."""
        print("\n🧠 Benchmarking Memory Efficiency Patterns")
        print("-" * 50)

        results = {}

        # Test 1: Memory-efficient sequential vs standard
        print("Test 1: Memory-Efficient Sequential")

        standard_model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(self.device)

        efficient_model = MemoryEfficientSequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(self.device)

        test_input = torch.randn(64, 512, device=self.device)

        # Time both models
        standard_times = self.time_execution(lambda: standard_model(test_input))
        efficient_times = self.time_execution(lambda: efficient_model(test_input))

        speedup = standard_times["mean_time_ms"] / efficient_times["mean_time_ms"]

        results["memory_efficient_sequential"] = {
            "standard_model": standard_times,
            "efficient_model": efficient_times,
            "speedup": speedup,
            "improvement_percent": (speedup - 1) * 100
        }

        print(f"  Standard model: {standard_times['mean_time_ms']:.2f}ms")
        print(f"  Efficient model: {efficient_times['mean_time_ms']:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")

        # Test 2: Memory pattern analysis
        print("\nTest 2: Memory Pattern Analysis")
        try:
            analysis = analyze_memory_access_patterns(standard_model, test_input)
            results["memory_analysis"] = {
                "patterns_found": len(analysis) if isinstance(analysis, dict) else 0,
                "analysis_successful": True
            }
            print(f"  Memory patterns identified: {results['memory_analysis']['patterns_found']}")
        except Exception as e:
            results["memory_analysis"] = {
                "patterns_found": 0,
                "analysis_successful": False,
                "error": str(e)
            }
            print(f"  Memory analysis failed: {str(e)[:50]}...")

        return results

    def benchmark_compute_intensity(self) -> Dict[str, Any]:
        """Benchmark compute intensity patterns."""
        print("\n⚡ Benchmarking Compute Intensity Patterns")
        print("-" * 50)

        results = {}

        # Test 1: Arithmetic intensity calculation
        print("Test 1: Arithmetic Intensity Analysis")

        operations = {
            "element_wise_add": lambda x, y: x + y,
            "matrix_multiply": lambda x, y: torch.matmul(x, y.transpose(-1, -2)),
            "convolution": lambda x, conv: conv(x)
        }

        x = torch.randn(64, 256, device=self.device)
        y = torch.randn(64, 256, device=self.device)
        conv_input = torch.randn(16, 3, 32, 32, device=self.device)
        conv_layer = nn.Conv2d(3, 64, 3, padding=1).to(self.device)

        intensity_results = {}
        for name, op in operations.items():
            try:
                if name == "convolution":
                    intensity = calculate_arithmetic_intensity(op, conv_input, conv_layer)
                else:
                    intensity = calculate_arithmetic_intensity(op, x, y)

                intensity_results[name] = {
                    "intensity": float(intensity) if torch.is_tensor(intensity) else intensity,
                    "success": True
                }
                print(f"  {name}: {intensity:.2f} FLOP/byte")
            except Exception as e:
                intensity_results[name] = {
                    "intensity": 0.0,
                    "success": False,
                    "error": str(e)[:50]
                }
                print(f"  {name}: Failed ({str(e)[:30]}...)")

        results["arithmetic_intensity"] = intensity_results

        # Test 2: Model compute intensity profile
        print("\nTest 2: Model Compute Intensity Profile")

        test_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        ).to(self.device)

        test_input = torch.randn(32, 256, device=self.device)

        try:
            profile = analyze_compute_intensity_profile(test_model, test_input)
            results["model_profile"] = {
                "total_layers": profile.get("total_layers", 0),
                "memory_bound_count": profile.get("memory_bound_count", 0),
                "compute_bound_count": profile.get("compute_bound_count", 0),
                "overall_intensity": profile.get("overall_intensity", 0.0),
                "success": True
            }

            print(f"  Total layers: {results['model_profile']['total_layers']}")
            print(f"  Memory-bound: {results['model_profile']['memory_bound_count']}")
            print(f"  Compute-bound: {results['model_profile']['compute_bound_count']}")
            print(f"  Overall intensity: {results['model_profile']['overall_intensity']:.2f} FLOP/byte")

        except Exception as e:
            results["model_profile"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  Model profiling failed: {str(e)[:50]}...")

        return results

    def benchmark_compiler_friendly(self) -> Dict[str, Any]:
        """Benchmark compiler-friendly patterns."""
        print("\n🔧 Benchmarking Compiler-Friendly Patterns")
        print("-" * 50)

        results = {}

        # Test 1: Optimized vs standard modules
        print("Test 1: Optimized Module Performance")

        # Standard modules
        standard_linear_gelu = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU()
        ).to(self.device)

        standard_transformer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        ).to(self.device)

        # Optimized modules
        optimized_linear_gelu = OptimizedLinearGELU(256, 512).to(self.device)

        optimized_transformer = OptimizedTransformerBlock(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024
        ).to(self.device)

        # Test inputs
        linear_input = torch.randn(32, 256, device=self.device)
        transformer_input = torch.randn(32, 128, 256, device=self.device)

        # Benchmark Linear+GELU
        standard_lg_times = self.time_execution(lambda: standard_linear_gelu(linear_input))
        optimized_lg_times = self.time_execution(lambda: optimized_linear_gelu(linear_input))

        lg_speedup = standard_lg_times["mean_time_ms"] / optimized_lg_times["mean_time_ms"]

        results["linear_gelu"] = {
            "standard": standard_lg_times,
            "optimized": optimized_lg_times,
            "speedup": lg_speedup,
            "improvement_percent": (lg_speedup - 1) * 100
        }

        print(f"  Linear+GELU:")
        print(f"    Standard: {standard_lg_times['mean_time_ms']:.2f}ms")
        print(f"    Optimized: {optimized_lg_times['mean_time_ms']:.2f}ms")
        print(f"    Speedup: {lg_speedup:.2f}x ({(lg_speedup-1)*100:+.1f}%)")

        # Benchmark Transformer
        standard_tf_times = self.time_execution(lambda: standard_transformer(transformer_input))
        optimized_tf_times = self.time_execution(lambda: optimized_transformer(transformer_input))

        tf_speedup = standard_tf_times["mean_time_ms"] / optimized_tf_times["mean_time_ms"]

        results["transformer"] = {
            "standard": standard_tf_times,
            "optimized": optimized_tf_times,
            "speedup": tf_speedup,
            "improvement_percent": (tf_speedup - 1) * 100
        }

        print(f"  Transformer Block:")
        print(f"    Standard: {standard_tf_times['mean_time_ms']:.2f}ms")
        print(f"    Optimized: {optimized_tf_times['mean_time_ms']:.2f}ms")
        print(f"    Speedup: {tf_speedup:.2f}x ({(tf_speedup-1)*100:+.1f}%)")

        # Test 2: Compilation compatibility
        print("\nTest 2: Compilation Compatibility")

        models_to_test = {
            "standard_linear": nn.Linear(128, 64),
            "optimized_linear_gelu": OptimizedLinearGELU(128, 64),
            "optimized_transformer": OptimizedTransformerBlock(embed_dim=64, num_heads=4, feedforward_dim=256)
        }

        compatibility_results = {}
        for name, model in models_to_test.items():
            model = model.to(self.device)
            test_input = torch.randn(16, 128 if "transformer" not in name else 32,
                                    128 if "transformer" not in name else 64, device=self.device)

            try:
                compatibility = check_compilation_compatibility(model, test_input)
                compatibility_results[name] = {
                    "score": compatibility.get("compatibility_score", 0.0),
                    "issues": len(compatibility.get("issues", [])),
                    "success": True
                }
                print(f"  {name}: {compatibility_results[name]['score']:.2f} compatibility score")
            except Exception as e:
                compatibility_results[name] = {
                    "score": 0.0,
                    "success": False,
                    "error": str(e)[:50]
                }
                print(f"  {name}: Compatibility check failed")

        results["compatibility"] = compatibility_results

        # Test 3: torch.compile performance (if available)
        print("\nTest 3: torch.compile Performance")

        try:
            simple_model = nn.Sequential(
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 256)
            ).to(self.device)

            optimized_model = optimize_for_torch_compile(simple_model)
            compiled_model = torch.compile(optimized_model)

            test_input = torch.randn(32, 256, device=self.device)

            # Time all three versions
            standard_times = self.time_execution(lambda: simple_model(test_input))
            optimized_times = self.time_execution(lambda: optimized_model(test_input))
            compiled_times = self.time_execution(lambda: compiled_model(test_input))

            results["torch_compile"] = {
                "standard": standard_times,
                "optimized": optimized_times,
                "compiled": compiled_times,
                "optimized_speedup": standard_times["mean_time_ms"] / optimized_times["mean_time_ms"],
                "compiled_speedup": standard_times["mean_time_ms"] / compiled_times["mean_time_ms"],
                "success": True
            }

            print(f"  Standard: {standard_times['mean_time_ms']:.2f}ms")
            print(f"  Optimized: {optimized_times['mean_time_ms']:.2f}ms")
            print(f"  Compiled: {compiled_times['mean_time_ms']:.2f}ms")
            print(f"  Compilation speedup: {results['torch_compile']['compiled_speedup']:.2f}x")

        except Exception as e:
            results["torch_compile"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  torch.compile failed: {str(e)[:50]}...")

        return results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all patterns."""
        print("📊 Comprehensive Optimization Patterns Benchmark")
        print("=" * 60)
        print(f"🔧 Device: {self.device}")
        print(f"⚡ Mode: {'Quick' if self.quick else 'Full'}")
        print(f"🔄 Runs per test: {self.num_runs}")

        start_time = time.time()

        # Run all benchmarks
        self.results["memory_efficiency"] = self.benchmark_memory_efficiency()
        self.results["compute_intensity"] = self.benchmark_compute_intensity()
        self.results["compiler_friendly"] = self.benchmark_compiler_friendly()

        total_time = time.time() - start_time

        # Generate summary
        self.results["benchmark_info"] = {
            "device": str(self.device),
            "quick_mode": self.quick,
            "num_runs": self.num_runs,
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__
        }

        print(f"\n✅ Benchmark completed in {total_time:.1f}s")
        return self.results

    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_patterns_benchmark_{timestamp}.json"

        # Create results directory
        results_dir = "tests/patterns/results"
        os.makedirs(results_dir, exist_ok=True)

        filepath = os.path.join(results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"📁 Results saved to: {filepath}")
        return filepath

    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            print("No benchmark results available")
            return

        print("\n📈 BENCHMARK SUMMARY")
        print("=" * 40)

        # Memory efficiency summary
        if "memory_efficiency" in self.results:
            mem_results = self.results["memory_efficiency"]
            if "memory_efficient_sequential" in mem_results:
                speedup = mem_results["memory_efficient_sequential"]["speedup"]
                print(f"🧠 Memory Efficiency: {speedup:.2f}x speedup")

        # Compute intensity summary
        if "compute_intensity" in self.results:
            comp_results = self.results["compute_intensity"]
            if "model_profile" in comp_results and comp_results["model_profile"]["success"]:
                intensity = comp_results["model_profile"]["overall_intensity"]
                print(f"⚡ Compute Intensity: {intensity:.2f} FLOP/byte")

        # Compiler friendly summary
        if "compiler_friendly" in self.results:
            cf_results = self.results["compiler_friendly"]
            if "linear_gelu" in cf_results:
                lg_speedup = cf_results["linear_gelu"]["speedup"]
                print(f"🔧 Compiler Optimization: {lg_speedup:.2f}x speedup")

        benchmark_info = self.results.get("benchmark_info", {})
        total_time = benchmark_info.get("total_time_seconds", 0)
        print(f"⏱️  Total benchmark time: {total_time:.1f}s")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark optimization patterns")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to run benchmarks on")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks (fewer iterations)")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")

    args = parser.parse_args()

    # Run benchmark
    benchmark = OptimizationPatternBenchmark(device=args.device, quick=args.quick)
    results = benchmark.run_comprehensive_benchmark()

    # Print summary
    benchmark.print_summary()

    # Save results if requested
    if args.save:
        benchmark.save_results()


if __name__ == "__main__":
    main()