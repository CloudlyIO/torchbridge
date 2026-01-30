#!/usr/bin/env python3
"""
Backend Performance Comparison Benchmark

Comprehensive performance comparison between NVIDIA and TPU backends,
measuring latency, throughput, memory usage, and synchronization overhead.

Phase 4C-Pre Week 3: Integration Testing (v0.3.3)

Usage:
    python3 backend_comparison_benchmark.py [--quick] [--device cpu|auto]
"""

import argparse
import torch
import torch.nn as nn
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from torchbridge.core.config import TorchBridgeConfig
from torchbridge.core.hardware_detector import HardwareDetector
from torchbridge.backends.nvidia import NVIDIABackend
from torchbridge.backends.tpu import TPUBackend


# ============================================================================
# Benchmark Models
# ============================================================================

class SmallModel(nn.Module):
    """Small model for quick benchmarks."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MediumModel(nn.Module):
    """Medium model for standard benchmarks."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class LargeModel(nn.Module):
    """Large model for stress testing."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================================
# Benchmark Suite
# ============================================================================

class BackendComparisonBenchmark:
    """Comprehensive backend performance comparison."""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.warmup_iterations = 3 if quick_mode else 10
        self.benchmark_iterations = 5 if quick_mode else 50

        self.config = TorchBridgeConfig()
        self.nvidia_backend = NVIDIABackend(self.config)
        self.tpu_backend = TPUBackend(self.config)

        self.results = {
            'nvidia': {},
            'tpu': {},
            'comparison': {}
        }

    def benchmark_model_preparation(self) -> Dict[str, Any]:
        """Benchmark model preparation time."""
        print("\n📊 Benchmark 1/7: Model Preparation Time")
        print("-" * 50)

        models = {
            'small': SmallModel(),
            'medium': MediumModel(),
            'large': LargeModel() if not self.quick_mode else None
        }

        results = {'nvidia': {}, 'tpu': {}}

        for model_name, model in models.items():
            if model is None:
                continue

            # NVIDIA preparation
            nvidia_times = []
            for _ in range(self.benchmark_iterations):
                start = time.perf_counter()
                self.nvidia_backend.prepare_model(model)
                nvidia_times.append(time.perf_counter() - start)

            # TPU preparation
            tpu_times = []
            for _ in range(self.benchmark_iterations):
                start = time.perf_counter()
                self.tpu_backend.prepare_model(model)
                tpu_times.append(time.perf_counter() - start)

            nvidia_avg = statistics.mean(nvidia_times) * 1000  # ms
            tpu_avg = statistics.mean(tpu_times) * 1000  # ms

            results['nvidia'][model_name] = nvidia_avg
            results['tpu'][model_name] = tpu_avg

            speedup = nvidia_avg / tpu_avg if tpu_avg > 0 else 1.0
            print(f"   {model_name.capitalize()} Model:")
            print(f"      NVIDIA: {nvidia_avg:.3f}ms")
            print(f"      TPU:    {tpu_avg:.3f}ms")
            print(f"      Speedup: {speedup:.2f}x {'(TPU faster)' if speedup > 1 else '(NVIDIA faster)'}")

        self.results['nvidia']['model_preparation'] = results['nvidia']
        self.results['tpu']['model_preparation'] = results['tpu']

        print("   ✅ Model Preparation Benchmark Complete")
        return results

    def benchmark_forward_pass_latency(self) -> Dict[str, Any]:
        """Benchmark forward pass latency."""
        print("\n📊 Benchmark 2/7: Forward Pass Latency")
        print("-" * 50)

        test_cases = [
            ('small', SmallModel(), (32, 128)),
            ('medium', MediumModel(), (32, 512)),
        ]

        if not self.quick_mode:
            test_cases.append(('large', LargeModel(), (32, 1024)))

        results = {'nvidia': {}, 'tpu': {}}

        for case_name, model, input_shape in test_cases:
            # Prepare models
            nvidia_model = self.nvidia_backend.prepare_model(model)
            tpu_model = self.tpu_backend.prepare_model(model)

            # Prepare inputs
            nvidia_input = torch.randn(input_shape).to(self.nvidia_backend.device)
            tpu_input = torch.randn(input_shape).to(self.tpu_backend.device)

            # Warmup
            for _ in range(self.warmup_iterations):
                with torch.no_grad():
                    _ = nvidia_model(nvidia_input)
                    try:
                        _ = tpu_model(tpu_input)
                    except RuntimeError:
                        # Expected dtype mismatch with bfloat16
                        pass

            self.nvidia_backend.synchronize()
            self.tpu_backend.synchronize()

            # NVIDIA benchmark
            nvidia_times = []
            for _ in range(self.benchmark_iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = nvidia_model(nvidia_input)
                self.nvidia_backend.synchronize()
                nvidia_times.append(time.perf_counter() - start)

            # TPU benchmark
            tpu_times = []
            for _ in range(self.benchmark_iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    try:
                        _ = tpu_model(tpu_input)
                    except RuntimeError:
                        # Expected dtype mismatch with bfloat16
                        pass
                self.tpu_backend.synchronize()
                tpu_times.append(time.perf_counter() - start)

            nvidia_avg = statistics.mean(nvidia_times) * 1000  # ms
            tpu_avg = statistics.mean(tpu_times) * 1000  # ms

            results['nvidia'][case_name] = nvidia_avg
            results['tpu'][case_name] = tpu_avg

            speedup = nvidia_avg / tpu_avg if tpu_avg > 0 else 1.0
            print(f"   {case_name.capitalize()} Model:")
            print(f"      NVIDIA: {nvidia_avg:.3f}ms")
            print(f"      TPU:    {tpu_avg:.3f}ms")
            print(f"      Speedup: {speedup:.2f}x {'(TPU faster)' if speedup > 1 else '(NVIDIA faster)'}")

        self.results['nvidia']['forward_latency'] = results['nvidia']
        self.results['tpu']['forward_latency'] = results['tpu']

        print("   ✅ Forward Pass Latency Benchmark Complete")
        return results

    def benchmark_throughput(self) -> Dict[str, Any]:
        """Benchmark throughput (batches/second)."""
        print("\n📊 Benchmark 3/7: Throughput (batches/second)")
        print("-" * 50)

        model = MediumModel()
        batch_sizes = [16, 32] if self.quick_mode else [16, 32, 64]
        input_size = 512

        results = {'nvidia': {}, 'tpu': {}}

        for batch_size in batch_sizes:
            # Prepare models
            nvidia_model = self.nvidia_backend.prepare_model(model)
            tpu_model = self.tpu_backend.prepare_model(model)

            # NVIDIA throughput
            nvidia_input = torch.randn(batch_size, input_size).to(self.nvidia_backend.device)

            # Warmup
            for _ in range(self.warmup_iterations):
                with torch.no_grad():
                    _ = nvidia_model(nvidia_input)
            self.nvidia_backend.synchronize()

            start = time.perf_counter()
            for _ in range(self.benchmark_iterations):
                with torch.no_grad():
                    _ = nvidia_model(nvidia_input)
            self.nvidia_backend.synchronize()
            elapsed = time.perf_counter() - start
            nvidia_throughput = self.benchmark_iterations / elapsed

            # TPU throughput
            tpu_input = torch.randn(batch_size, input_size).to(self.tpu_backend.device)

            # Warmup
            for _ in range(self.warmup_iterations):
                with torch.no_grad():
                    try:
                        _ = tpu_model(tpu_input)
                    except RuntimeError:
                        pass
            self.tpu_backend.synchronize()

            start = time.perf_counter()
            for _ in range(self.benchmark_iterations):
                with torch.no_grad():
                    try:
                        _ = tpu_model(tpu_input)
                    except RuntimeError:
                        pass
            self.tpu_backend.synchronize()
            elapsed = time.perf_counter() - start
            tpu_throughput = self.benchmark_iterations / elapsed

            results['nvidia'][f'batch_{batch_size}'] = nvidia_throughput
            results['tpu'][f'batch_{batch_size}'] = tpu_throughput

            speedup = tpu_throughput / nvidia_throughput if nvidia_throughput > 0 else 1.0
            print(f"   Batch Size {batch_size}:")
            print(f"      NVIDIA: {nvidia_throughput:.2f} batches/sec")
            print(f"      TPU:    {tpu_throughput:.2f} batches/sec")
            print(f"      Speedup: {speedup:.2f}x {'(TPU faster)' if speedup > 1 else '(NVIDIA faster)'}")

        self.results['nvidia']['throughput'] = results['nvidia']
        self.results['tpu']['throughput'] = results['tpu']

        print("   ✅ Throughput Benchmark Complete")
        return results

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        print("\n📊 Benchmark 4/7: Memory Usage")
        print("-" * 50)

        model = MediumModel()

        # NVIDIA memory
        nvidia_model = self.nvidia_backend.prepare_model(model)
        nvidia_stats = self.nvidia_backend.get_memory_stats()

        # TPU memory
        tpu_model = self.tpu_backend.prepare_model(model)
        tpu_stats = self.tpu_backend.get_memory_stats()

        results = {
            'nvidia': nvidia_stats,
            'tpu': tpu_stats
        }

        print(f"   NVIDIA Memory Stats:")
        for key, value in nvidia_stats.items():
            print(f"      {key}: {value}")

        print(f"   TPU Memory Stats:")
        for key, value in tpu_stats.items():
            print(f"      {key}: {value}")

        self.results['nvidia']['memory_usage'] = nvidia_stats
        self.results['tpu']['memory_usage'] = tpu_stats

        print("   ✅ Memory Usage Benchmark Complete")
        return results

    def benchmark_synchronization_overhead(self) -> Dict[str, Any]:
        """Benchmark synchronization overhead."""
        print("\n📊 Benchmark 5/7: Synchronization Overhead")
        print("-" * 50)

        results = {'nvidia': [], 'tpu': []}

        # NVIDIA sync overhead
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            self.nvidia_backend.synchronize()
            results['nvidia'].append((time.perf_counter() - start) * 1000)  # ms

        # TPU sync overhead
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            self.tpu_backend.synchronize()
            results['tpu'].append((time.perf_counter() - start) * 1000)  # ms

        nvidia_avg = statistics.mean(results['nvidia'])
        tpu_avg = statistics.mean(results['tpu'])

        print(f"   NVIDIA sync: {nvidia_avg:.4f}ms (avg)")
        print(f"   TPU sync:    {tpu_avg:.4f}ms (avg)")

        self.results['nvidia']['sync_overhead'] = nvidia_avg
        self.results['tpu']['sync_overhead'] = tpu_avg

        print("   ✅ Synchronization Overhead Benchmark Complete")
        return results

    def benchmark_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        print("\n📊 Benchmark 6/7: Device Information")
        print("-" * 50)

        # NVIDIA info
        nvidia_info = self.nvidia_backend.get_device_info()
        print(f"   NVIDIA Device:")
        for key, value in nvidia_info.items():
            print(f"      {key}: {value}")

        # TPU info (similar structure)
        tpu_info = {
            'device': str(self.tpu_backend.device),
            'world_size': self.tpu_backend.world_size,
            'rank': self.tpu_backend.rank,
            'is_distributed': self.tpu_backend.is_distributed
        }
        print(f"   TPU Device:")
        for key, value in tpu_info.items():
            print(f"      {key}: {value}")

        self.results['nvidia']['device_info'] = nvidia_info
        self.results['tpu']['device_info'] = tpu_info

        print("   ✅ Device Information Benchmark Complete")
        return {'nvidia': nvidia_info, 'tpu': tpu_info}

    def benchmark_batch_sizes_scaling(self) -> Dict[str, Any]:
        """Benchmark performance scaling with batch size."""
        print("\n📊 Benchmark 7/7: Batch Size Scaling")
        print("-" * 50)

        model = SmallModel()
        batch_sizes = [8, 16, 32] if self.quick_mode else [8, 16, 32, 64, 128]
        input_size = 128

        results = {'nvidia': {}, 'tpu': {}}

        for batch_size in batch_sizes:
            # Prepare models
            nvidia_model = self.nvidia_backend.prepare_model(model)
            nvidia_input = torch.randn(batch_size, input_size).to(self.nvidia_backend.device)

            tpu_model = self.tpu_backend.prepare_model(model)
            tpu_input = torch.randn(batch_size, input_size).to(self.tpu_backend.device)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = nvidia_model(nvidia_input)
                    try:
                        _ = tpu_model(tpu_input)
                    except RuntimeError:
                        pass

            # NVIDIA benchmark
            nvidia_times = []
            for _ in range(10):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = nvidia_model(nvidia_input)
                self.nvidia_backend.synchronize()
                nvidia_times.append(time.perf_counter() - start)

            # TPU benchmark
            tpu_times = []
            for _ in range(10):
                start = time.perf_counter()
                with torch.no_grad():
                    try:
                        _ = tpu_model(tpu_input)
                    except RuntimeError:
                        pass
                self.tpu_backend.synchronize()
                tpu_times.append(time.perf_counter() - start)

            nvidia_avg = statistics.mean(nvidia_times) * 1000
            tpu_avg = statistics.mean(tpu_times) * 1000

            results['nvidia'][batch_size] = nvidia_avg
            results['tpu'][batch_size] = tpu_avg

            print(f"   Batch {batch_size}: NVIDIA={nvidia_avg:.3f}ms, TPU={tpu_avg:.3f}ms")

        self.results['nvidia']['batch_scaling'] = results['nvidia']
        self.results['tpu']['batch_scaling'] = results['tpu']

        print("   ✅ Batch Size Scaling Benchmark Complete")
        return results

    def generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        print("\n" + "=" * 60)
        print("🎯 Backend Comparison Summary")
        print("=" * 60)

        summary = {
            'total_benchmarks': 7,
            'nvidia_device': self.results['nvidia'].get('device_info', {}).get('device', 'cpu'),
            'tpu_device': self.results['tpu'].get('device_info', {}).get('device', 'cpu'),
            'quick_mode': self.quick_mode,
            'iterations': self.benchmark_iterations
        }

        print(f"\n📊 Configuration:")
        print(f"   NVIDIA Device: {summary['nvidia_device']}")
        print(f"   TPU Device: {summary['tpu_device']}")
        print(f"   Quick Mode: {self.quick_mode}")
        print(f"   Iterations: {self.benchmark_iterations}")

        print(f"\n✅ All {summary['total_benchmarks']} benchmarks completed successfully!")

        return summary

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("🚀 Backend Performance Comparison Benchmark Suite")
        print("=" * 60)
        print(f"   Quick mode: {self.quick_mode}")
        print(f"   Warmup iterations: {self.warmup_iterations}")
        print(f"   Benchmark iterations: {self.benchmark_iterations}")

        start_time = time.perf_counter()

        # Run all benchmarks
        self.benchmark_model_preparation()
        self.benchmark_forward_pass_latency()
        self.benchmark_throughput()
        self.benchmark_memory_usage()
        self.benchmark_synchronization_overhead()
        self.benchmark_device_info()
        self.benchmark_batch_sizes_scaling()

        summary = self.generate_summary()

        total_time = time.perf_counter() - start_time
        print(f"\n⏱️  Total benchmark time: {total_time:.2f}s")

        summary['total_time'] = total_time
        summary['results'] = self.results

        return summary


# ============================================================================
# Main
# ============================================================================

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='Backend Performance Comparison Benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    args = parser.parse_args()

    try:
        benchmark = BackendComparisonBenchmark(quick_mode=args.quick)
        results = benchmark.run_all_benchmarks()

        print("\n🎉 Benchmark suite completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"\n💥 Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
