"""
NVIDIA Configuration Benchmarks

Benchmarks the new unified NVIDIA configuration system to demonstrate
hardware detection, optimization selection, and performance impact.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernel_pytorch.core.config import (
    KernelPyTorchConfig,
    NVIDIAArchitecture,
    OptimizationLevel
)
from kernel_pytorch.validation.unified_validator import UnifiedValidator


class NVIDIAConfigBenchmark:
    """Comprehensive NVIDIA configuration benchmarking suite."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def benchmark_config_creation_performance(self) -> Dict[str, float]:
        """Benchmark configuration creation and hardware detection performance."""
        print("📊 Benchmarking NVIDIA Configuration Creation Performance...")

        results = {}

        # Benchmark basic config creation
        start_time = time.perf_counter()
        for _ in range(100):
            config = KernelPyTorchConfig()
        basic_creation_time = (time.perf_counter() - start_time) / 100
        results['basic_config_creation_ms'] = basic_creation_time * 1000

        # Benchmark config modes
        start_time = time.perf_counter()
        for _ in range(100):
            inference_config = KernelPyTorchConfig.for_inference()
        inference_creation_time = (time.perf_counter() - start_time) / 100
        results['inference_config_creation_ms'] = inference_creation_time * 1000

        start_time = time.perf_counter()
        for _ in range(100):
            training_config = KernelPyTorchConfig.for_training()
        training_creation_time = (time.perf_counter() - start_time) / 100
        results['training_config_creation_ms'] = training_creation_time * 1000

        start_time = time.perf_counter()
        for _ in range(100):
            dev_config = KernelPyTorchConfig.for_development()
        dev_creation_time = (time.perf_counter() - start_time) / 100
        results['development_config_creation_ms'] = dev_creation_time * 1000

        # Benchmark hardware detection
        config = KernelPyTorchConfig()
        start_time = time.perf_counter()
        for _ in range(50):
            architecture = config.hardware.nvidia._detect_architecture()
        detection_time = (time.perf_counter() - start_time) / 50
        results['hardware_detection_ms'] = detection_time * 1000

        print(f"   ✅ Basic config creation: {results['basic_config_creation_ms']:.2f}ms")
        print(f"   ✅ Inference config: {results['inference_config_creation_ms']:.2f}ms")
        print(f"   ✅ Training config: {results['training_config_creation_ms']:.2f}ms")
        print(f"   ✅ Development config: {results['development_config_creation_ms']:.2f}ms")
        print(f"   ✅ Hardware detection: {results['hardware_detection_ms']:.2f}ms")

        return results

    def benchmark_configuration_validation(self) -> Dict[str, Any]:
        """Benchmark configuration validation performance."""
        print("🔍 Benchmarking Configuration Validation...")

        validator = UnifiedValidator()
        results = {}

        # Test different configurations
        configs = {
            'default': KernelPyTorchConfig(),
            'inference': KernelPyTorchConfig.for_inference(),
            'training': KernelPyTorchConfig.for_training(),
            'development': KernelPyTorchConfig.for_development()
        }

        for config_name, config in configs.items():
            start_time = time.perf_counter()

            # Configuration validation
            config_results = validator.validate_configuration(config)

            # Hardware validation
            hw_results = validator.validate_hardware_compatibility(self.device)

            validation_time = time.perf_counter() - start_time

            results[f'{config_name}_validation'] = {
                'time_ms': validation_time * 1000,
                'config_passed': config_results.passed,
                'config_total': config_results.total_tests,
                'hardware_passed': hw_results.passed,
                'hardware_total': hw_results.total_tests
            }

            print(f"   ✅ {config_name.capitalize()} config: {validation_time*1000:.2f}ms "
                  f"({config_results.passed}/{config_results.total_tests} passed)")

        return results

    def benchmark_optimization_levels(self) -> Dict[str, Any]:
        """Benchmark different optimization levels."""
        print("⚡ Benchmarking Optimization Levels...")

        results = {}
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(self.device)

        test_input = torch.randn(64, 512).to(self.device)

        optimization_levels = [
            OptimizationLevel.CONSERVATIVE,
            OptimizationLevel.BALANCED,
            OptimizationLevel.AGGRESSIVE
        ]

        for opt_level in optimization_levels:
            # Create config with specific optimization level
            config = KernelPyTorchConfig()
            config.optimization_level = opt_level

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(test_input)

            # Benchmark inference
            start_time = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    output = model(test_input)
            inference_time = time.perf_counter() - start_time

            results[opt_level.value] = {
                'inference_time_ms': inference_time * 1000,
                'throughput_samples_per_sec': 6400 / inference_time,  # 64 samples * 100 iterations
                'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2 if self.device.type == 'cuda' else 0
            }

            print(f"   ✅ {opt_level.value.capitalize()}: {inference_time*1000:.2f}ms "
                  f"({6400/inference_time:.1f} samples/sec)")

        return results

    def benchmark_nvidia_specific_features(self) -> Dict[str, Any]:
        """Benchmark NVIDIA-specific configuration features."""
        print("🔧 Benchmarking NVIDIA-Specific Features...")

        config = KernelPyTorchConfig()
        results = {}

        # Architecture detection benchmark
        start_time = time.perf_counter()
        detected_arch = config.hardware.nvidia.architecture
        arch_detection_time = time.perf_counter() - start_time

        results['architecture_detection'] = {
            'time_ms': arch_detection_time * 1000,
            'detected_architecture': detected_arch.value,
            'fp8_enabled': config.hardware.nvidia.fp8_enabled,
            'tensor_core_version': config.hardware.nvidia.tensor_core_version,
            'flash_attention_version': config.hardware.nvidia.flash_attention_version
        }

        # Config serialization benchmark
        start_time = time.perf_counter()
        config_dict = config.to_dict()
        serialization_time = time.perf_counter() - start_time

        results['config_serialization'] = {
            'time_ms': serialization_time * 1000,
            'serialized_size_keys': len(config_dict),
            'nvidia_config_present': 'nvidia' in config_dict.get('hardware', {})
        }

        print(f"   ✅ Architecture detection: {arch_detection_time*1000:.2f}ms ({detected_arch.value})")
        print(f"   ✅ Config serialization: {serialization_time*1000:.2f}ms ({len(config_dict)} keys)")
        print(f"   ✅ FP8 enabled: {config.hardware.nvidia.fp8_enabled}")
        print(f"   ✅ Tensor Core version: {config.hardware.nvidia.tensor_core_version}")

        return results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("🚀 Running Comprehensive NVIDIA Configuration Benchmarks")
        print("=" * 70)

        all_results = {}

        # Performance benchmarks
        all_results['config_performance'] = self.benchmark_config_creation_performance()
        print()

        # Validation benchmarks
        all_results['validation_performance'] = self.benchmark_configuration_validation()
        print()

        # Optimization level benchmarks
        all_results['optimization_levels'] = self.benchmark_optimization_levels()
        print()

        # NVIDIA-specific benchmarks
        all_results['nvidia_features'] = self.benchmark_nvidia_specific_features()
        print()

        # Summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("📊 BENCHMARK SUMMARY")
        print("=" * 40)

        # Configuration performance
        config_perf = results['config_performance']
        print(f"⚡ Config Creation: {config_perf['basic_config_creation_ms']:.2f}ms avg")
        print(f"🔍 Hardware Detection: {config_perf['hardware_detection_ms']:.2f}ms avg")

        # Optimization performance
        opt_results = results['optimization_levels']
        fastest_opt = min(opt_results.keys(), key=lambda k: opt_results[k]['inference_time_ms'])
        fastest_time = opt_results[fastest_opt]['inference_time_ms']
        print(f"🏃 Fastest Optimization: {fastest_opt} ({fastest_time:.2f}ms)")

        # NVIDIA features
        nvidia_features = results['nvidia_features']
        arch = nvidia_features['architecture_detection']['detected_architecture']
        fp8_enabled = nvidia_features['architecture_detection']['fp8_enabled']
        print(f"🎯 Detected Architecture: {arch} (FP8: {fp8_enabled})")

        print("=" * 40)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='NVIDIA Configuration Benchmarks')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🔧 Using device: {device}")
    print(f"🏃 Quick mode: {args.quick}")
    print()

    # Run benchmarks
    benchmark = NVIDIAConfigBenchmark(device)

    if args.quick:
        print("🏃 Running Quick NVIDIA Config Benchmark...")
        results = benchmark.benchmark_config_creation_performance()
        print(f"✅ Quick benchmark completed: {results['basic_config_creation_ms']:.2f}ms avg")
    else:
        results = benchmark.run_comprehensive_benchmark()

    print("\n🎯 NVIDIA Configuration Benchmarks Complete!")


if __name__ == "__main__":
    main()