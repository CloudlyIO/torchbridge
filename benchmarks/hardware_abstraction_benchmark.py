#!/usr/bin/env python3
"""
Hardware Abstraction Layer (HAL) Performance Benchmark

Comprehensive benchmarking suite for the Hardware Abstraction Layer:
- HAL overhead measurement
- Cross-vendor mesh creation performance
- Device discovery latency
- Memory optimization effectiveness
- Communication backend selection performance

This benchmark validates that HAL provides value without significant overhead.
"""

import sys
import time
import statistics
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import torch.nn as nn
    import numpy as np

    # Hardware abstraction imports
    from kernel_pytorch.hardware_abstraction.hal_core import (
        HardwareAbstractionLayer, DeviceSpec, HardwareCapabilities,
        HardwareVendor, ComputeCapability
    )
    from kernel_pytorch.hardware_abstraction.vendor_adapters import (
        NVIDIAAdapter, IntelAdapter, CPUAdapter
    )
    from kernel_pytorch.distributed_scale.hardware_adapter import HardwareAdapter

    # Import optimization components
    from kernel_pytorch.components import AttentionLayer
    from kernel_pytorch.compiler_optimized import FusedGELU

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the kernel_pytorch package is available.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HALBenchmarkSuite:
    """Comprehensive HAL benchmarking suite"""

    def __init__(self, enable_hal: bool = True, warmup_runs: int = 5, benchmark_runs: int = 50):
        self.enable_hal = enable_hal
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}

        # Initialize components
        self.hal = HardwareAbstractionLayer() if enable_hal else None
        self.hardware_adapter = HardwareAdapter(enable_hal=enable_hal, enable_monitoring=False)

    def print_header(self, title: str):
        """Print benchmark section header"""
        print(f"\n{'=' * 80}")
        print(f" {title}")
        print(f"{'=' * 80}")

    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function execution time with high precision"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def benchmark_hal_overhead(self) -> Dict[str, Any]:
        """Benchmark HAL overhead vs direct operations"""
        self.print_header("🔍 HAL Overhead Analysis")

        results = {
            'hal_device_lookup': [],
            'direct_device_lookup': [],
            'hal_capability_query': [],
            'direct_capability_query': [],
            'overhead_percentage': {}
        }

        print("📊 Measuring HAL operational overhead...")

        if not self.hal:
            print("⚠️  HAL not available, skipping overhead tests")
            return results

        # Register some test devices for benchmarking
        test_device = DeviceSpec(
            device_id=999,
            vendor=HardwareVendor.NVIDIA,
            capabilities=HardwareCapabilities(
                vendor=HardwareVendor.NVIDIA,
                device_name="Test Device",
                compute_capability="7.0",
                memory_gb=16.0,
                peak_flops_fp32=15e12,
                peak_flops_fp16=30e12,
                memory_bandwidth_gbps=900.0,
                supported_precisions=[ComputeCapability.FP32, ComputeCapability.FP16],
                tensor_core_support=True,
                interconnect_type="NVLink"
            )
        )
        self.hal.register_device(test_device)

        # Warmup
        for _ in range(self.warmup_runs):
            _ = test_device.device_id in self.hal.devices
            _ = test_device.capabilities.memory_gb

        # Benchmark HAL device lookup
        print("  Testing device lookup...")
        for _ in range(self.benchmark_runs):
            _, hal_time = self.measure_execution_time(
                lambda: test_device.device_id in self.hal.devices
            )
            results['hal_device_lookup'].append(hal_time)

        # Benchmark direct device lookup
        direct_devices = {999: test_device}
        for _ in range(self.benchmark_runs):
            _, direct_time = self.measure_execution_time(
                lambda: 999 in direct_devices
            )
            results['direct_device_lookup'].append(direct_time)

        # Benchmark HAL capability query
        print("  Testing capability queries...")
        for _ in range(self.benchmark_runs):
            _, hal_time = self.measure_execution_time(
                lambda: self.hal.devices[999].capabilities.memory_gb
            )
            results['hal_capability_query'].append(hal_time)

        # Benchmark direct capability query
        for _ in range(self.benchmark_runs):
            _, direct_time = self.measure_execution_time(
                lambda: test_device.capabilities.memory_gb
            )
            results['direct_capability_query'].append(direct_time)

        # Calculate overhead
        hal_lookup_avg = statistics.mean(results['hal_device_lookup']) * 1e6  # Convert to microseconds
        direct_lookup_avg = statistics.mean(results['direct_device_lookup']) * 1e6
        lookup_overhead = ((hal_lookup_avg - direct_lookup_avg) / direct_lookup_avg) * 100

        hal_capability_avg = statistics.mean(results['hal_capability_query']) * 1e6
        direct_capability_avg = statistics.mean(results['direct_capability_query']) * 1e6
        capability_overhead = ((hal_capability_avg - direct_capability_avg) / direct_capability_avg) * 100

        results['overhead_percentage'] = {
            'device_lookup': lookup_overhead,
            'capability_query': capability_overhead
        }

        print(f"\n📈 Overhead Analysis Results:")
        print(f"   Device Lookup:")
        print(f"     HAL: {hal_lookup_avg:.2f} μs")
        print(f"     Direct: {direct_lookup_avg:.2f} μs")
        print(f"     Overhead: {lookup_overhead:.1f}%")
        print(f"   Capability Query:")
        print(f"     HAL: {hal_capability_avg:.2f} μs")
        print(f"     Direct: {direct_capability_avg:.2f} μs")
        print(f"     Overhead: {capability_overhead:.1f}%")

        return results

    def benchmark_device_discovery(self) -> Dict[str, Any]:
        """Benchmark hardware discovery performance"""
        self.print_header("🔍 Hardware Discovery Performance")

        results = {
            'hal_discovery_times': [],
            'fallback_discovery_times': [],
            'device_counts': {},
            'discovery_methods': {}
        }

        print("📊 Measuring hardware discovery performance...")

        # Benchmark HAL discovery
        if self.hal:
            print("  Testing HAL-based discovery...")
            for run in range(min(10, self.benchmark_runs)):  # Fewer runs for discovery
                _, discovery_time = self.measure_execution_time(
                    self.hal.discover_all_hardware
                )
                results['hal_discovery_times'].append(discovery_time)

                if run == 0:  # Count devices on first run
                    inventory = self.hal.discover_all_hardware()
                    for vendor, devices in inventory.items():
                        results['device_counts'][vendor.value] = len(devices)

        # Benchmark fallback discovery
        print("  Testing fallback discovery...")
        for run in range(min(10, self.benchmark_runs)):
            _, discovery_time = self.measure_execution_time(
                self._fallback_device_discovery
            )
            results['fallback_discovery_times'].append(discovery_time)

        # Calculate statistics
        if results['hal_discovery_times']:
            hal_avg = statistics.mean(results['hal_discovery_times']) * 1000  # Convert to ms
            hal_std = statistics.stdev(results['hal_discovery_times']) * 1000 if len(results['hal_discovery_times']) > 1 else 0
            results['discovery_methods']['hal'] = {'avg_ms': hal_avg, 'std_ms': hal_std}

        if results['fallback_discovery_times']:
            fallback_avg = statistics.mean(results['fallback_discovery_times']) * 1000
            fallback_std = statistics.stdev(results['fallback_discovery_times']) * 1000 if len(results['fallback_discovery_times']) > 1 else 0
            results['discovery_methods']['fallback'] = {'avg_ms': fallback_avg, 'std_ms': fallback_std}

        print(f"\n📈 Discovery Performance Results:")
        if 'hal' in results['discovery_methods']:
            hal_metrics = results['discovery_methods']['hal']
            print(f"   HAL Discovery: {hal_metrics['avg_ms']:.1f} ± {hal_metrics['std_ms']:.1f} ms")

        if 'fallback' in results['discovery_methods']:
            fallback_metrics = results['discovery_methods']['fallback']
            print(f"   Fallback Discovery: {fallback_metrics['avg_ms']:.1f} ± {fallback_metrics['std_ms']:.1f} ms")

        print(f"   Devices Discovered: {sum(results['device_counts'].values())}")
        for vendor, count in results['device_counts'].items():
            print(f"     {vendor}: {count}")

        return results

    def _fallback_device_discovery(self) -> Dict[str, List]:
        """Fallback device discovery for comparison"""
        devices = {}

        # CUDA discovery
        if torch.cuda.is_available():
            nvidia_devices = []
            for i in range(torch.cuda.device_count()):
                nvidia_devices.append(f"cuda:{i}")
            devices['nvidia'] = nvidia_devices

        # CPU discovery
        devices['cpu'] = ['cpu']

        return devices

    def benchmark_mesh_creation(self) -> Dict[str, Any]:
        """Benchmark device mesh creation performance"""
        self.print_header("🕸️  Device Mesh Creation Performance")

        results = {
            'mesh_creation_times': [],
            'mesh_sizes': [],
            'topology_performance': {}
        }

        if not self.hal:
            print("⚠️  HAL not available, skipping mesh creation tests")
            return results

        print("📊 Measuring mesh creation performance...")

        # Create test devices for mesh benchmarking
        test_devices = []
        for i in range(8):  # Create 8 test devices
            vendor = HardwareVendor.NVIDIA if i % 2 == 0 else HardwareVendor.INTEL
            device = DeviceSpec(
                device_id=i + 1000,  # Use high IDs to avoid conflicts
                vendor=vendor,
                capabilities=HardwareCapabilities(
                    vendor=vendor,
                    device_name=f"Test Device {i}",
                    compute_capability="1.0",
                    memory_gb=16.0,
                    peak_flops_fp32=10e12,
                    peak_flops_fp16=20e12,
                    memory_bandwidth_gbps=500.0,
                    supported_precisions=[ComputeCapability.FP32, ComputeCapability.FP16],
                    tensor_core_support=False,
                    interconnect_type="PCIe"
                )
            )
            test_devices.append(device)
            self.hal.register_device(device)

        # Test different mesh sizes
        mesh_sizes = [2, 4, 6, 8]
        topologies = ['ring', 'tree', 'mesh']

        for topology in topologies:
            topology_times = []

            for mesh_size in mesh_sizes:
                devices_for_mesh = test_devices[:mesh_size]

                # Warmup
                for _ in range(self.warmup_runs):
                    try:
                        mesh = self.hal.create_cross_vendor_mesh(
                            devices=devices_for_mesh,
                            mesh_id=f"warmup_{mesh_size}",
                            topology=topology
                        )
                    except Exception:
                        pass

                # Benchmark
                mesh_times = []
                for run in range(min(10, self.benchmark_runs)):
                    try:
                        _, creation_time = self.measure_execution_time(
                            self.hal.create_cross_vendor_mesh,
                            devices=devices_for_mesh,
                            mesh_id=f"benchmark_{topology}_{mesh_size}_{run}",
                            topology=topology
                        )
                        mesh_times.append(creation_time)
                        results['mesh_creation_times'].append(creation_time)
                        results['mesh_sizes'].append(mesh_size)
                    except Exception as e:
                        logger.warning(f"Mesh creation failed for {topology}/{mesh_size}: {e}")

                if mesh_times:
                    avg_time = statistics.mean(mesh_times) * 1000  # Convert to ms
                    topology_times.append((mesh_size, avg_time))

            results['topology_performance'][topology] = topology_times

        # Print results
        print(f"\n📈 Mesh Creation Performance Results:")
        for topology, times in results['topology_performance'].items():
            print(f"   {topology.capitalize()} Topology:")
            for mesh_size, avg_time in times:
                print(f"     {mesh_size} devices: {avg_time:.1f} ms")

        if results['mesh_creation_times']:
            overall_avg = statistics.mean(results['mesh_creation_times']) * 1000
            print(f"\n   Overall Average: {overall_avg:.1f} ms")

        return results

    def benchmark_memory_optimization(self) -> Dict[str, Any]:
        """Benchmark memory optimization effectiveness"""
        self.print_header("💾 Memory Optimization Performance")

        results = {
            'optimization_times': [],
            'memory_reduction': [],
            'tensor_types': []
        }

        print("📊 Measuring memory optimization performance...")

        # Create test tensors of different types
        test_tensors = [
            ('fp32_large', torch.randn(1000, 1000, dtype=torch.float32)),
            ('fp32_medium', torch.randn(500, 500, dtype=torch.float32)),
            ('fp16_large', torch.randn(1000, 1000, dtype=torch.float16)),
            ('int32_medium', torch.randint(0, 100, (500, 500), dtype=torch.int32)),
        ]

        # Test device specs for optimization
        test_device_nvidia = DeviceSpec(
            device_id=2000,
            vendor=HardwareVendor.NVIDIA,
            capabilities=HardwareCapabilities(
                vendor=HardwareVendor.NVIDIA,
                device_name="Test NVIDIA Device",
                compute_capability="7.0",
                memory_gb=24.0,
                peak_flops_fp32=15e12,
                peak_flops_fp16=30e12,
                memory_bandwidth_gbps=900.0,
                supported_precisions=[
                    ComputeCapability.FP32, ComputeCapability.FP16,
                    ComputeCapability.MIXED_PRECISION, ComputeCapability.TENSOR_CORES
                ],
                tensor_core_support=True,
                interconnect_type="NVLink"
            )
        )

        # Test with NVIDIA adapter
        nvidia_adapter = NVIDIAAdapter()

        for tensor_name, tensor in test_tensors:
            original_memory = tensor.element_size() * tensor.nelement()

            # Warmup
            for _ in range(self.warmup_runs):
                try:
                    _ = nvidia_adapter.optimize_memory_layout(tensor, test_device_nvidia)
                except Exception:
                    pass

            # Benchmark
            optimization_times = []
            for _ in range(self.benchmark_runs):
                try:
                    optimized_tensor, opt_time = self.measure_execution_time(
                        nvidia_adapter.optimize_memory_layout,
                        tensor.clone(),  # Use clone to avoid modifying original
                        test_device_nvidia
                    )

                    optimization_times.append(opt_time)

                    # Calculate memory usage (approximation)
                    optimized_memory = optimized_tensor.element_size() * optimized_tensor.nelement()
                    memory_reduction = (original_memory - optimized_memory) / original_memory * 100

                    results['optimization_times'].append(opt_time)
                    results['memory_reduction'].append(memory_reduction)
                    results['tensor_types'].append(tensor_name)

                except Exception as e:
                    logger.warning(f"Memory optimization failed for {tensor_name}: {e}")

            if optimization_times:
                avg_time = statistics.mean(optimization_times) * 1e6  # Convert to microseconds
                print(f"   {tensor_name}: {avg_time:.1f} μs")

        # Calculate overall statistics
        if results['optimization_times']:
            overall_avg_time = statistics.mean(results['optimization_times']) * 1e6
            overall_avg_reduction = statistics.mean(results['memory_reduction'])

            print(f"\n📈 Memory Optimization Results:")
            print(f"   Average optimization time: {overall_avg_time:.1f} μs")
            print(f"   Average memory impact: {overall_avg_reduction:.1f}%")

        return results

    def benchmark_communication_backend_selection(self) -> Dict[str, Any]:
        """Benchmark communication backend selection"""
        self.print_header("📡 Communication Backend Selection Performance")

        results = {
            'selection_times': [],
            'backend_types': [],
            'vendor_combinations': []
        }

        if not self.hal:
            print("⚠️  HAL not available, skipping backend selection tests")
            return results

        print("📊 Measuring communication backend selection...")

        # Test different vendor combinations
        vendor_combinations = [
            ([HardwareVendor.NVIDIA], "all_nvidia"),
            ([HardwareVendor.INTEL], "all_intel"),
            ([HardwareVendor.NVIDIA, HardwareVendor.INTEL], "nvidia_intel_mix"),
            ([HardwareVendor.NVIDIA, HardwareVendor.AMD], "nvidia_amd_mix"),
            ([HardwareVendor.NVIDIA, HardwareVendor.INTEL, HardwareVendor.AMD], "multi_vendor"),
        ]

        for vendors, combo_name in vendor_combinations:
            # Create mock device groups
            vendor_groups = {}
            for i, vendor in enumerate(vendors):
                device = DeviceSpec(
                    device_id=3000 + i,
                    vendor=vendor,
                    capabilities=HardwareCapabilities(
                        vendor=vendor,
                        device_name=f"Test {vendor.value} Device",
                        compute_capability="1.0",
                        memory_gb=16.0,
                        peak_flops_fp32=10e12,
                        peak_flops_fp16=20e12,
                        memory_bandwidth_gbps=500.0,
                        supported_precisions=[ComputeCapability.FP32],
                        tensor_core_support=False,
                        interconnect_type="PCIe"
                    )
                )
                if vendor not in vendor_groups:
                    vendor_groups[vendor] = []
                vendor_groups[vendor].append(device)

            # Benchmark backend selection
            selection_times = []
            for _ in range(self.benchmark_runs):
                try:
                    _, selection_time = self.measure_execution_time(
                        self.hal._select_communication_backend,
                        vendor_groups
                    )
                    selection_times.append(selection_time)
                except Exception as e:
                    logger.warning(f"Backend selection failed for {combo_name}: {e}")

            if selection_times:
                avg_time = statistics.mean(selection_times) * 1e6  # Convert to microseconds
                results['selection_times'].extend(selection_times)
                results['vendor_combinations'].append(combo_name)

                print(f"   {combo_name}: {avg_time:.1f} μs")

        # Overall statistics
        if results['selection_times']:
            overall_avg = statistics.mean(results['selection_times']) * 1e6
            print(f"\n📈 Backend Selection Results:")
            print(f"   Average selection time: {overall_avg:.1f} μs")

        return results

    def benchmark_end_to_end_workflow(self) -> Dict[str, Any]:
        """Benchmark complete end-to-end HAL workflow"""
        self.print_header("🔄 End-to-End Workflow Performance")

        results = {
            'total_workflow_time': 0.0,
            'component_times': {},
            'success': False
        }

        if not self.hal:
            print("⚠️  HAL not available, skipping end-to-end tests")
            return results

        print("📊 Measuring complete HAL workflow...")

        try:
            workflow_start = time.perf_counter()

            # Step 1: Hardware Discovery
            discovery_start = time.perf_counter()
            inventory = self.hal.discover_all_hardware()
            discovery_time = time.perf_counter() - discovery_start
            results['component_times']['hardware_discovery'] = discovery_time

            # Step 2: Capability Analysis
            capability_start = time.perf_counter()
            capabilities = self.hal.get_cross_vendor_capabilities()
            capability_time = time.perf_counter() - capability_start
            results['component_times']['capability_analysis'] = capability_time

            # Step 3: Device Selection
            selection_start = time.perf_counter()
            optimal_device = self.hal.get_optimal_device(
                memory_requirement_gb=8.0,
                compute_requirement_tflops=10.0
            )
            selection_time = time.perf_counter() - selection_start
            results['component_times']['device_selection'] = selection_time

            # Step 4: Mesh Creation (if enough devices)
            if capabilities.total_devices >= 2:
                mesh_start = time.perf_counter()
                available_devices = [d for d in self.hal.devices.values() if d.is_available][:2]
                if len(available_devices) >= 2:
                    mesh = self.hal.create_cross_vendor_mesh(
                        devices=available_devices,
                        mesh_id="workflow_test",
                        topology="ring"
                    )
                mesh_time = time.perf_counter() - mesh_start
                results['component_times']['mesh_creation'] = mesh_time

            workflow_total = time.perf_counter() - workflow_start
            results['total_workflow_time'] = workflow_total
            results['success'] = True

            print(f"\n📈 End-to-End Workflow Results:")
            print(f"   Total workflow time: {workflow_total * 1000:.1f} ms")
            print(f"   Component breakdown:")
            for component, comp_time in results['component_times'].items():
                percentage = (comp_time / workflow_total) * 100
                print(f"     {component}: {comp_time * 1000:.1f} ms ({percentage:.1f}%)")

        except Exception as e:
            logger.error(f"End-to-end workflow failed: {e}")
            results['error'] = str(e)
            print(f"   ❌ Workflow failed: {e}")

        return results

    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        self.print_header("📋 HAL Benchmark Report")

        # Run all benchmarks
        print("🚀 Running comprehensive HAL benchmark suite...")

        benchmark_start = time.perf_counter()

        self.results['hal_overhead'] = self.benchmark_hal_overhead()
        self.results['device_discovery'] = self.benchmark_device_discovery()
        self.results['mesh_creation'] = self.benchmark_mesh_creation()
        self.results['memory_optimization'] = self.benchmark_memory_optimization()
        self.results['backend_selection'] = self.benchmark_communication_backend_selection()
        self.results['end_to_end'] = self.benchmark_end_to_end_workflow()

        total_benchmark_time = time.perf_counter() - benchmark_start

        # Generate summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_benchmark_time': total_benchmark_time,
            'hal_enabled': self.enable_hal,
            'warmup_runs': self.warmup_runs,
            'benchmark_runs': self.benchmark_runs,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'results': self.results,
            'performance_summary': self._generate_performance_summary()
        }

        print(f"\n📊 Benchmark Summary:")
        print(f"   Total benchmark time: {total_benchmark_time:.1f} seconds")
        print(f"   HAL overhead: {summary['performance_summary'].get('hal_overhead_avg', 'N/A')}")
        print(f"   Discovery performance: {summary['performance_summary'].get('discovery_performance', 'N/A')}")
        print(f"   End-to-end success: {summary['performance_summary'].get('end_to_end_success', False)}")

        return summary

    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from benchmark results"""
        summary = {}

        # HAL overhead summary
        if 'hal_overhead' in self.results and self.results['hal_overhead']['overhead_percentage']:
            overheads = self.results['hal_overhead']['overhead_percentage']
            avg_overhead = statistics.mean(overheads.values())
            summary['hal_overhead_avg'] = f"{avg_overhead:.1f}%"

        # Discovery performance
        if 'device_discovery' in self.results:
            discovery = self.results['device_discovery']
            if 'hal' in discovery['discovery_methods']:
                hal_time = discovery['discovery_methods']['hal']['avg_ms']
                summary['discovery_performance'] = f"{hal_time:.1f} ms"

        # End-to-end success
        if 'end_to_end' in self.results:
            summary['end_to_end_success'] = self.results['end_to_end'].get('success', False)
            if summary['end_to_end_success']:
                total_time = self.results['end_to_end']['total_workflow_time'] * 1000
                summary['end_to_end_time'] = f"{total_time:.1f} ms"

        return summary

    def cleanup(self):
        """Cleanup benchmark resources"""
        try:
            self.hardware_adapter.shutdown()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description='HAL Performance Benchmark Suite')
    parser.add_argument('--no-hal', action='store_true', help='Disable HAL for comparison')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark (fewer iterations)')
    parser.add_argument('--warmup-runs', type=int, default=5, help='Number of warmup runs')
    parser.add_argument('--benchmark-runs', type=int, default=50, help='Number of benchmark runs')
    parser.add_argument('--output', type=str, help='Output file for results')

    args = parser.parse_args()

    print("🏁 Hardware Abstraction Layer (HAL) Benchmark Suite")
    print("=" * 80)

    # Adjust runs for quick mode
    warmup_runs = 2 if args.quick else args.warmup_runs
    benchmark_runs = 10 if args.quick else args.benchmark_runs

    # Initialize benchmark suite
    benchmark = HALBenchmarkSuite(
        enable_hal=not args.no_hal,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs
    )

    try:
        # Run benchmarks
        report = benchmark.generate_benchmark_report()

        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")

        print("\n🎉 HAL Benchmark Complete!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\n❌ Benchmark failed: {e}")
        return 1

    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    exit(main())