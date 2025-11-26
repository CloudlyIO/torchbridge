#!/usr/bin/env python3
"""
Multi-Vendor Hardware Abstraction Demo

Demonstrates the new Hardware Abstraction Layer (HAL) capabilities:
- Automatic detection of multiple hardware vendors
- Optimal device selection across vendors
- Cross-vendor workload placement
- Backward compatibility with existing code

This demo shows how the HAL enhances existing functionality while maintaining
full backward compatibility with legacy hardware discovery.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from typing import Dict, List, Any

# Import existing hardware components (backward compatibility)
from kernel_pytorch.distributed_scale.hardware_adapter import HardwareAdapter
from kernel_pytorch.distributed_scale.hardware_discovery import HardwareVendor

# Import new HAL components (graceful fallback if not available)
try:
    from kernel_pytorch.hardware_abstraction.hal_core import HardwareAbstractionLayer
    from kernel_pytorch.hardware_abstraction.vendor_adapters import get_available_vendors, auto_detect_best_adapter
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False
    print("⚠️  Hardware Abstraction Layer not available, running in legacy mode")


class MultiVendorDemo:
    """
    Demonstrates multi-vendor hardware capabilities

    Shows both legacy functionality and new HAL features,
    with automatic fallback for backward compatibility.
    """

    def __init__(self, enable_hal: bool = True):
        self.enable_hal = enable_hal and HAL_AVAILABLE

        # Initialize hardware adapter (enhanced with HAL if available)
        self.hardware_adapter = HardwareAdapter(enable_hal=self.enable_hal)

        print(f"🔧 Hardware Adapter initialized (HAL: {'✅ enabled' if self.hardware_adapter.is_hal_enabled() else '❌ disabled'})")

    def demonstrate_vendor_detection(self) -> Dict[str, Any]:
        """Demonstrate automatic vendor detection"""
        print("\n" + "="*60)
        print("🔍 VENDOR DETECTION DEMONSTRATION")
        print("="*60)

        results = {
            'legacy_detection': {},
            'hal_detection': {},
            'available_vendors': []
        }

        # Legacy vendor detection (always works)
        print("\n📊 Legacy Hardware Detection:")
        try:
            cluster_stats = self.hardware_adapter.get_cluster_statistics()
            if 'topology' in cluster_stats:
                topology = cluster_stats['topology']
                print(f"  • Total devices: {topology.get('total_devices', 0)}")
                print(f"  • Total memory: {topology.get('total_memory_gb', 0):.1f} GB")
                vendor_dist = topology.get('vendor_distribution', {})
                for vendor, count in vendor_dist.items():
                    print(f"  • {vendor}: {count} device(s)")

                results['legacy_detection'] = {
                    'total_devices': topology.get('total_devices', 0),
                    'vendor_distribution': vendor_dist,
                    'total_memory_gb': topology.get('total_memory_gb', 0)
                }
            else:
                print("  • No topology information available")

        except Exception as e:
            print(f"  ❌ Legacy detection error: {e}")

        # HAL vendor detection (enhanced features)
        if self.enable_hal and HAL_AVAILABLE:
            print("\n🚀 HAL Enhanced Detection:")
            try:
                # Get available vendors
                available_vendors = get_available_vendors()
                results['available_vendors'] = [v.value for v in available_vendors]

                print(f"  • Available vendors: {[v.value for v in available_vendors]}")

                # Get cross-vendor capabilities
                capabilities = self.hardware_adapter.get_cross_vendor_capabilities()
                if capabilities.get('hal_enabled', False):
                    print(f"  • Cross-vendor devices: {capabilities.get('cross_vendor_devices', 0)}")
                    print(f"  • Available memory: {capabilities.get('available_memory_gb', 0):.1f} GB")
                    print(f"  • Average utilization: {capabilities.get('average_utilization', 0):.1f}%")

                    # Show detailed device information
                    device_details = capabilities.get('device_details', [])
                    if device_details:
                        print("\n  📋 Device Details:")
                        for device in device_details[:5]:  # Show first 5 devices
                            vendor = device.get('vendor', 'unknown')
                            name = device.get('name', 'Unknown Device')
                            util = device.get('utilization', 0)
                            memory_used = device.get('memory_used_gb', 0)
                            memory_total = device.get('memory_total_gb', 0)
                            print(f"    • {vendor}: {name}")
                            print(f"      Utilization: {util:.1f}%, Memory: {memory_used:.1f}/{memory_total:.1f} GB")

                    results['hal_detection'] = capabilities

            except Exception as e:
                print(f"  ❌ HAL detection error: {e}")
        else:
            print("\n⚠️  HAL not available - using legacy detection only")

        return results

    def demonstrate_optimal_device_selection(self) -> Dict[str, Any]:
        """Demonstrate optimal device selection across vendors"""
        print("\n" + "="*60)
        print("🎯 OPTIMAL DEVICE SELECTION DEMONSTRATION")
        print("="*60)

        results = {
            'legacy_selection': None,
            'hal_selection': None,
            'comparison': {}
        }

        # Test requirements
        memory_req = 4.0  # GB
        compute_req = 5.0  # TFLOPS

        print(f"\n📋 Requirements: {memory_req} GB memory, {compute_req} TFLOPS compute")

        # Legacy device selection
        print("\n📊 Legacy Device Selection:")
        try:
            start_time = time.time()
            legacy_devices = self.hardware_adapter.get_optimal_device_placement(
                memory_requirement_gb=memory_req,
                compute_requirement_tflops=compute_req
            )
            legacy_time = time.time() - start_time

            if legacy_devices:
                print(f"  ✅ Found {len(legacy_devices)} suitable device(s)")
                print(f"  ⚡ Selection time: {legacy_time*1000:.1f} ms")
                print(f"  🎯 Best device ID: {legacy_devices[0] if legacy_devices else 'None'}")

                results['legacy_selection'] = {
                    'device_count': len(legacy_devices),
                    'best_device': legacy_devices[0] if legacy_devices else None,
                    'selection_time_ms': legacy_time * 1000
                }
            else:
                print("  ❌ No suitable devices found")

        except Exception as e:
            print(f"  ❌ Legacy selection error: {e}")

        # HAL device selection (enhanced)
        if self.enable_hal:
            print("\n🚀 HAL Enhanced Selection:")
            try:
                start_time = time.time()
                hal_device = self.hardware_adapter.get_optimal_device_hal(
                    memory_requirement_gb=memory_req,
                    compute_requirement_tflops=compute_req,
                    preferred_vendors=[HardwareVendor.NVIDIA, HardwareVendor.INTEL]
                )
                hal_time = time.time() - start_time

                if hal_device:
                    print(f"  ✅ Found optimal device")
                    print(f"  ⚡ Selection time: {hal_time*1000:.1f} ms")

                    # Show device details if available
                    if hasattr(hal_device, 'vendor'):
                        print(f"  🏷️  Vendor: {hal_device.vendor.value}")
                    if hasattr(hal_device, 'capabilities'):
                        caps = hal_device.capabilities
                        print(f"  💾 Memory: {caps.memory_gb:.1f} GB")
                        print(f"  ⚡ Peak FLOPS: {caps.peak_flops_fp32/1e12:.1f} TFLOPS")
                        print(f"  🔧 Device: {caps.device_name}")

                    results['hal_selection'] = {
                        'device_found': True,
                        'selection_time_ms': hal_time * 1000,
                        'vendor': hal_device.vendor.value if hasattr(hal_device, 'vendor') else 'unknown',
                        'memory_gb': hal_device.capabilities.memory_gb if hasattr(hal_device, 'capabilities') else 0,
                        'peak_tflops': hal_device.capabilities.peak_flops_fp32/1e12 if hasattr(hal_device, 'capabilities') else 0
                    }
                else:
                    print("  ❌ No optimal device found")
                    results['hal_selection'] = {'device_found': False}

            except Exception as e:
                print(f"  ❌ HAL selection error: {e}")
                results['hal_selection'] = {'error': str(e)}

        # Performance comparison
        if results['legacy_selection'] and results['hal_selection'] and results['hal_selection'].get('device_found'):
            legacy_time = results['legacy_selection']['selection_time_ms']
            hal_time = results['hal_selection']['selection_time_ms']

            print(f"\n📈 Performance Comparison:")
            print(f"  • Legacy method: {legacy_time:.1f} ms")
            print(f"  • HAL method: {hal_time:.1f} ms")
            if hal_time > 0:
                speedup = legacy_time / hal_time
                print(f"  • Speedup: {speedup:.2f}x {'⚡ faster' if speedup > 1 else '⚠️ slower'}")

            results['comparison'] = {
                'legacy_time_ms': legacy_time,
                'hal_time_ms': hal_time,
                'speedup': legacy_time / hal_time if hal_time > 0 else 0
            }

        return results

    def demonstrate_cross_vendor_workload(self) -> Dict[str, Any]:
        """Demonstrate cross-vendor workload placement"""
        print("\n" + "="*60)
        print("🔄 CROSS-VENDOR WORKLOAD DEMONSTRATION")
        print("="*60)

        results = {'workload_executed': False, 'performance_metrics': {}}

        # Create a simple test workload
        print("\n🧪 Creating Test Workload:")
        batch_size = 32
        hidden_size = 512

        # Simple neural network for testing
        model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

        # Test data
        test_input = torch.randn(batch_size, hidden_size)

        print(f"  • Model: 3-layer neural network")
        print(f"  • Input shape: {test_input.shape}")
        print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test on different devices/vendors
        devices_tested = []
        performance_metrics = {}

        # Test on CPU (always available)
        print("\n🖥️  Testing on CPU:")
        try:
            start_time = time.time()
            model_cpu = model.cpu()
            input_cpu = test_input.cpu()

            with torch.no_grad():
                output_cpu = model_cpu(input_cpu)

            cpu_time = time.time() - start_time
            print(f"  ✅ CPU execution successful")
            print(f"  ⚡ Time: {cpu_time*1000:.1f} ms")
            print(f"  📊 Output shape: {output_cpu.shape}")

            devices_tested.append('cpu')
            performance_metrics['cpu'] = {
                'execution_time_ms': cpu_time * 1000,
                'throughput_samples_per_sec': batch_size / cpu_time,
                'device_type': 'cpu'
            }

        except Exception as e:
            print(f"  ❌ CPU test failed: {e}")

        # Test on GPU (if available)
        if torch.cuda.is_available():
            print("\n🎮 Testing on NVIDIA GPU:")
            try:
                start_time = time.time()
                model_cuda = model.cuda()
                input_cuda = test_input.cuda()

                # Warm up
                with torch.no_grad():
                    _ = model_cuda(input_cuda)

                torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    output_cuda = model_cuda(input_cuda)

                torch.cuda.synchronize()
                gpu_time = time.time() - start_time

                print(f"  ✅ NVIDIA GPU execution successful")
                print(f"  ⚡ Time: {gpu_time*1000:.1f} ms")
                print(f"  📊 Output shape: {output_cuda.shape}")

                # Performance comparison
                if 'cpu' in performance_metrics:
                    speedup = performance_metrics['cpu']['execution_time_ms'] / (gpu_time * 1000)
                    print(f"  🚀 Speedup vs CPU: {speedup:.2f}x")

                devices_tested.append('cuda')
                performance_metrics['cuda'] = {
                    'execution_time_ms': gpu_time * 1000,
                    'throughput_samples_per_sec': batch_size / gpu_time,
                    'device_type': 'cuda',
                    'gpu_name': torch.cuda.get_device_name(0)
                }

            except Exception as e:
                print(f"  ❌ NVIDIA GPU test failed: {e}")

        # HAL-enhanced workload placement
        if self.enable_hal:
            print("\n🚀 HAL-Enhanced Workload Placement:")
            try:
                # Get optimal device using HAL
                optimal_device = self.hardware_adapter.get_optimal_device_hal(
                    memory_requirement_gb=1.0,  # Small workload
                    compute_requirement_tflops=1.0,
                    preferred_vendors=[HardwareVendor.NVIDIA, HardwareVendor.INTEL, HardwareVendor.UNKNOWN]
                )

                if optimal_device:
                    print(f"  ✅ HAL selected optimal device")
                    if hasattr(optimal_device, 'vendor'):
                        print(f"  🏷️  Optimal vendor: {optimal_device.vendor.value}")
                    if hasattr(optimal_device, 'capabilities'):
                        print(f"  💾 Device memory: {optimal_device.capabilities.memory_gb:.1f} GB")
                        print(f"  🔧 Device name: {optimal_device.capabilities.device_name}")
                else:
                    print("  ⚠️  HAL could not find optimal device")

            except Exception as e:
                print(f"  ❌ HAL workload placement error: {e}")

        # Summary
        print(f"\n📋 Workload Execution Summary:")
        print(f"  • Devices tested: {', '.join(devices_tested)}")
        print(f"  • Total execution scenarios: {len(performance_metrics)}")

        if len(performance_metrics) > 1:
            fastest_device = min(performance_metrics.items(), key=lambda x: x[1]['execution_time_ms'])
            print(f"  • Fastest device: {fastest_device[0]} ({fastest_device[1]['execution_time_ms']:.1f} ms)")

        results = {
            'workload_executed': len(devices_tested) > 0,
            'devices_tested': devices_tested,
            'performance_metrics': performance_metrics
        }

        return results

    def demonstrate_backward_compatibility(self) -> Dict[str, Any]:
        """Demonstrate that existing code still works unchanged"""
        print("\n" + "="*60)
        print("🔒 BACKWARD COMPATIBILITY DEMONSTRATION")
        print("="*60)

        results = {'legacy_methods_working': [], 'all_compatible': True}

        # Test existing methods still work
        legacy_tests = [
            ('get_cluster_statistics', lambda: self.hardware_adapter.get_cluster_statistics()),
            ('get_health_report', lambda: self.hardware_adapter.get_health_report(hours=1)),
            ('get_optimal_device_placement', lambda: self.hardware_adapter.get_optimal_device_placement(
                memory_requirement_gb=2.0, compute_requirement_tflops=1.0)),
        ]

        print("\n🧪 Testing Legacy Methods:")
        for method_name, method_call in legacy_tests:
            try:
                result = method_call()
                if result is not None:
                    print(f"  ✅ {method_name}: Working")
                    results['legacy_methods_working'].append(method_name)
                else:
                    print(f"  ⚠️  {method_name}: Returns None (may be normal)")
                    results['legacy_methods_working'].append(method_name)

            except Exception as e:
                print(f"  ❌ {method_name}: Failed ({e})")
                results['all_compatible'] = False

        # Test that HAL doesn't break existing functionality
        if self.enable_hal:
            print("\n🔄 Testing HAL Impact on Legacy Methods:")
            try:
                # Disable HAL temporarily
                hal_backup = self.hardware_adapter.hal
                self.hardware_adapter.hal_enabled = False
                self.hardware_adapter.hal = None

                # Test legacy methods work without HAL
                legacy_stats = self.hardware_adapter.get_cluster_statistics()

                # Restore HAL
                self.hardware_adapter.hal = hal_backup
                self.hardware_adapter.hal_enabled = True

                # Test legacy methods work with HAL
                hal_stats = self.hardware_adapter.get_cluster_statistics()

                print(f"  ✅ Legacy methods work with and without HAL")
                print(f"  🔍 Legacy keys: {list(legacy_stats.keys()) if legacy_stats else []}")
                print(f"  🔍 HAL keys: {list(hal_stats.keys()) if hal_stats else []}")

            except Exception as e:
                print(f"  ❌ HAL compatibility test failed: {e}")
                results['all_compatible'] = False

        compatibility_percentage = len(results['legacy_methods_working']) / len(legacy_tests) * 100
        print(f"\n📊 Compatibility Results:")
        print(f"  • Methods tested: {len(legacy_tests)}")
        print(f"  • Methods working: {len(results['legacy_methods_working'])}")
        print(f"  • Compatibility: {compatibility_percentage:.1f}%")
        print(f"  • Overall status: {'✅ Fully compatible' if results['all_compatible'] else '⚠️ Issues detected'}")

        return results

    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run all demonstration components"""
        print("🚀 MULTI-VENDOR HARDWARE ABSTRACTION DEMO")
        print("="*80)
        print("Demonstrating PyTorch optimization framework hardware abstraction")
        print(f"HAL Status: {'✅ Available' if HAL_AVAILABLE else '❌ Not Available'}")
        print(f"HAL Enabled: {'✅ Yes' if self.enable_hal else '❌ No'}")

        all_results = {}

        try:
            # Run demonstrations
            all_results['vendor_detection'] = self.demonstrate_vendor_detection()
            all_results['device_selection'] = self.demonstrate_optimal_device_selection()
            all_results['workload_placement'] = self.demonstrate_cross_vendor_workload()
            all_results['backward_compatibility'] = self.demonstrate_backward_compatibility()

            # Final summary
            print("\n" + "="*80)
            print("📊 DEMONSTRATION SUMMARY")
            print("="*80)

            # Vendor detection summary
            vendor_results = all_results['vendor_detection']
            print(f"\n🔍 Vendor Detection:")
            print(f"  • Legacy detection: {'✅' if vendor_results['legacy_detection'] else '❌'}")
            print(f"  • HAL detection: {'✅' if vendor_results['hal_detection'] else '❌'}")
            print(f"  • Available vendors: {len(vendor_results['available_vendors'])}")

            # Device selection summary
            selection_results = all_results['device_selection']
            print(f"\n🎯 Device Selection:")
            legacy_sel = selection_results['legacy_selection']
            hal_sel = selection_results['hal_selection']
            print(f"  • Legacy method: {'✅' if legacy_sel else '❌'}")
            print(f"  • HAL method: {'✅' if hal_sel and hal_sel.get('device_found') else '❌'}")

            if selection_results.get('comparison'):
                comp = selection_results['comparison']
                print(f"  • Performance improvement: {comp.get('speedup', 0):.2f}x")

            # Workload execution summary
            workload_results = all_results['workload_placement']
            print(f"\n🔄 Workload Execution:")
            print(f"  • Workload executed: {'✅' if workload_results['workload_executed'] else '❌'}")
            print(f"  • Devices tested: {len(workload_results['devices_tested'])}")
            print(f"  • Performance data: {len(workload_results['performance_metrics'])} device(s)")

            # Compatibility summary
            compat_results = all_results['backward_compatibility']
            print(f"\n🔒 Backward Compatibility:")
            print(f"  • Legacy methods working: {len(compat_results['legacy_methods_working'])}")
            print(f"  • Fully compatible: {'✅' if compat_results['all_compatible'] else '⚠️'}")

            print(f"\n✨ Demo completed successfully!")
            print(f"🎯 Result: Multi-vendor hardware abstraction is {'functional' if self.enable_hal else 'available in legacy mode'}")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            all_results['error'] = str(e)

        return all_results


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='Multi-Vendor Hardware Abstraction Demo')
    parser.add_argument('--disable-hal', action='store_true',
                       help='Disable HAL to test legacy mode')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version of demo')

    args = parser.parse_args()

    # Initialize demo
    demo = MultiVendorDemo(enable_hal=not args.disable_hal)

    if args.quick:
        # Quick demo - just vendor detection and device selection
        print("🏃‍♂️ Running Quick Demo")
        demo.demonstrate_vendor_detection()
        demo.demonstrate_optimal_device_selection()
    else:
        # Full comprehensive demo
        demo.run_comprehensive_demo()

    return 0


if __name__ == '__main__':
    exit(main())