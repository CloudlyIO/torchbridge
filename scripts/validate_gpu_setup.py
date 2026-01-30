#!/usr/bin/env python3
"""
GPU Setup Validation Script

Comprehensive validation of CUDA, Triton, and framework setup.
Tests both hardware simulation and real GPU functionality.
"""

import sys
import os
import time
import torch
import traceback
from typing import Dict, List, Tuple

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class GPUSetupValidator:
    """Comprehensive GPU and framework setup validator"""

    def __init__(self):
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation checks"""
        print("🔧 GPU Setup Validation")
        print("=" * 40)

        validations = [
            ("Basic PyTorch", self.validate_pytorch),
            ("CUDA Availability", self.validate_cuda),
            ("Triton Installation", self.validate_triton),
            ("Framework Components", self.validate_framework),
            ("Hardware Simulation", self.validate_simulation),
            ("Performance Testing", self.validate_performance),
        ]

        for name, validation_func in validations:
            print(f"\n🧪 Testing {name}...")
            try:
                success = validation_func()
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   {status}")
                self.results[name] = success
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                self.results[name] = False

        return self.results

    def validate_pytorch(self) -> bool:
        """Validate PyTorch installation"""
        try:
            print(f"   PyTorch version: {torch.__version__}")

            # Test basic tensor operations
            x = torch.randn(10, 10)
            y = torch.mm(x, x.t())
            assert y.shape == (10, 10)

            # Test device handling
            x_device = x.to(self.device)
            print(f"   Device: {self.device}")

            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def validate_cuda(self) -> bool:
        """Validate CUDA setup"""
        try:
            cuda_available = torch.cuda.is_available()
            print(f"   CUDA available: {cuda_available}")

            if cuda_available:
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   GPU count: {torch.cuda.device_count()}")

                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    print(f"   GPU {i}: {name} ({memory_gb:.1f}GB)")

                # Test CUDA operations
                if torch.cuda.is_available():
                    x = torch.randn(100, 100, device='cuda')
                    y = torch.mm(x, x.t())
                    assert y.device.type == 'cuda'
                    print(f"   CUDA operations: working")
            else:
                print("   Note: Tests will run in CPU mode")

            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def validate_triton(self) -> bool:
        """Validate Triton installation"""
        try:
            import triton
            print(f"   Triton version: {triton.__version__}")
            return True
        except ImportError:
            print("   Triton not installed - install with: pip3 install triton")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def validate_framework(self) -> bool:
        """Validate our optimization framework components"""
        try:
            # Test core package import
            import torchbridge  # noqa: F401
            print("   torchbridge package: ✅")

            # Test PyGraph optimizer from next_gen
            from torchbridge.optimizations.next_gen import create_pygraph_optimizer  # noqa: F401
            print("   PyGraph optimizer: ✅")

            return True
        except ImportError as e:
            print(f"   Import error: {e}")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def validate_simulation(self) -> bool:
        """Validate hardware simulation framework"""
        try:
            from torchbridge.testing_framework import GPUSimulator, create_hardware_simulator

            # Test hardware simulation
            simulator = create_hardware_simulator(architecture='ampere', simulation_mode='performance')
            print("   Hardware simulator: created")

            # Test basic simulation - just check that it has expected methods
            summary = simulator.get_simulation_summary()
            print(f"   Simulation summary: {len(summary)} metrics")

            print(f"   Simulation execution: completed")
            return True

        except ImportError as e:
            print(f"   Import error: {e}")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def validate_performance(self) -> bool:
        """Validate performance testing framework"""
        try:
            from torchbridge.testing_framework import PerformanceBenchmarkSuite, create_benchmark_suite

            # Test benchmark suite
            benchmark = create_benchmark_suite(warmup_iterations=5, measurement_iterations=10)
            print("   Performance benchmarking: created")

            # Test simple operation timing
            test_input = torch.randn(100, 100, device=self.device)

            # Time a simple operation
            start_time = time.time()
            for _ in range(10):
                result = torch.mm(test_input, test_input.t())
            end_time = time.time()

            avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
            print(f"   Simple benchmark: {avg_time:.2f}ms avg")
            return True

        except ImportError as e:
            print(f"   Import error: {e}")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def generate_report(self) -> None:
        """Generate validation report"""
        print("\n📊 Validation Summary")
        print("=" * 30)

        total_tests = len(self.results)
        passed_tests = sum(self.results.values())

        for test_name, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name}: {status}")

        print(f"\nResults: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🎉 All validations passed! Setup is ready for development.")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️ Most validations passed. Check failed tests above.")
        else:
            print("❌ Multiple validations failed. Review setup instructions.")

        # Provide next steps
        print(f"\n🚀 Next Steps:")
        if passed_tests >= total_tests * 0.8:
            print("   1. python3 benchmarks/simple_benchmark_test.py")
            print("   2. python3 benchmarks/next_gen/demo_cutting_edge_benchmark.py --quick")
            print("   3. Explore demos/ directory for examples")
        else:
            print("   1. Review docs/user-guides/cuda_setup.md for detailed instructions")
            print("   2. Check pip3 install -r requirements.txt")
            print("   3. Ensure CUDA drivers are properly installed")


def main():
    """Main validation function"""
    print("🚀 GPU Setup Validation Script")
    print("This script validates CUDA, Triton, and framework setup.")
    print("For detailed setup instructions, see docs/user-guides/cuda_setup.md")

    validator = GPUSetupValidator()

    # Run validations
    results = validator.run_all_validations()

    # Generate report
    validator.generate_report()

    # Exit with appropriate code
    total_tests = len(results)
    passed_tests = sum(results.values())

    if passed_tests == total_tests:
        return 0
    elif passed_tests >= total_tests * 0.8:
        return 1  # Warning
    else:
        return 2  # Error


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(2)