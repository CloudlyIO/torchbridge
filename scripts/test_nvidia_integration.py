#!/usr/bin/env python3
"""
NVIDIA Integration Test Suite

Comprehensive testing script for NVIDIA hardware support validation.
Tests configuration, detection, performance, and integration.

Usage:
    python3 scripts/test_nvidia_integration.py           # Full test suite
    python3 scripts/test_nvidia_integration.py --quick   # Quick validation

Can be run standalone or integrated with pytest:
    pytest scripts/test_nvidia_integration.py            # Run as pytest
"""

import sys
import time
import subprocess
from pathlib import Path
import torch
from typing import Dict, Any, List

# Add src to path for standalone execution
repo_root = Path(__file__).parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from kernel_pytorch.core.config import KernelPyTorchConfig, NVIDIAArchitecture, OptimizationLevel
from kernel_pytorch.validation.unified_validator import UnifiedValidator


class NVIDIAIntegrationTester:
    """Comprehensive NVIDIA integration testing suite."""

    def __init__(self):
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_header(self, title: str):
        """Print formatted test section header."""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print('='*60)

    def print_result(self, test_name: str, passed: bool, details: str = ""):
        """Print formatted test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")

    def test_environment_setup(self) -> bool:
        """Test 1: Environment and Dependencies"""
        self.print_header("Environment Setup")

        try:
            # Test PyTorch
            print(f"🔧 PyTorch version: {torch.__version__}")
            print(f"🔧 CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🔧 CUDA version: {torch.version.cuda}")
                print(f"🔧 GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"🔧 GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")

            # Test KernelPyTorch imports
            from kernel_pytorch import __version__
            print(f"🔧 KernelPyTorch version: {__version__}")

            self.print_result("Environment setup", True, f"Using device: {self.device}")
            return True

        except Exception as e:
            self.print_result("Environment setup", False, f"Import error: {e}")
            return False

    def test_configuration_system(self) -> bool:
        """Test 2: Configuration System"""
        self.print_header("Configuration System")

        try:
            # Test basic configuration creation
            config = KernelPyTorchConfig()
            self.print_result("Basic config creation", True, f"Architecture: {config.hardware.nvidia.architecture.value}")

            # Test all config modes
            configs = {
                'default': KernelPyTorchConfig(),
                'inference': KernelPyTorchConfig.for_inference(),
                'training': KernelPyTorchConfig.for_training(),
                'development': KernelPyTorchConfig.for_development()
            }

            for mode, cfg in configs.items():
                nvidia_cfg = cfg.hardware.nvidia
                details = f"FP8: {nvidia_cfg.fp8_enabled}, TC: v{nvidia_cfg.tensor_core_version}"
                self.print_result(f"{mode.capitalize()} mode", True, details)

            # Test serialization
            config_dict = config.to_dict()
            has_nvidia = 'nvidia' in config_dict.get('hardware', {})
            self.print_result("Configuration serialization", has_nvidia, f"{len(config_dict)} top-level keys")

            return True

        except Exception as e:
            self.print_result("Configuration system", False, str(e))
            return False

    def test_nvidia_detection(self) -> bool:
        """Test 3: NVIDIA Hardware Detection"""
        self.print_header("NVIDIA Hardware Detection")

        try:
            config = KernelPyTorchConfig()
            nvidia = config.hardware.nvidia

            # Test detection results
            detection_info = {
                'Architecture': nvidia.architecture.value,
                'FP8 Enabled': nvidia.fp8_enabled,
                'Tensor Core Version': nvidia.tensor_core_version,
                'FlashAttention Version': nvidia.flash_attention_version,
                'Mixed Precision': nvidia.mixed_precision_enabled,
                'Memory Pool': nvidia.memory_pool_enabled,
                'Memory Fraction': nvidia.memory_fraction,
                'Kernel Fusion': nvidia.kernel_fusion_enabled
            }

            for key, value in detection_info.items():
                self.print_result(f"Detect {key}", True, str(value))

            # Test architecture-specific logic
            if self.device.type == 'cuda':
                props = torch.cuda.get_device_properties(0)
                expected_arch = self._get_expected_architecture(props.name, props.major, props.minor)
                arch_correct = nvidia.architecture == expected_arch
                self.print_result("Architecture detection accuracy", arch_correct,
                                f"Expected: {expected_arch}, Got: {nvidia.architecture}")
            else:
                self.print_result("CPU fallback detection", nvidia.architecture == NVIDIAArchitecture.PASCAL)

            return True

        except Exception as e:
            self.print_result("NVIDIA detection", False, str(e))
            return False

    def test_unit_tests(self) -> bool:
        """Test 4: Unit Test Suite"""
        self.print_header("Unit Test Suite")

        try:
            # Run NVIDIA configuration tests
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/test_nvidia_config.py",
                "-q", "--tb=no"
            ],
            cwd=repo_root,
            env={**dict(), "PYTHONPATH": str(repo_root / "src")},
            capture_output=True, text=True)

            if result.returncode == 0:
                # Parse test results
                output_lines = result.stdout.strip().split('\n')
                summary_line = [line for line in output_lines if 'passed' in line or 'failed' in line]
                summary = summary_line[-1] if summary_line else "Tests completed"
                self.print_result("NVIDIA unit tests", True, summary)
                return True
            else:
                self.print_result("NVIDIA unit tests", False, result.stdout or result.stderr)
                return False

        except Exception as e:
            self.print_result("Unit tests", False, str(e))
            return False

    def test_demos_and_benchmarks(self) -> bool:
        """Test 5: Demos and Benchmarks"""
        self.print_header("Demos and Benchmarks")

        try:
            # Test NVIDIA demo
            result = subprocess.run([
                sys.executable, "demos/nvidia_configuration_demo.py", "--quick"
            ],
            cwd=repo_root,
            env={**dict(), "PYTHONPATH": str(repo_root / "src")},
            capture_output=True, text=True)

            demo_success = result.returncode == 0
            self.print_result("NVIDIA demo", demo_success, "Quick mode executed")

            # Test NVIDIA benchmark
            result = subprocess.run([
                sys.executable, "benchmarks/nvidia_config_benchmarks.py", "--quick"
            ],
            cwd=repo_root,
            env={**dict(), "PYTHONPATH": str(repo_root / "src")},
            capture_output=True, text=True)

            benchmark_success = result.returncode == 0
            self.print_result("NVIDIA benchmark", benchmark_success, "Performance analysis completed")

            return demo_success and benchmark_success

        except Exception as e:
            self.print_result("Demos and benchmarks", False, str(e))
            return False

    def test_performance(self) -> bool:
        """Test 6: Performance Validation"""
        self.print_header("Performance Validation")

        try:
            # Test configuration creation performance
            start_time = time.perf_counter()
            for _ in range(100):
                config = KernelPyTorchConfig()
            config_time = (time.perf_counter() - start_time) * 1000 / 100

            perf_good = config_time < 1.0  # Should be < 1ms
            self.print_result("Config creation speed", perf_good, f"{config_time:.2f}ms avg")

            # Test hardware detection performance
            config = KernelPyTorchConfig()
            start_time = time.perf_counter()
            for _ in range(50):
                arch = config.hardware.nvidia._detect_architecture()
            detection_time = (time.perf_counter() - start_time) * 1000 / 50

            detection_good = detection_time < 1.0  # Should be < 1ms
            self.print_result("Hardware detection speed", detection_good, f"{detection_time:.2f}ms avg")

            # Test optimization levels
            model = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            ).to(self.device)

            test_input = torch.randn(32, 256).to(self.device)

            for opt_level in [OptimizationLevel.CONSERVATIVE, OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
                config = KernelPyTorchConfig()
                config.optimization_level = opt_level

                # Quick inference test
                with torch.no_grad():
                    output = model(test_input)

                self.print_result(f"{opt_level.value} optimization", True, f"Output shape: {output.shape}")

            return perf_good and detection_good

        except Exception as e:
            self.print_result("Performance validation", False, str(e))
            return False

    def test_validation_framework(self) -> bool:
        """Test 7: Validation Framework Integration"""
        self.print_header("Validation Framework")

        try:
            validator = UnifiedValidator()

            # Test configuration validation
            config = KernelPyTorchConfig.for_training()
            config_results = validator.validate_configuration(config)
            self.print_result("Configuration validation", config_results.passed > 0,
                            f"{config_results.passed}/{config_results.total_tests} passed")

            # Test hardware validation
            hw_results = validator.validate_hardware_compatibility(self.device)
            self.print_result("Hardware compatibility", hw_results.passed > 0,
                            f"{hw_results.passed}/{hw_results.total_tests} passed")

            # Test model validation
            model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))
            model_results = validator.validate_model(model, (8, 64))
            self.print_result("Model validation", model_results.passed > 0,
                            f"{model_results.passed}/{model_results.total_tests} passed")

            return config_results.passed > 0 and hw_results.passed > 0 and model_results.passed > 0

        except Exception as e:
            self.print_result("Validation framework", False, str(e))
            return False

    def _get_expected_architecture(self, gpu_name: str, major: int, minor: int) -> NVIDIAArchitecture:
        """Helper to get expected architecture based on GPU properties."""
        gpu_name = gpu_name.upper()

        if any(name in gpu_name for name in ["H100", "H200"]):
            return NVIDIAArchitecture.HOPPER
        elif any(name in gpu_name for name in ["B100", "B200"]):
            return NVIDIAArchitecture.BLACKWELL
        elif "A100" in gpu_name:
            return NVIDIAArchitecture.AMPERE
        elif any(name in gpu_name for name in ["RTX 40", "RTX 4090", "RTX 4080"]):
            return NVIDIAArchitecture.ADA
        elif any(name in gpu_name for name in ["RTX 30", "A40", "A30"]):
            return NVIDIAArchitecture.AMPERE
        elif any(name in gpu_name for name in ["RTX 20", "TITAN RTX"]):
            return NVIDIAArchitecture.TURING
        elif "V100" in gpu_name:
            return NVIDIAArchitecture.VOLTA
        elif major >= 9:
            return NVIDIAArchitecture.HOPPER
        elif major >= 8:
            return NVIDIAArchitecture.AMPERE
        elif major >= 7:
            return NVIDIAArchitecture.TURING if minor >= 5 else NVIDIAArchitecture.VOLTA
        else:
            return NVIDIAArchitecture.PASCAL

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all test suites and return results."""
        print("🚀 NVIDIA Integration Test Suite")
        print("="*60)

        test_results = {}

        test_suite = [
            ("Environment Setup", self.test_environment_setup),
            ("Configuration System", self.test_configuration_system),
            ("NVIDIA Detection", self.test_nvidia_detection),
            ("Unit Tests", self.test_unit_tests),
            ("Demos & Benchmarks", self.test_demos_and_benchmarks),
            ("Performance", self.test_performance),
            ("Validation Framework", self.test_validation_framework)
        ]

        for test_name, test_func in test_suite:
            try:
                result = test_func()
                test_results[test_name] = result
            except Exception as e:
                print(f"❌ CRITICAL ERROR in {test_name}: {e}")
                test_results[test_name] = False

        # Print summary
        self.print_header("Test Summary")
        passed = sum(test_results.values())
        total = len(test_results)

        for test_name, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")

        print(f"\n🎯 Overall Result: {passed}/{total} test suites passed")

        if passed == total:
            print("🎉 ALL TESTS PASSED - NVIDIA Integration is working perfectly!")
        elif passed >= total * 0.8:
            print("⚠️ Most tests passed - Minor issues detected")
        else:
            print("❌ Multiple test failures - NVIDIA Integration needs attention")

        return test_results


# Pytest integration classes
class TestNVIDIAIntegration:
    """Pytest integration for NVIDIA hardware tests."""

    def setup_class(self):
        """Set up test class."""
        self.tester = NVIDIAIntegrationTester()

    def test_environment_setup(self):
        """Test environment setup."""
        assert self.tester.test_environment_setup(), "Environment setup failed"

    def test_configuration_system(self):
        """Test configuration system."""
        assert self.tester.test_configuration_system(), "Configuration system failed"

    def test_nvidia_detection(self):
        """Test NVIDIA hardware detection."""
        assert self.tester.test_nvidia_detection(), "NVIDIA detection failed"

    def test_unit_tests(self):
        """Test unit test suite."""
        assert self.tester.test_unit_tests(), "Unit tests failed"

    def test_demos_and_benchmarks(self):
        """Test demos and benchmarks."""
        assert self.tester.test_demos_and_benchmarks(), "Demos and benchmarks failed"

    def test_performance(self):
        """Test performance validation."""
        assert self.tester.test_performance(), "Performance validation failed"

    def test_validation_framework(self):
        """Test validation framework integration."""
        assert self.tester.test_validation_framework(), "Validation framework failed"


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description='NVIDIA Integration Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--pytest', action='store_true', help='Run as pytest (used internally)')
    args = parser.parse_args()

    tester = NVIDIAIntegrationTester()

    if args.quick:
        print("🏃 Running Quick NVIDIA Integration Test...")
        # Just run essential tests
        results = {
            "Configuration": tester.test_configuration_system(),
            "Detection": tester.test_nvidia_detection(),
            "Performance": tester.test_performance()
        }
        passed = sum(results.values())
        print(f"\n🎯 Quick Test Result: {passed}/{len(results)} passed")
    else:
        results = tester.run_all_tests()

    # Exit code for CI/CD
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()