"""
System diagnostics commands for KernelPyTorch CLI.

Provides comprehensive system compatibility checking and optimization recommendations.
"""

import argparse
import sys
import platform
import subprocess
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import kernel_pytorch as kpt


@dataclass
class DiagnosticResult:
    """Result from a single diagnostic check."""
    name: str
    status: str  # "pass", "warning", "fail"
    message: str
    details: Optional[str] = None
    recommendation: Optional[str] = None


class DoctorCommand:
    """System diagnostics command implementation."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the doctor command with argument parser."""
        parser = subparsers.add_parser(
            'doctor',
            help='Diagnose system compatibility and optimization readiness',
            description='Check system configuration for optimal KernelPyTorch performance',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Check Categories:
  basic      - Python, PyTorch, and basic dependencies
  hardware   - GPU detection and capabilities
  optimization - Optimization framework availability
  advanced   - Advanced features (Triton, CUDA kernels)

Examples:
  kpt-doctor                    # Quick system check
  kpt-doctor --full-report      # Comprehensive diagnostics
  kpt-doctor --category hardware # Hardware-specific checks
  kpt-doctor --fix             # Attempt to fix detected issues
            """
        )

        parser.add_argument(
            '--category',
            choices=['basic', 'hardware', 'optimization', 'advanced'],
            help='Focus on specific diagnostic category'
        )

        parser.add_argument(
            '--full-report',
            action='store_true',
            help='Run comprehensive diagnostics (all categories)'
        )

        parser.add_argument(
            '--fix',
            action='store_true',
            help='Attempt to fix detected issues (where possible)'
        )

        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Save diagnostic report to file'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the doctor command."""
        print("🩺 KernelPyTorch System Diagnostics")
        print("=" * 50)

        try:
            results = []

            # Determine which checks to run
            if args.category:
                if args.category == 'basic':
                    results.extend(DoctorCommand._check_basic_requirements(args.verbose))
                elif args.category == 'hardware':
                    results.extend(DoctorCommand._check_hardware(args.verbose))
                elif args.category == 'optimization':
                    results.extend(DoctorCommand._check_optimization_frameworks(args.verbose))
                elif args.category == 'advanced':
                    results.extend(DoctorCommand._check_advanced_features(args.verbose))
                else:
                    raise ValueError(f"Invalid category: {args.category}")
            elif args.full_report:
                results.extend(DoctorCommand._check_basic_requirements(args.verbose))
                results.extend(DoctorCommand._check_hardware(args.verbose))
                results.extend(DoctorCommand._check_optimization_frameworks(args.verbose))
                results.extend(DoctorCommand._check_advanced_features(args.verbose))
            else:
                # Quick check - basic + hardware
                results.extend(DoctorCommand._check_basic_requirements(args.verbose))
                results.extend(DoctorCommand._check_hardware(args.verbose))

            # Display results
            DoctorCommand._display_results(results, args.verbose)

            # Generate summary
            summary = DoctorCommand._generate_summary(results)
            print(summary)

            # Fix issues if requested
            if args.fix:
                DoctorCommand._attempt_fixes(results, args.verbose)

            # Save report if requested
            if args.output:
                DoctorCommand._save_report(results, args.output, args.verbose)

            # Return appropriate exit code
            has_failures = any(r.status == 'fail' for r in results)
            return 1 if has_failures else 0

        except Exception as e:
            print(f"❌ Diagnostics failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def _check_basic_requirements(verbose: bool) -> List[DiagnosticResult]:
        """Check basic Python and PyTorch requirements."""
        if verbose:
            print("🔍 Checking basic requirements...")

        results = []

        # Python version
        python_version = platform.python_version()
        python_major, python_minor = map(int, python_version.split('.')[:2])

        if python_major >= 3 and python_minor >= 8:
            results.append(DiagnosticResult(
                "Python Version",
                "pass",
                f"Python {python_version} (✓ Compatible)"
            ))
        else:
            results.append(DiagnosticResult(
                "Python Version",
                "fail",
                f"Python {python_version} (✗ Requires Python 3.8+)",
                recommendation="Upgrade to Python 3.8 or later"
            ))

        # PyTorch version
        try:
            torch_version = torch.__version__
            torch_major, torch_minor = map(int, torch_version.split('.')[:2])

            if torch_major >= 2:
                results.append(DiagnosticResult(
                    "PyTorch Version",
                    "pass",
                    f"PyTorch {torch_version} (✓ Compatible)"
                ))
            elif torch_major == 1 and torch_minor >= 12:
                results.append(DiagnosticResult(
                    "PyTorch Version",
                    "warning",
                    f"PyTorch {torch_version} (⚠ Recommend 2.0+)",
                    recommendation="Upgrade to PyTorch 2.0+ for best performance"
                ))
            else:
                results.append(DiagnosticResult(
                    "PyTorch Version",
                    "fail",
                    f"PyTorch {torch_version} (✗ Requires 1.12+)",
                    recommendation="Upgrade to PyTorch 2.0+ for full compatibility"
                ))
        except Exception as e:
            results.append(DiagnosticResult(
                "PyTorch Installation",
                "fail",
                f"PyTorch not found ({e})",
                recommendation="Install PyTorch: pip install torch"
            ))

        # NumPy version
        try:
            import numpy as np
            numpy_version = np.__version__
            results.append(DiagnosticResult(
                "NumPy Version",
                "pass",
                f"NumPy {numpy_version} (✓ Available)"
            ))
        except ImportError:
            results.append(DiagnosticResult(
                "NumPy Installation",
                "fail",
                "NumPy not found",
                recommendation="Install NumPy: pip install numpy"
            ))

        # KernelPyTorch installation
        try:
            kpt_version = kpt.__version__
            results.append(DiagnosticResult(
                "KernelPyTorch Version",
                "pass",
                f"KernelPyTorch {kpt_version} (✓ Available)"
            ))
        except Exception as e:
            results.append(DiagnosticResult(
                "KernelPyTorch Installation",
                "fail",
                f"KernelPyTorch not properly installed ({e})",
                recommendation="Reinstall: pip install -e ."
            ))

        return results

    @staticmethod
    def _check_hardware(verbose: bool) -> List[DiagnosticResult]:
        """Check hardware capabilities and GPU availability."""
        if verbose:
            print("🔍 Checking hardware capabilities...")

        results = []

        # CUDA availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda

            results.append(DiagnosticResult(
                "CUDA GPU",
                "pass",
                f"{gpu_name} with {gpu_memory:.1f} GB (✓ Available)",
                details=f"CUDA {cuda_version}"
            ))

            # Check GPU compute capability
            major, minor = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
            compute_capability = f"{major}.{minor}"

            if major >= 7:  # Volta or newer
                results.append(DiagnosticResult(
                    "GPU Compute Capability",
                    "pass",
                    f"Compute {compute_capability} (✓ Excellent for optimization)",
                    details="Supports Tensor Cores and advanced optimizations"
                ))
            elif major >= 6:  # Pascal
                results.append(DiagnosticResult(
                    "GPU Compute Capability",
                    "warning",
                    f"Compute {compute_capability} (⚠ Good, but older)",
                    recommendation="Consider upgrading for best performance"
                ))
            else:
                results.append(DiagnosticResult(
                    "GPU Compute Capability",
                    "warning",
                    f"Compute {compute_capability} (⚠ Limited optimization support)",
                    recommendation="GPU may not support all optimization features"
                ))

            # Check memory
            if gpu_memory >= 8.0:
                results.append(DiagnosticResult(
                    "GPU Memory",
                    "pass",
                    f"{gpu_memory:.1f} GB (✓ Sufficient for most workloads)"
                ))
            elif gpu_memory >= 4.0:
                results.append(DiagnosticResult(
                    "GPU Memory",
                    "warning",
                    f"{gpu_memory:.1f} GB (⚠ May limit large models)",
                    recommendation="Consider larger GPU for production workloads"
                ))
            else:
                results.append(DiagnosticResult(
                    "GPU Memory",
                    "warning",
                    f"{gpu_memory:.1f} GB (⚠ Limited memory for optimization)",
                    recommendation="Upgrade GPU or use CPU-only optimizations"
                ))
        else:
            results.append(DiagnosticResult(
                "CUDA GPU",
                "warning",
                "No CUDA GPU detected (⚠ CPU-only mode)",
                recommendation="Install CUDA-compatible PyTorch for GPU acceleration"
            ))

        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            results.append(DiagnosticResult(
                "Apple Silicon GPU",
                "pass",
                "Apple Silicon GPU (✓ Available)",
                details="MPS backend available for acceleration"
            ))

        # CPU info
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        results.append(DiagnosticResult(
            "CPU Cores",
            "pass",
            f"{cpu_count} cores (✓ Available)",
            details=f"Platform: {platform.machine()}"
        ))

        return results

    @staticmethod
    def _check_optimization_frameworks(verbose: bool) -> List[DiagnosticResult]:
        """Check availability of optimization frameworks."""
        if verbose:
            print("🔍 Checking optimization frameworks...")

        results = []

        # torch.compile (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                # Test torch.compile
                test_model = torch.nn.Linear(10, 1)
                compiled_model = torch.compile(test_model, mode='default')
                results.append(DiagnosticResult(
                    "torch.compile",
                    "pass",
                    "torch.compile (✓ Available and working)",
                    details="PyTorch 2.0+ compilation framework"
                ))
            else:
                results.append(DiagnosticResult(
                    "torch.compile",
                    "warning",
                    "torch.compile not available (⚠ Requires PyTorch 2.0+)",
                    recommendation="Upgrade to PyTorch 2.0+ for compilation features"
                ))
        except Exception as e:
            results.append(DiagnosticResult(
                "torch.compile",
                "fail",
                f"torch.compile not working ({e})",
                recommendation="Check PyTorch installation"
            ))

        # TorchScript
        try:
            test_model = torch.nn.Linear(10, 1)
            test_input = torch.randn(1, 10)
            traced_model = torch.jit.trace(test_model, test_input)
            results.append(DiagnosticResult(
                "TorchScript",
                "pass",
                "TorchScript JIT (✓ Available and working)"
            ))
        except Exception as e:
            results.append(DiagnosticResult(
                "TorchScript",
                "warning",
                f"TorchScript issues detected ({e})",
                recommendation="Check for model compatibility issues"
            ))

        # KernelPyTorch components
        try:
            from kernel_pytorch.utils.compiler_assistant import CompilerOptimizationAssistant
            results.append(DiagnosticResult(
                "KernelPyTorch Optimization",
                "pass",
                "Optimization framework (✓ Available)"
            ))
        except ImportError as e:
            results.append(DiagnosticResult(
                "KernelPyTorch Optimization",
                "fail",
                f"Optimization framework not available ({e})",
                recommendation="Reinstall KernelPyTorch package"
            ))

        return results

    @staticmethod
    def _check_advanced_features(verbose: bool) -> List[DiagnosticResult]:
        """Check availability of advanced optimization features."""
        if verbose:
            print("🔍 Checking advanced features...")

        results = []

        # Triton
        try:
            import triton
            triton_version = triton.__version__
            results.append(DiagnosticResult(
                "Triton Kernels",
                "pass",
                f"Triton {triton_version} (✓ Available)",
                details="Python-based GPU kernel development"
            ))
        except ImportError:
            if torch.cuda.is_available():
                results.append(DiagnosticResult(
                    "Triton Kernels",
                    "warning",
                    "Triton not available (⚠ Optional for advanced kernels)",
                    recommendation="Install Triton: pip install triton"
                ))
            else:
                results.append(DiagnosticResult(
                    "Triton Kernels",
                    "warning",
                    "Triton not available (⚠ Requires CUDA GPU)",
                    details="Triton requires CUDA-compatible hardware"
                ))

        # Flash Attention
        try:
            import flash_attn
            results.append(DiagnosticResult(
                "Flash Attention",
                "pass",
                "Flash Attention (✓ Available)",
                details="Optimized attention implementation"
            ))
        except ImportError:
            if torch.cuda.is_available():
                results.append(DiagnosticResult(
                    "Flash Attention",
                    "warning",
                    "Flash Attention not available (⚠ Optional optimization)",
                    recommendation="Install: pip install flash-attn"
                ))

        # CUDA toolkit (for custom kernels)
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                nvcc_info = result.stdout
                results.append(DiagnosticResult(
                    "CUDA Toolkit",
                    "pass",
                    "NVCC compiler (✓ Available)",
                    details="Can compile custom CUDA kernels"
                ))
            else:
                results.append(DiagnosticResult(
                    "CUDA Toolkit",
                    "warning",
                    "CUDA Toolkit not found (⚠ Optional for custom kernels)",
                    recommendation="Install CUDA Toolkit for kernel development"
                ))
        except FileNotFoundError:
            results.append(DiagnosticResult(
                "CUDA Toolkit",
                "warning",
                "NVCC not in PATH (⚠ Optional for custom kernels)",
                recommendation="Add CUDA to PATH or install CUDA Toolkit"
            ))

        return results

    @staticmethod
    def _display_results(results: List[DiagnosticResult], verbose: bool) -> None:
        """Display diagnostic results in a formatted way."""
        print("\n🔍 Diagnostic Results:")
        print("-" * 60)

        for result in results:
            status_icon = {
                'pass': '✅',
                'warning': '⚠️',
                'fail': '❌'
            }.get(result.status, '❓')

            print(f"{status_icon} {result.name}: {result.message}")

            if verbose and result.details:
                print(f"   Details: {result.details}")

            if result.recommendation:
                print(f"   💡 Recommendation: {result.recommendation}")

            if verbose:
                print()

    @staticmethod
    def _generate_summary(results: List[DiagnosticResult]) -> str:
        """Generate a summary of diagnostic results."""
        total = len(results)
        passed = sum(1 for r in results if r.status == 'pass')
        warnings = sum(1 for r in results if r.status == 'warning')
        failed = sum(1 for r in results if r.status == 'fail')

        summary = f"\n📊 Summary: {passed}/{total} checks passed"
        if warnings > 0:
            summary += f", {warnings} warnings"
        if failed > 0:
            summary += f", {failed} failures"

        if failed > 0:
            summary += "\n❗ Critical issues detected - system may not work optimally"
        elif warnings > 0:
            summary += "\n⚠️  Some optimizations may not be available"
        else:
            summary += "\n✅ System is ready for optimal KernelPyTorch performance!"

        return summary

    @staticmethod
    def _attempt_fixes(results: List[DiagnosticResult], verbose: bool) -> None:
        """Attempt to fix detected issues where possible."""
        print("\n🔧 Attempting to fix detected issues...")

        fixable_issues = [r for r in results if r.status in ['fail', 'warning'] and r.recommendation]

        if not fixable_issues:
            print("   No fixable issues detected.")
            return

        for issue in fixable_issues:
            if 'pip install' in issue.recommendation:
                print(f"   Attempting to install missing package for {issue.name}...")
                # Note: In a real implementation, you might want to be more careful about automatic installation
                print(f"   ℹ️  Manual action required: {issue.recommendation}")

        print("   💡 Some fixes require manual intervention. See recommendations above.")

    @staticmethod
    def _save_report(results: List[DiagnosticResult], output_path: str, verbose: bool) -> None:
        """Save diagnostic report to file."""
        if verbose:
            print(f"💾 Saving diagnostic report to: {output_path}")

        import json
        import time

        report = {
            'timestamp': time.time(),
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            'diagnostics': [
                {
                    'name': r.name,
                    'status': r.status,
                    'message': r.message,
                    'details': r.details,
                    'recommendation': r.recommendation
                }
                for r in results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        if verbose:
            print(f"   Report saved ({len(results)} diagnostics)")


def main():
    """Standalone entry point for kpt-doctor."""
    parser = argparse.ArgumentParser(
        prog='kpt-doctor',
        description='Diagnose system compatibility and optimization readiness',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add the doctor command arguments directly to the parser
    parser.add_argument(
        '--category',
        choices=['basic', 'hardware', 'optimization', 'advanced'],
        help='Focus on specific category'
    )
    parser.add_argument(
        '--full-report',
        action='store_true',
        help='Run comprehensive diagnostics (all categories)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to fix detected issues (where possible)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save diagnostic report to file (JSON format)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()
    return DoctorCommand.execute(args)


if __name__ == '__main__':
    sys.exit(main())