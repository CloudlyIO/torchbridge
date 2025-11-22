#!/usr/bin/env python3
"""
Automated Demo Runner for PyTorch Optimization Showcase

Orchestrates execution of all demos with multiple modes:
- Quick: Fast validation (5-10 minutes)
- Full: Complete demonstration (1.5-2 hours)
- Interactive: Step-by-step learning experience
- Validate: Comprehensive correctness testing

Usage:
    python3 demos/run_all_demos.py --quick
    python3 demos/run_all_demos.py --full
    python3 demos/run_all_demos.py --interactive
    python3 demos/run_all_demos.py --validate
"""

import sys
import os
import time
import subprocess
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import importlib.util

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class DemoMode(Enum):
    QUICK = "quick"
    FULL = "full"
    INTERACTIVE = "interactive"
    VALIDATE = "validate"

@dataclass
class DemoInfo:
    name: str
    path: str
    category: str
    difficulty: str
    estimated_time: str
    description: str
    requires_gpu: bool = False
    requires_cuda: bool = False

@dataclass
class DemoResult:
    demo: DemoInfo
    success: bool
    execution_time: float
    output: str
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict] = None

class DemoRunner:
    """Orchestrates execution of all optimization demos"""

    def __init__(self, mode: DemoMode):
        self.mode = mode
        self.results: List[DemoResult] = []
        self.start_time = time.time()

        # Demo catalog
        self.demos = self._build_demo_catalog()

        # Environment info
        self.env_info = self._get_environment_info()

    def _build_demo_catalog(self) -> List[DemoInfo]:
        """Build comprehensive demo catalog"""
        demos = [
            # 01_getting_started
            DemoInfo(
                "Basic Optimizations",
                "01_getting_started/basic_optimizations_demo.py",
                "Getting Started",
                "🟢 Beginner",
                "5-8 min",
                "Fundamental PyTorch optimization patterns"
            ),

            # 02_compiler_optimizations
            DemoInfo(
                "FlashLight Compiler",
                "02_compiler_optimizations/flashlight_demo.py",
                "Compiler Optimizations",
                "🟡 Intermediate",
                "8-12 min",
                "Automatic attention kernel generation",
                requires_gpu=True
            ),
            DemoInfo(
                "Integrated Compiler Demo",
                "02_compiler_optimizations/integrated_compiler_demo.py",
                "Compiler Optimizations",
                "🟠 Advanced",
                "10-15 min",
                "All compiler optimizations working together"
            ),

            # 03_advanced_attention
            DemoInfo(
                "Ring Attention",
                "03_advanced_attention/ring_attention_demo.py",
                "Advanced Attention",
                "🟡 Intermediate",
                "10-15 min",
                "Ring attention for extremely long sequences"
            ),
            DemoInfo(
                "Sparse Attention",
                "03_advanced_attention/sparse_attention_demo.py",
                "Advanced Attention",
                "🟡 Intermediate",
                "8-12 min",
                "Sparse attention patterns and optimization"
            ),

            # 04_gpu_integration
            DemoInfo(
                "CUDA Graphs",
                "04_gpu_integration/cuda_graphs_demo.py",
                "GPU Integration",
                "🟠 Advanced",
                "12-18 min",
                "CUDA graphs and advanced GPU optimization",
                requires_gpu=True
            ),

            # 05_next_generation
            DemoInfo(
                "Neuromorphic Simulation",
                "05_next_generation/neuromorphic_simulation_demo.py",
                "Next Generation",
                "🔴 Expert",
                "15-20 min",
                "Neuromorphic computing and next-gen paradigms"
            ),

            # 06_testing_framework
            DemoInfo(
                "Optimization Validation",
                "06_testing_framework/optimization_validation_demo.py",
                "Testing Framework",
                "🟡 Intermediate",
                "8-12 min",
                "Validating optimization correctness"
            ),

            # 07_production_ready
            DemoInfo(
                "Deployment Optimization",
                "07_production_ready/deployment_optimization_demo.py",
                "Production Ready",
                "🟠 Advanced",
                "15-20 min",
                "Production deployment and monitoring"
            )
        ]

        return demos

    def _get_environment_info(self) -> Dict:
        """Get environment information for demo compatibility"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if cuda_available else 0
            cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "None"
        except ImportError:
            cuda_available = False
            cuda_device_count = 0
            cuda_device_name = "PyTorch not available"

        return {
            "python_version": sys.version,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "platform": sys.platform
        }

    def run_all_demos(self) -> List[DemoResult]:
        """Run all demos according to selected mode"""
        self._print_header()

        if self.mode == DemoMode.INTERACTIVE:
            return self._run_interactive_mode()
        elif self.mode == DemoMode.QUICK:
            return self._run_quick_mode()
        elif self.mode == DemoMode.VALIDATE:
            return self._run_validation_mode()
        else:  # FULL mode
            return self._run_full_mode()

    def _print_header(self):
        """Print demo session header"""
        print("🚀 PyTorch Optimization Demos")
        print("=" * 60)
        print(f"Mode: {self.mode.value.upper()}")
        print(f"Total Demos: {len(self.demos)}")
        print(f"CUDA Available: {'✅' if self.env_info['cuda_available'] else '❌'}")
        if self.env_info['cuda_available']:
            print(f"GPU: {self.env_info['cuda_device_name']}")
        print("=" * 60)

    def _run_quick_mode(self) -> List[DemoResult]:
        """Run quick validation mode (5-10 minutes)"""
        print("\n🚀 Quick Validation Mode")
        print("Running essential demos for functionality validation...\n")

        # Select representative demos from each category
        quick_demos = [
            "01_getting_started/basic_optimizations_demo.py",
            "02_compiler_optimizations/flashlight_demo.py",
            "03_advanced_attention/ring_attention_demo.py",
            "06_testing_framework/optimization_validation_demo.py"
        ]

        selected_demos = [demo for demo in self.demos if demo.path in quick_demos]
        return self._execute_demos(selected_demos, quick_mode=True)

    def _run_full_mode(self) -> List[DemoResult]:
        """Run full demo suite (1.5-2 hours)"""
        print("\n🔍 Full Demo Mode")
        print("Running complete demonstration suite...\n")

        # Filter demos based on hardware availability
        available_demos = self._filter_demos_by_hardware(self.demos)
        return self._execute_demos(available_demos, quick_mode=False)

    def _run_interactive_mode(self) -> List[DemoResult]:
        """Run interactive learning mode"""
        print("\n🎓 Interactive Learning Mode")
        print("Step-by-step demonstration with explanations...\n")

        results = []
        categories = self._group_demos_by_category()

        for category, demos in categories.items():
            print(f"\n📂 Category: {category}")
            print("-" * 40)

            for demo in demos:
                if not self._is_demo_compatible(demo):
                    print(f"⏭️  Skipping {demo.name} (hardware incompatible)")
                    continue

                print(f"\n🎯 {demo.name}")
                print(f"   Description: {demo.description}")
                print(f"   Difficulty: {demo.difficulty}")
                print(f"   Estimated Time: {demo.estimated_time}")

                response = input("\n   Run this demo? [y/n/q]: ").lower()

                if response == 'q':
                    break
                elif response == 'y':
                    result = self._execute_single_demo(demo, interactive=True)
                    results.append(result)
                    input("\n   Press Enter to continue...")

        return results

    def _run_validation_mode(self) -> List[DemoResult]:
        """Run comprehensive validation mode"""
        print("\n🧪 Validation Mode")
        print("Comprehensive correctness and performance testing...\n")

        # Run all demos with validation checks
        available_demos = self._filter_demos_by_hardware(self.demos)
        return self._execute_demos(available_demos, validate=True)

    def _filter_demos_by_hardware(self, demos: List[DemoInfo]) -> List[DemoInfo]:
        """Filter demos based on available hardware"""
        available_demos = []

        for demo in demos:
            if demo.requires_cuda and not self.env_info['cuda_available']:
                print(f"⏭️  Skipping {demo.name} (CUDA required)")
                continue
            elif demo.requires_gpu and not self.env_info['cuda_available']:
                print(f"⏭️  Skipping {demo.name} (GPU recommended)")
                # Still include but note performance may be limited

            available_demos.append(demo)

        return available_demos

    def _is_demo_compatible(self, demo: DemoInfo) -> bool:
        """Check if demo is compatible with current hardware"""
        if demo.requires_cuda and not self.env_info['cuda_available']:
            return False
        return True

    def _group_demos_by_category(self) -> Dict[str, List[DemoInfo]]:
        """Group demos by category for organized execution"""
        categories = {}
        for demo in self.demos:
            if demo.category not in categories:
                categories[demo.category] = []
            categories[demo.category].append(demo)
        return categories

    def _execute_demos(self, demos: List[DemoInfo], quick_mode: bool = False,
                      validate: bool = False) -> List[DemoResult]:
        """Execute a list of demos"""
        results = []

        for i, demo in enumerate(demos, 1):
            print(f"\n[{i}/{len(demos)}] Running {demo.name}...")

            if not self._is_demo_compatible(demo):
                print(f"   ⏭️  Skipped (hardware incompatible)")
                continue

            result = self._execute_single_demo(demo, quick_mode, validate)
            results.append(result)

            # Print immediate feedback
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"   {status} ({result.execution_time:.1f}s)")

            if not result.success and result.error_message:
                print(f"   Error: {result.error_message}")

        return results

    def _execute_single_demo(self, demo: DemoInfo, quick_mode: bool = False,
                            validate: bool = False, interactive: bool = False) -> DemoResult:
        """Execute a single demo and capture results"""
        demo_path = os.path.join(os.path.dirname(__file__), demo.path)

        if not os.path.exists(demo_path):
            return DemoResult(
                demo=demo,
                success=False,
                execution_time=0,
                output="",
                error_message=f"Demo file not found: {demo_path}"
            )

        try:
            start_time = time.time()

            # Build command
            cmd = [sys.executable, demo_path]
            if quick_mode:
                cmd.append("--quick")
            if validate:
                cmd.append("--validate")
            if interactive:
                cmd.append("--interactive")

            # Execute demo
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '..', 'src')

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )

            execution_time = time.time() - start_time

            return DemoResult(
                demo=demo,
                success=result.returncode == 0,
                execution_time=execution_time,
                output=result.stdout,
                error_message=result.stderr if result.returncode != 0 else None
            )

        except subprocess.TimeoutExpired:
            return DemoResult(
                demo=demo,
                success=False,
                execution_time=300,
                output="",
                error_message="Demo timed out after 5 minutes"
            )
        except Exception as e:
            return DemoResult(
                demo=demo,
                success=False,
                execution_time=0,
                output="",
                error_message=str(e)
            )

    def print_summary(self, results: List[DemoResult]):
        """Print comprehensive summary of demo results"""
        print("\n" + "=" * 60)
        print("📊 Demo Results Summary")
        print("=" * 60)

        # Overall statistics
        total_demos = len(results)
        passed_demos = sum(1 for r in results if r.success)
        total_time = sum(r.execution_time for r in results)

        print(f"Total Demos Run: {total_demos}")
        print(f"Passed: {passed_demos} ✅")
        print(f"Failed: {total_demos - passed_demos} ❌")
        print(f"Success Rate: {passed_demos/total_demos*100:.1f}%")
        print(f"Total Execution Time: {total_time:.1f}s ({total_time/60:.1f} min)")

        # Category breakdown
        print(f"\n📂 Results by Category:")
        categories = {}
        for result in results:
            cat = result.demo.category
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if result.success:
                categories[cat]["passed"] += 1

        for category, stats in categories.items():
            rate = stats["passed"] / stats["total"] * 100
            print(f"   {category}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

        # Failed demos details
        failed_demos = [r for r in results if not r.success]
        if failed_demos:
            print(f"\n❌ Failed Demos:")
            for result in failed_demos:
                print(f"   • {result.demo.name}: {result.error_message}")

        # Performance highlights
        successful_demos = [r for r in results if r.success]
        if successful_demos:
            fastest = min(successful_demos, key=lambda r: r.execution_time)
            slowest = max(successful_demos, key=lambda r: r.execution_time)

            print(f"\n⚡ Performance Highlights:")
            print(f"   Fastest: {fastest.demo.name} ({fastest.execution_time:.1f}s)")
            print(f"   Slowest: {slowest.demo.name} ({slowest.execution_time:.1f}s)")

        print(f"\n🎉 Demo session completed!")

        if passed_demos == total_demos:
            print("All demos passed successfully! 🚀")
        elif passed_demos >= total_demos * 0.8:
            print("Most demos passed - excellent results! ✨")
        elif passed_demos >= total_demos * 0.6:
            print("Good results - some demos need attention. 👍")
        else:
            print("Several demos failed - check setup and requirements. ⚠️")

def main():
    """Main demo runner entry point"""
    parser = argparse.ArgumentParser(description="PyTorch Optimization Demo Runner")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation mode (5-10 minutes)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full demo suite (1.5-2 hours)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive learning mode"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run comprehensive validation mode"
    )

    args = parser.parse_args()

    # Determine mode
    if args.interactive:
        mode = DemoMode.INTERACTIVE
    elif args.validate:
        mode = DemoMode.VALIDATE
    elif args.quick:
        mode = DemoMode.QUICK
    else:
        mode = DemoMode.FULL

    # Run demos
    runner = DemoRunner(mode)
    results = runner.run_all_demos()
    runner.print_summary(results)

    # Exit with appropriate code
    success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
    sys.exit(0 if success_rate >= 0.8 else 1)

if __name__ == "__main__":
    main()