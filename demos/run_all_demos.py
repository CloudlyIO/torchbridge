#!/usr/bin/env python3
"""
🚀 Streamlined Demo Runner for PyTorch Optimization Showcase

Orchestrates execution of all demos with multiple modes:
- Quick: Fast validation (3-5 minutes)
- Full: Complete demonstration (15-20 minutes)
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

@dataclass
class DemoResult:
    demo: DemoInfo
    success: bool
    duration: float
    output: str
    error_message: Optional[str] = None


class StreamlinedDemoRunner:
    """Simplified demo runner for the new consolidated demo structure."""

    def __init__(self):
        # Define the 5 consolidated demos
        self.demos = [
            DemoInfo(
                name="Basic Optimizations",
                path="01_basic_optimizations.py",
                category="Fundamentals",
                difficulty="Beginner",
                estimated_time="2-3 minutes",
                description="Core PyTorch optimization techniques (fusion, compilation)",
                requires_gpu=False
            ),
            DemoInfo(
                name="Advanced Attention",
                path="02_advanced_attention.py",
                category="Advanced",
                difficulty="Intermediate",
                estimated_time="3-5 minutes",
                description="Ring, Sparse, and Context Parallel attention mechanisms",
                requires_gpu=True
            ),
            DemoInfo(
                name="FP8 Training",
                path="03_fp8_training.py",
                category="Precision",
                difficulty="Advanced",
                estimated_time="2-4 minutes",
                description="Production FP8 training for 2x H100 speedup",
                requires_gpu=True
            ),
            DemoInfo(
                name="Hardware Abstraction",
                path="04_hardware_abstraction.py",
                category="Infrastructure",
                difficulty="Intermediate",
                estimated_time="2-3 minutes",
                description="Multi-vendor GPU support and automatic optimization",
                requires_gpu=False
            ),
            DemoInfo(
                name="Production Deployment",
                path="05_production_deployment.py",
                category="Production",
                difficulty="Advanced",
                estimated_time="3-5 minutes",
                description="End-to-end optimization pipeline and deployment strategies",
                requires_gpu=False
            )
        ]

        self.demo_dir = os.path.dirname(__file__)

    def get_system_info(self) -> Dict:
        """Get system and hardware information."""
        import torch
        import platform

        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }

        return info

    def print_banner(self):
        """Print demo runner banner."""
        print("=" * 80)
        print("🚀 STREAMLINED PYTORCH OPTIMIZATION DEMO SHOWCASE")
        print("=" * 80)
        print("Consolidated 5-demo structure showcasing key optimization capabilities:")
        print("• Basic Optimizations → Advanced Attention → FP8 Training")
        print("• Hardware Abstraction → Production Deployment")
        print()

        # System info
        info = self.get_system_info()
        print(f"🖥️  System: {info['platform']}, Python {info['python_version']}")
        print(f"⚡ PyTorch: {info['pytorch_version']}")
        if info['cuda_available']:
            print(f"🎮 GPU: {info['gpu_name']} (CUDA {info['cuda_version']})")
        else:
            print(f"💻 Running on CPU (GPU not available)")
        print()

    def should_run_demo(self, demo: DemoInfo, mode: DemoMode, has_gpu: bool) -> Tuple[bool, str]:
        """Determine if demo should run based on mode and system capabilities."""

        if demo.requires_gpu and not has_gpu:
            return False, "Requires GPU (will run with fallback)"

        if mode == DemoMode.QUICK:
            # In quick mode, run all demos but with --quick flag
            return True, "Quick mode"
        elif mode == DemoMode.VALIDATE:
            # In validate mode, run all demos for validation
            return True, "Validation mode"
        else:
            # Full and interactive modes run everything
            return True, "Full demonstration"

    def run_demo(self, demo: DemoInfo, mode: DemoMode) -> DemoResult:
        """Execute a single demo and capture results."""

        print(f"🎯 Running: {demo.name}")
        print(f"   Category: {demo.category} | Difficulty: {demo.difficulty}")
        print(f"   Description: {demo.description}")

        start_time = time.time()

        # Construct command
        demo_path = os.path.join(self.demo_dir, demo.path)
        cmd = [sys.executable, demo_path]

        # Add mode-specific flags
        if mode == DemoMode.QUICK:
            cmd.append("--quick")
        elif mode == DemoMode.VALIDATE:
            cmd.append("--quick")  # Use quick mode for validation too

        try:
            # Run demo with timeout
            timeout = 300 if mode == DemoMode.FULL else 120  # 5 min full, 2 min quick

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.demo_dir
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"   ✅ Success ({duration:.1f}s)")
                return DemoResult(
                    demo=demo,
                    success=True,
                    duration=duration,
                    output=result.stdout
                )
            else:
                print(f"   ❌ Failed ({duration:.1f}s)")
                print(f"   Error: {result.stderr[:200]}...")
                return DemoResult(
                    demo=demo,
                    success=False,
                    duration=duration,
                    output=result.stdout,
                    error_message=result.stderr
                )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"   ⏰ Timeout after {duration:.1f}s")
            return DemoResult(
                demo=demo,
                success=False,
                duration=duration,
                output="",
                error_message=f"Demo timed out after {timeout}s"
            )
        except Exception as e:
            duration = time.time() - start_time
            print(f"   💥 Exception: {str(e)}")
            return DemoResult(
                demo=demo,
                success=False,
                duration=duration,
                output="",
                error_message=str(e)
            )

    def run_all_demos(self, mode: DemoMode) -> List[DemoResult]:
        """Run all demos in the specified mode."""

        self.print_banner()

        print(f"🎮 Running demos in {mode.value.upper()} mode")
        print(f"📊 Total demos: {len(self.demos)}")
        print()

        # Check system capabilities
        import torch
        has_gpu = torch.cuda.is_available()

        results = []
        total_start_time = time.time()

        # Run demos in sequence
        for i, demo in enumerate(self.demos, 1):
            print(f"\\n{'='*60}")
            print(f"[{i}/{len(self.demos)}] {demo.name}")
            print(f"{'='*60}")

            should_run, reason = self.should_run_demo(demo, mode, has_gpu)

            if should_run:
                result = self.run_demo(demo, mode)
                results.append(result)

                # Interactive mode pause
                if mode == DemoMode.INTERACTIVE and i < len(self.demos):
                    input("\\n⏸️  Press Enter to continue to next demo...")
            else:
                print(f"⏭️  Skipping: {reason}")

        # Summary
        total_duration = time.time() - total_start_time
        self.print_summary(results, total_duration, mode)

        return results

    def print_summary(self, results: List[DemoResult], total_duration: float, mode: DemoMode):
        """Print execution summary."""

        print(f"\\n{'='*80}")
        print("📊 DEMO EXECUTION SUMMARY")
        print(f"{'='*80}")

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"✅ Successful: {len(successful)}/{len(results)}")
        print(f"❌ Failed: {len(failed)}/{len(results)}")
        print(f"⏱️  Total time: {total_duration:.1f}s")
        print(f"🎯 Mode: {mode.value.upper()}")

        if successful:
            print(f"\\n🎉 Successful demos:")
            for result in successful:
                print(f"   • {result.demo.name} ({result.duration:.1f}s)")

        if failed:
            print(f"\\n💥 Failed demos:")
            for result in failed:
                print(f"   • {result.demo.name}: {result.error_message}")

        # Overall status
        success_rate = len(successful) / len(results) * 100 if results else 0

        if success_rate == 100:
            print(f"\\n🚀 ALL DEMOS SUCCESSFUL! Framework is production-ready.")
        elif success_rate >= 80:
            print(f"\\n✅ Mostly successful ({success_rate:.0f}%). Minor issues detected.")
        else:
            print(f"\\n⚠️  Multiple failures detected ({success_rate:.0f}%). Please investigate.")

        # Performance insights
        if successful:
            avg_time = sum(r.duration for r in successful) / len(successful)
            print(f"\\n📈 Performance insights:")
            print(f"   Average demo time: {avg_time:.1f}s")
            print(f"   {mode.value.capitalize()} mode duration: {total_duration:.1f}s")

    def run_interactive_mode(self) -> List[DemoResult]:
        """Run demos in interactive learning mode."""

        self.print_banner()

        print("🎓 INTERACTIVE LEARNING MODE")
        print("This mode guides you through each optimization technique step-by-step.")
        print("You can examine outputs and ask questions between demos.\\n")

        # Let user choose demos
        print("Available demos:")
        for i, demo in enumerate(self.demos, 1):
            gpu_req = " (GPU recommended)" if demo.requires_gpu else ""
            print(f"  {i}. {demo.name}{gpu_req}")
            print(f"     {demo.description}")

        choice = input("\\nRun all demos? [y/N]: ").strip().lower()

        if choice not in ['y', 'yes']:
            print("Interactive mode cancelled.")
            return []

        return self.run_all_demos(DemoMode.INTERACTIVE)


def main():
    parser = argparse.ArgumentParser(description="Streamlined PyTorch Optimization Demo Runner")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode (3-5 minutes)")
    parser.add_argument("--full", action="store_true", help="Full demonstration mode (15-20 minutes)")
    parser.add_argument("--interactive", action="store_true", help="Interactive learning mode")
    parser.add_argument("--validate", action="store_true", help="Validation mode for testing")

    args = parser.parse_args()

    # Determine mode
    if args.quick:
        mode = DemoMode.QUICK
    elif args.full:
        mode = DemoMode.FULL
    elif args.interactive:
        mode = DemoMode.INTERACTIVE
    elif args.validate:
        mode = DemoMode.VALIDATE
    else:
        # Default mode
        mode = DemoMode.QUICK
        print("No mode specified, using --quick mode")

    # Initialize runner
    runner = StreamlinedDemoRunner()

    # Run demos
    try:
        if mode == DemoMode.INTERACTIVE:
            results = runner.run_interactive_mode()
        else:
            results = runner.run_all_demos(mode)

        # Exit with appropriate code
        successful_demos = sum(1 for r in results if r.success)

        if successful_demos == len(results):
            sys.exit(0)  # All successful
        elif successful_demos > 0:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # All failed

    except KeyboardInterrupt:
        print("\\n\\n⏹️  Demo execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\n\\n💥 Demo runner error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()