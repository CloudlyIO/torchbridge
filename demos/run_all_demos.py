#!/usr/bin/env python3
"""
🚀 KernelPyTorch Demo Runner

Runs all demos with clean, organized structure.

Usage:
    python run_all_demos.py --quick
    python run_all_demos.py --validate
"""

import sys
import os
import subprocess
import argparse
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def print_header():
    """Print demo suite header."""
    print("🚀 KernelPyTorch Demo Suite")
    print("=" * 50)
    print("Running PyTorch optimization demonstrations")
    print()

def run_demo(demo_path, mode="quick"):
    """Run a single demo."""
    if not os.path.exists(demo_path):
        return False, f"Demo not found: {demo_path}"

    cmd = [sys.executable, demo_path, f"--{mode}"]

    try:
        start_time = time.time()
        result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=300,  # 5 minute timeout
                              cwd=os.path.dirname(__file__))

        duration = time.time() - start_time

        if result.returncode == 0:
            return True, f"Completed in {duration:.1f}s"
        else:
            return False, f"Failed: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return False, "Timeout (5 minutes)"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Run KernelPyTorch demos")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode")
    parser.add_argument("--validate", action="store_true", help="Validation mode")
    args = parser.parse_args()

    mode = "validate" if args.validate else "quick"

    print_header()
    print(f"Mode: {mode.upper()}")
    print()

    # Key demos to run in order of importance
    demos = [
        ("🎯 Adaptive Precision", "precision/adaptive.py"),
        ("🧠 Neural Operator Fusion", "attention/fusion.py"),
        ("💾 Deep Optimizer States", "memory/deep_states.py"),
        ("⚡ Dynamic Shapes", "compiler/shapes.py"),
        ("🚀 Ultra Precision", "experimental/ultra_precision.py")
    ]

    results = []
    total_start = time.time()

    for i, (name, path) in enumerate(demos, 1):
        print(f"[{i}/{len(demos)}] {name}")
        print("-" * 40)

        success, message = run_demo(path, mode)
        results.append((name, success, message))

        if success:
            print(f"   ✅ {message}")
        else:
            print(f"   ❌ {message}")
        print()

    # Summary
    total_time = time.time() - total_start
    successful = sum(1 for _, success, _ in results if success)

    print("=" * 50)
    print("📊 DEMO RESULTS SUMMARY")
    print("=" * 50)
    print(f"✅ Successful: {successful}/{len(demos)}")
    print(f"⏱️  Total time: {total_time:.1f}s")

    if successful < len(demos):
        print("\n❌ Failed demos:")
        for name, success, message in results:
            if not success:
                print(f"   • {name}: {message}")
    else:
        print("\n🎉 All demos completed successfully!")
        print("   Ready for production optimization!")

if __name__ == "__main__":
    main()