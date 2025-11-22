#!/usr/bin/env python3
"""
Test Performance Profiler

This script helps identify slow tests in the codebase and suggests optimizations.
"""

import subprocess
import time
import sys
import re
from typing import List, Dict, Tuple

def profile_test_file(test_file: str) -> Dict[str, float]:
    """Profile all tests in a test file and return timing results"""
    print(f"📊 Profiling {test_file}...")

    # Get list of tests in the file
    try:
        result = subprocess.run([
            'python3', '-m', 'pytest', test_file, '--collect-only', '-q'
        ], env={'PYTHONPATH': 'src'}, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"❌ Failed to collect tests from {test_file}")
            return {}

    except Exception as e:
        print(f"❌ Error collecting tests: {e}")
        return {}

    # Extract test names
    test_names = re.findall(r'::test_\w+', result.stdout)
    test_timings = {}

    print(f"Found {len(test_names)} tests")

    # Profile each test individually
    for i, test_name in enumerate(test_names[:5], 1):  # Limit to first 5 for demo
        test_path = f"{test_file}{test_name}"
        print(f"  [{i}/{min(5, len(test_names))}] Testing {test_name}...")

        start_time = time.time()
        try:
            result = subprocess.run([
                'python3', '-m', 'pytest', test_path, '-v'
            ], env={'PYTHONPATH': 'src'}, capture_output=True, text=True, timeout=60)

            duration = time.time() - start_time
            test_timings[test_name] = duration

            status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
            print(f"    {status} {duration:.3f}s")

        except subprocess.TimeoutExpired:
            duration = 60.0
            test_timings[test_name] = duration
            print(f"    ⏱️ TIMEOUT {duration:.3f}s")
        except Exception as e:
            print(f"    🚨 ERROR: {e}")

    return test_timings

def analyze_timings(timings: Dict[str, float]) -> None:
    """Analyze test timings and provide recommendations"""
    if not timings:
        print("No timing data available")
        return

    print("\n" + "="*60)
    print("📈 PERFORMANCE ANALYSIS")
    print("="*60)

    # Sort by duration
    sorted_tests = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    total_time = sum(timings.values())

    print(f"Total test time: {total_time:.3f}s")
    print(f"Average test time: {total_time/len(timings):.3f}s")
    print(f"Slowest test: {sorted_tests[0][1]:.3f}s")

    print("\n🐌 SLOWEST TESTS (optimization candidates):")
    for test_name, duration in sorted_tests:
        percentage = (duration / total_time) * 100

        if duration > 5.0:
            priority = "🔴 HIGH"
        elif duration > 1.0:
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"

        print(f"  {priority:12} {test_name:40} {duration:8.3f}s ({percentage:5.1f}%)")

def suggest_optimizations(test_name: str, duration: float) -> List[str]:
    """Suggest optimizations based on test name and duration"""
    suggestions = []

    if duration > 10.0:
        suggestions.append("🚀 Reduce tensor dimensions (batch_size=1, seq_len=64)")
        suggestions.append("🎯 Split into fast + comprehensive versions")

    if "compilation" in test_name.lower():
        suggestions.append("⚡ Mock torch.compile for faster testing")
        suggestions.append("🔄 Cache compilation results")

    if "pattern" in test_name.lower():
        suggestions.append("📋 Test only essential patterns in fast version")
        suggestions.append("🧪 Use smaller test patterns")

    if duration > 5.0:
        suggestions.append("🏷️  Add @pytest.mark.slow decorator")
        suggestions.append("🔧 Consider mocking expensive operations")

    return suggestions

def main():
    """Main profiling script"""
    print("🔍 Test Performance Profiler")
    print("="*50)

    # Test files to profile
    test_files = [
        "tests/test_priority1_compiler_integration.py",
        "tests/test_next_gen_optimizations.py",
        "tests/test_advanced_optimizations.py"
    ]

    all_timings = {}

    for test_file in test_files:
        timings = profile_test_file(test_file)
        all_timings.update({f"{test_file}{k}": v for k, v in timings.items()})

        if timings:
            analyze_timings(timings)
            print()

    print("\n" + "="*60)
    print("🎯 OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    # Find slowest tests across all files
    if all_timings:
        slowest_tests = sorted(all_timings.items(), key=lambda x: x[1], reverse=True)[:3]

        for test_path, duration in slowest_tests:
            test_name = test_path.split("::")[-1]
            print(f"\n🐌 {test_name} ({duration:.3f}s):")

            suggestions = suggest_optimizations(test_name, duration)
            for suggestion in suggestions:
                print(f"   {suggestion}")

    print("\n✨ Run 'pytest -m \"not slow\"' for fast tests only!")

if __name__ == "__main__":
    main()