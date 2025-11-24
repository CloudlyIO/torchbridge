#!/usr/bin/env python3
"""
Test Performance Profiler

This script helps identify slow tests in the current codebase and suggests optimizations.
Profiles tests from the actual test suite to identify performance bottlenecks.
"""

import subprocess
import time
import sys
import re
from typing import List, Dict, Tuple
from pathlib import Path


def profile_test_file(test_file: str) -> Dict[str, float]:
    """Profile all tests in a test file and return timing results"""
    print(f"📊 Profiling {test_file}...")

    # Check if test file exists
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return {}

    # Get list of tests in the file
    try:
        result = subprocess.run([
            'python3', '-m', 'pytest', test_file, '--collect-only', '-q'
        ], env={'PYTHONPATH': 'src'}, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"❌ Failed to collect tests from {test_file}")
            print(f"   Error: {result.stderr}")
            return {}

    except Exception as e:
        print(f"❌ Error collecting tests: {e}")
        return {}

    # Extract test names
    test_names = re.findall(r'::test_\w+', result.stdout)
    test_timings = {}

    print(f"Found {len(test_names)} tests")

    # Profile each test individually (limit to prevent timeouts)
    max_tests = min(len(test_names), 10)  # Profile at most 10 tests per file

    for i, test_name in enumerate(test_names[:max_tests], 1):
        test_path = f"{test_file}{test_name}"
        print(f"  [{i}/{max_tests}] Testing {test_name}...")

        start_time = time.time()
        try:
            result = subprocess.run([
                'python3', '-m', 'pytest', test_path, '-v', '--tb=no'
            ], env={'PYTHONPATH': 'src'}, capture_output=True, text=True, timeout=120)

            duration = time.time() - start_time
            test_timings[test_name] = duration

            if result.returncode == 0:
                status = "✅ PASS"
            elif "SKIPPED" in result.stdout:
                status = "⚠️ SKIP"
            else:
                status = "❌ FAIL"

            print(f"    {status} {duration:.3f}s")

        except subprocess.TimeoutExpired:
            duration = 120.0
            test_timings[test_name] = duration
            print(f"    ⏱️ TIMEOUT {duration:.3f}s")
        except Exception as e:
            print(f"    🚨 ERROR: {e}")

    return test_timings


def analyze_timings(timings: Dict[str, float], file_name: str) -> None:
    """Analyze test timings and provide recommendations"""
    if not timings:
        print(f"No timing data available for {file_name}")
        return

    print(f"\n" + "="*60)
    print(f"📈 PERFORMANCE ANALYSIS - {file_name}")
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

        if duration > 10.0:
            priority = "🔴 HIGH"
        elif duration > 3.0:
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"

        print(f"  {priority:12} {test_name:50} {duration:8.3f}s ({percentage:5.1f}%)")


def suggest_optimizations(test_name: str, duration: float) -> List[str]:
    """Suggest optimizations based on test name and duration"""
    suggestions = []

    if duration > 30.0:
        suggestions.append("🚀 Critical: Reduce problem size dramatically")
        suggestions.append("🔄 Consider mocking heavy components")

    if duration > 10.0:
        suggestions.append("🎯 Split into fast + comprehensive versions")
        suggestions.append("📦 Use smaller tensor dimensions")

    if "integration" in test_name.lower():
        suggestions.append("⚡ Mock expensive integration points")
        suggestions.append("🔄 Cache setup/teardown operations")

    if "compiler" in test_name.lower() or "optimization" in test_name.lower():
        suggestions.append("⚡ Mock torch.compile for unit tests")
        suggestions.append("🔄 Cache compilation results")

    if "attention" in test_name.lower():
        suggestions.append("📋 Use minimal sequence lengths (seq_len=32)")
        suggestions.append("🧪 Test with smaller embedding dimensions")

    if "benchmark" in test_name.lower():
        suggestions.append("🏷️ Add @pytest.mark.slow decorator")
        suggestions.append("⚡ Create separate fast benchmark version")

    if duration > 5.0:
        suggestions.append("🏷️ Consider @pytest.mark.slow decorator")
        suggestions.append("🔧 Mock expensive external dependencies")

    if not suggestions:
        suggestions.append("✨ Performance looks good!")

    return suggestions


def get_current_test_files() -> List[str]:
    """Get list of current test files in the repository"""
    test_files = []

    # Find all test files in the tests directory
    tests_dir = Path("tests")
    if tests_dir.exists():
        for test_file in tests_dir.glob("test_*.py"):
            test_files.append(str(test_file))

    return test_files


def main():
    """Main profiling script"""
    print("🔍 Test Performance Profiler")
    print("="*50)
    print("Profiling current test suite for performance optimization\n")

    # Get current test files
    test_files = get_current_test_files()

    if not test_files:
        print("❌ No test files found in tests/ directory")
        return

    print(f"Found {len(test_files)} test files to profile:")
    for test_file in test_files:
        print(f"  • {test_file}")
    print()

    all_timings = {}
    file_summaries = {}

    # Profile each test file
    for test_file in test_files:
        timings = profile_test_file(test_file)

        if timings:
            all_timings.update({f"{test_file}{k}": v for k, v in timings.items()})
            file_summaries[test_file] = timings
            analyze_timings(timings, test_file)
            print()

    # Global analysis
    print("\n" + "="*60)
    print("🎯 GLOBAL OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    if all_timings:
        # Find slowest tests across all files
        slowest_tests = sorted(all_timings.items(), key=lambda x: x[1], reverse=True)[:5]

        print("🐌 TOP 5 SLOWEST TESTS ACROSS ALL FILES:")
        for i, (test_path, duration) in enumerate(slowest_tests, 1):
            file_name = test_path.split("::")[0]
            test_name = test_path.split("::")[-1]
            print(f"\n{i}. {test_name} in {file_name} ({duration:.3f}s):")

            suggestions = suggest_optimizations(test_name, duration)
            for suggestion in suggestions:
                print(f"   {suggestion}")

    # Summary statistics
    if file_summaries:
        total_tests = sum(len(timings) for timings in file_summaries.values())
        total_time = sum(sum(timings.values()) for timings in file_summaries.values())

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total tests profiled: {total_tests}")
        print(f"   Total execution time: {total_time:.3f}s")
        print(f"   Average test time: {total_time/total_tests:.3f}s")

        # Count slow tests
        slow_tests = sum(1 for timing in all_timings.values() if timing > 5.0)
        very_slow_tests = sum(1 for timing in all_timings.values() if timing > 10.0)

        print(f"   Tests > 5s: {slow_tests}")
        print(f"   Tests > 10s: {very_slow_tests}")

    print(f"\n💡 OPTIMIZATION TIPS:")
    print(f"   • Run 'pytest -m \"not slow\"' to skip slow tests")
    print(f"   • Use 'pytest --durations=10' to see slowest tests")
    print(f"   • Add @pytest.mark.slow to tests > 5s")
    print(f"   • Mock expensive operations in unit tests")
    print(f"   • Use smaller dimensions for faster testing")


if __name__ == "__main__":
    main()