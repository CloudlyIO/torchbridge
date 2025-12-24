#!/usr/bin/env python3
"""
Test Execution Manager

Provides convenient commands for running different categories of tests
with appropriate configurations and reporting.
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return timing/results"""
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=False, capture_output=False, text=True)
        duration = time.time() - start_time

        print(f"\n⏱️  Completed in {duration:.2f}s")
        print(f"📊 Exit code: {result.returncode}")

        return result.returncode == 0, duration

    except KeyboardInterrupt:
        print("\n❌ Test run interrupted by user")
        return False, time.time() - start_time
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return False, time.time() - start_time

def run_unit_tests():
    """Run fast unit tests for development"""
    cmd = [
        "python3", "-m", "pytest",
        "tests/",
        "-m", "not (integration or stress or slow or gpu)",
        "-v", "--tb=short",
        "--durations=10"  # Show 10 slowest tests
    ]

    success, duration = run_command(cmd, "Running unit tests (fast, basic functionality)")

    if success:
        print("✅ All unit tests passed!")
    else:
        print("❌ Some unit tests failed")

    return success, duration

def run_integration_tests():
    """Run integration tests with realistic data"""
    cmd = [
        "python3", "-m", "pytest",
        "tests/",
        "-m", "integration",
        "-v", "--tb=short",
        "--durations=10"
    ]

    success, duration = run_command(cmd, "Running integration tests (realistic scale)")

    if success:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed")

    return success, duration

def run_stress_tests():
    """Run stress tests with large-scale data"""
    cmd = [
        "python3", "-m", "pytest",
        "tests/",
        "-m", "stress",
        "-v", "--tb=short",
        "-s",  # Don't capture output for stress tests
        "--durations=0"  # Show all test durations
    ]

    success, duration = run_command(cmd, "Running stress tests (large scale, performance)")

    if success:
        print("✅ All stress tests passed!")
    else:
        print("❌ Some stress tests failed")

    return success, duration

def run_comprehensive_suite():
    """Run the full comprehensive test suite"""
    cmd = [
        "python3", "-m", "pytest",
        "tests/test_comprehensive_integration.py",
        "-v", "--tb=short",
        "-s",
        "--durations=0"
    ]

    success, duration = run_command(cmd, "Running comprehensive test suite (full validation)")

    if success:
        print("✅ Comprehensive test suite passed!")
    else:
        print("❌ Comprehensive test suite failed")

    return success, duration

def run_performance_analysis():
    """Run tests with performance analysis"""
    cmd = [
        "python3", "-m", "pytest",
        "tests/",
        "-v", "--tb=short",
        "--durations=0",
        "--benchmark-disable",  # Disable benchmarking if installed
        "-m", "not (gpu or distributed)"  # Skip hardware-dependent tests
    ]

    success, duration = run_command(cmd, "Running performance analysis")

    if success:
        print("✅ Performance analysis completed!")
    else:
        print("❌ Performance analysis failed")

    return success, duration

def run_ci_pipeline():
    """Run tests suitable for CI pipeline"""
    print("🔄 Running CI Pipeline Test Sequence")
    print("=" * 60)

    total_start = time.time()
    results = {}

    # Stage 1: Unit Tests
    print("\n📝 Stage 1: Unit Tests")
    success, duration = run_unit_tests()
    results['unit'] = {'success': success, 'duration': duration}

    if not success:
        print("❌ Unit tests failed - stopping CI pipeline")
        return False

    # Stage 2: Integration Tests
    print("\n🔗 Stage 2: Integration Tests")
    success, duration = run_integration_tests()
    results['integration'] = {'success': success, 'duration': duration}

    if not success:
        print("❌ Integration tests failed - continuing with warnings")

    total_duration = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("📊 CI PIPELINE SUMMARY")
    print("=" * 60)

    for stage, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{stage.capitalize():15} {status:8} {result['duration']:8.2f}s")

    print(f"{'Total':15} {'':8} {total_duration:8.2f}s")

    all_passed = all(r['success'] for r in results.values())

    if all_passed:
        print("\n🎉 CI Pipeline completed successfully!")
    else:
        print("\n⚠️ CI Pipeline completed with some failures")

    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Test Execution Manager")
    parser.add_argument("command", choices=[
        "unit", "integration", "stress", "comprehensive",
        "performance", "ci", "all"
    ], help="Type of tests to run")

    parser.add_argument("--env", choices=["local", "ci", "gpu"],
                        default="local", help="Test environment")

    args = parser.parse_args()

    # Set environment variables
    env_vars = {
        "PYTHONPATH": "src"
    }

    if args.env == "gpu":
        env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    # Apply environment variables
    for key, value in env_vars.items():
        import os
        os.environ[key] = value

    print(f"🧪 Test Execution Manager - {args.command.upper()} mode")
    print(f"🌍 Environment: {args.env}")
    print(f"📍 Working directory: {Path.cwd()}")
    print()

    # Route to appropriate test function
    if args.command == "unit":
        success, _ = run_unit_tests()
    elif args.command == "integration":
        success, _ = run_integration_tests()
    elif args.command == "stress":
        success, _ = run_stress_tests()
    elif args.command == "comprehensive":
        success, _ = run_comprehensive_suite()
    elif args.command == "performance":
        success, _ = run_performance_analysis()
    elif args.command == "ci":
        success = run_ci_pipeline()
    elif args.command == "all":
        print("🎯 Running ALL test categories...")
        success = True
        for test_type in ["unit", "integration", "stress"]:
            print(f"\n{'='*20} {test_type.upper()} TESTS {'='*20}")
            if test_type == "unit":
                test_success, _ = run_unit_tests()
            elif test_type == "integration":
                test_success, _ = run_integration_tests()
            elif test_type == "stress":
                test_success, _ = run_stress_tests()

            if not test_success:
                success = False
                print(f"❌ {test_type} tests failed")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()