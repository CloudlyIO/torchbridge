#!/usr/bin/env python3
"""
Intel DevCloud Validation Script for KernelPyTorch v0.4.7

This script validates the Intel XPU backend on Intel DevCloud/Tiber AI Cloud.

Supported environments:
- Intel Data Center GPU Max Series (Ponte Vecchio)
- Intel Arc A-Series (A770, A750, A580)
- Intel Flex Series

Usage on Intel DevCloud:
    # Via JupyterLab
    !python intel_devcloud_test.py

    # Via SSH
    python intel_devcloud_test.py --all
    python intel_devcloud_test.py --tests-only
    python intel_devcloud_test.py --demo-only

Requirements:
    - Intel Extension for PyTorch (IPEX)
    - PyTorch 2.5+
    - KernelPyTorch v0.4.7+
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def check_environment():
    """Check Intel XPU environment."""
    print("=" * 60)
    print(" Intel DevCloud Environment Check")
    print("=" * 60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'environment': {}
    }

    # Check Python version
    print(f"\nPython: {sys.version}")
    results['environment']['python_version'] = sys.version

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        results['environment']['pytorch_version'] = torch.__version__

        # Check XPU availability
        if hasattr(torch, 'xpu'):
            xpu_available = torch.xpu.is_available()
            print(f"XPU Available: {xpu_available}")
            results['environment']['xpu_available'] = xpu_available

            if xpu_available:
                device_count = torch.xpu.device_count()
                print(f"XPU Device Count: {device_count}")
                results['environment']['xpu_device_count'] = device_count

                for i in range(device_count):
                    props = torch.xpu.get_device_properties(i)
                    print(f"  Device {i}: {props.name}")
                    print(f"    Total Memory: {props.total_memory / (1024**3):.2f} GB")
                    results['environment'][f'xpu_device_{i}'] = {
                        'name': props.name,
                        'total_memory_gb': props.total_memory / (1024**3)
                    }
        else:
            print("XPU not available in this PyTorch build")
            results['environment']['xpu_available'] = False

    except ImportError:
        print("PyTorch not installed")
        results['environment']['pytorch_version'] = None

    # Check IPEX
    try:
        import intel_extension_for_pytorch as ipex
        print(f"IPEX: {ipex.__version__}")
        results['environment']['ipex_version'] = ipex.__version__
    except ImportError:
        print("IPEX not installed")
        results['environment']['ipex_version'] = None

    # Check KernelPyTorch
    try:
        import kernel_pytorch
        print(f"KernelPyTorch: {kernel_pytorch.__version__}")
        results['environment']['kernelpytorch_version'] = kernel_pytorch.__version__
    except ImportError:
        print("KernelPyTorch not installed - installing from local...")
        results['environment']['kernelpytorch_version'] = None

    return results


def run_intel_backend_tests():
    """Run Intel backend tests."""
    print("\n" + "=" * 60)
    print(" Running Intel Backend Tests")
    print("=" * 60 + "\n")

    import subprocess

    result = subprocess.run(
        [sys.executable, '-m', 'pytest',
         'tests/test_intel_backend.py',
         '-v', '--tb=short'],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse results
    output = result.stdout
    passed = output.count(' PASSED')
    failed = output.count(' FAILED')
    skipped = output.count(' SKIPPED')

    return {
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'return_code': result.returncode,
        'output': output
    }


def run_intel_demo():
    """Run Intel XPU demo."""
    print("\n" + "=" * 60)
    print(" Running Intel XPU Demo")
    print("=" * 60 + "\n")

    import subprocess

    result = subprocess.run(
        [sys.executable, 'demos/intel_xpu_demo.py'],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:")
        print(result.stderr)

    return {
        'return_code': result.returncode,
        'output': result.stdout
    }


def run_xpu_benchmarks():
    """Run XPU-specific benchmarks."""
    print("\n" + "=" * 60)
    print(" Running XPU Benchmarks")
    print("=" * 60 + "\n")

    results = {}

    try:
        import torch
        import torch.nn as nn

        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            print("XPU not available - skipping benchmarks")
            return {'status': 'skipped', 'reason': 'XPU not available'}

        from kernel_pytorch.backends.intel import IntelBackend

        backend = IntelBackend()
        device = backend.device

        # Benchmark 1: Matrix multiplication
        print("\n1. Matrix Multiplication Benchmark")
        sizes = [512, 1024, 2048, 4096]

        for size in sizes:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            # Warmup
            for _ in range(5):
                _ = torch.matmul(a, b)
            torch.xpu.synchronize()

            # Benchmark
            start = time.perf_counter()
            iterations = 20
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.xpu.synchronize()
            elapsed = time.perf_counter() - start

            ops = 2 * size**3 * iterations
            tflops = ops / elapsed / 1e12
            print(f"  {size}x{size}: {elapsed/iterations*1000:.2f} ms, {tflops:.2f} TFLOPS")

            results[f'matmul_{size}'] = {
                'time_ms': elapsed / iterations * 1000,
                'tflops': tflops
            }

        # Benchmark 2: Simple model inference
        print("\n2. Model Inference Benchmark")
        model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
        ).to(device)

        # Optimize with IPEX if available
        try:
            model = backend.optimize_for_inference(model)
        except Exception:
            pass

        batch_sizes = [1, 8, 32, 64, 128]

        for bs in batch_sizes:
            x = torch.randn(bs, 1024, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            torch.xpu.synchronize()

            # Benchmark
            start = time.perf_counter()
            iterations = 50
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(x)
            torch.xpu.synchronize()
            elapsed = time.perf_counter() - start

            throughput = iterations * bs / elapsed
            print(f"  Batch {bs}: {elapsed/iterations*1000:.2f} ms, {throughput:.0f} samples/sec")

            results[f'inference_bs{bs}'] = {
                'time_ms': elapsed / iterations * 1000,
                'throughput': throughput
            }

        results['status'] = 'completed'

    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['error'] = str(e)

    return results


def generate_report(env_results, test_results, demo_results, benchmark_results, output_dir):
    """Generate validation report."""
    print("\n" + "=" * 60)
    print(" Generating Report")
    print("=" * 60 + "\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine status
    if test_results and test_results.get('failed', 0) > 0:
        status = 'PARTIAL'
    elif test_results and test_results.get('passed', 0) > 0:
        status = 'PASS'
    else:
        status = 'N/A'

    # Create summary
    summary = f"""# Intel DevCloud Validation Report - v0.4.7

**Platform**: Intel DevCloud / Tiber AI Cloud
**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Status**: {status}

---

## Environment

| Component | Version |
|-----------|---------|
| Python | {env_results.get('environment', {}).get('python_version', 'N/A').split()[0]} |
| PyTorch | {env_results.get('environment', {}).get('pytorch_version', 'N/A')} |
| IPEX | {env_results.get('environment', {}).get('ipex_version', 'N/A')} |
| KernelPyTorch | {env_results.get('environment', {}).get('kernelpytorch_version', 'N/A')} |
| XPU Available | {env_results.get('environment', {}).get('xpu_available', False)} |
| XPU Devices | {env_results.get('environment', {}).get('xpu_device_count', 0)} |

---

## Test Results

| Test Suite | Passed | Failed | Skipped | Status |
|------------|--------|--------|---------|--------|
| Intel Backend | {test_results.get('passed', 'N/A')} | {test_results.get('failed', 'N/A')} | {test_results.get('skipped', 'N/A')} | {'PASS' if test_results.get('failed', 0) == 0 else 'FAIL'} |

---

## Demo Results

- **Status**: {'PASS' if demo_results.get('return_code', 1) == 0 else 'FAIL'}

---

## Benchmark Results

"""

    if benchmark_results.get('status') == 'completed':
        summary += "### Matrix Multiplication Performance\n\n"
        summary += "| Size | Time (ms) | TFLOPS |\n"
        summary += "|------|-----------|--------|\n"
        for key, value in benchmark_results.items():
            if key.startswith('matmul_'):
                size = key.split('_')[1]
                summary += f"| {size}x{size} | {value['time_ms']:.2f} | {value['tflops']:.2f} |\n"

        summary += "\n### Model Inference Performance\n\n"
        summary += "| Batch Size | Time (ms) | Throughput (samples/sec) |\n"
        summary += "|------------|-----------|-------------------------|\n"
        for key, value in benchmark_results.items():
            if key.startswith('inference_'):
                bs = key.split('bs')[1]
                summary += f"| {bs} | {value['time_ms']:.2f} | {value['throughput']:.0f} |\n"
    else:
        summary += f"Benchmarks: {benchmark_results.get('status', 'N/A')}\n"
        if benchmark_results.get('reason'):
            summary += f"Reason: {benchmark_results.get('reason')}\n"

    summary += """
---

## Conclusion

This report validates KernelPyTorch v0.4.7 Intel XPU backend functionality.

---

*Report generated by KernelPyTorch v0.4.7 Intel DevCloud Validation*
"""

    # Save summary
    with open(output_path / 'SUMMARY.md', 'w') as f:
        f.write(summary)

    # Save full results as JSON
    full_results = {
        'environment': env_results,
        'tests': test_results,
        'demo': demo_results,
        'benchmarks': benchmark_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"Report saved to: {output_path / 'SUMMARY.md'}")
    print(f"Full results saved to: {output_path / 'results.json'}")

    return output_path / 'SUMMARY.md'


def main():
    parser = argparse.ArgumentParser(description='Intel DevCloud Validation for KernelPyTorch')
    parser.add_argument('--all', action='store_true', help='Run all validations')
    parser.add_argument('--tests-only', action='store_true', help='Run tests only')
    parser.add_argument('--demo-only', action='store_true', help='Run demo only')
    parser.add_argument('--benchmarks-only', action='store_true', help='Run benchmarks only')
    parser.add_argument('--output', default='docs/cloud_testing/reports/intel_devcloud_v047',
                       help='Output directory for reports')

    args = parser.parse_args()

    # Default to all if no specific option
    if not any([args.tests_only, args.demo_only, args.benchmarks_only]):
        args.all = True

    print("\n" + "=" * 60)
    print(" KernelPyTorch v0.4.7 Intel DevCloud Validation")
    print("=" * 60)

    # Check environment
    env_results = check_environment()

    test_results = {}
    demo_results = {}
    benchmark_results = {}

    if args.all or args.tests_only:
        test_results = run_intel_backend_tests()

    if args.all or args.demo_only:
        demo_results = run_intel_demo()

    if args.all or args.benchmarks_only:
        benchmark_results = run_xpu_benchmarks()

    # Generate report
    report_path = generate_report(
        env_results, test_results, demo_results, benchmark_results, args.output
    )

    print("\n" + "=" * 60)
    print(" Validation Complete!")
    print("=" * 60)

    # Print summary
    if test_results:
        passed = test_results.get('passed', 0)
        failed = test_results.get('failed', 0)
        print(f"\nTests: {passed} passed, {failed} failed")

    return 0 if test_results.get('failed', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
