#!/bin/bash
# =============================================================================
# AMD Backend Validation - AMD Developer Cloud (MI300X)
# KernelPyTorch v0.3.7
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/utils.sh"

export WORK_DIR="${WORK_DIR:-$HOME/kernel_pytorch_test}"
export REPORT_DIR="$WORK_DIR/reports"
export BACKEND="amd"
export PLATFORM="amd_cloud"

mkdir -p "$REPORT_DIR"

print_header "AMD Backend Validation (MI300X)"

# =============================================================================
# Setup
# =============================================================================
log_step "1/5" "Environment Setup"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR/kernel_pytorch:$PYTHONPATH"

# Check for AMD GPU via ROCm
if command_exists rocm-smi; then
    log_success "ROCm detected"
    rocm-smi --showproductname
else
    log_warning "rocm-smi not found - ROCm may not be installed"
fi

# Check ROCm version
if [ -f /opt/rocm/.info/version ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version)
    log_info "ROCm Version: $ROCM_VERSION"
fi

# Install dependencies
install_python_deps

# Verify PyTorch with ROCm
if ! check_python_package torch; then
    log_warning "PyTorch not found, installing with ROCm support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0 -q
fi

# =============================================================================
# GPU Info
# =============================================================================
log_step "2/5" "GPU Configuration"

python3 << 'PYEOF'
import torch
import json
import os

gpu_info = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),  # ROCm uses CUDA API
    "gpu_count": 0,
    "gpus": [],
    "backend": "rocm"
}

if torch.cuda.is_available():
    gpu_info["gpu_count"] = torch.cuda.device_count()
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info["gpus"].append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_gb": round(props.total_memory / 1024**3, 1),
            "compute_capability": f"{props.major}.{props.minor}"
        })
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: PyTorch cannot access AMD GPU via ROCm/HIP")

report_dir = os.environ.get('REPORT_DIR', '.')
with open(f'{report_dir}/gpu_info.json', 'w') as f:
    json.dump(gpu_info, f, indent=2)
PYEOF

# =============================================================================
# Tests
# =============================================================================
log_step "3/5" "Running AMD Backend Tests"

warmup_gpu

run_pytest "tests/test_amd_backend.py" "$REPORT_DIR" "amd_test"
TEST_EXIT=$?

echo ""
log_info "Test Summary:"
parse_pytest_results "$REPORT_DIR/amd_test_results.json"

# =============================================================================
# Benchmarks
# =============================================================================
log_step "4/5" "Running AMD Benchmarks"

# Integration benchmark
python3 benchmarks/amd_integration_benchmark.py 2>&1 | tee "$REPORT_DIR/amd_benchmark_output.txt"

# Additional performance tests
python3 << 'PYEOF'
import torch
import time
import json
import os

results = {"matrix_operations": [], "memory_bandwidth": []}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running performance tests on: {device}")

# Matrix multiplication benchmarks
print("\nMatrix Multiplication (FP16):")
sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]

for M, N in sizes:
    K = M
    try:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            c = torch.matmul(a, b)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations * 1000
        tflops = (2 * M * N * K) / (avg_time / 1000) / 1e12

        print(f"  {M}x{K} @ {K}x{N}: {avg_time:.3f}ms, {tflops:.2f} TFLOPS")
        results["matrix_operations"].append({
            "size": f"{M}x{K}x{N}",
            "time_ms": avg_time,
            "tflops": tflops
        })
    except Exception as e:
        print(f"  {M}x{K}: Failed - {e}")

report_dir = os.environ.get('REPORT_DIR', '.')
with open(f'{report_dir}/amd_perf_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nPerformance results saved")
PYEOF

# =============================================================================
# Report
# =============================================================================
log_step "5/5" "Generating Report"

python3 << PYEOF
import json
from datetime import datetime

report_dir = "$REPORT_DIR"

# Load data
with open(f'{report_dir}/gpu_info.json') as f:
    gpu_info = json.load(f)
with open(f'{report_dir}/amd_test_results.json') as f:
    test_results = json.load(f)

summary = test_results.get('summary', {})

# Load perf results if available
perf_results = {}
try:
    with open(f'{report_dir}/amd_perf_results.json') as f:
        perf_results = json.load(f)
except:
    pass

report = f"""# AMD Backend Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Platform:** AMD Developer Cloud
**Backend:** AMD ROCm

## Summary

| Metric | Value |
|--------|-------|
| Tests Passed | {summary.get('passed', 0)}/{summary.get('total', 0)} |
| Tests Failed | {summary.get('failed', 0)} |
| Tests Skipped | {summary.get('skipped', 0)} |
| Duration | {summary.get('duration', 0):.2f}s |

## GPU Configuration

| Property | Value |
|----------|-------|
| GPU | {gpu_info['gpus'][0]['name'] if gpu_info.get('gpus') else 'N/A'} |
| Memory | {gpu_info['gpus'][0]['memory_gb'] if gpu_info.get('gpus') else 'N/A'} GB |
| PyTorch | {gpu_info.get('pytorch_version', 'N/A')} |
| Backend | ROCm/HIP |

"""

# Add performance results if available
if perf_results.get('matrix_operations'):
    report += """## Performance Results

### Matrix Operations (FP16)

| Size | Time (ms) | TFLOPS |
|------|-----------|--------|
"""
    for op in perf_results['matrix_operations']:
        report += f"| {op['size']} | {op['time_ms']:.3f} | {op['tflops']:.2f} |\n"

report += f"""
## Status

{"**PASSED** - All tests successful" if summary.get('failed', 0) == 0 else "**FAILED** - Review failed tests"}
"""

with open(f'{report_dir}/AMD_CLOUD_REPORT.md', 'w') as f:
    f.write(report)

print(f"Report saved: {report_dir}/AMD_CLOUD_REPORT.md")
PYEOF

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Validation Complete!"
echo "=============================================="
echo ""
echo "Reports: $REPORT_DIR"
ls -la "$REPORT_DIR"

if [ $TEST_EXIT -eq 0 ]; then
    log_success "All tests passed!"
else
    log_error "Some tests failed (exit code: $TEST_EXIT)"
fi

exit $TEST_EXIT
