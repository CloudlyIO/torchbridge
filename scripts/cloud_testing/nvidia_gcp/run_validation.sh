#!/bin/bash
# =============================================================================
# NVIDIA Backend Validation - GCP (L4/A100)
# KernelPyTorch v0.3.7
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/utils.sh"

export WORK_DIR="${WORK_DIR:-$HOME/kernel_pytorch_test}"
export REPORT_DIR="$WORK_DIR/reports"
export BACKEND="nvidia"
export PLATFORM="gcp"

mkdir -p "$REPORT_DIR"

print_header "NVIDIA Backend Validation (GCP)"

# =============================================================================
# Setup
# =============================================================================
log_step "1/5" "Environment Setup"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR/kernel_pytorch:$PYTHONPATH"

# Check for NVIDIA GPU
if command_exists nvidia-smi; then
    log_success "NVIDIA driver detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    log_error "nvidia-smi not found"
    exit 1
fi

# Install dependencies
install_python_deps

# Verify PyTorch CUDA
if ! check_python_package torch; then
    log_warning "PyTorch not found, installing..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
fi

# =============================================================================
# GPU Info
# =============================================================================
log_step "2/5" "GPU Configuration"

get_gpu_info_json > "$REPORT_DIR/gpu_info.json"
print_gpu_info

# =============================================================================
# Tests
# =============================================================================
log_step "3/5" "Running NVIDIA Backend Tests"

warmup_gpu

run_pytest "tests/test_nvidia_backend.py" "$REPORT_DIR" "nvidia_test"
TEST_EXIT=$?

echo ""
log_info "Test Summary:"
parse_pytest_results "$REPORT_DIR/nvidia_test_results.json"

# =============================================================================
# Benchmarks
# =============================================================================
log_step "4/5" "Running NVIDIA Benchmarks"

# Integration benchmark
python3 benchmarks/nvidia_integration_benchmark.py 2>&1 | tee "$REPORT_DIR/nvidia_benchmark_output.txt"

# Config benchmark
python3 benchmarks/nvidia_config_benchmarks.py 2>&1 | tee -a "$REPORT_DIR/nvidia_benchmark_output.txt"

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
with open(f'{report_dir}/nvidia_test_results.json') as f:
    test_results = json.load(f)

summary = test_results.get('summary', {})

report = f"""# NVIDIA Backend Validation Report (GCP)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Platform:** GCP
**Backend:** NVIDIA

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

## Status

{"**PASSED** - All tests successful" if summary.get('failed', 0) == 0 else "**FAILED** - Review failed tests"}
"""

with open(f'{report_dir}/NVIDIA_GCP_REPORT.md', 'w') as f:
    f.write(report)

print(f"Report saved: {report_dir}/NVIDIA_GCP_REPORT.md")
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
