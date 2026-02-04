#!/bin/bash
# =============================================================================
# TPU Backend Validation - GCP (v5e/v5p)
# TorchBridge v0.3.7
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/utils.sh"

export WORK_DIR="${WORK_DIR:-$HOME/torchbridge_test}"
export REPORT_DIR="$WORK_DIR/reports"
export BACKEND="tpu"
export PLATFORM="gcp"

mkdir -p "$REPORT_DIR"

print_header "TPU Backend Validation (GCP)"

# =============================================================================
# Setup
# =============================================================================
log_step "1/5" "Environment Setup"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR/torchbridge:$PYTHONPATH"

# Install dependencies
install_python_deps

# Check/Install torch_xla
if ! check_python_package torch_xla; then
    log_warning "torch_xla not found, installing..."
    pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 \
        -f https://storage.googleapis.com/libtpu-releases/index.html -q
fi

# =============================================================================
# TPU Info
# =============================================================================
log_step "2/5" "TPU Configuration"

python3 << 'PYEOF'
import json
import os

tpu_info = {
    "tpu_available": False,
    "pytorch_version": "",
    "torch_xla_version": "",
    "device": ""
}

try:
    import torch
    tpu_info["pytorch_version"] = torch.__version__

    import torch_xla
    tpu_info["torch_xla_version"] = torch_xla.__version__

    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    tpu_info["tpu_available"] = True
    tpu_info["device"] = str(device)
    print(f"TPU Available: True")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch_xla: {torch_xla.__version__}")
except Exception as e:
    print(f"TPU check failed: {e}")
    tpu_info["error"] = str(e)

report_dir = os.environ.get('REPORT_DIR', '.')
with open(f'{report_dir}/tpu_info.json', 'w') as f:
    json.dump(tpu_info, f, indent=2)
PYEOF

# =============================================================================
# Tests
# =============================================================================
log_step "3/5" "Running TPU Backend Tests"

run_pytest "tests/test_tpu_backend.py" "$REPORT_DIR" "tpu_test"
TEST_EXIT=$?

echo ""
log_info "Test Summary:"
parse_pytest_results "$REPORT_DIR/tpu_test_results.json"

# =============================================================================
# Benchmarks
# =============================================================================
log_step "4/5" "Running TPU Benchmarks"

python3 benchmarks/tpu_integration_benchmark.py 2>&1 | tee "$REPORT_DIR/tpu_benchmark_output.txt"

# =============================================================================
# Report
# =============================================================================
log_step "5/5" "Generating Report"

python3 << PYEOF
import json
from datetime import datetime

report_dir = "$REPORT_DIR"

# Load data
with open(f'{report_dir}/tpu_info.json') as f:
    tpu_info = json.load(f)
with open(f'{report_dir}/tpu_test_results.json') as f:
    test_results = json.load(f)

summary = test_results.get('summary', {})

report = f"""# TPU Backend Validation Report (GCP)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Platform:** GCP
**Backend:** TPU

## Summary

| Metric | Value |
|--------|-------|
| Tests Passed | {summary.get('passed', 0)}/{summary.get('total', 0)} |
| Tests Failed | {summary.get('failed', 0)} |
| Tests Skipped | {summary.get('skipped', 0)} |
| Duration | {summary.get('duration', 0):.2f}s |

## TPU Configuration

| Property | Value |
|----------|-------|
| TPU Available | {tpu_info.get('tpu_available', False)} |
| Device | {tpu_info.get('device', 'N/A')} |
| PyTorch | {tpu_info.get('pytorch_version', 'N/A')} |
| torch_xla | {tpu_info.get('torch_xla_version', 'N/A')} |

## Status

{"**PASSED** - All tests successful" if summary.get('failed', 0) == 0 else "**FAILED** - Review failed tests"}
"""

with open(f'{report_dir}/TPU_GCP_REPORT.md', 'w') as f:
    f.write(report)

print(f"Report saved: {report_dir}/TPU_GCP_REPORT.md")
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
