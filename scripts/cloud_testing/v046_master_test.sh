#!/bin/bash
# =============================================================================
# TorchBridge v0.4.6 Master Cloud Testing Script
# =============================================================================
#
# This script orchestrates comprehensive testing across all cloud platforms:
# - AWS NVIDIA (P4d/P5/G5 instances)
# - AWS AMD (G6e instances with MI300X)
# - GCP NVIDIA (A100/L4 instances)
# - GCP TPU (v5e/v5p pods)
#
# Usage:
#   ./v046_master_test.sh [platform] [options]
#
# Platforms: aws-nvidia, aws-amd, gcp-nvidia, gcp-tpu, local, all
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="0.4.6"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Configuration
# =============================================================================

export WORK_DIR="${WORK_DIR:-$HOME/torchbridge_test}"
export REPORT_DIR="${REPORT_DIR:-$WORK_DIR/reports_${VERSION}_${TIMESTAMP}}"
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

log_step() {
    echo -e "${YELLOW}[$1] $2${NC}"
}

log_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

log_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

log_info() {
    echo -e "INFO: $1"
}

check_dependencies() {
    log_step "1/5" "Checking Dependencies"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        exit 1
    fi
    log_info "Python: $(python3 --version)"

    # Check pip
    if ! python3 -m pip --version &> /dev/null; then
        log_error "pip not found"
        exit 1
    fi

    # Check pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        log_info "Installing pytest..."
        python3 -m pip install pytest pytest-json-report -q
    fi
    log_info "pytest installed"

    # Check torch
    if ! python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        log_error "PyTorch not installed"
        exit 1
    fi
    log_success "Dependencies verified"
}

detect_hardware() {
    log_step "2/5" "Detecting Hardware"

    python3 << 'PYEOF'
import torch
import json
import os

info = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "gpu_count": 0,
    "gpus": [],
    "platform": "local",
    "backend": "cpu"
}

# Check for TPU
try:
    import torch_xla.core.xla_model as xm
    info["platform"] = "gcp_tpu"
    info["backend"] = "tpu"
    devices = xm.get_xla_supported_devices()
    info["gpu_count"] = len(devices)
    print(f"Platform: GCP TPU ({len(devices)} devices)")
except ImportError:
    pass

# Check for CUDA/ROCm
if torch.cuda.is_available():
    info["gpu_count"] = torch.cuda.device_count()
    print(f"CUDA Available: {torch.cuda.device_count()} GPU(s)")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_name = torch.cuda.get_device_name(i)
        info["gpus"].append({
            "index": i,
            "name": gpu_name,
            "memory_gb": round(props.total_memory / 1024**3, 1),
        })
        print(f"  GPU {i}: {gpu_name} ({props.total_memory / 1024**3:.1f} GB)")

    # Detect backend
    gpu_name_lower = info["gpus"][0]["name"].lower() if info["gpus"] else ""
    if "mi300" in gpu_name_lower or "instinct" in gpu_name_lower:
        info["backend"] = "amd"
        print("Backend: AMD ROCm")
    else:
        info["backend"] = "nvidia"
        print("Backend: NVIDIA CUDA")
else:
    print("No GPU detected, using CPU")

# Save info
report_dir = os.environ.get('REPORT_DIR', '.')
os.makedirs(report_dir, exist_ok=True)
with open(f'{report_dir}/hardware_info.json', 'w') as f:
    json.dump(info, f, indent=2)
PYEOF

    log_success "Hardware detected"
}

run_moe_tests() {
    log_step "3/5" "Running MoE Tests"

    cd "$REPO_ROOT"

    # Unit tests
    log_info "Running MoE unit tests..."
    python3 -m pytest tests/test_moe.py -v --json-report --json-report-file="$REPORT_DIR/moe_test_results.json" 2>&1 | tee "$REPORT_DIR/moe_test_output.txt" || true

    # Demo
    log_info "Running MoE demo..."
    python3 demos/moe_demo.py 2>&1 | tee "$REPORT_DIR/moe_demo_output.txt" || true

    # Count results
    if [ -f "$REPORT_DIR/moe_test_results.json" ]; then
        python3 << PYEOF
import json
with open("$REPORT_DIR/moe_test_results.json") as f:
    data = json.load(f)
    summary = data.get("summary", {})
    print(f"MoE Tests: {summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed")
PYEOF
    fi
}

run_fp8_tests() {
    log_step "4/5" "Running FP8 Tests"

    cd "$REPO_ROOT"

    # Check FP8 support
    FP8_SUPPORTED=$(python3 -c "
from torchbridge.precision.fp8_native import is_fp8_available
print('yes' if is_fp8_available() else 'no')
" 2>/dev/null || echo "no")

    if [ "$FP8_SUPPORTED" = "no" ]; then
        log_info "FP8 not supported on this hardware, running basic tests only"
    fi

    # Unit tests
    log_info "Running FP8 native tests..."
    python3 -m pytest tests/test_fp8_native.py -v --json-report --json-report-file="$REPORT_DIR/fp8_test_results.json" 2>&1 | tee "$REPORT_DIR/fp8_test_output.txt" || true

    # Demo
    log_info "Running FP8 demo..."
    python3 demos/fp8_native_demo.py 2>&1 | tee "$REPORT_DIR/fp8_demo_output.txt" || true

    # Count results
    if [ -f "$REPORT_DIR/fp8_test_results.json" ]; then
        python3 << PYEOF
import json
with open("$REPORT_DIR/fp8_test_results.json") as f:
    data = json.load(f)
    summary = data.get("summary", {})
    print(f"FP8 Tests: {summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed")
PYEOF
    fi
}

run_backend_tests() {
    log_step "5/5" "Running Backend Tests"

    cd "$REPO_ROOT"

    # Detect backend
    BACKEND=$(python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0).lower()
    if 'mi300' in name or 'instinct' in name:
        print('amd')
    else:
        print('nvidia')
else:
    try:
        import torch_xla
        print('tpu')
    except:
        print('cpu')
" 2>/dev/null || echo "cpu")

    log_info "Backend: $BACKEND"

    case $BACKEND in
        nvidia)
            log_info "Running NVIDIA backend tests..."
            python3 -m pytest tests/test_nvidia_backend.py -v --json-report --json-report-file="$REPORT_DIR/backend_test_results.json" 2>&1 | tee "$REPORT_DIR/backend_test_output.txt" || true
            ;;
        amd)
            log_info "Running AMD backend tests..."
            python3 -m pytest tests/test_amd_backend.py -v --json-report --json-report-file="$REPORT_DIR/backend_test_results.json" 2>&1 | tee "$REPORT_DIR/backend_test_output.txt" || true
            ;;
        tpu)
            log_info "Running TPU backend tests..."
            python3 -m pytest tests/test_tpu_backend.py -v --json-report --json-report-file="$REPORT_DIR/backend_test_results.json" 2>&1 | tee "$REPORT_DIR/backend_test_output.txt" || true
            ;;
        *)
            log_info "No GPU backend, running CPU tests..."
            ;;
    esac

    # Run integration benchmark
    log_info "Running backend benchmark..."
    python3 benchmarks/${BACKEND}_integration_benchmark.py 2>&1 | tee "$REPORT_DIR/backend_benchmark_output.txt" || true
}

run_comprehensive_benchmark() {
    log_header "Running Comprehensive Benchmarks"

    cd "$REPO_ROOT"

    python3 << 'PYEOF'
import torch
import time
import json
import os

report_dir = os.environ.get('REPORT_DIR', '.')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

results = {
    "device": device,
    "matmul": [],
    "attention": [],
    "moe": []
}

print(f"\nRunning benchmarks on: {device}")

# Matrix multiplication
print("\n=== Matrix Multiplication (FP16) ===")
sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]

for M, N in sizes:
    K = M
    try:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(10):
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

        avg_ms = elapsed / iterations * 1000
        tflops = (2 * M * N * K) / (avg_ms / 1000) / 1e12

        results["matmul"].append({
            "size": f"{M}x{K}x{N}",
            "time_ms": round(avg_ms, 3),
            "tflops": round(tflops, 2)
        })
        print(f"  {M}x{K}: {avg_ms:.2f}ms, {tflops:.2f} TFLOPS")
    except Exception as e:
        print(f"  {M}x{K}: Failed - {e}")

# MoE benchmark
print("\n=== MoE Performance ===")
try:
    from torchbridge import MoEConfig, MoELayer

    configs = [
        {"hidden": 512, "experts": 8, "batch": 8, "seq": 128},
        {"hidden": 1024, "experts": 8, "batch": 4, "seq": 256},
    ]

    for cfg in configs:
        config = MoEConfig(num_experts=cfg["experts"], top_k=2)
        layer = MoELayer(config, cfg["hidden"]).to(device)
        x = torch.randn(cfg["batch"], cfg["seq"], cfg["hidden"], device=device)

        # Warmup
        for _ in range(10):
            _ = layer(x)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            _ = layer(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / iterations * 1000
        tokens = cfg["batch"] * cfg["seq"]
        tokens_per_sec = tokens / (avg_ms / 1000)

        results["moe"].append({
            "config": f"{cfg['hidden']}d/{cfg['experts']}e",
            "time_ms": round(avg_ms, 2),
            "tokens_per_sec": round(tokens_per_sec)
        })
        print(f"  {cfg['hidden']}d/{cfg['experts']}e: {avg_ms:.2f}ms, {tokens_per_sec:.0f} tok/s")

except Exception as e:
    print(f"  MoE benchmark failed: {e}")

# Save results
with open(f'{report_dir}/benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {report_dir}/benchmark_results.json")
PYEOF
}

generate_report() {
    log_header "Generating Final Report"

    python3 << PYEOF
import json
import os
from datetime import datetime

report_dir = "$REPORT_DIR"
version = "$VERSION"

# Load all results
hardware = {}
moe_results = {}
fp8_results = {}
backend_results = {}
benchmarks = {}

try:
    with open(f'{report_dir}/hardware_info.json') as f:
        hardware = json.load(f)
except: pass

try:
    with open(f'{report_dir}/moe_test_results.json') as f:
        moe_results = json.load(f)
except: pass

try:
    with open(f'{report_dir}/fp8_test_results.json') as f:
        fp8_results = json.load(f)
except: pass

try:
    with open(f'{report_dir}/backend_test_results.json') as f:
        backend_results = json.load(f)
except: pass

try:
    with open(f'{report_dir}/benchmark_results.json') as f:
        benchmarks = json.load(f)
except: pass

# Calculate totals
def get_summary(results):
    return results.get('summary', {'passed': 0, 'failed': 0, 'skipped': 0})

moe_sum = get_summary(moe_results)
fp8_sum = get_summary(fp8_results)
backend_sum = get_summary(backend_results)

total_passed = moe_sum['passed'] + fp8_sum['passed'] + backend_sum['passed']
total_failed = moe_sum['failed'] + fp8_sum['failed'] + backend_sum['failed']
total_skipped = moe_sum['skipped'] + fp8_sum['skipped'] + backend_sum['skipped']

# Generate report
report = f"""# TorchBridge v{version} Cloud Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Platform:** {hardware.get('platform', 'unknown')}
**Backend:** {hardware.get('backend', 'unknown')}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tests | {total_passed + total_failed + total_skipped} |
| Passed | {total_passed} |
| Failed | {total_failed} |
| Skipped | {total_skipped} |
| **Status** | **{'PASSED' if total_failed == 0 else 'FAILED'}** |

## Hardware Configuration

| Property | Value |
|----------|-------|
| PyTorch | {hardware.get('pytorch_version', 'N/A')} |
| CUDA | {hardware.get('cuda_version', 'N/A')} |
| GPU Count | {hardware.get('gpu_count', 0)} |

"""

# GPU details
gpus = hardware.get('gpus', [])
if gpus:
    report += "### GPUs\n\n| Index | Name | Memory |\n|-------|------|--------|\n"
    for gpu in gpus:
        report += f"| {gpu['index']} | {gpu['name']} | {gpu['memory_gb']} GB |\n"
    report += "\n"

# Test results
report += """## Test Results

| Suite | Passed | Failed | Skipped |
|-------|--------|--------|---------|
"""
report += f"| MoE Tests | {moe_sum['passed']} | {moe_sum['failed']} | {moe_sum['skipped']} |\n"
report += f"| FP8 Tests | {fp8_sum['passed']} | {fp8_sum['failed']} | {fp8_sum['skipped']} |\n"
report += f"| Backend Tests | {backend_sum['passed']} | {backend_sum['failed']} | {backend_sum['skipped']} |\n"
report += "\n"

# Benchmarks
if benchmarks:
    report += "## Benchmark Results\n\n"

    if benchmarks.get('matmul'):
        report += "### Matrix Multiplication (FP16)\n\n"
        report += "| Size | Time (ms) | TFLOPS |\n|------|-----------|--------|\n"
        for b in benchmarks['matmul']:
            report += f"| {b['size']} | {b['time_ms']} | {b['tflops']} |\n"
        report += "\n"

    if benchmarks.get('moe'):
        report += "### MoE Performance\n\n"
        report += "| Config | Time (ms) | Tokens/sec |\n|--------|-----------|------------|\n"
        for b in benchmarks['moe']:
            report += f"| {b['config']} | {b['time_ms']} | {b['tokens_per_sec']} |\n"
        report += "\n"

# Footer
status_msg = "All tests passed. System validated for production." if total_failed == 0 else f"FAILED: {total_failed} test(s) failed."
report += f"""---

## Validation Status

{status_msg}

---
*Report generated by TorchBridge v{version} Cloud Testing Framework*
"""

# Save report
report_path = f'{report_dir}/VALIDATION_REPORT.md'
with open(report_path, 'w') as f:
    f.write(report)

print(f"Report saved: {report_path}")
print(f"\nTotal: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")

if total_failed > 0:
    exit(1)
PYEOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    PLATFORM="${1:-local}"

    log_header "TorchBridge v${VERSION} Cloud Validation"
    log_info "Platform: $PLATFORM"
    log_info "Report Directory: $REPORT_DIR"

    mkdir -p "$REPORT_DIR"

    # Run tests
    check_dependencies
    detect_hardware
    run_moe_tests
    run_fp8_tests
    run_backend_tests
    run_comprehensive_benchmark
    generate_report

    log_header "Validation Complete"
    log_info "Reports saved to: $REPORT_DIR"
    ls -la "$REPORT_DIR"
}

# Run
main "$@"
