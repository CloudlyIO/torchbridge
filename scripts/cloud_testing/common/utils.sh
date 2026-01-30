#!/bin/bash
# =============================================================================
# Common Utilities for Cloud Testing Scripts
# TorchBridge v0.3.7
# =============================================================================

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export CYAN='\033[0;36m'
export NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}[$1]${NC} $2"
    echo "----------------------------------------"
}

# Print header
print_header() {
    local title="$1"
    local version="${2:-0.3.7}"
    echo "=============================================="
    echo "  $title"
    echo "  TorchBridge v$version"
    echo "  $(date)"
    echo "=============================================="
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check Python package
check_python_package() {
    python3 -c "import $1" 2>/dev/null
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    pip install --upgrade pip -q
    pip install pytest pytest-json-report psutil numpy -q
    log_success "Python dependencies installed"
}

# Get GPU info as JSON
get_gpu_info_json() {
    python3 << 'PYEOF'
import torch
import json

gpu_info = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": 0,
    "gpus": []
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

print(json.dumps(gpu_info, indent=2))
PYEOF
}

# Print GPU info
print_gpu_info() {
    python3 << 'PYEOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("No GPU available")
PYEOF
}

# Run pytest with JSON report
run_pytest() {
    local test_file="$1"
    local report_dir="$2"
    local report_name="$3"

    python3 -m pytest "$test_file" \
        -v \
        --tb=short \
        --json-report \
        --json-report-file="$report_dir/${report_name}_results.json" \
        2>&1 | tee "$report_dir/${report_name}_output.txt"

    return ${PIPESTATUS[0]}
}

# Parse pytest JSON results
parse_pytest_results() {
    local json_file="$1"

    python3 << PYEOF
import json
try:
    with open('$json_file', 'r') as f:
        results = json.load(f)
    summary = results.get('summary', {})
    print(f"Total:   {summary.get('total', 0)}")
    print(f"Passed:  {summary.get('passed', 0)}")
    print(f"Failed:  {summary.get('failed', 0)}")
    print(f"Skipped: {summary.get('skipped', 0)}")
    print(f"Duration: {summary.get('duration', 0):.2f}s")
except Exception as e:
    print(f"Could not parse results: {e}")
PYEOF
}

# Create tarball of reports
package_reports() {
    local report_dir="$1"
    local output_name="$2"

    tar -czf "$output_name" -C "$report_dir" .
    log_success "Reports packaged: $output_name"
}

# Warmup GPU
warmup_gpu() {
    python3 << 'PYEOF'
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    for _ in range(10):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("GPU warmup complete")
else:
    print("No GPU - skipping warmup")
PYEOF
}
