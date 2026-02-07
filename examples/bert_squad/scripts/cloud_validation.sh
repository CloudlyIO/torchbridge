#!/bin/bash
# =============================================================================
# BERT SQuAD Cross-Backend Cloud Validation
# TorchBridge v0.5.3
#
# Validates BERT QA model produces consistent outputs across backends.
# Run on AWS (CUDA), GCP (TPU), AMD Developer Cloud (ROCm).
#
# Usage:
#   ./scripts/cloud_validation.sh              # Run on detected backend
#   ./scripts/cloud_validation.sh --quick      # Quick validation (fewer iterations)
#   ./scripts/cloud_validation.sh --backend cuda  # Force specific backend
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VERSION="0.5.3"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Parse arguments
QUICK_MODE=""
FORCE_BACKEND=""
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/cloud}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE="--quick"; shift ;;
        --backend) FORCE_BACKEND="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

log_step() {
    echo ""
    echo "[$1] $2"
    echo "------------------------------------------------------------"
}

log_success() { echo "  [OK] $1"; }
log_warning() { echo "  [WARN] $1"; }
log_error() { echo "  [ERROR] $1"; }
log_info() { echo "  [INFO] $1"; }

# =============================================================================
# Backend Detection
# =============================================================================

detect_backend() {
    if [ -n "$FORCE_BACKEND" ]; then
        echo "$FORCE_BACKEND"
        return
    fi

    # Check for NVIDIA GPU (CUDA)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "cuda"
        return
    fi

    # Check for AMD GPU (ROCm)
    if command -v rocm-smi &> /dev/null; then
        echo "rocm"
        return
    fi

    # Check for TPU
    if [ -d "/dev/accel0" ] || python3 -c "import torch_xla" 2>/dev/null; then
        echo "tpu"
        return
    fi

    # Check for Intel XPU
    if python3 -c "import torch; torch.xpu.is_available()" 2>/dev/null; then
        echo "xpu"
        return
    fi

    # Check for Apple MPS
    if python3 -c "import torch; torch.backends.mps.is_available()" 2>/dev/null; then
        echo "mps"
        return
    fi

    echo "cpu"
}

get_gpu_info() {
    python3 << 'PYEOF'
import torch
import json

info = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    "xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available(),
    "devices": []
}

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_gb": round(props.total_memory / 1024**3, 1),
            "backend": "rocm" if "AMD" in torch.cuda.get_device_name(i) else "cuda"
        })
elif info["mps_available"]:
    info["devices"].append({"index": 0, "name": "Apple MPS", "backend": "mps"})
elif info["xpu_available"]:
    info["devices"].append({"index": 0, "name": "Intel XPU", "backend": "xpu"})
else:
    info["devices"].append({"index": 0, "name": "CPU", "backend": "cpu"})

print(json.dumps(info, indent=2))
PYEOF
}

# =============================================================================
# Main Validation
# =============================================================================

print_header "BERT SQuAD Cross-Backend Validation v$VERSION"
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Project: $PROJECT_DIR"
echo "Output: $OUTPUT_DIR"

# Detect backend
BACKEND=$(detect_backend)
log_step "1/6" "Backend Detection"
log_info "Detected backend: $BACKEND"

# Save GPU info
get_gpu_info > "$OUTPUT_DIR/hardware_info.json"
cat "$OUTPUT_DIR/hardware_info.json"

# =============================================================================
# Install Dependencies
# =============================================================================
log_step "2/6" "Environment Setup"

PYTHON="${PYTHON:-python3}"
log_info "Python: $($PYTHON --version)"

# Check/install dependencies
$PYTHON -c "import transformers, datasets, torch" 2>/dev/null || {
    log_warning "Installing dependencies..."
    pip install -r requirements.txt -q
}

# Check TorchBridge
if $PYTHON -c "import torchbridge" 2>/dev/null; then
    TB_VERSION=$($PYTHON -c "import torchbridge; print(torchbridge.__version__)")
    log_success "TorchBridge v$TB_VERSION available"
else
    log_warning "TorchBridge not installed (will use vanilla PyTorch)"
fi

# =============================================================================
# Run pytest Cross-Backend Tests
# =============================================================================
log_step "3/6" "Cross-Backend Consistency Tests"

$PYTHON -m pytest tests/test_consistency.py -v \
    --tb=short \
    --json-report \
    --json-report-file="$OUTPUT_DIR/test_results.json" \
    2>&1 | tee "$OUTPUT_DIR/test_output.txt"

TEST_EXIT=${PIPESTATUS[0]}

# Parse results
$PYTHON << PYEOF
import json
with open("$OUTPUT_DIR/test_results.json") as f:
    results = json.load(f)
summary = results.get("summary", {})
print(f"  Tests: {summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed, {summary.get('skipped', 0)} skipped")
print(f"  Duration: {summary.get('duration', 0):.2f}s")
PYEOF

# =============================================================================
# Run Cross-Backend Validation Script
# =============================================================================
log_step "4/6" "Cross-Backend Numerical Validation"

$PYTHON validate_cross_backend.py \
    --tolerance 1e-4 \
    --output "$OUTPUT_DIR/cross_backend_validation.json" \
    2>&1 | tee "$OUTPUT_DIR/validation_output.txt"

VALIDATION_EXIT=${PIPESTATUS[0]}

# =============================================================================
# Run Inference Benchmark
# =============================================================================
log_step "5/6" "Inference Benchmark"

ITERATIONS=100
if [ -n "$QUICK_MODE" ]; then
    ITERATIONS=20
fi

$PYTHON inference.py \
    --benchmark \
    --iterations $ITERATIONS \
    --output "$OUTPUT_DIR/inference_benchmark.json" \
    2>&1 | tee "$OUTPUT_DIR/benchmark_output.txt"

# =============================================================================
# Generate Report
# =============================================================================
log_step "6/6" "Generating Report"

$PYTHON << PYEOF
import json
from datetime import datetime
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
backend = "$BACKEND"
version = "$VERSION"

# Load results
hardware_info = json.load(open(output_dir / "hardware_info.json"))
test_results = json.load(open(output_dir / "test_results.json"))
validation_results = json.load(open(output_dir / "cross_backend_validation.json"))
benchmark_results = json.load(open(output_dir / "inference_benchmark.json"))

test_summary = test_results.get("summary", {})
benchmark = benchmark_results.get("benchmark", {})

# Determine device name
device_name = "CPU"
if hardware_info.get("devices"):
    device_name = hardware_info["devices"][0].get("name", "Unknown")

# Check if passed
all_passed = (
    test_summary.get("failed", 0) == 0 and
    validation_results.get("all_passed", False)
)

report = f"""# BERT SQuAD Cloud Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**TorchBridge Version:** {version}
**Backend:** {backend.upper()}
**Device:** {device_name}

---

## Summary

| Metric | Value |
|--------|-------|
| Overall Status | {"PASSED" if all_passed else "FAILED"} |
| Pytest Tests | {test_summary.get('passed', 0)} passed / {test_summary.get('total', 0)} total |
| Cross-Backend Validation | {"PASSED" if validation_results.get('all_passed') else "FAILED"} |
| Tolerance | {validation_results.get('tolerance', 'N/A')} |

---

## Hardware Configuration

| Property | Value |
|----------|-------|
| PyTorch Version | {hardware_info.get('pytorch_version', 'N/A')} |
| Device | {device_name} |
| Backend | {backend} |

---

## Cross-Backend Consistency

Reference backend: **{validation_results.get('reference_backend', 'cpu').upper()}**

Backends tested: {', '.join(b.upper() for b in validation_results.get('backends_tested', []))}

"""

# Add comparison details
for backend_name, comparison in validation_results.get("comparisons", {}).items():
    status = "PASS" if comparison.get("passed") else "FAIL"
    start_diff = comparison.get("start_logits", {}).get("max_diff", 0)
    end_diff = comparison.get("end_logits", {}).get("max_diff", 0)
    start_cos = comparison.get("start_logits", {}).get("cosine_sim", 0)
    end_cos = comparison.get("end_logits", {}).get("cosine_sim", 0)

    report += f"""
### CPU vs {backend_name.upper()}: [{status}]

| Metric | Start Logits | End Logits |
|--------|--------------|------------|
| Max Diff | {start_diff:.2e} | {end_diff:.2e} |
| Cosine Sim | {start_cos:.6f} | {end_cos:.6f} |
"""

report += f"""
---

## Inference Performance

| Metric | Value |
|--------|-------|
| Mean Latency | {benchmark.get('mean_ms', 0):.2f} ms |
| P50 Latency | {benchmark.get('p50_ms', 0):.2f} ms |
| P95 Latency | {benchmark.get('p95_ms', 0):.2f} ms |
| P99 Latency | {benchmark.get('p99_ms', 0):.2f} ms |
| Throughput | {benchmark.get('throughput_qps', 0):.1f} queries/sec |

---

## Test Details

| Test | Result |
|------|--------|
"""

for test in test_results.get("tests", []):
    name = test.get("nodeid", "").split("::")[-1]
    outcome = test.get("outcome", "unknown").upper()
    report += f"| {name} | {outcome} |\n"

report += f"""
---

## Files Generated

- `hardware_info.json` - Hardware detection results
- `test_results.json` - Pytest detailed results
- `cross_backend_validation.json` - Numerical consistency data
- `inference_benchmark.json` - Performance metrics

---

## Verification Commands

```bash
# Re-run validation
python validate_cross_backend.py --tolerance 1e-4

# Run specific tests
pytest tests/test_consistency.py -v -k "cosine_similarity"

# Benchmark inference
python inference.py --benchmark --iterations 100
```
"""

# Save report
report_file = output_dir / f"BERT_SQUAD_{backend.upper()}_REPORT.md"
with open(report_file, "w") as f:
    f.write(report)

print(f"Report saved: {report_file}")

# Print summary
print("")
print("=" * 60)
print(f"  Validation {'PASSED' if all_passed else 'FAILED'}")
print("=" * 60)
PYEOF

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  BERT SQuAD Cloud Validation Complete"
echo "============================================================"
echo ""
echo "Backend: $BACKEND"
echo "Results: $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR"
echo ""

if [ $TEST_EXIT -eq 0 ] && [ $VALIDATION_EXIT -eq 0 ]; then
    log_success "All validations passed!"
    exit 0
else
    log_error "Some validations failed"
    echo "  - Pytest exit code: $TEST_EXIT"
    echo "  - Validation exit code: $VALIDATION_EXIT"
    exit 1
fi
