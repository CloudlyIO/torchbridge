#!/bin/bash
# Run BERT SQuAD training and validation across all available backends
#
# Usage:
#   ./scripts/run_all_backends.sh         # Full training
#   ./scripts/run_all_backends.sh --quick # Quick validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  BERT SQuAD - Cross-Backend Training & Validation"
echo "============================================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# Parse arguments
QUICK_MODE=""
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE="--quick"
    echo "Mode: Quick validation"
else
    echo "Mode: Full training"
fi
echo ""

# Detect Python
PYTHON="${PYTHON:-python3}"
echo "Python: $($PYTHON --version)"
echo ""

# Install dependencies if needed
echo "[1/5] Checking dependencies..."
$PYTHON -c "import transformers, datasets, torch" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt -q
}
echo "  Dependencies OK"
echo ""

# Detect available backends
echo "[2/5] Detecting available backends..."
BACKENDS=$($PYTHON -c "
import torch
backends = ['cpu']
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    if 'AMD' in name or 'Radeon' in name:
        backends.append('rocm')
    else:
        backends.append('cuda')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    backends.append('mps')
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    backends.append('xpu')
print(' '.join(backends))
")
echo "  Available backends: $BACKENDS"
echo ""

# Run training on best backend
echo "[3/5] Training on best available backend..."
mkdir -p results checkpoints

$PYTHON train.py $QUICK_MODE --epochs 1 2>&1 | tee results/training_log.txt
echo ""

# Run cross-backend validation
echo "[4/5] Running cross-backend validation..."
$PYTHON validate_cross_backend.py \
    --output results/cross_backend_validation.json \
    2>&1 | tee results/validation_log.txt
echo ""

# Run inference benchmark
echo "[5/5] Running inference benchmark..."
$PYTHON inference.py \
    --benchmark \
    --iterations 50 \
    --output results/inference_benchmark.json \
    2>&1 | tee results/benchmark_log.txt
echo ""

# Summary
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - results/training_log.txt"
echo "  - results/cross_backend_validation.json"
echo "  - results/inference_benchmark.json"
echo ""

# Show validation result
if [ -f results/cross_backend_validation.json ]; then
    PASSED=$($PYTHON -c "import json; print(json.load(open('results/cross_backend_validation.json'))['all_passed'])")
    if [ "$PASSED" == "True" ]; then
        echo "Cross-backend validation: PASSED"
    else
        echo "Cross-backend validation: FAILED"
    fi
fi

echo ""
echo "Done!"
