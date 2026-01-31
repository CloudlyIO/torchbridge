#!/bin/bash
# TorchBridge v0.4.40 — Cloud GPU Use Case Validation
# Runs all 5 use case scripts on NVIDIA GPU hardware (AWS or GCP)
#
# Tested platforms:
#   AWS: g5.xlarge (A10G 24GB) — Deep Learning AMI PyTorch 2.9, Ubuntu 24.04
#   GCP: g2-standard-4 (L4 24GB) — pytorch-2-7-cu128-ubuntu-2404-nvidia-570
#
# Usage:
#   SSH mode:    Copy to instance, then: bash v0440_gpu_validation.sh
#   Startup:     Pass as user-data (AWS) or startup-script (GCP)
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support
#   - Code tarball at GCS_URL below (or override with TORCHBRIDGE_CODE_URL env var)
#
# Generated: 2026-01-31

set -uo pipefail  # no -e: let individual use cases fail without killing the script

GCS_URL="${TORCHBRIDGE_CODE_URL:-https://storage.googleapis.com/torchbridge-validation-v0440/torchbridge_v0440.tar.gz}"
LOG=/tmp/torchbridge_validation.log

exec > >(tee "$LOG") 2>&1

echo "============================================================"
echo "TorchBridge v0.4.40 Cloud GPU Validation"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Hostname:  $(hostname)"
echo "============================================================"

# ── 0. Find Python with PyTorch ──────────────────────────────────────
echo ""
echo "[0/8] Setting up Python environment..."

PYTHON=""
PIP=""

# Strategy 1: AWS Deep Learning AMI — PyTorch in /opt/pytorch/
if [ -x /opt/pytorch/bin/python3 ]; then
    /opt/pytorch/bin/python3 -c "import torch" 2>/dev/null && {
        PYTHON=/opt/pytorch/bin/python3
        PIP=/opt/pytorch/bin/pip
        echo "Found PyTorch in /opt/pytorch/ (AWS DL AMI)"
    }
fi

# Strategy 2: Conda environment
if [ -z "$PYTHON" ] && [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    . /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch 2>/dev/null || conda activate base 2>/dev/null || true
    python3 -c "import torch" 2>/dev/null && {
        PYTHON=$(which python3)
        PIP=$(which pip)
        echo "Found PyTorch via conda: $PYTHON"
    }
fi

# Strategy 3: GCP DL VM — system python with torch pre-installed
if [ -z "$PYTHON" ]; then
    python3 -c "import torch" 2>/dev/null && {
        PYTHON=$(which python3)
        PIP="$(which pip 2>/dev/null || which pip3 2>/dev/null || echo pip)"
        echo "Found PyTorch in system python: $PYTHON"
    }
fi

# Strategy 4: Search conda environments
if [ -z "$PYTHON" ] && [ -d /opt/conda/envs ]; then
    for env_dir in /opt/conda/envs/*/; do
        if [ -f "${env_dir}bin/python3" ]; then
            "${env_dir}bin/python3" -c "import torch" 2>/dev/null && {
                PYTHON="${env_dir}bin/python3"
                PIP="${env_dir}bin/pip"
                echo "Found PyTorch in conda env: ${env_dir}"
                break
            }
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "FATAL: Could not find Python with PyTorch installed"
    exit 1
fi

echo "Python: $PYTHON"
echo "Pip:    $PIP"

# ── 1. Hardware Detection ────────────────────────────────────────────
echo ""
echo "[1/8] Detecting hardware..."
$PYTHON -c "
import torch
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA:        {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU Memory:  {props.total_memory / 1e9:.1f} GB')
    print(f'Compute Cap: {props.major}.{props.minor}')
    print(f'GPU Count:   {torch.cuda.device_count()}')
" || echo "Hardware detection had warnings (non-fatal)"

# Detect cloud provider
PROVIDER="unknown"
INSTANCE="unknown"
if curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type > /dev/null 2>&1; then
    INSTANCE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
    PROVIDER="aws"
    echo "Provider:    AWS"
    echo "Instance:    $INSTANCE"
elif curl -s --max-time 2 -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type > /dev/null 2>&1; then
    MACHINE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type)
    INSTANCE=$(basename "$MACHINE")
    PROVIDER="gcp"
    echo "Provider:    GCP"
    echo "Machine:     $INSTANCE"
fi

# ── 2. Download Code ──────────────────────────────────────────────────
echo ""
echo "[2/8] Downloading code..."
cd /tmp
rm -rf torchbridge_val
mkdir -p torchbridge_val
cd torchbridge_val

curl -sL -o code.tar.gz "$GCS_URL"
if [ ! -s code.tar.gz ]; then
    echo "FATAL: Failed to download code tarball from $GCS_URL"
    exit 1
fi
tar xzf code.tar.gz && rm code.tar.gz
echo "Code extracted: $(ls -d */ 2>/dev/null | tr '\n' ' ')"

# ── 3. Install Dependencies ──────────────────────────────────────────
echo ""
echo "[3/8] Installing dependencies..."

# Detect pip flags and install method
PIP_CMD="$PIP install -q"

# GCP DL VMs enforce PEP 668 — need --break-system-packages
if $PIP install --dry-run psutil 2>&1 | grep -q "externally-managed"; then
    PIP_CMD="$PIP install -q --break-system-packages"
    echo "  PEP 668 detected, using --break-system-packages"
fi

# AWS DL AMIs have read-only /opt/pytorch/ — need sudo
if ! $PIP_CMD psutil 2>&1 | grep -q "Permission denied"; then
    echo "  Standard pip install"
else
    PIP_CMD="sudo $PYTHON -m pip install -q"
    echo "  Read-only venv detected, using sudo"
fi

# Install all deps
$PIP_CMD \
    psutil \
    matplotlib \
    onnx \
    onnxruntime \
    onnxscript \
    safetensors \
    transformers \
    accelerate \
    2>&1 | tail -5

# Verify key packages are importable
export PYTHONPATH=/tmp/torchbridge_val/src
echo ""
echo "  Verifying packages:"
for pkg in torch torchbridge transformers safetensors onnx psutil; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        echo "    $pkg: OK"
    else
        echo "    $pkg: MISSING"
    fi
done

# ── 4-8. Run Use Cases ───────────────────────────────────────────────
UC1_EXIT=1; UC2_EXIT=1; UC3_EXIT=1; UC4_EXIT=1; UC5_EXIT=1

echo ""
echo "============================================================"
echo "[4/8] USE CASE 1: Hardware-Agnostic Export Pipeline"
echo "============================================================"
$PYTHON examples/usecase1_export_pipeline.py 2>&1 && UC1_EXIT=0 || UC1_EXIT=$?
echo "Use Case 1 exit code: $UC1_EXIT"

echo ""
echo "============================================================"
echo "[5/8] USE CASE 2: LLM Optimization for Inference"
echo "============================================================"
$PYTHON examples/usecase2_llm_optimization.py 2>&1 && UC2_EXIT=0 || UC2_EXIT=$?
echo "Use Case 2 exit code: $UC2_EXIT"

echo ""
echo "============================================================"
echo "[6/8] USE CASE 3: CI/CD Hardware Validation"
echo "============================================================"
$PYTHON examples/usecase3_cicd_validation.py 2>&1 && UC3_EXIT=0 || UC3_EXIT=$?
echo "Use Case 3 exit code: $UC3_EXIT"

echo ""
echo "============================================================"
echo "[7/8] USE CASE 4: Backend-Agnostic Training"
echo "============================================================"
$PYTHON examples/usecase4_backend_agnostic_training.py 2>&1 && UC4_EXIT=0 || UC4_EXIT=$?
echo "Use Case 4 exit code: $UC4_EXIT"

echo ""
echo "============================================================"
echo "[8/8] USE CASE 5: Cross-Backend Validation"
echo "============================================================"
$PYTHON examples/usecase5_cross_backend_validation.py 2>&1 && UC5_EXIT=0 || UC5_EXIT=$?
echo "Use Case 5 exit code: $UC5_EXIT"

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "CLOUD VALIDATION COMPLETE"
echo "============================================================"
GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "CPU")
echo "Provider:  $PROVIDER"
echo "Instance:  $INSTANCE"
echo "GPU:       $GPU_NAME"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Results:"
echo "  Use Case 1 (Export Pipeline):        $([ $UC1_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Use Case 2 (LLM Optimization):       $([ $UC2_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Use Case 3 (CI/CD Validation):       $([ $UC3_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Use Case 4 (Backend Training):       $([ $UC4_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Use Case 5 (Cross-Backend Val):      $([ $UC5_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"

TOTAL_PASS=0
for code in $UC1_EXIT $UC2_EXIT $UC3_EXIT $UC4_EXIT $UC5_EXIT; do
    [ "$code" -eq 0 ] && TOTAL_PASS=$((TOTAL_PASS + 1))
done
echo ""
echo "Total: $TOTAL_PASS/5 passed"

# Upload log to GCS (best-effort)
TIMESTAMP=$(date +%s)
gsutil cp "$LOG" \
    "gs://torchbridge-validation-v0440/logs/${PROVIDER}_${INSTANCE}_${TIMESTAMP}.log" 2>/dev/null || true

echo ""
echo "Log: $LOG"

# Auto-shutdown (only when running as startup script, not SSH)
if [ "${AUTO_SHUTDOWN:-0}" = "1" ]; then
    echo "AUTO_SHUTDOWN=1: Shutting down in 30 seconds..."
    sleep 30
    shutdown -h now
fi
