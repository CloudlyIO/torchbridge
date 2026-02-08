#!/bin/bash
# =============================================================================
# Model Examples Validation - Cloud GPU
# TorchBridge v0.5.8
#
# Runs all model examples and benchmark suite on a cloud GPU instance.
# Adapts quantization based on available VRAM.
#
# Usage:
#   ./scripts/cloud_testing/validate_model_examples.sh [--results-dir DIR]
#
# Supports: AWS (A10G/H100), GCP (T4/L4), AMD (MI300X)
# =============================================================================

set -eo pipefail

# ─── Configuration ───

RESULTS_DIR="${1:-$HOME/model_validation_results}"
REPO_URL="https://github.com/CloudlyIO/torchbridge.git"
WORK_DIR="$HOME/torchbridge"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; }
log_section() { echo -e "\n${CYAN}══════════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}══════════════════════════════════════════════════${NC}\n"; }

mkdir -p "$RESULTS_DIR"

# ─── System Detection ───

log_section "System Detection"

detect_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
        GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)
        GPU_MEM_GB=$((GPU_MEM_MB / 1024))
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
        BACKEND="CUDA"
        log_info "GPU: $GPU_NAME ($GPU_MEM_GB GB)"
        log_info "Driver: $DRIVER"
    elif command -v rocm-smi &>/dev/null; then
        GPU_NAME=$(rocm-smi --showproductname | grep -oP 'Card.*' | head -1)
        GPU_MEM_GB=192  # Default for MI300X
        BACKEND="ROCm"
        log_info "GPU: $GPU_NAME (~$GPU_MEM_GB GB)"
    else
        GPU_NAME="CPU"
        GPU_MEM_GB=0
        BACKEND="CPU"
        log_warning "No GPU detected, running CPU-only"
    fi

    # Determine quantization based on VRAM
    if [ "$GPU_MEM_GB" -ge 40 ]; then
        LLAMA4_QUANT="none"       # FP16 fits on 40GB+
        LLM_QUANT="none"          # FP16 for 7-8B models
        GEMMA_QUANT="none"        # FP16 for 12B
        GEMMA_MODEL="google/gemma-3-12b-it"
    elif [ "$GPU_MEM_GB" -ge 20 ]; then
        LLAMA4_QUANT="int4"       # INT4 for Llama 4 on 24GB
        LLM_QUANT="none"          # FP16 for 7-8B models
        GEMMA_QUANT="int8"        # INT8 for 12B on 24GB
        GEMMA_MODEL="google/gemma-3-12b-it"
    elif [ "$GPU_MEM_GB" -ge 14 ]; then
        LLAMA4_QUANT="skip"       # Can't fit Llama 4 on 16GB
        LLM_QUANT="int8"          # INT8 for 7-8B models
        GEMMA_QUANT="int4"        # INT4 for 12B
        GEMMA_MODEL="google/gemma-3-4b-it"
    else
        LLAMA4_QUANT="skip"
        LLM_QUANT="int4"
        GEMMA_QUANT="int4"
        GEMMA_MODEL="google/gemma-3-4b-it"
    fi

    log_info "Backend: $BACKEND"
    log_info "LLM quantization: $LLM_QUANT"
    log_info "Llama 4 quantization: $LLAMA4_QUANT"
    log_info "Gemma model: $GEMMA_MODEL ($GEMMA_QUANT)"
}

detect_gpu

# Save system info
cat > "$RESULTS_DIR/system_info.json" <<SYSEOF
{
    "gpu_name": "$GPU_NAME",
    "gpu_memory_gb": $GPU_MEM_GB,
    "backend": "$BACKEND",
    "timestamp": "$TIMESTAMP",
    "hostname": "$(hostname)",
    "python_version": "$(python3 --version 2>&1)",
    "torch_version": "$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
}
SYSEOF

# ─── Setup ───

log_section "Setup"

if [ ! -d "$WORK_DIR" ]; then
    log_info "Cloning TorchBridge..."
    git clone "$REPO_URL" "$WORK_DIR"
else
    log_info "Updating existing repo..."
    cd "$WORK_DIR" && git pull
fi

cd "$WORK_DIR"

log_info "Installing dependencies..."
pip install --break-system-packages -q torch transformers accelerate bitsandbytes 2>/dev/null || \
pip install -q torch transformers accelerate bitsandbytes 2>/dev/null || \
log_warning "Some dependencies may have failed to install"

# Verify
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"

export PYTHONPATH="$WORK_DIR:$WORK_DIR/src:${PYTHONPATH:-}"

# ─── Run Model Examples ───

run_example() {
    local name="$1"
    local script="$2"
    shift 2
    local args=("$@")

    log_info "Running: $name"
    local start_time=$(date +%s)
    local exit_code=0

    python3 "$script" "${args[@]}" > "$RESULTS_DIR/${name}.log" 2>&1 || exit_code=$?
    cat "$RESULTS_DIR/${name}.log"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ "$exit_code" -eq 0 ]; then
        log_success "$name completed in ${duration}s"
        echo "$name,PASS,$duration" >> "$RESULTS_DIR/summary.csv"
    else
        log_error "$name failed after ${duration}s (exit $exit_code)"
        echo "$name,FAIL,$duration" >> "$RESULTS_DIR/summary.csv"
    fi
}

# Initialize summary
echo "test,status,duration_s" > "$RESULTS_DIR/summary.csv"

# ─── SAM 3 (lightweight, run first) ───

log_section "SAM 3 (Segment Anything Model 3)"

run_example "sam3_inference" \
    "examples/models/vision/sam3_optimization.py" \
    --output-json "$RESULTS_DIR/sam3_inference.json"

run_example "sam3_benchmark" \
    "examples/models/vision/sam3_optimization.py" \
    --benchmark \
    --output-json "$RESULTS_DIR/sam3_benchmark.json"

# ─── DeepSeek R1 Distill 7B ───

log_section "DeepSeek R1 Distill 7B"

run_example "deepseek_inference" \
    "examples/models/medium/deepseek_optimization.py" \
    --quantization "$LLM_QUANT" \
    --max-new-tokens 256 \
    --output-json "$RESULTS_DIR/deepseek_inference.json"

run_example "deepseek_benchmark" \
    "examples/models/medium/deepseek_optimization.py" \
    --quantization "$LLM_QUANT" \
    --benchmark \
    --output-json "$RESULTS_DIR/deepseek_benchmark.json"

run_example "deepseek_moe_analysis" \
    "examples/models/medium/deepseek_optimization.py" \
    --analyze-experts \
    --output-json "$RESULTS_DIR/deepseek_moe.json"

# ─── Qwen 3 8B ───

log_section "Qwen 3 8B"

run_example "qwen3_inference" \
    "examples/models/medium/qwen3_optimization.py" \
    --quantization "$LLM_QUANT" \
    --max-new-tokens 128 \
    --output-json "$RESULTS_DIR/qwen3_inference.json"

run_example "qwen3_multilingual" \
    "examples/models/medium/qwen3_optimization.py" \
    --quantization "$LLM_QUANT" \
    --multilingual \
    --output-json "$RESULTS_DIR/qwen3_multilingual.json"

run_example "qwen3_benchmark" \
    "examples/models/medium/qwen3_optimization.py" \
    --quantization "$LLM_QUANT" \
    --benchmark \
    --output-json "$RESULTS_DIR/qwen3_benchmark.json"

# ─── Gemma 3 ───

log_section "Gemma 3 ($GEMMA_MODEL)"

run_example "gemma3_inference" \
    "examples/models/small/gemma3_optimization.py" \
    --model "$GEMMA_MODEL" \
    --quantization "$GEMMA_QUANT" \
    --max-new-tokens 128 \
    --output-json "$RESULTS_DIR/gemma3_inference.json"

run_example "gemma3_benchmark" \
    "examples/models/small/gemma3_optimization.py" \
    --model "$GEMMA_MODEL" \
    --quantization "$GEMMA_QUANT" \
    --benchmark \
    --output-json "$RESULTS_DIR/gemma3_benchmark.json"

# ─── Llama 4 Scout (if VRAM allows) ───

if [ "$LLAMA4_QUANT" != "skip" ]; then
    log_section "Llama 4 Scout (17B active, 16 experts)"

    run_example "llama4_inference" \
        "examples/models/medium/llama4_optimization.py" \
        --quantization "$LLAMA4_QUANT" \
        --max-new-tokens 128 \
        --output-json "$RESULTS_DIR/llama4_inference.json"

    run_example "llama4_benchmark" \
        "examples/models/medium/llama4_optimization.py" \
        --quantization "$LLAMA4_QUANT" \
        --benchmark \
        --output-json "$RESULTS_DIR/llama4_benchmark.json"
else
    log_warning "Skipping Llama 4 Scout — insufficient VRAM ($GPU_MEM_GB GB)"
    echo "llama4_inference,SKIPPED,0" >> "$RESULTS_DIR/summary.csv"
    echo "llama4_benchmark,SKIPPED,0" >> "$RESULTS_DIR/summary.csv"
fi

# ─── Cross-Backend Benchmark Suite ───

log_section "Cross-Backend Benchmark Suite"

# Determine which models the benchmark suite can run
BENCH_MODELS="deepseek qwen3"
if [ "$LLAMA4_QUANT" != "skip" ]; then
    BENCH_MODELS="llama4 deepseek qwen3"
fi

run_example "benchmark_suite" \
    "scripts/benchmark_suite.py" \
    --models $BENCH_MODELS \
    --quantization "$LLM_QUANT" \
    --num-runs 10 \
    --max-new-tokens 100 \
    --output-dir "$RESULTS_DIR/benchmark_suite"

# ─── Summary Report ───

log_section "Validation Summary"

echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Test Results:"
echo "─────────────────────────────────────────"
column -t -s',' "$RESULTS_DIR/summary.csv" 2>/dev/null || cat "$RESULTS_DIR/summary.csv"
echo ""

PASS_COUNT=$(grep -c ',PASS,' "$RESULTS_DIR/summary.csv" 2>/dev/null || echo 0)
FAIL_COUNT=$(grep -c ',FAIL,' "$RESULTS_DIR/summary.csv" 2>/dev/null || echo 0)
SKIP_COUNT=$(grep -c ',SKIPPED,' "$RESULTS_DIR/summary.csv" 2>/dev/null || echo 0)

echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo ""

# Create tar for easy download
TAR_NAME="model_validation_${BACKEND}_${GPU_NAME// /_}_${TIMESTAMP}.tar.gz"
cd "$(dirname "$RESULTS_DIR")"
tar czf "$HOME/$TAR_NAME" "$(basename "$RESULTS_DIR")"
echo "Tarball: ~/$TAR_NAME"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    log_success "All validations passed!"
else
    log_error "$FAIL_COUNT validation(s) failed"
fi
