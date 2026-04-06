#!/bin/bash
# TPU v5e Validation Script for TorchBridge
#
# Validates Qwen3-0.6B cross-backend consistency on Google Cloud TPU v5e.
# Quota: 16 chips approved in us-central1 (since Dec 2025).
#
# Usage:
#   bash scripts/validation/tpu_v5e_validation.sh
#
# Prerequisites:
#   - gcloud CLI authenticated with your GCP project
#   - TPU v5 Lite PodSlice quota in us-central1
#
# Cost: ~$1.20/chip/hr, validation takes ~15 min = ~$2.40 total
#
# CRITICAL: This script terminates the TPU VM after validation.

set -euo pipefail

PROJECT="${GCP_PROJECT:-your-gcp-project}"
ZONE="us-central1-a"
TPU_NAME="tb-tpu-val-$(date +%s)"
REPORT_DIR="reports/cloud_validation/$(date +%Y-%m-%d)"

echo "=== TorchBridge TPU v5e Validation ==="
echo "Project: $PROJECT"
echo "Zone: $ZONE"
echo "TPU Name: $TPU_NAME"
echo ""

# Step 1: Create TPU VM
echo "[1/5] Creating TPU VM..."
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --accelerator-type=v5litepod-8 \
  --version=v2-alpha-tpuv5-lite \
  --quiet

echo "TPU VM created: $TPU_NAME"

# Step 2: Wait for SSH readiness
echo "[2/5] Waiting for TPU VM to be ready (60s)..."
sleep 60

# Step 3: Install dependencies and verify TPU
echo "[3/5] Installing dependencies on TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="
    pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html -q 2>/dev/null
    pip install transformers -q 2>/dev/null
    echo 'Dependencies installed.'
    PJRT_DEVICE=TPU python3 -c \"
import torch_xla.core.xla_model as xm
devices = xm.get_xla_supported_devices('TPU')
print(f'TPU devices: {devices}')
print(f'Device count: {len(devices)}')
\"
  "

# Step 4: Run validation
echo "[4/5] Running Qwen3-0.6B validation..."
VALIDATION_OUTPUT=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="
PJRT_DEVICE=TPU python3 << 'VALEOF'
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import time
import json

print('=== Qwen3-0.6B TPU Validation ===')

model_name = 'Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
inputs = tokenizer('The capital of France is', return_tensors='pt')

# CPU baseline
model.eval()
with torch.no_grad():
    cpu_out = model(**inputs)
cpu_logits = cpu_out.logits[:, -1, :]

# TPU inference
tpu_device = xm.xla_device()
print(f'TPU device: {tpu_device}')

model_tpu = model.to(tpu_device)
inputs_tpu = {k: v.to(tpu_device) for k, v in inputs.items()}

with torch.no_grad():
    tpu_out = model_tpu(**inputs_tpu)

tpu_logits = tpu_out.logits[:, -1, :].cpu()
xm.mark_step()

max_diff = torch.abs(cpu_logits - tpu_logits).max().item()
cos_sim = F.cosine_similarity(
    cpu_logits.flatten().unsqueeze(0),
    tpu_logits.flatten().unsqueeze(0)
).item()

# Latency benchmark
for _ in range(3):
    model_tpu(**inputs_tpu)
    xm.mark_step()

t0 = time.perf_counter()
for _ in range(50):
    model_tpu(**inputs_tpu)
    xm.mark_step()
latency = (time.perf_counter() - t0) / 50 * 1000

status = 'PASSED' if max_diff < 0.5 and cos_sim > 0.999 else 'FAILED'

print(f'Max diff: {max_diff:.2e}')
print(f'Cosine sim: {cos_sim:.6f}')
print(f'Status: {status}')
print(f'Latency: {latency:.1f} ms')

# Output JSON for capture
results = {
    'date': '$(date +%Y-%m-%d)',
    'backend': 'tpu',
    'hardware': 'TPU v5e',
    'provider': 'GCP',
    'instance_type': 'v5litepod-8',
    'model': 'Qwen/Qwen3-0.6B',
    'pytorch_version': torch.__version__,
    'sdk_version': f'torch_xla (XLA)',
    'max_diff': max_diff,
    'cosine_sim': cos_sim,
    'latency_ms': round(latency, 1),
    'status': status,
    'notes': 'First TPU v5e validation'
}

print('JSON_RESULTS:' + json.dumps(results))
VALEOF
  ")

echo "$VALIDATION_OUTPUT"

# Step 5: TERMINATE TPU VM (CRITICAL!)
echo "[5/5] Terminating TPU VM..."
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --quiet

echo "TPU VM terminated: $TPU_NAME"

# Save results locally
mkdir -p "$REPORT_DIR"
JSON_LINE=$(echo "$VALIDATION_OUTPUT" | grep "JSON_RESULTS:" | sed 's/JSON_RESULTS://')
if [ -n "$JSON_LINE" ]; then
  echo "$JSON_LINE" | python3 -m json.tool > "$REPORT_DIR/gcp_tpu_v5e_qwen3.json"
  echo "Results saved to $REPORT_DIR/gcp_tpu_v5e_qwen3.json"
else
  echo "WARNING: Could not extract JSON results. Raw output saved."
  echo "$VALIDATION_OUTPUT" > "$REPORT_DIR/gcp_tpu_v5e_raw.txt"
fi

echo ""
echo "=== TPU Validation Complete ==="
echo "Remember to verify no TPU VMs are running:"
echo "  gcloud compute tpus tpu-vm list --project=$PROJECT --zone=$ZONE"
