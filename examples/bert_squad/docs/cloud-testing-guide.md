# BERT SQuAD Cross-Backend Cloud Validation

This guide explains how to validate BERT SQuAD across heterogeneous backends:
- **AWS CUDA** (NVIDIA A10G/A100)
- **AMD ROCm** (MI300X on AMD Developer Cloud)
- **GCP TPU** (v5e)

## Quick Start

### Automated (GitHub Actions)

Trigger the workflow manually:

```bash
# Via GitHub CLI
gh workflow run bert-squad-validation.yml \
  -f backends="cuda,rocm,tpu" \
  -f quick_mode=false

# Check status
gh run list --workflow=bert-squad-validation.yml
```

### Manual Execution

Run on any cloud instance:

```bash
cd examples/bert_squad
./scripts/cloud_validation.sh
```

---

## AWS CUDA (NVIDIA A10G/A100)

### Launch Instance

```bash
# Launch g5.xlarge (A10G) spot instance
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g5.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --instance-market-options '{"MarketType":"spot"}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=bert-squad-cuda}]'
```

### Run Validation

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Setup
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge/examples/bert_squad

# Install dependencies (Deep Learning AMI has PyTorch pre-installed)
pip install -r requirements.txt
pip install -e ../..

# Run validation
./scripts/cloud_validation.sh --output results/aws_cuda

# View results
cat results/aws_cuda/BERT_SQUAD_CUDA_REPORT.md
```

### Expected Output

```
Backend: CUDA (NVIDIA A10G)
Cross-Backend Validation: PASSED
  - Max diff: 1.26e-06
  - Cosine similarity: 1.000000
Inference: 8.5 ms (117 QPS)
```

---

## AMD ROCm (MI300X)

### Access AMD Developer Cloud

1. Sign up at https://www.amd.com/en/developer.html
2. Request access to MI300X instances
3. Launch instance with ROCm 6.0 image

### Run Validation

```bash
# SSH into AMD instance
ssh user@<amd-instance>

# Setup
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge/examples/bert_squad

# Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
pip install -e ../..

# Verify ROCm
rocm-smi --showproductname
python -c "import torch; print(f'CUDA/ROCm: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Run validation (use higher tolerance for ROCm due to SDPA differences)
./scripts/cloud_validation.sh --output results/amd_rocm

# View results
cat results/amd_rocm/BERT_SQUAD_ROCM_REPORT.md
```

### ROCm SDPA Note

ROCm's flash attention uses online softmax with different accumulation order, causing higher numerical divergence (~1e-3 vs 1e-6 for CUDA). This is expected and documented in `docs/blog/sdpa-divergence-rocm.md`.

To force exact parity, disable flash attention:

```python
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
```

---

## GCP TPU (v5e)

### Create TPU VM

```bash
# Set project
gcloud config set project your-project

# Create TPU v5e VM
gcloud compute tpus tpu-vm create bert-squad-tpu \
  --zone=us-central1-a \
  --accelerator-type=v5e-8 \
  --version=tpu-ubuntu2204-base \
  --preemptible
```

### Run Validation

```bash
# SSH into TPU VM
gcloud compute tpus tpu-vm ssh bert-squad-tpu --zone=us-central1-a

# Setup
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge/examples/bert_squad

# Install PyTorch/XLA for TPU
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install -r requirements.txt
pip install -e ../..

# Verify TPU
python -c "import torch_xla; import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"

# Run TPU-specific validation
python << 'EOF'
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn.functional as F

device = xm.xla_device()
print(f"TPU Device: {device}")

# Load model on TPU
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
model.eval()

# Test input
inputs = tokenizer("What is the capital?", "Paris is the capital of France.",
                  max_length=384, truncation=True, padding="max_length", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Inference
with torch.no_grad():
    outputs = model(**inputs)
xm.mark_step()

# Compare with CPU
cpu_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
cpu_inputs = tokenizer("What is the capital?", "Paris is the capital of France.",
                       max_length=384, truncation=True, padding="max_length", return_tensors="pt")
with torch.no_grad():
    cpu_outputs = cpu_model(**cpu_inputs)

# Results
tpu_start = outputs.start_logits.cpu()
cpu_start = cpu_outputs.start_logits
diff = torch.abs(cpu_start - tpu_start).max().item()
cos_sim = F.cosine_similarity(cpu_start.flatten().unsqueeze(0), tpu_start.flatten().unsqueeze(0)).item()

print(f"\nCPU vs TPU:")
print(f"  Max diff: {diff:.2e}")
print(f"  Cosine sim: {cos_sim:.6f}")
print(f"  Status: {'PASSED' if diff < 1e-3 else 'FAILED'}")
EOF
```

### Cleanup

```bash
gcloud compute tpus tpu-vm delete bert-squad-tpu --zone=us-central1-a
```

---

## Validation Thresholds

| Backend | Max Diff Tolerance | Cosine Sim Threshold | Notes |
|---------|-------------------|---------------------|-------|
| CUDA    | 1e-4              | 0.9999              | Exact parity expected |
| MPS     | 1e-4              | 0.9999              | Apple Silicon |
| ROCm    | 1e-3              | 0.999               | SDPA divergence expected |
| TPU     | 1e-3              | 0.999               | XLA compilation differences |
| XPU     | 1e-4              | 0.9999              | Intel Arc/Flex |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python inference.py --benchmark --batch-size 1

# Clear cache between runs
python -c "import torch; torch.cuda.empty_cache()"
```

### ROCm Device Not Found

```bash
# Check ROCm installation
rocminfo
rocm-smi

# Set HIP visible devices
export HIP_VISIBLE_DEVICES=0
```

### TPU XLA Compilation Slow

```bash
# Enable persistent compilation cache
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Use torch.compile with XLA
model = torch.compile(model, backend="openxla")
```

---

## Results

After running validation on all backends, results are saved to:

```
results/
├── cpu/
│   ├── test_results.json
│   ├── cross_backend_validation.json
│   └── BERT_SQUAD_CPU_REPORT.md
├── cuda/
│   ├── test_results.json
│   ├── cross_backend_validation.json
│   └── BERT_SQUAD_CUDA_REPORT.md
├── rocm/
│   └── ...
└── tpu/
    └── ...
```

Upload results to the repo:

```bash
# Copy to docs
cp -r results/ ../../docs/cloud_testing/reports/bert_squad/

# Commit
git add -A
git commit -m "docs: add BERT SQuAD cloud validation results"
git push
```
