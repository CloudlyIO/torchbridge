# Claude Code Instructions for TorchBridge

**READ THIS FILE FIRST WHEN WORKING ON THIS REPO.**

This file contains permanent configuration and instructions that Claude must follow.

---

## Cloud Validation — Complete Step-by-Step Procedures

### BERT SQuAD Validation Script (Use on ALL Backends)

```python
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn.functional as F
import time

print("=== BERT SQuAD Cross-Backend Validation ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
inputs = tokenizer("What is the capital?", "Paris is the capital of France.",
                   max_length=384, truncation=True, padding="max_length", return_tensors="pt")

model.eval()
with torch.no_grad():
    cpu_out = model(**inputs)

if torch.cuda.is_available():
    model_gpu = model.to(device)
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        gpu_out = model_gpu(**inputs_gpu)

    max_diff = torch.abs(cpu_out.start_logits - gpu_out.start_logits.cpu()).max().item()
    cos_sim = F.cosine_similarity(cpu_out.start_logits.flatten().unsqueeze(0),
                                   gpu_out.start_logits.cpu().flatten().unsqueeze(0)).item()

    for _ in range(3): model_gpu(**inputs_gpu)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): model_gpu(**inputs_gpu)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) / 100 * 1000

    print(f"Max diff: {max_diff:.2e}")
    print(f"Cosine sim: {cos_sim:.6f}")
    status = "PASSED" if max_diff < 1e-3 else "FAILED"
    print(f"Status: {status}")
    print(f"Latency: {latency:.1f} ms")
```

---

## 1. AWS CUDA (NVIDIA A10G) — Complete Procedure

### Configuration
| Setting | Value |
|---------|-------|
| Region | **us-east-1** (NOT us-west-2 — no quota there) |
| Zone | us-east-1d |
| Key Pair | shahmod-gpu-key-east1 |
| Key File | `~/.ssh/shahmod-gpu-key-east1.pem` |
| Instance | g5.xlarge (A10G, 24GB VRAM) |
| User | ubuntu |

### Step 1: Launch Instance
```bash
# Find latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
  --region us-east-1 \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning Base OSS Nvidia*Ubuntu*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

echo "Using AMI: $AMI_ID"

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
  --region us-east-1 \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --key-name shahmod-gpu-key-east1 \
  --placement AvailabilityZone=us-east-1d \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=torchbridge-val},{Key=Project,Value=TorchBridge}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: $INSTANCE_ID"
```

### Step 2: Wait for Instance and Get IP
```bash
aws ec2 wait instance-running --region us-east-1 --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances --region us-east-1 --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Public IP: $PUBLIC_IP"
```

### Step 3: SSH and Run Validation
```bash
# Wait for SSH to be ready (may take 1-2 minutes after instance running)
sleep 60

ssh -o StrictHostKeyChecking=no -i ~/.ssh/shahmod-gpu-key-east1.pem ubuntu@$PUBLIC_IP
```

### Step 4: On the Instance — Run Validation
```bash
# Install dependencies
pip install transformers --break-system-packages -q

# Verify GPU
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Run validation (copy the validation script above or use inline)
python3 -c "
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn.functional as F
import time

device = torch.device('cuda')
print(f'Device: {torch.cuda.get_device_name(0)}')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
inputs = tokenizer('What is the capital?', 'Paris is the capital of France.', max_length=384, truncation=True, padding='max_length', return_tensors='pt')

model.eval()
with torch.no_grad(): cpu_out = model(**inputs)

model_gpu = model.to(device)
inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad(): gpu_out = model_gpu(**inputs_gpu)

max_diff = torch.abs(cpu_out.start_logits - gpu_out.start_logits.cpu()).max().item()
cos_sim = F.cosine_similarity(cpu_out.start_logits.flatten().unsqueeze(0), gpu_out.start_logits.cpu().flatten().unsqueeze(0)).item()

for _ in range(3): model_gpu(**inputs_gpu)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100): model_gpu(**inputs_gpu)
torch.cuda.synchronize()
latency = (time.perf_counter() - t0) / 100 * 1000

print(f'Max diff: {max_diff:.2e}')
print(f'Cosine sim: {cos_sim:.6f}')
print(f'Status: {\"PASSED\" if max_diff < 1e-4 else \"FAILED\"}')
print(f'Latency: {latency:.1f} ms')
"
```

### Step 5: TERMINATE INSTANCE (CRITICAL!)
```bash
aws ec2 terminate-instances --region us-east-1 --instance-ids $INSTANCE_ID
```

### Expected Results
```
Device: NVIDIA A10G
Max diff: 2.34e-06
Cosine sim: 1.000000
Status: PASSED
Latency: 7.4 ms
```

---

## 2. GCP CUDA (NVIDIA T4) — Complete Procedure

### Configuration
| Setting | Value |
|---------|-------|
| Project | shahmod-kernel-pytorch |
| Zone | us-central1-a |
| Machine | n1-standard-4 + T4 |
| Image | pytorch-2-7-cu128-ubuntu-2404-nvidia-570 |
| Preemptible | Yes (cost savings) |

### Step 1: Launch Instance
```bash
INSTANCE_NAME="tb-val-$(date +%s)"

gcloud compute instances create $INSTANCE_NAME \
  --project=shahmod-kernel-pytorch \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --preemptible

echo "Instance: $INSTANCE_NAME"
```

### Step 2: SSH and Run Validation
```bash
# Wait for instance to be ready
sleep 60

gcloud compute ssh $INSTANCE_NAME --zone=us-central1-a --project=shahmod-kernel-pytorch
```

### Step 3: On the Instance — Run Validation
```bash
# Install dependencies
pip install transformers -q

# Verify GPU
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Run validation (same script as AWS)
python3 -c "
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn.functional as F
import time

device = torch.device('cuda')
print(f'Device: {torch.cuda.get_device_name(0)}')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
inputs = tokenizer('What is the capital?', 'Paris is the capital of France.', max_length=384, truncation=True, padding='max_length', return_tensors='pt')

model.eval()
with torch.no_grad(): cpu_out = model(**inputs)

model_gpu = model.to(device)
inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad(): gpu_out = model_gpu(**inputs_gpu)

max_diff = torch.abs(cpu_out.start_logits - gpu_out.start_logits.cpu()).max().item()
cos_sim = F.cosine_similarity(cpu_out.start_logits.flatten().unsqueeze(0), gpu_out.start_logits.cpu().flatten().unsqueeze(0)).item()

for _ in range(3): model_gpu(**inputs_gpu)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100): model_gpu(**inputs_gpu)
torch.cuda.synchronize()
latency = (time.perf_counter() - t0) / 100 * 1000

print(f'Max diff: {max_diff:.2e}')
print(f'Cosine sim: {cos_sim:.6f}')
print(f'Status: {\"PASSED\" if max_diff < 1e-4 else \"FAILED\"}')
print(f'Latency: {latency:.1f} ms')
"
```

### Step 4: TERMINATE INSTANCE (CRITICAL!)
```bash
gcloud compute instances delete $INSTANCE_NAME --zone=us-central1-a --project=shahmod-kernel-pytorch --quiet
```

### Expected Results
```
Device: Tesla T4
Max diff: 2.52e-06
Cosine sim: 1.000000
Status: PASSED
Latency: 21.8 ms
```

---

## 3. AMD ROCm (MI300X) — Complete Procedure

### Configuration
| Setting | Value |
|---------|-------|
| Portal | https://www.amd.com/en/developer/resources/ai-cloud.html |
| SSH Key | `~/.ssh/id_ed25520` |
| User | root |
| Instance Type | 2.6.0---ROCm-7.0-gpu-mi300x1-192gb-devcloud-atl1 |
| Public IP | 129.212.181.53 (update if instance relaunched) |
| GPU | AMD Instinct MI300X VF (gfx942, CDNA3, 192GB HBM3) |

### Step 1: SSH to Instance
```bash
ssh -i ~/.ssh/id_ed25520 root@129.212.181.53
```

**If connection refused:**
1. Go to AMD Developer Cloud portal
2. Relaunch instance: `2.6.0---ROCm-7.0-gpu-mi300x1-192gb-devcloud-atl1`
3. Get new public IP and update this file
4. Retry SSH

### Step 2: Install Dependencies (if needed)
```bash
# Check if PyTorch ROCm is installed
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || \
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 --break-system-packages -q

pip install transformers --break-system-packages -q
```

### Step 3: Verify GPU
```bash
rocm-smi --showproductname

python3 -c "import torch; print(f'PyTorch {torch.__version__}, ROCm: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Run Validation
```bash
python3 -c "
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn.functional as F
import time

device = torch.device('cuda')
print(f'Device: {torch.cuda.get_device_name(0)}')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
inputs = tokenizer('What is the capital?', 'Paris is the capital of France.', max_length=384, truncation=True, padding='max_length', return_tensors='pt')

model.eval()
with torch.no_grad(): cpu_out = model(**inputs)

model_gpu = model.to(device)
inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad(): gpu_out = model_gpu(**inputs_gpu)

max_diff = torch.abs(cpu_out.start_logits - gpu_out.start_logits.cpu()).max().item()
cos_sim = F.cosine_similarity(cpu_out.start_logits.flatten().unsqueeze(0), gpu_out.start_logits.cpu().flatten().unsqueeze(0)).item()

for _ in range(3): model_gpu(**inputs_gpu)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100): model_gpu(**inputs_gpu)
torch.cuda.synchronize()
latency = (time.perf_counter() - t0) / 100 * 1000

print(f'Max diff: {max_diff:.2e}')
print(f'Cosine sim: {cos_sim:.6f}')
status = 'PASSED' if max_diff < 1e-3 else 'FAILED'
print(f'Status: {status}')
print(f'Latency: {latency:.1f} ms')
"
```

### Expected Results
```
Device: AMD Instinct MI300X VF
Max diff: 2.03e-06
Cosine sim: 1.000000
Status: PASSED
Latency: 5.5 ms
```

**Note:** ROCm tolerance is 1e-3 (vs 1e-4 for CUDA) due to SDPA flash attention divergence. However, actual results are often better.

---

## Validation Thresholds

| Backend | Max Diff Tolerance | Cosine Sim Threshold | Notes |
|---------|-------------------|---------------------|-------|
| CUDA (NVIDIA) | 1e-4 | 0.9999 | Exact parity expected |
| ROCm (AMD) | 1e-3 | 0.999 | SDPA flash attention divergence |
| MPS (Apple) | 1e-4 | 0.9999 | Apple Silicon |
| TPU (XLA) | 1e-3 | 0.999 | XLA compilation differences |

---

## Cost Efficiency — CRITICAL

**ALWAYS terminate instances after validation!**

### Check Running Instances
```bash
# AWS
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:Project,Values=TorchBridge" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,PublicIpAddress]' --output table

# GCP
gcloud compute instances list --filter="name~tb-val" --project=shahmod-kernel-pytorch
```

### Terminate All
```bash
# AWS - terminate specific instance
aws ec2 terminate-instances --region us-east-1 --instance-ids INSTANCE_ID

# GCP - delete specific instance
gcloud compute instances delete INSTANCE_NAME --zone=us-central1-a --project=shahmod-kernel-pytorch --quiet
```

### Cost Estimates
| Provider | Instance | GPU | Cost/Hour |
|----------|----------|-----|-----------|
| AWS | g5.xlarge | A10G | ~$1.00 on-demand, ~$0.30 spot |
| GCP | n1-standard-4 + T4 | T4 | ~$0.35 preemptible |
| AMD | Developer Cloud | MI300X | Free (developer program) |

---

## Validation Results History

| Date | Backend | GPU | Max Diff | Cosine Sim | Latency | Status |
|------|---------|-----|----------|------------|---------|--------|
| 2026-02-07 | AMD ROCm | MI300X | 2.03e-06 | 1.000000 | 5.5ms | PASSED |
| 2026-02-07 | AWS CUDA | A10G | 2.34e-06 | 1.000000 | 7.4ms | PASSED |
| 2026-02-07 | GCP CUDA | T4 | 2.52e-06 | 1.000000 | 21.8ms | PASSED |
| 2026-02-03 | AMD ROCm | MI300X | - | - | - | PASSED (1611 tests) |
| 2026-01-28 | AWS CUDA | A10G | - | - | - | PASSED (66 tests) |

### Performance Comparison (CPU Baseline: 450ms)
| GPU | Latency | Speedup vs CPU |
|-----|---------|----------------|
| AMD MI300X | 5.5ms | **82x** |
| NVIDIA A10G | 7.4ms | 61x |
| NVIDIA T4 | 21.8ms | 21x |

---

## Reports Location

Validation reports are saved to: `reports/cloud_validation/YYYY-MM-DD/`

```
reports/cloud_validation/
└── 2026-02-07/
    ├── amd_mi300x_bert_squad.json
    ├── aws_a10g_bert_squad.json
    ├── gcp_t4_bert_squad.json
    └── summary.json
```

---

## Troubleshooting

### AWS: "MaxSpotInstanceCountExceeded"
- Use on-demand instead of spot, or request quota increase

### GCP: "GPUS_ALL_REGIONS quota exceeded"
- Request GPU quota increase at https://console.cloud.google.com/iam-admin/quotas

### AMD: "Connection refused"
1. Instance may have been terminated (ephemeral)
2. Relaunch from AMD Developer Cloud portal
3. Update IP in this file
4. SSH may take a few minutes to become available after instance launch

### PyTorch not found on AMD
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 --break-system-packages
```

---

## Presentation Deck — MANDATORY Release Step

**File:** `TorchBridge_Presentation.pptx` (local only, gitignored — NEVER commit)

**TorchBridge is NOT open-source. Never add open-source references to the deck or marketing materials.**

### On Every Release, Update the Deck

Use `python-pptx` to programmatically update the `.pptx` file. The deck has 20 slides.
Key slides to update:

| Slide | Content | What to Update |
|-------|---------|----------------|
| 1 | Title | Version number (TextBox 5) |
| 4 | Solution | Cloud validation status (TextBox 19) |
| 9 | CLI | Command count + descriptions (TextBox 3, 32) |
| 10 | Performance | Benchmark data if new hardware tested |
| 12 | CI/CD | Test count (TextBox 33), CLI test count (TextBox 35) |
| 15 | Code Quality | Lines (TextBox 5), modules (TextBox 9), tests (TextBox 13), release cadence (TextBox 37) |
| 19 | Roadmap | Milestone statuses, version labels, descriptions |
| 20 | Get Started | pip install command (TextBox 4, 5), footer tagline (TextBox 6) |

### How to Update (Pattern)

```python
from pptx import Presentation
prs = Presentation('TorchBridge_Presentation.pptx')

# Replace text in a specific shape, preserving formatting:
slide = prs.slides[SLIDE_INDEX]  # 0-indexed
for shape in slide.shapes:
    if shape.name == 'TARGET_SHAPE_NAME':
        for para in shape.text_frame.paragraphs:
            if para.runs:
                para.runs[0].text = 'NEW TEXT'
                for run in para.runs[1:]:
                    run.text = ''

prs.save('TorchBridge_Presentation.pptx')
```

### Checklist

1. Update version number on Slide 1 and Slide 20
2. Update code metrics (run `pytest --co` count, count source lines/modules)
3. Update roadmap milestones (mark completed items, add new planned items)
4. Update cloud validation status with any new hardware results
5. Update CLI command count if new commands were added
6. Verify NO "open source" or "open-source" text exists anywhere in the deck
7. Verify pip install uses `torchbridge-ml` (the PyPI package name)
8. Verify all benchmark data references real measured results only
