# Claude Code Instructions for TorchBridge

**READ THIS FILE FIRST WHEN WORKING ON THIS REPO.**

This file contains permanent configuration and instructions that Claude must follow.

---

## Cloud Validation - EXACT COMMANDS

### AWS CUDA (A10G) - WORKING CONFIG

```bash
# Region: us-east-1 (NOT us-west-2)
# Key: shahmod-gpu-key-east1
# Zone: us-east-1d

# Find latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
  --region us-east-1 \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning Base OSS Nvidia*Ubuntu*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

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

# Wait and get IP
aws ec2 wait instance-running --region us-east-1 --instance-ids $INSTANCE_ID
PUBLIC_IP=$(aws ec2 describe-instances --region us-east-1 --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# SSH and run validation
ssh -i ~/.ssh/shahmod-gpu-key-east1.pem ubuntu@$PUBLIC_IP

# ALWAYS TERMINATE AFTER VALIDATION
aws ec2 terminate-instances --region us-east-1 --instance-ids $INSTANCE_ID
```

### GCP CUDA (T4/L4) - WORKING CONFIG

```bash
# Project: shahmod-kernel-pytorch
# Zone: us-central1-a

# Launch T4 instance
gcloud compute instances create tb-val-$(date +%s) \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --preemptible

# ALWAYS TERMINATE AFTER VALIDATION
gcloud compute instances delete INSTANCE_NAME --zone=us-central1-a --quiet
```

### AMD ROCm - CONFIGURATION NEEDED

**Previous validation (Feb 3, 2026):**
- Device: AMD Instinct MI300X VF (gfx942, CDNA3, 192GB VRAM)
- Result: 1611 passed, 76 skipped, 0 failed
- Commit: 9faf7c4

**AWS AMD GPU options:**
- g4ad.xlarge (Radeon Pro V520) - requires vCPU quota increase
- Request quota at: https://aws.amazon.com/contact-us/ec2-request

**AMD Developer Cloud:**
- Access: https://www.amd.com/en/developer/resources/ai-cloud.html
- $100 credits via AI Developer Program
- MI300X instances with 192GB HBM3

**TODO: Add AMD cloud SSH access details here when available**

```bash
# After getting access, run:
./scripts/cloud_testing/amd_cloud/run_validation.sh
```

### Validation Script on Any Instance

```bash
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge
pip install --break-system-packages torch transformers -q
python3 examples/bert_squad/validate_cross_backend.py --tolerance 1e-4
```

---

## AWS Configuration Details

| Setting | Value |
|---------|-------|
| Region | **us-east-1** (NOT us-west-2) |
| Zone | us-east-1d |
| Key Pair | shahmod-gpu-key-east1 |
| Key File | ~/.ssh/shahmod-gpu-key-east1.pem |
| AMI | Deep Learning Base OSS Nvidia Ubuntu (latest) |
| Instance | g5.xlarge (A10G, 24GB) |

## GCP Configuration Details

| Setting | Value |
|---------|-------|
| Project | shahmod-kernel-pytorch |
| Zone | us-central1-a |
| Machine | n1-standard-4 + T4, or g2-standard-4 (L4) |
| Image | pytorch-2-7-cu128-ubuntu-2404-nvidia-570 |
| GPU Quota | 1 GPU (request increase for more) |

---

## Cost Efficiency - CRITICAL

**ALWAYS terminate instances after validation:**

```bash
# Check running instances
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:Project,Values=TorchBridge" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output table

gcloud compute instances list --filter="name~tb-val"

# Terminate ALL
aws ec2 terminate-instances --region us-east-1 --instance-ids INSTANCE_ID
gcloud compute instances delete INSTANCE_NAME --zone=us-central1-a --quiet
```

**Estimated costs:**
- AWS g5.xlarge: ~$1.00/hour on-demand, ~$0.30/hour spot
- GCP n1-standard-4 + T4: ~$0.35/hour preemptible
- Always use preemptible/spot when possible

---

## Previous Successful Validations

| Date | Backend | GPU | Result | Latency |
|------|---------|-----|--------|---------|
| 2026-02-07 | AWS CUDA | A10G | PASSED (diff 2.34e-06) | 7.4ms |
| 2026-02-07 | GCP CUDA | T4 | PASSED (diff 2.52e-06) | 21.8ms |
| 2026-01-28 | AWS CUDA | A10G | PASSED (66 tests) | - |

---

## Quick Reference

```bash
# Run cloud validation script (auto-detects backend)
python scripts/validation/run_cloud_validation.py --provider aws --tier dev

# BERT SQuAD specific
cd examples/bert_squad
./scripts/cloud_validation.sh

# List available instance types
python scripts/validation/run_cloud_validation.py --list
```
