# Claude Code Instructions for TorchBridge

This file contains instructions for Claude Code to remember how to work with this repository.

## Cloud Validation - HOW TO RUN

**IMPORTANT: When asked to run cloud validation, use these commands:**

### Quick Validation (AWS + GCP spot instances)

```bash
# Run validation on AWS g5.xlarge (A10G) and GCP g2-standard-4 (L4)
python scripts/validation/run_cloud_validation.py --provider all --tier spot
```

### AWS Only

```bash
# Single instance (A10G)
python scripts/validation/run_cloud_validation.py --provider aws --instance-type g5.xlarge

# Multiple instances
python scripts/validation/run_cloud_validation.py --provider aws --tier spot

# Full validation (includes A100)
python scripts/validation/run_cloud_validation.py --provider aws --tier full
```

### GCP Only

```bash
# Single instance (T4)
python scripts/validation/run_cloud_validation.py --provider gcp --machine-type n1-standard-4

# L4 GPU
python scripts/validation/run_cloud_validation.py --provider gcp --machine-type g2-standard-4

# A100 GPU
python scripts/validation/run_cloud_validation.py --provider gcp --machine-type a2-highgpu-1g
```

### AMD ROCm (MI300X)

```bash
# Use AMD Developer Cloud script
./scripts/cloud_testing/amd_cloud/run_validation.sh
```

### Intel XPU (DevCloud)

```bash
# Submit to Intel DevCloud
ssh devcloud 'cd torchbridge && qsub -I -l nodes=1:gpu:ppn=2 scripts/cloud_testing/intel_devcloud/run_validation.sh'
```

### BERT SQuAD Specific

```bash
# Run BERT cross-backend validation on cloud
cd examples/bert_squad
./scripts/cloud_validation.sh

# Or trigger GitHub workflow
gh workflow run "BERT SQuAD Cross-Backend Validation" -f backends="cuda,rocm,cpu"
```

## Available Instance Types

### AWS
| Instance | GPU | Spot Price |
|----------|-----|------------|
| g5.xlarge | 1x A10G | ~$0.50/hr |
| g5.2xlarge | 1x A10G | ~$0.80/hr |
| p4d.24xlarge | 8x A100 | ~$15/hr |
| p5.48xlarge | 8x H100 | ~$40/hr |

### GCP
| Machine | GPU | Notes |
|---------|-----|-------|
| n1-standard-4 | 1x T4 | Cheapest |
| g2-standard-4 | 1x L4 | Good balance |
| a2-highgpu-1g | 1x A100 | Premium |

## Checking Results

```bash
# Results are saved to
ls reports/cloud_validation/

# Check instance status
aws ec2 describe-instances --filters "Name=tag:Project,Values=TorchBridge" --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType]'

# GCP instances
gcloud compute instances list --filter="name~tb-val"
```

## Dry Run (Preview without launching)

```bash
python scripts/validation/run_cloud_validation.py --provider all --tier spot --dry-run
```

## Cost Efficiency - IMPORTANT

**Always terminate instances after validation:**

```bash
# List running instances
gcloud compute instances list --filter="name~tb-val"
aws ec2 describe-instances --filters "Name=tag:Project,Values=TorchBridge" "Name=instance-state-name,Values=running"

# Terminate GCP instances
gcloud compute instances delete INSTANCE_NAME --zone=ZONE --quiet

# Terminate AWS instances
aws ec2 terminate-instances --instance-ids INSTANCE_ID
```

**Prefer preemptible/spot instances** (auto-terminate, cheaper):
- AWS: Uses spot by default (~70% cheaper)
- GCP: Uses preemptible by default (~80% cheaper)

## Credentials Required

- AWS: `aws configure` (uses ~/.aws/credentials)
- GCP: `gcloud auth login` (uses ~/.config/gcloud)
- AWS Key Pair: `shahmod-gpu-key-east1` (or set AWS_KEY_NAME env var)
