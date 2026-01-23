# Validated Cloud Hardware Testing Guide

> **Version**: v0.4.15 | **Status**: ✅ Current | **Last Updated**: Jan 22, 2026

**Tested and verified steps for cloud GPU/TPU testing.**

## Quick Reference

| Platform | Backend | Instance Type | GPU | Tests | Benchmarks | Status |
|----------|---------|---------------|-----|-------|------------|--------|
| GCP | NVIDIA | g2-standard-4 | L4 (23GB) | 66 | 1300 | Validated |
| AWS | NVIDIA | g5.xlarge | A10G (23GB) | 66 | 1300 | Validated |
| GCP | TPU | v5litepod-1 | TPU v5e | 56 | 7 | Validated |
| AMD | - | See AMD section | MI300X | 41 | 20 | Local only |

---

## GCP NVIDIA Testing (Validated)

### Prerequisites
```bash
# Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Request GPU Quota (Required First Time)
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter: `GPUS_ALL_REGIONS`
3. Request increase to at least 1
4. Wait for approval (usually 1-24 hours)

### Launch GPU Instance
```bash
# Create instance with L4 GPU (validated working)
gcloud compute instances create shahmod-nvidia-test \
  --zone=us-west1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --project=YOUR_PROJECT_ID

# Alternative zones if capacity exhausted:
# us-west1-b, us-central1-a, us-east1-b
```

### Upload Code and Run Tests
```bash
# Upload code (from local machine)
gcloud compute scp --recurse src/kernel_pytorch shahmod-nvidia-test:~/kernel_pytorch --zone=us-west1-a
gcloud compute scp --recurse tests shahmod-nvidia-test:~/tests --zone=us-west1-a
gcloud compute scp --recurse benchmarks shahmod-nvidia-test:~/benchmarks --zone=us-west1-a

# SSH and run tests
gcloud compute ssh shahmod-nvidia-test --zone=us-west1-a --command="
  pip install pytest psutil -q && \
  cd ~ && \
  PYTHONPATH=/home/\$USER python3 -m pytest tests/test_nvidia_backend.py -v --tb=short
"

# Run benchmarks
gcloud compute ssh shahmod-nvidia-test --zone=us-west1-a --command="
  cd ~ && \
  PYTHONPATH=/home/\$USER python3 benchmarks/nvidia_integration_benchmark.py
"
```

### Cleanup (Important!)
```bash
gcloud compute instances delete shahmod-nvidia-test --zone=us-west1-a --quiet
```

---

## AWS NVIDIA Testing (Validated)

### Prerequisites
```bash
# Configure AWS CLI
aws configure
# Set region (us-east-2 has GPU quota)
export AWS_DEFAULT_REGION=us-east-2
```

### Request GPU Quota (Required First Time)
1. Go to: https://us-east-2.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA
2. Request increase to at least 4 vCPUs for "Running On-Demand G and VT instances"
3. Wait for approval (usually 1-48 hours)

### Setup Security Group and Key Pair
```bash
# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --query "Subnets[0].SubnetId" --output text)

# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name shahmod-gpu-sg \
  --description "GPU testing security group" \
  --vpc-id $VPC_ID \
  --query "GroupId" --output text)

# Allow SSH
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Create key pair
aws ec2 create-key-pair \
  --key-name shahmod-gpu-key \
  --query "KeyMaterial" --output text > ~/.ssh/shahmod-gpu-key.pem
chmod 600 ~/.ssh/shahmod-gpu-key.pem
```

### Launch GPU Instance
```bash
# Find Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning*Ubuntu*" \
  --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
  --output text)

# Launch g5.xlarge (A10G GPU)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --key-name shahmod-gpu-key \
  --security-group-ids $SG_ID \
  --subnet-id $SUBNET_ID \
  --associate-public-ip-address \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=shahmod-nvidia-test}]' \
  --query "Instances[0].InstanceId" --output text)

# Wait for instance
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo "Instance IP: $PUBLIC_IP"
```

### Upload Code and Run Tests
```bash
# Wait for SSH to be ready
sleep 30

# Upload code
scp -i ~/.ssh/shahmod-gpu-key.pem -o StrictHostKeyChecking=no -r \
  src/kernel_pytorch tests benchmarks \
  ubuntu@$PUBLIC_IP:~/

# Install dependencies and run tests
ssh -i ~/.ssh/shahmod-gpu-key.pem ubuntu@$PUBLIC_IP "
  pip install torch --index-url https://download.pytorch.org/whl/cu121 -q && \
  pip install pytest psutil numpy -q && \
  cd ~ && \
  PYTHONPATH=kernel_pytorch python3 -m pytest tests/test_nvidia_backend.py -v --tb=short
"

# Run benchmarks
ssh -i ~/.ssh/shahmod-gpu-key.pem ubuntu@$PUBLIC_IP "
  cd ~ && \
  PYTHONPATH=/home/ubuntu python3 benchmarks/nvidia_integration_benchmark.py
"
```

### Cleanup (Important!)
```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

---

## GCP TPU Testing (Validated)

### Request TPU Quota
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter: `TPU v5 lite PodSlice chips`
3. Request increase to at least 1
4. Wait for approval

### Launch TPU VM
```bash
gcloud compute tpus tpu-vm create shahmod-tpu-test \
  --zone=us-central1-a \
  --accelerator-type=v5litepod-1 \
  --version=tpu-ubuntu2204-base \
  --project=YOUR_PROJECT_ID
```

### Upload Code and Run Tests
```bash
# Upload code
gcloud compute tpus tpu-vm scp --recurse src/kernel_pytorch shahmod-tpu-test:~/kernel_pytorch --zone=us-central1-a
gcloud compute tpus tpu-vm scp --recurse tests shahmod-tpu-test:~/tests --zone=us-central1-a

# Install dependencies
gcloud compute tpus tpu-vm ssh shahmod-tpu-test --zone=us-central1-a --command="
  pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html -q && \
  pip install pytest psutil numpy -q
"

# Run tests
gcloud compute tpus tpu-vm ssh shahmod-tpu-test --zone=us-central1-a --command="
  cd ~ && \
  PYTHONPATH=kernel_pytorch python3 -m pytest tests/test_tpu_backend.py -v --tb=short
"
```

### Cleanup
```bash
gcloud compute tpus tpu-vm delete shahmod-tpu-test --zone=us-central1-a --quiet
```

---

## AMD GPU Testing Options

AMD GPUs require ROCm environment. Cloud options include:

### AMD Developer Cloud (Free Credits)
- URL: https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html
- GPUs: MI300X (192GB HBM3)
- Credits: 25 hours free for qualified developers
- Best for: Initial testing and validation

### Commercial Providers
- **Crusoe Cloud**: MI300X instances
- **CUDO Compute**: MI250/MI300 instances
- **Cirrascale**: AMD Instinct Series Cloud

### Local Testing (No ROCm)
```bash
# Run AMD tests locally (CPU fallback)
PYTHONPATH=src python3 -m pytest tests/test_amd_backend.py -v
# Expected: 41 passed, 3 skipped

# Run AMD benchmarks locally
PYTHONPATH=src python3 benchmarks/amd_integration_benchmark.py --quick
# Expected: 20 benchmarks passed
```

---

## Complete Test Suite Commands

### Full NVIDIA Test Suite
```bash
# Tests
PYTHONPATH=kernel_pytorch python3 -m pytest tests/test_nvidia_backend.py -v --tb=short

# Integration Benchmark
PYTHONPATH=/home/$USER python3 benchmarks/nvidia_integration_benchmark.py

# Config Benchmark
PYTHONPATH=/home/$USER python3 benchmarks/nvidia_config_benchmarks.py
```

### Full TPU Test Suite
```bash
# Tests
PYTHONPATH=kernel_pytorch python3 -m pytest tests/test_tpu_backend.py -v --tb=short

# Benchmark
PYTHONPATH=/home/$USER python3 benchmarks/tpu_integration_benchmark.py
```

### Full AMD Test Suite
```bash
# Tests
PYTHONPATH=kernel_pytorch python3 -m pytest tests/test_amd_backend.py -v --tb=short

# Benchmark
PYTHONPATH=/home/$USER python3 benchmarks/amd_integration_benchmark.py
```

---

## Troubleshooting

### GCP: "GPUS_ALL_REGIONS exceeded"
Request GPU quota increase at https://console.cloud.google.com/iam-admin/quotas

### GCP: "Zone does not have enough resources"
Try alternative zones: us-west1-a, us-central1-b, us-east1-c

### AWS: "VcpuLimitExceeded"
Request quota increase for "Running On-Demand G and VT instances" in Service Quotas

### PyTorch CUDA not detecting GPU
```bash
# Check CUDA version compatibility
nvidia-smi
python3 -c "import torch; print(torch.version.cuda)"

# Reinstall matching version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Missing dependencies
```bash
pip install pytest psutil numpy torch
```

---

## Cost Monitoring

### Check for running resources
```bash
# GCP
gcloud compute instances list
gcloud compute tpus tpu-vm list --zone=us-central1-a
gcloud compute disks list

# AWS
aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId,State.Name]"
aws ec2 describe-volumes
```

### Estimated Costs
| Resource | Cost/Hour |
|----------|-----------|
| GCP g2-standard-4 (L4) | ~$0.70 |
| AWS g5.xlarge (A10G) | ~$1.00 |
| GCP TPU v5litepod-1 | ~$1.20 |

---

## Expected Results Summary

### NVIDIA Backend (66 tests, 1300 benchmarks)
```
Performance Metrics (GCP L4):
- Backend creation: 0.30ms
- Model preparation: 1.21ms
- FlashAttention forward: 5.28ms
- Optimization throughput: ~450K samples/sec

Performance Metrics (AWS A10G):
- Backend creation: 0.28ms
- Model preparation: 1.41ms
- FlashAttention forward: 7.01ms
```

### TPU Backend (56 tests, 7 benchmarks)
```
- XLA compilation working
- Memory manager functional
- torch_xla 2.9.0 compatible
```

### AMD Backend (41 tests, 20 benchmarks)
```
- All core functionality tested
- 3 tests skipped (require ROCm hardware)
- CPU fallback working correctly
```
