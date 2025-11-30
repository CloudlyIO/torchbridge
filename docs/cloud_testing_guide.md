# 🌩️ Cloud Platform Testing Guide for CUDA & Triton

**Complete instructions for testing PyTorch kernel optimizations on public cloud platforms.**

This guide covers hardware testing setup for:
- **AWS** - EC2 GPU instances (P3, P4, G4, G5)
- **Azure** - NC/ND/NV series GPU VMs
- **GCP** - GPU-enabled Compute Engine instances
- **OCP** - OpenShift Container Platform with GPU operators

## 🎯 Overview

### Why Cloud Testing?

1. **Hardware Access** - Test on different GPU generations (V100, A100, H100)
2. **Scalability** - Multi-GPU and distributed testing
3. **Cost Efficiency** - Pay-per-use for expensive hardware
4. **CI/CD Integration** - Automated testing pipelines
5. **Production Simulation** - Real deployment environment testing

### Quick Start Matrix

| Platform | GPU Types | Best For | Setup Time |
|----------|-----------|----------|------------|
| **AWS EC2** | P3 (V100), P4 (A100), G4 (T4) | Production testing, large-scale | 10-15 min |
| **Azure** | NC (K80/P100), ND (V100), NV (A10/A100) | Enterprise integration | 10-15 min |
| **GCP** | T4, V100, A100, TPU | Research, cost optimization | 10-15 min |
| **OCP** | Node Feature Discovery | Container-native, Kubernetes | 15-20 min |

## 🚀 AWS EC2 GPU Testing

### Instance Selection

**Recommended Instances:**
```bash
# P3 instances (V100) - Balanced performance/cost
# P3.2xlarge:  1 V100 (16GB), $3.06/hour
# P3.8xlarge:  4 V100 (64GB), $12.24/hour

# P4 instances (A100) - Latest generation
# P4d.24xlarge: 8 A100 (320GB), $32.77/hour

# G4 instances (T4) - Cost-effective
# G4dn.xlarge: 1 T4 (16GB), $0.526/hour
```

### Setup Instructions

#### 1. Launch Instance
```bash
# Using AWS CLI
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type p3.2xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --user-data file://cloud-init.sh

# Or use AWS Console:
# 1. Go to EC2 → Launch Instance
# 2. Choose "Deep Learning AMI (Ubuntu 20.04)"
# 3. Select P3.2xlarge or P4d.24xlarge
# 4. Configure security group (SSH port 22)
```

#### 2. Instance Setup Script
```bash
# Save as cloud-init.sh
#!/bin/bash
set -e

# Update system
apt-get update && apt-get upgrade -y

# Install basic tools
apt-get install -y git wget curl vim htop tmux

# Verify NVIDIA driver and CUDA
nvidia-smi
nvcc --version

# Install Python and pip
apt-get install -y python3.11 python3.11-pip python3.11-venv

# Create project directory
mkdir -p /opt/pytorch-optimization
cd /opt/pytorch-optimization

# Clone repository (replace with your repo)
git clone https://github.com/yourusername/pytorch-optimization.git .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Triton and other dependencies
pip install triton numpy scipy pytest

# Install project requirements
pip install -r requirements.txt

echo "Setup complete! SSH into instance and run tests."
```

#### 3. Connect and Test
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Navigate to project
cd /opt/pytorch-optimization
source venv/bin/activate

# Verify GPU availability
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Run comprehensive tests
export PYTHONPATH=src:$PYTHONPATH
python3 -m pytest tests/ -v -m gpu
```

### AWS-Specific Optimizations

#### Spot Instances for Cost Savings
```bash
# Launch spot instance (up to 90% savings)
aws ec2 request-spot-instances \
    --spot-price "1.50" \
    --launch-specification '{
        "ImageId": "ami-0c02fb55956c7d316",
        "InstanceType": "p3.2xlarge",
        "KeyName": "your-key-pair",
        "SecurityGroupIds": ["sg-xxxxxxxxx"],
        "SubnetId": "subnet-xxxxxxxxx"
    }'
```

#### EBS Optimization for Large Datasets
```bash
# Create high-performance EBS volume
aws ec2 create-volume \
    --size 500 \
    --volume-type gp3 \
    --iops 10000 \
    --throughput 1000 \
    --availability-zone us-west-2a

# Attach to instance
aws ec2 attach-volume \
    --volume-id vol-xxxxxxxxx \
    --instance-id i-xxxxxxxxx \
    --device /dev/sdf
```

## 🔵 Azure GPU Testing

### VM Selection

**Recommended Series:**
```bash
# NC series (K80, older generation)
# Standard_NC6: 1 K80 (12GB), ~$0.90/hour

# NCv3 series (V100)
# Standard_NC6s_v3: 1 V100 (16GB), ~$3.06/hour
# Standard_NC24s_v3: 4 V100 (64GB), ~$12.24/hour

# ND series (A100)
# Standard_ND96asr_v4: 8 A100 (320GB), ~$27.20/hour
```

### Setup Instructions

#### 1. Create Resource Group and VM
```bash
# Using Azure CLI
az group create --name pytorch-optimization-rg --location eastus

# Create VM with GPU
az vm create \
    --resource-group pytorch-optimization-rg \
    --name pytorch-gpu-vm \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
    --size Standard_NC6s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys \
    --custom-data cloud-init.sh
```

#### 2. Azure Setup Script
```bash
# Save as azure-setup.sh
#!/bin/bash
set -e

# Update package list
apt-get update

# Install NVIDIA drivers for Azure
apt-get install -y linux-headers-$(uname -r)

# Install NVIDIA driver
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get -y install cuda-drivers

# Reboot required for driver loading
reboot
```

#### 3. Post-Reboot Setup
```bash
# SSH into VM
ssh azureuser@your-vm-ip

# Verify GPU
nvidia-smi

# Install CUDA toolkit and Python
sudo apt-get install -y cuda-toolkit-12-1 python3.11 python3.11-pip python3.11-venv

# Clone and setup project
git clone https://github.com/yourusername/pytorch-optimization.git
cd pytorch-optimization

python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton -r requirements.txt

# Test setup
export PYTHONPATH=src:$PYTHONPATH
python3 tests/test_gpu_setup.py
```

### Azure-Specific Features

#### Azure Machine Learning Integration
```python
# azureml-setup.py
from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute

# Connect to workspace
ws = Workspace.from_config()

# Create GPU cluster
compute_config = AmlCompute.provisioning_configuration(
    vm_size="Standard_NC6s_v3",
    max_nodes=4,
    min_nodes=0,
    idle_seconds_before_scaledown=1800
)

gpu_cluster = ComputeTarget.create(
    workspace=ws,
    name="pytorch-gpu-cluster",
    provisioning_configuration=compute_config
)

# Submit training job
experiment = Experiment(workspace=ws, name="pytorch-optimization-test")
```

## 🟠 Google Cloud Platform (GCP) Testing

### Instance Types

**Recommended Configurations:**
```bash
# N1 with T4 (cost-effective)
# n1-standard-4 + 1 T4: ~$0.35/hour + $0.35/hour

# N1 with V100
# n1-standard-8 + 1 V100: ~$0.38/hour + $2.48/hour

# A2 with A100 (latest)
# a2-highgpu-1g: 1 A100 (40GB), ~$3.67/hour
```

### Setup Instructions

#### 1. Create Instance
```bash
# Using gcloud CLI
gcloud compute instances create pytorch-gpu-vm \
    --zone=us-central1-c \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd
```

#### 2. Connect and Setup
```bash
# SSH into instance
gcloud compute ssh pytorch-gpu-vm --zone=us-central1-c

# Verify GPU and drivers
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Clone project
git clone https://github.com/yourusername/pytorch-optimization.git
cd pytorch-optimization

# Install dependencies (PyTorch pre-installed in deep learning images)
pip install triton -r requirements.txt

# Run tests
export PYTHONPATH=src:$PYTHONPATH
python3 -m pytest tests/test_comprehensive_integration.py -v
```

### GCP-Specific Optimizations

#### Preemptible Instances
```bash
# Create preemptible instance (up to 80% savings)
gcloud compute instances create pytorch-preemptible \
    --zone=us-central1-c \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --preemptible \
    --metadata="install-nvidia-driver=True"
```

#### TPU Integration (Alternative to GPU)
```python
# tpu-setup.py
import torch_xla.core.xla_model as xm

# Check TPU availability
device = xm.xla_device()
print(f'TPU device: {device}')

# Run model on TPU
model = YourModel().to(device)
data = torch.randn(32, 768).to(device)
output = model(data)
```

## 🔴 OpenShift Container Platform (OCP) Testing

### Prerequisites

**Requirements:**
- OpenShift 4.8+ cluster with GPU operators
- `oc` CLI tool installed
- Cluster admin access or GPU project permissions

### Setup Instructions

#### 1. Install GPU Operators
```yaml
# gpu-operator.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nvidia-gpu-operator
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: nvidia-gpu-operator-group
  namespace: nvidia-gpu-operator
spec:
  targetNamespaces:
  - nvidia-gpu-operator
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: gpu-operator-certified
  namespace: nvidia-gpu-operator
spec:
  channel: v23.9
  name: gpu-operator-certified
  source: certified-operators
  sourceNamespace: openshift-marketplace
```

```bash
# Apply GPU operator
oc apply -f gpu-operator.yaml

# Wait for operator installation
oc get pods -n nvidia-gpu-operator -w
```

## 🧪 Common Testing Scenarios

### 1. Performance Benchmarking
```bash
# Create benchmark script
cat > cloud_benchmark.py << 'EOF'
#!/usr/bin/env python3
"""Cloud GPU Performance Benchmark"""

import torch
import time
import json
from datetime import datetime

def benchmark_attention(batch_size=32, seq_len=512, embed_dim=768):
    """Benchmark attention computation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Standard attention
    start_time = time.perf_counter()
    with torch.no_grad():
        attn = torch.nn.MultiheadAttention(embed_dim, 8).to(device)
        output, _ = attn(x, x, x)
    standard_time = time.perf_counter() - start_time

    return {
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        'batch_size': batch_size,
        'seq_len': seq_len,
        'embed_dim': embed_dim,
        'standard_time': standard_time,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == '__main__':
    results = benchmark_attention()
    print(json.dumps(results, indent=2))
EOF

# Run benchmark
python3 cloud_benchmark.py
```

### 2. Multi-GPU Testing
```python
# multi_gpu_test.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    """Setup distributed training"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def test_distributed(rank, world_size):
    """Test distributed computation"""
    setup(rank, world_size)

    # Create model and data
    model = torch.nn.Linear(1024, 512).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    data = torch.randn(32, 1024).to(rank)
    output = model(data)

    print(f'Rank {rank}: Output shape {output.shape}')
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(test_distributed, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
```

## 📊 Monitoring and Debugging

### GPU Utilization Monitoring
```bash
# monitor_gpu.sh
#!/bin/bash
echo "=== GPU Monitoring ==="
while true; do
    clear
    echo "$(date): GPU Status"
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo ""
    echo "Running Processes:"
    nvidia-smi pmon -c 1
    sleep 5
done
```

### Performance Profiling
```python
# profiling.py
import torch.profiler

def profile_model():
    """Profile model execution"""
    model = torch.nn.Linear(1024, 512).cuda()
    x = torch.randn(32, 1024).cuda()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            output = model(x)
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == '__main__':
    profile_model()
```

## 💰 Cost Optimization

### Spot/Preemptible Instance Strategies
```bash
# cost_optimizer.sh
#!/bin/bash

# Function to get current spot prices
get_spot_price() {
    local instance_type=$1
    local region=$2

    aws ec2 describe-spot-price-history \
        --instance-types $instance_type \
        --product-descriptions "Linux/UNIX" \
        --max-items 1 \
        --region $region \
        --output table
}

# Check prices for different GPU instances
echo "Current Spot Prices:"
get_spot_price "p3.2xlarge" "us-west-2"
get_spot_price "p4d.24xlarge" "us-west-2"
get_spot_price "g4dn.xlarge" "us-west-2"
```

### Auto-shutdown Scripts
```python
# auto_shutdown.py
#!/usr/bin/env python3
"""Auto-shutdown script to prevent runaway costs"""

import time
import subprocess
import psutil

def check_gpu_utilization():
    """Check if GPU is being utilized"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        utilization = int(result.stdout.strip())
        return utilization
    except:
        return 0

def check_system_activity():
    """Check system activity"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent, memory_percent

def main():
    """Monitor system and shutdown if idle"""
    idle_threshold = 300  # 5 minutes
    idle_time = 0

    while True:
        gpu_util = check_gpu_utilization()
        cpu_util, mem_util = check_system_activity()

        print(f"GPU: {gpu_util}%, CPU: {cpu_util:.1f}%, Memory: {mem_util:.1f}%")

        # Check if system is idle
        if gpu_util < 5 and cpu_util < 10:
            idle_time += 60
            print(f"System idle for {idle_time} seconds")

            if idle_time >= idle_threshold:
                print("Shutting down due to inactivity...")
                subprocess.run(['sudo', 'shutdown', '-h', 'now'])
                break
        else:
            idle_time = 0

        time.sleep(60)

if __name__ == '__main__':
    main()
```

## 🔧 Troubleshooting Common Issues

### CUDA Driver Issues
```bash
# fix_cuda_driver.sh
#!/bin/bash

# Check current driver version
nvidia-smi

# If driver is missing or incompatible:
# 1. Update package list
sudo apt update

# 2. Remove existing NVIDIA packages
sudo apt remove --purge nvidia* libnvidia*

# 3. Install latest driver
sudo ubuntu-drivers autoinstall

# 4. Reboot
sudo reboot

# 5. Verify installation
nvidia-smi
nvcc --version
```

### Container Runtime Issues
```bash
# fix_container_runtime.sh
#!/bin/bash

# For Docker runtime issues with GPU
# 1. Install nvidia-container-runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# 2. Configure Docker
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# 3. Restart Docker
sudo systemctl restart docker

# 4. Test GPU in container
docker run --rm --gpus all pytorch/pytorch:latest nvidia-smi
```

## 📋 Quick Reference Commands

### Instance Management
```bash
# AWS
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running"
aws ec2 stop-instances --instance-ids i-xxxxxxxxx
aws ec2 terminate-instances --instance-ids i-xxxxxxxxx

# Azure
az vm list --show-details -o table
az vm stop --resource-group pytorch-optimization-rg --name pytorch-gpu-vm
az vm delete --resource-group pytorch-optimization-rg --name pytorch-gpu-vm

# GCP
gcloud compute instances list
gcloud compute instances stop pytorch-gpu-vm --zone=us-central1-c
gcloud compute instances delete pytorch-gpu-vm --zone=us-central1-c
```

### Performance Testing
```bash
# Quick GPU test
python3 -c "
import torch
import time

x = torch.randn(1000, 1000).cuda()
start = time.time()
y = torch.matmul(x, x)
torch.cuda.synchronize()
print(f'GPU matmul time: {time.time() - start:.4f}s')
"

# Memory test
python3 -c "
import torch
print(f'Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
"

# Framework test
export PYTHONPATH=src:$PYTHONPATH
python3 -m pytest tests/test_distributed_scale.py::TestMultiGPUIntegration -v
```

---

## 🎯 Next Steps

1. **Choose your platform** based on requirements and budget
2. **Follow platform-specific setup** instructions above
3. **Run comprehensive tests** to validate functionality
4. **Implement monitoring** for production deployments
5. **Set up cost controls** to prevent unexpected charges

For platform-specific issues, refer to:
- [AWS Documentation](https://docs.aws.amazon.com/ec2/latest/userguide/accelerated-computing-instances.html)
- [Azure GPU VMs](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)
- [GCP GPUs](https://cloud.google.com/compute/docs/gpus)
- [OpenShift GPU Operators](https://docs.openshift.com/container-platform/latest/support/remote_health_monitoring/about-remote-health-monitoring.html)

**🚀 Ready for cloud-scale GPU optimization testing!**