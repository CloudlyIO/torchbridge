# GCP Cloud Testing Setup Guide

This guide covers setting up Google Cloud Platform infrastructure for KernelPyTorch cloud testing, including both GPU instances and TPU pods.

## Prerequisites

- GCP Account with billing enabled
- gcloud CLI installed and configured
- Python 3.10+ with google-cloud-compute
- Service account with appropriate permissions

## Service Account Setup

```bash
# Create service account
gcloud iam service-accounts create kernelpytorch-testing \
    --display-name="KernelPyTorch Testing"

# Grant necessary roles
PROJECT_ID=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:kernelpytorch-testing@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.instanceAdmin.v1"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:kernelpytorch-testing@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:kernelpytorch-testing@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/tpu.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:kernelpytorch-testing@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"

# Download credentials
gcloud iam service-accounts keys create credentials.json \
    --iam-account=kernelpytorch-testing@$PROJECT_ID.iam.gserviceaccount.com
```

## gcloud CLI Configuration

```bash
# Initialize gcloud
gcloud init

# Set default project and region
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable tpu.googleapis.com
gcloud services enable monitoring.googleapis.com
```

## Instance Types for Testing

### NVIDIA GPU Instances

| Machine Type | GPU | Memory | vCPUs | Cost/hr (On-Demand) |
|--------------|-----|--------|-------|---------------------|
| a3-highgpu-8g | 8x H100 | 640 GB HBM3 | 208 | ~$98.32 |
| a2-highgpu-1g | 1x A100 | 40 GB HBM2e | 12 | ~$3.67 |
| a2-highgpu-2g | 2x A100 | 80 GB HBM2e | 24 | ~$7.35 |
| a2-highgpu-4g | 4x A100 | 160 GB HBM2e | 48 | ~$14.69 |
| a2-highgpu-8g | 8x A100 | 320 GB HBM2e | 96 | ~$29.39 |
| g2-standard-4 | 1x L4 | 24 GB | 4 | ~$0.84 |
| g2-standard-8 | 1x L4 | 24 GB | 8 | ~$1.10 |

### TPU Instances

| TPU Type | Cores | HBM | Cost/hr (On-Demand) |
|----------|-------|-----|---------------------|
| v5litepod-1 | 1 | 16 GB | ~$1.20 |
| v5litepod-4 | 4 | 64 GB | ~$4.80 |
| v5litepod-8 | 8 | 128 GB | ~$9.60 |
| v5litepod-16 | 16 | 256 GB | ~$19.20 |
| v5p-8 | 8 | 192 GB | ~$12.00 |
| v6e-1 | 1 | 32 GB | ~$2.40 |

## VM Image Selection

```bash
# List available Deep Learning VM images
gcloud compute images list \
    --project deeplearning-platform-release \
    --filter="name~pytorch"

# Recommended images
# GPU: pytorch-2-1-cu121-notebooks-v20240105-debian-11
# TPU: tpu-pytorch-2-1-v20240105
```

## VPC and Firewall Setup

```bash
# Create VPC network (optional, can use default)
gcloud compute networks create kernelpytorch-vpc \
    --subnet-mode=auto

# Create firewall rule for SSH
gcloud compute firewall-rules create allow-ssh \
    --network=kernelpytorch-vpc \
    --allow=tcp:22 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=kernelpytorch-testing
```

## GCS Bucket Setup

```bash
# Create bucket for benchmark results
gsutil mb -l us-central1 gs://kernelpytorch-benchmarks/

# Set lifecycle policy
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF
gsutil lifecycle set lifecycle.json gs://kernelpytorch-benchmarks/
```

## Using the GCP Test Harness

### GPU Testing

```python
from tests.cloud_testing import GCPTestHarness, GCPInstanceConfig, GCPMachineType

# Configure GPU instance
config = GCPInstanceConfig(
    machine_type=GCPMachineType.A2_HIGHGPU_1G,
    project_id="your-project-id",
    zone="us-central1-a",
    image_family="pytorch-2-1-cu121",
    image_project="deeplearning-platform-release",
    preemptible=True,  # Use preemptible for cost savings
)

# Create harness
harness = GCPTestHarness(config)

# Run tests
try:
    instance_name = harness.launch_instance()
    result = harness.run_tests(
        test_path="tests/backends/nvidia/",
        pytest_args=["--tb=short", "-v"],
        timeout_seconds=3600,
    )
    print(f"Tests passed: {result.tests_passed}")
    print(f"Tests failed: {result.tests_failed}")
    print(f"Duration: {result.duration_seconds:.1f}s")
finally:
    harness.terminate_instance()
```

### TPU Testing

```python
from tests.cloud_testing import TPUTestHarness, TPUConfig, TPUType

# Configure TPU
config = TPUConfig(
    tpu_type=TPUType.V5E_8,
    project_id="your-project-id",
    zone="us-central1-a",
    software_version="tpu-pytorch-2.1",
    preemptible=True,
)

# Create TPU harness
harness = TPUTestHarness(config)

# Run TPU tests
try:
    tpu_name = harness.create_tpu()
    result = harness.run_tests(
        test_path="tests/backends/tpu/",
        pytest_args=["--tb=short", "-v"],
    )
    print(f"Tests passed: {result.tests_passed}")
finally:
    harness.delete_tpu()
```

## Environment Variables

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"
export KERNELPYTORCH_GCS_BUCKET="kernelpytorch-benchmarks"
```

## TPU Setup Notes

### TPU VM vs TPU Node

GCP offers two TPU deployment models:

1. **TPU VM** (Recommended): Direct SSH access to the TPU host
   ```bash
   gcloud compute tpus tpu-vm create my-tpu \
       --zone=us-central1-a \
       --accelerator-type=v5litepod-8 \
       --version=tpu-pytorch-2.1
   ```

2. **TPU Node**: Network-attached TPU (legacy)
   ```bash
   gcloud compute tpus create my-tpu \
       --zone=us-central1-a \
       --accelerator-type=v5litepod-8 \
       --version=pytorch-2.1
   ```

### TPU Software Versions

```bash
# List available TPU software versions
gcloud compute tpus versions list --zone=us-central1-a

# Common versions:
# - tpu-pytorch-2.1 (recommended)
# - tpu-pytorch-2.0
# - tpu-ubuntu2204-base
```

## Cloud Monitoring Dashboard

Import the provided dashboard configuration:

```bash
# Import dashboard
gcloud monitoring dashboards create \
    --config-from-file=monitoring/cloud_dashboards/gcp_monitoring_dashboard.json
```

## Preemptible Instance Best Practices

1. **Always save checkpoints** when using preemptible instances
2. **Use startup scripts** to auto-configure environment
3. **Implement graceful shutdown** handlers
4. **Monitor preemption events**

```bash
# Check for preemption
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/preempted
```

## Quotas and Limits

```bash
# Check GPU quotas
gcloud compute regions describe us-central1 \
    --format="value(quotas)" | grep -i gpu

# Request quota increase
# Go to: https://console.cloud.google.com/iam-admin/quotas
```

## Troubleshooting

### Instance Creation Failures

```bash
# Check available zones for A100
gcloud compute accelerator-types list \
    --filter="name=nvidia-tesla-a100"

# Check quota
gcloud compute project-info describe \
    --format="value(quotas)" | grep GPU
```

### TPU Issues

```bash
# Check TPU status
gcloud compute tpus tpu-vm describe my-tpu --zone=us-central1-a

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central1-a
```

## Next Steps

- [Instance Selection Guide](instance_selection.md)
- [Cost Optimization](cost_optimization.md)
- [Team Workflow](team_workflow.md)
