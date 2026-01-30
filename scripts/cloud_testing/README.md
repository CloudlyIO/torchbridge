# Cloud Testing Scripts

Unified validation scripts for testing TorchBridge across cloud platforms.

## Structure

```
scripts/cloud_testing/
├── common/
│   └── utils.sh          # Shared utilities (logging, pytest, GPU info)
├── nvidia_gcp/
│   └── run_validation.sh # NVIDIA validation on GCP (L4/A100)
├── nvidia_aws/
│   └── run_validation.sh # NVIDIA validation on AWS (A10G/A100)
├── tpu_gcp/
│   └── run_validation.sh # TPU validation on GCP (v5e/v5p)
└── amd_cloud/
    └── run_validation.sh # AMD validation (MI300X)
```

## Usage

Each backend has a single `run_validation.sh` script that performs:

1. **Environment Setup** - Install dependencies, verify hardware
2. **Hardware Info** - Collect GPU/TPU configuration
3. **Tests** - Run backend-specific pytest tests
4. **Benchmarks** - Run performance benchmarks
5. **Report** - Generate markdown validation report

### Prerequisites

```bash
# Clone repository
git clone https://github.com/shahmodthesecond/torchbridge.git
cd torchbridge

# Set work directory
export WORK_DIR=$(pwd)
```

### Running Validation

```bash
# NVIDIA on GCP
./scripts/cloud_testing/nvidia_gcp/run_validation.sh

# NVIDIA on AWS
./scripts/cloud_testing/nvidia_aws/run_validation.sh

# TPU on GCP
./scripts/cloud_testing/tpu_gcp/run_validation.sh

# AMD on AMD Developer Cloud
./scripts/cloud_testing/amd_cloud/run_validation.sh
```

### Output

Reports are saved to `$WORK_DIR/reports/`:

| Backend | Report File |
|---------|-------------|
| NVIDIA GCP | `NVIDIA_GCP_REPORT.md` |
| NVIDIA AWS | `NVIDIA_AWS_REPORT.md` |
| TPU GCP | `TPU_GCP_REPORT.md` |
| AMD | `AMD_CLOUD_REPORT.md` |

Additional files:
- `*_test_results.json` - pytest JSON report
- `*_benchmark_output.txt` - benchmark logs
- `gpu_info.json` / `tpu_info.json` - hardware info

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `$HOME/torchbridge_test` | Working directory |
| `REPORT_DIR` | `$WORK_DIR/reports` | Output directory |

## Shared Utilities

`common/utils.sh` provides:

- **Logging**: `log_info`, `log_success`, `log_warning`, `log_error`, `log_step`
- **Checks**: `command_exists`, `check_python_package`
- **Setup**: `install_python_deps`
- **GPU**: `get_gpu_info_json`, `print_gpu_info`, `warmup_gpu`
- **Testing**: `run_pytest`, `parse_pytest_results`
- **Packaging**: `package_reports`

## Cloud Platform Setup

### GCP (NVIDIA/TPU)

```bash
# Create instance with GPU
gcloud compute instances create kernel-test \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release

# Or TPU
gcloud compute tpus tpu-vm create kernel-test \
    --zone=us-central1-a \
    --accelerator-type=v5litepod-1 \
    --version=tpu-ubuntu2204-base
```

### AWS (NVIDIA)

```bash
# Launch g5.xlarge (A10G) or p4d.24xlarge (A100)
aws ec2 run-instances \
    --instance-type g5.xlarge \
    --image-id ami-0abcdef1234567890  # Deep Learning AMI
```

### AMD Developer Cloud

Sign up at https://www.amd.com/en/developer/resources/ai-cloud.html

- $100 credits via AI Developer Program
- MI300X instances with 192GB HBM3
