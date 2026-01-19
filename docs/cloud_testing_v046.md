# KernelPyTorch v0.4.6 Cloud Testing Guide

This guide provides step-by-step instructions for running comprehensive tests across all supported cloud platforms.

## Supported Platforms

| Platform | Instance Types | GPUs | Backend |
|----------|----------------|------|---------|
| AWS NVIDIA | P5.48xlarge, P4d.24xlarge, G5.xlarge | H100, A100, A10G | CUDA |
| AWS AMD | P5e.48xlarge, G6e.xlarge | MI300X | ROCm |
| GCP NVIDIA | a3-highgpu-8g, a2-highgpu-8g, g2-standard-8 | H100, A100, L4 | CUDA |
| GCP TPU | v5litepod-8, v5p-8 | TPU v5e, v5p | XLA |

## Quick Start

### Local Testing

Run all tests locally to verify the framework:

```bash
# From repository root
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Run all core tests
python3 -m pytest tests/test_moe.py tests/test_fp8_native.py -v

# Run demos
python3 demos/moe_demo.py
python3 demos/fp8_native_demo.py

# Run comprehensive test (generates reports)
python3 tests/cloud_testing/v046_comprehensive_test.py --output-dir ./test_results
```

### Cloud Testing (All Platforms)

```bash
# Use master test script
./scripts/cloud_testing/v046_master_test.sh [platform]

# Platforms: local, aws-nvidia, aws-amd, gcp-nvidia, gcp-tpu
```

## AWS Testing

### NVIDIA Instances (P4d/P5/G5)

1. **Launch Instance**
   ```bash
   # Recommended: Deep Learning AMI (Ubuntu 22.04) with PyTorch 2.x
   # Instance types: p4d.24xlarge (A100), p5.48xlarge (H100), g5.xlarge (A10G)

   aws ec2 run-instances \
     --image-id ami-xxx \
     --instance-type p4d.24xlarge \
     --key-name your-key \
     --security-group-ids sg-xxx \
     --region us-west-2
   ```

2. **Setup and Test**
   ```bash
   # SSH to instance
   ssh -i key.pem ubuntu@<instance-ip>

   # Clone and setup
   git clone https://github.com/shahrahman-fb/shahmod.git
   cd shahmod
   pip install -e .

   # Run tests
   ./scripts/cloud_testing/v046_master_test.sh aws-nvidia
   ```

3. **Expected Results**
   - MoE Tests: 48 passed
   - FP8 Tests: 51 passed (FP8 native on H100/A100)
   - Backend Tests: 10+ passed
   - Benchmarks: Matrix mult 100+ TFLOPS (H100)

### AMD Instances (MI300X)

1. **Launch Instance**
   ```bash
   # Note: AMD MI300X instances (p5e, g6e) are in limited preview
   # Check availability in us-east-1 or us-west-2

   aws ec2 run-instances \
     --image-id ami-xxx \  # ROCm-enabled AMI
     --instance-type g6e.xlarge \
     --key-name your-key \
     --region us-east-1
   ```

2. **Setup and Test**
   ```bash
   # Install ROCm PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

   # Run tests
   ./scripts/cloud_testing/amd_cloud/run_validation.sh
   # Or
   ./scripts/cloud_testing/v046_master_test.sh aws-amd
   ```

3. **Expected Results**
   - AMD Backend Tests: Pass
   - MoE Tests: 48 passed
   - Benchmarks: Matrix mult 200+ TFLOPS (MI300X)

## GCP Testing

### NVIDIA Instances (A100/L4)

1. **Create Instance**
   ```bash
   gcloud compute instances create kernel-pytorch-test \
     --zone=us-central1-a \
     --machine-type=a2-highgpu-1g \
     --accelerator=type=nvidia-a100-40gb,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --boot-disk-size=200GB
   ```

2. **Run Tests**
   ```bash
   gcloud compute ssh kernel-pytorch-test --zone=us-central1-a

   git clone https://github.com/shahrahman-fb/shahmod.git
   cd shahmod
   pip install -e .

   ./scripts/cloud_testing/v046_master_test.sh gcp-nvidia
   ```

### TPU Testing (v5e/v5p)

1. **Create TPU VM**
   ```bash
   gcloud compute tpus tpu-vm create kernel-pytorch-tpu \
     --zone=us-central1-a \
     --accelerator-type=v5litepod-8 \
     --version=tpu-ubuntu2204-base
   ```

2. **Setup PyTorch XLA**
   ```bash
   gcloud compute tpus tpu-vm ssh kernel-pytorch-tpu --zone=us-central1-a

   # Install PyTorch XLA
   pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

   # Clone and test
   git clone https://github.com/shahrahman-fb/shahmod.git
   cd shahmod
   pip install -e .

   ./scripts/cloud_testing/tpu_gcp/run_validation.sh
   ```

3. **Expected Results**
   - TPU Backend Tests: Pass
   - XLA Compilation: Working
   - MoE on TPU: Basic functionality

## Test Coverage Matrix

### v0.4.6 Feature Tests

| Feature | NVIDIA | AMD | TPU | CPU |
|---------|--------|-----|-----|-----|
| MoE Standard | ✅ | ✅ | ✅ | ✅ |
| MoE Sparse | ✅ | ✅ | ✅ | ✅ |
| MoE Switch | ✅ | ✅ | ✅ | ✅ |
| MoE GLaM | ✅ | ✅ | ✅ | ✅ |
| FP8 Native | ✅ H100/A100 | ⚠️ Limited | ❌ | ⚠️ Simulated |
| FP8 Quantization | ✅ | ⚠️ | ❌ | ✅ |
| FlexAttention | ✅ | ⚠️ | ⚠️ | ✅ |
| Memory Opt | ✅ | ✅ | ✅ | ✅ |

### Benchmark Targets

| Platform | MatMul TFLOPS (FP16) | MoE tok/s | Memory BW |
|----------|---------------------|-----------|-----------|
| H100 | 150-200 | 500K+ | 2TB/s |
| A100-80GB | 100-150 | 300K+ | 1.5TB/s |
| A100-40GB | 80-120 | 250K+ | 1.5TB/s |
| MI300X | 150-250 | 400K+ | 2.6TB/s |
| TPU v5p | N/A | 300K+ | N/A |
| A10G | 30-50 | 100K+ | 600GB/s |
| L4 | 40-60 | 120K+ | 400GB/s |

## Output Files

After running tests, check the report directory:

```
test_results/
├── hardware_info.json          # Detected hardware
├── moe_test_results.json       # MoE pytest results
├── fp8_test_results.json       # FP8 pytest results
├── backend_test_results.json   # Backend-specific tests
├── benchmark_results.json      # Performance benchmarks
├── moe_demo_output.txt         # MoE demo output
├── fp8_demo_output.txt         # FP8 demo output
└── VALIDATION_REPORT.md        # Final validation report
```

## Troubleshooting

### Common Issues

1. **CUDA not detected**
   ```bash
   # Verify CUDA installation
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **ROCm not detected**
   ```bash
   # Verify ROCm installation
   rocm-smi
   python3 -c "import torch; print(torch.cuda.is_available())"  # ROCm uses CUDA API
   ```

3. **TPU not detected**
   ```bash
   # Verify TPU
   python3 -c "import torch_xla; import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"
   ```

4. **FP8 not available**
   - FP8 requires H100 or A100 with PyTorch 2.1+
   - On other GPUs, tests will use simulated FP8

### Getting Help

- Report issues: https://github.com/shahrahman-fb/shahmod/issues
- Check logs in `test_results/` directory
- Run with verbose: `python3 -m pytest tests/ -v -s`

## Cost Estimates

| Platform | Instance | Cost/hour | Recommended Duration |
|----------|----------|-----------|---------------------|
| AWS P5 | p5.48xlarge | ~$98 | 30 min |
| AWS P4d | p4d.24xlarge | ~$32 | 30 min |
| AWS G5 | g5.xlarge | ~$1 | 30 min |
| AWS AMD | g6e.xlarge | ~$3 | 30 min |
| GCP A100 | a2-highgpu-1g | ~$3 | 30 min |
| GCP L4 | g2-standard-8 | ~$1 | 30 min |
| GCP TPU | v5litepod-8 | ~$2 | 30 min |

**Tip**: Use spot/preemptible instances for 60-90% cost savings.
