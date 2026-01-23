# Instance Selection Guide

This guide helps you choose the right cloud instances for KernelPyTorch testing based on your requirements.

## Quick Reference

### By Test Type

| Test Type | Recommended Instance | Cost/hr | Notes |
|-----------|---------------------|---------|-------|
| Quick smoke tests | AWS g5.xlarge / GCP g2-standard-4 | ~$1 | Fast startup, low cost |
| NVIDIA backend tests | AWS p4d.24xlarge / GCP a2-highgpu-8g | ~$30 | Full A100 capability |
| H100/FP8 tests | AWS p5.48xlarge / GCP a3-highgpu-8g | ~$98 | Latest GPU features |
| TPU tests | GCP v5litepod-8 | ~$10 | TPU-specific validation |
| Performance benchmarks | AWS p5.48xlarge / GCP a3-highgpu-8g | ~$98 | Maximum performance |

### By Budget

| Budget | AWS | GCP | Use Case |
|--------|-----|-----|----------|
| < $5/test | g5.xlarge (spot) | g2-standard-4 (preemptible) | Development, quick checks |
| $5-20/test | p4d.24xlarge (spot) | a2-highgpu-1g | Standard testing |
| $20-50/test | p4d.24xlarge (on-demand) | a2-highgpu-4g | CI/CD integration |
| $50+/test | p5.48xlarge | a3-highgpu-8g | Full performance validation |

## AWS Instance Details

### P5 Instances (H100)

**p5.48xlarge**
- 8x NVIDIA H100 80GB HBM3
- 192 vCPUs, 2TB RAM
- 3200 Gbps networking
- Best for: FP8 testing, maximum performance benchmarks
- Cost: ~$98.32/hr on-demand, ~$30-40/hr spot

### P4d Instances (A100)

**p4d.24xlarge**
- 8x NVIDIA A100 40GB HBM2e
- 96 vCPUs, 1.1TB RAM
- 400 Gbps networking (EFA enabled)
- Best for: Standard NVIDIA backend testing
- Cost: ~$32.77/hr on-demand, ~$10-15/hr spot

### G5 Instances (A10G)

**g5.xlarge - g5.48xlarge**
- 1-8x NVIDIA A10G 24GB
- 4-192 vCPUs
- Best for: Quick validation, development
- Cost: ~$1-16/hr

### Instance Selection Flow (AWS)

```
Need H100/FP8 features?
├── Yes → p5.48xlarge
└── No → Need multi-GPU?
         ├── Yes → Need 8 GPUs?
         │        ├── Yes → p4d.24xlarge
         │        └── No → g5.12xlarge (4 GPUs)
         └── No → g5.xlarge (1 GPU)
```

## GCP Instance Details

### A3 Instances (H100)

**a3-highgpu-8g**
- 8x NVIDIA H100 80GB HBM3
- 208 vCPUs, 1.8TB RAM
- 3200 Gbps networking
- Best for: FP8 testing, maximum performance
- Cost: ~$98.32/hr on-demand

### A2 Instances (A100)

**a2-highgpu-1g to a2-highgpu-8g**
- 1-8x NVIDIA A100 40GB HBM2e
- 12-96 vCPUs
- Best for: Standard NVIDIA backend testing
- Cost: ~$3.67-29.39/hr

### G2 Instances (L4)

**g2-standard-4 to g2-standard-96**
- 1-8x NVIDIA L4 24GB
- 4-96 vCPUs
- Best for: Quick validation, development
- Cost: ~$0.84-6.72/hr

### TPU Selection

| TPU Type | Cores | Memory | Use Case | Cost/hr |
|----------|-------|--------|----------|---------|
| v5litepod-1 | 1 | 16GB | Quick TPU tests | ~$1.20 |
| v5litepod-4 | 4 | 64GB | Standard TPU tests | ~$4.80 |
| v5litepod-8 | 8 | 128GB | Full TPU validation | ~$9.60 |
| v5p-8 | 8 | 192GB | Performance benchmarks | ~$12.00 |
| v6e-1 | 1 | 32GB | Latest TPU features | ~$2.40 |

## Test Suite Recommendations

### Unit Tests

```python
# Minimal instance for unit tests
AWS: g5.xlarge (1x A10G)
GCP: g2-standard-4 (1x L4)
TPU: v5litepod-1

# Estimated runtime: 10-20 minutes
# Estimated cost: $0.20-0.50
```

### Integration Tests

```python
# Standard instance for integration tests
AWS: p4d.24xlarge (8x A100)
GCP: a2-highgpu-1g (1x A100)
TPU: v5litepod-4

# Estimated runtime: 30-60 minutes
# Estimated cost: $5-15
```

### Performance Benchmarks

```python
# High-performance instance for benchmarks
AWS: p5.48xlarge (8x H100)
GCP: a3-highgpu-8g (8x H100)
TPU: v5p-8

# Estimated runtime: 1-2 hours
# Estimated cost: $50-200
```

## Cost Optimization Matrix

| Strategy | AWS | GCP | Savings |
|----------|-----|-----|---------|
| Spot/Preemptible | spot_instance=True | preemptible=True | 60-70% |
| Right-sizing | Use g5 for quick tests | Use g2 for quick tests | 50-80% |
| Region selection | us-west-2, us-east-1 | us-central1, us-east4 | 10-20% |
| Reserved instances | 1-year commitment | 1-year commitment | 30-40% |

## Hardware Feature Matrix

| Feature | G5/G2 | P4d/A2 | P5/A3 | TPU |
|---------|-------|--------|-------|-----|
| FP32 | Yes | Yes | Yes | Yes |
| FP16 | Yes | Yes | Yes | Yes |
| BF16 | Yes | Yes | Yes | Yes |
| FP8 | No | No | Yes | Partial |
| TF32 | Yes | Yes | Yes | No |
| Tensor Cores | Gen 3 | Gen 3 | Gen 4 | MXU |
| FlashAttention | Yes | Yes | Yes | No |
| NVLink | No | Yes | Yes | N/A |
| HBM | 24GB | 40-80GB | 80GB | 16-192GB |

## Decision Framework

### 1. Determine Test Requirements

```python
def select_instance(test_suite, budget_per_test, need_fp8=False, need_tpu=False):
    if need_tpu:
        if budget_per_test < 5:
            return "GCP v5litepod-1 (preemptible)"
        elif budget_per_test < 15:
            return "GCP v5litepod-8 (preemptible)"
        else:
            return "GCP v5p-8"

    if need_fp8:
        return "AWS p5.48xlarge (spot)" if budget_per_test < 50 else "AWS p5.48xlarge"

    if test_suite == "unit":
        return "AWS g5.xlarge (spot)" if budget_per_test < 2 else "GCP g2-standard-4"
    elif test_suite == "integration":
        return "AWS p4d.24xlarge (spot)" if budget_per_test < 15 else "GCP a2-highgpu-1g"
    else:  # performance
        return "AWS p5.48xlarge (spot)" if budget_per_test < 50 else "GCP a3-highgpu-8g"
```

### 2. Check Availability

```bash
# AWS: Check spot availability
aws ec2 describe-spot-instance-requests \
    --filters "Name=launch.instance-type,Values=p4d.24xlarge"

# GCP: Check preemptible availability
gcloud compute instances list \
    --filter="scheduling.preemptible=true"
```

### 3. Validate Quotas

```bash
# AWS
aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-7212CCBC

# GCP
gcloud compute project-info describe \
    --format="value(quotas)" | grep GPU
```

## Multi-Platform Testing Strategy

For comprehensive testing, use a combination:

```python
test_matrix = {
    "quick_validation": [
        ("aws", "g5.xlarge", "spot"),
        ("gcp", "g2-standard-4", "preemptible"),
    ],
    "nvidia_backend": [
        ("aws", "p4d.24xlarge", "spot"),
        ("gcp", "a2-highgpu-1g", "preemptible"),
    ],
    "h100_fp8": [
        ("aws", "p5.48xlarge", "spot"),
        ("gcp", "a3-highgpu-8g", "preemptible"),
    ],
    "tpu_backend": [
        ("gcp", "v5litepod-8", "preemptible"),
        ("gcp", "v5p-8", "preemptible"),
    ],
}
```

## Next Steps

- [AWS Setup Guide](aws_setup.md)
- [GCP Setup Guide](gcp_setup.md)
- [Cost Optimization](cost_optimization.md)
