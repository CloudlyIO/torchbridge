# Cost Optimization Guide

This guide provides strategies for minimizing cloud testing costs while maintaining comprehensive test coverage.

## Cost Overview

### AWS Pricing (On-Demand vs Spot)

| Instance | On-Demand | Spot (typical) | Savings |
|----------|-----------|----------------|---------|
| p5.48xlarge | $98.32/hr | $29-40/hr | 60-70% |
| p4d.24xlarge | $32.77/hr | $10-15/hr | 55-70% |
| g5.xlarge | $1.01/hr | $0.30-0.40/hr | 60-70% |

### GCP Pricing (On-Demand vs Preemptible)

| Instance | On-Demand | Preemptible | Savings |
|----------|-----------|-------------|---------|
| a3-highgpu-8g | $98.32/hr | $29.50/hr | 70% |
| a2-highgpu-1g | $3.67/hr | $1.10/hr | 70% |
| g2-standard-4 | $0.84/hr | $0.25/hr | 70% |
| v5litepod-8 | $9.60/hr | $2.88/hr | 70% |

## Strategy 1: Use Spot/Preemptible Instances

### AWS Spot Configuration

```python
from tests.cloud_testing import AWSTestHarness, AWSInstanceConfig, AWSInstanceType

config = AWSInstanceConfig(
    instance_type=AWSInstanceType.P4D_24XLARGE,
    spot_instance=True,
    spot_max_price=20.0,  # Set max at 60% of on-demand
    # ... other config
)
```

### GCP Preemptible Configuration

```python
from tests.cloud_testing import GCPTestHarness, GCPInstanceConfig, GCPMachineType

config = GCPInstanceConfig(
    machine_type=GCPMachineType.A2_HIGHGPU_1G,
    preemptible=True,
    # ... other config
)
```

### Best Practices for Spot/Preemptible

1. **Implement checkpointing** - Save state periodically
2. **Use graceful shutdown handlers** - Clean up on preemption
3. **Set appropriate max prices** - 60-70% of on-demand
4. **Use multiple availability zones** - Better availability

## Strategy 2: Right-Size Instances

### Test Type to Instance Mapping

| Test Type | Overkill | Right-Sized | Savings |
|-----------|----------|-------------|---------|
| Unit tests | p4d.24xlarge | g5.xlarge | 97% |
| Quick validation | a2-highgpu-8g | g2-standard-4 | 88% |
| Integration tests | p5.48xlarge | p4d.24xlarge | 67% |

### Auto-Scaling Configuration

```python
def get_optimal_instance(test_count: int, test_type: str) -> str:
    """Select instance based on test requirements."""
    if test_type == "unit" and test_count < 100:
        return "g5.xlarge"  # $1/hr
    elif test_type == "integration" and test_count < 500:
        return "g5.4xlarge"  # $1.62/hr
    elif test_type == "benchmark":
        return "p4d.24xlarge"  # $32.77/hr
    else:
        return "p5.48xlarge"  # $98.32/hr
```

## Strategy 3: Optimize Test Runtime

### Parallel Test Execution

```bash
# Run tests in parallel to reduce instance time
pytest tests/backends/nvidia/ -n auto --dist loadgroup
```

### Test Prioritization

```python
# Run critical tests first, skip slow tests on quick runs
pytest tests/ -m "not slow" --maxfail=3
```

### Caching

```python
# Cache model weights and compiled kernels
export TORCH_HOME=/shared/torch_cache
export XLA_FLAGS="--xla_dump_to=/shared/xla_cache"
```

## Strategy 4: Schedule Testing Wisely

### Off-Peak Hours

Spot/preemptible prices are typically lower during:
- Late night (10 PM - 6 AM local time)
- Weekends
- Holidays

### Batch Testing

```python
# Accumulate changes and test in batches
def should_run_cloud_tests():
    # Only run cloud tests for significant changes
    changed_files = get_changed_files()
    significant_changes = [
        f for f in changed_files
        if f.startswith(("src/kernel_pytorch/backends/", "tests/backends/"))
    ]
    return len(significant_changes) > 0
```

## Strategy 5: Regional Optimization

### AWS Regions by Cost

| Region | p4d.24xlarge | Notes |
|--------|--------------|-------|
| us-east-2 | $32.77/hr | Often lowest spot |
| us-west-2 | $32.77/hr | Good availability |
| eu-west-1 | $35.92/hr | Higher price |

### GCP Regions by Cost

| Region | a2-highgpu-1g | Notes |
|--------|---------------|-------|
| us-central1 | $3.67/hr | Lowest, best availability |
| us-east4 | $3.67/hr | Good alternative |
| europe-west4 | $4.03/hr | Higher price |

## Strategy 6: Reserved Capacity

### When to Use Reserved Instances

- Regular testing schedule (daily CI/CD)
- Predictable usage patterns
- Long-term projects

### AWS Savings Plans

```
1-year commitment: 30-40% savings
3-year commitment: 50-60% savings
```

### GCP Committed Use Discounts

```
1-year commitment: 37% savings
3-year commitment: 55% savings
```

## Cost Tracking

### Using the Benchmark Database

```python
from tests.cloud_testing import BenchmarkDatabase, BenchmarkRecord

db = BenchmarkDatabase("benchmarks.db")

# Query cost data
records = db.query(
    cloud_provider="aws",
    start_date=datetime(2024, 1, 1),
)

total_cost = sum(r.cost_usd for r in records)
print(f"Total AWS spend: ${total_cost:.2f}")
```

### Cost Alerts

```python
# AWS CloudWatch cost alert
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_alarm(
    AlarmName='KernelPyTorch-DailyCost',
    MetricName='EstimatedCharges',
    Namespace='AWS/Billing',
    Statistic='Maximum',
    Period=86400,
    Threshold=100.0,  # Alert at $100/day
    ComparisonOperator='GreaterThanThreshold',
)
```

## Monthly Budget Examples

### Small Team ($500/month)

```
Weekly allocation:
- 2x quick validation runs (g5.xlarge): $5
- 1x integration test (p4d.24xlarge spot): $15
- Monthly buffer for debugging: $30

Total: ~$100/week = $400/month
```

### Medium Team ($2000/month)

```
Weekly allocation:
- Daily smoke tests (g5.xlarge): $7
- 3x integration tests (p4d.24xlarge spot): $45
- 1x full benchmark (p5.48xlarge spot): $50
- TPU testing (v5litepod-8): $20

Total: ~$125/week = $500/month + buffer
```

### Large Team ($10000/month)

```
Weekly allocation:
- CI/CD per-commit tests: $200
- Nightly integration: $300
- Weekly benchmarks: $500
- TPU validation: $200
- On-demand debugging: $300

Total: ~$1500/week = $6000/month + buffer
```

## Cost Reduction Checklist

- [ ] Use spot/preemptible instances for non-critical tests
- [ ] Right-size instances based on test requirements
- [ ] Run tests in parallel to minimize instance time
- [ ] Cache model weights and compiled kernels
- [ ] Schedule tests during off-peak hours
- [ ] Use optimal regions for pricing
- [ ] Set up cost alerts and budgets
- [ ] Review spending weekly
- [ ] Consider reserved capacity for regular usage
- [ ] Clean up unused resources (snapshots, volumes, IPs)

## Cleanup Commands

```bash
# AWS: Find and clean unused resources
aws ec2 describe-volumes --filters "Name=status,Values=available"
aws ec2 describe-snapshots --owner-ids self
aws ec2 describe-addresses --filters "Name=domain,Values=vpc"

# GCP: Find unused resources
gcloud compute disks list --filter="NOT users:*"
gcloud compute snapshots list
gcloud compute addresses list --filter="status=RESERVED"
```

## Next Steps

- [Instance Selection Guide](instance_selection.md)
- [Team Workflow](team_workflow.md)
- [Result Sharing](result_sharing.md)
