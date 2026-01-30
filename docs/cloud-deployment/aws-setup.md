# AWS Cloud Testing Setup Guide

This guide covers setting up AWS infrastructure for TorchBridge cloud testing.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI v2 installed and configured
- Python 3.10+ with boto3
- SSH key pair for EC2 access

## IAM Permissions

Create an IAM role with the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:RunInstances",
                "ec2:TerminateInstances",
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus",
                "ec2:CreateTags",
                "ec2:DescribeImages",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeSubnets",
                "ec2:DescribeVpcs"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::torchbridge-benchmarks",
                "arn:aws:s3:::torchbridge-benchmarks/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "cloudwatch:GetMetricData"
            ],
            "Resource": "*"
        }
    ]
}
```

## AWS CLI Configuration

```bash
# Configure AWS CLI
aws configure
# AWS Access Key ID: [Your access key]
# AWS Secret Access Key: [Your secret key]
# Default region name: us-west-2
# Default output format: json

# Verify configuration
aws sts get-caller-identity
```

## Instance Types for Testing

### NVIDIA GPU Instances

| Instance Type | GPU | Memory | vCPUs | Cost/hr (On-Demand) |
|---------------|-----|--------|-------|---------------------|
| p5.48xlarge | 8x H100 | 640 GB HBM3 | 192 | ~$98.32 |
| p4d.24xlarge | 8x A100 | 320 GB HBM2e | 96 | ~$32.77 |
| g5.xlarge | 1x A10G | 24 GB | 4 | ~$1.01 |
| g5.2xlarge | 1x A10G | 24 GB | 8 | ~$1.21 |
| g5.4xlarge | 1x A10G | 24 GB | 16 | ~$1.62 |

### AMD GPU Instances (ROCm)

| Instance Type | GPU | Memory | vCPUs | Cost/hr (On-Demand) |
|---------------|-----|--------|-------|---------------------|
| p5.48xlarge | Coming soon | - | - | - |

## AMI Selection

### Deep Learning AMIs

```bash
# Find latest PyTorch DLAMI
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning AMI*PyTorch*" \
    --query 'Images[*].[ImageId,Name,CreationDate]' \
    --output table \
    --region us-west-2

# Recommended AMIs by instance type
# P5 (H100): Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)
# P4d (A100): Deep Learning AMI GPU PyTorch 2.x (Ubuntu 20.04)
# G5 (A10G): Deep Learning AMI GPU PyTorch 2.x (Ubuntu 20.04)
```

## Security Group Setup

```bash
# Create security group
aws ec2 create-security-group \
    --group-name torchbridge-testing \
    --description "Security group for TorchBridge testing"

# Allow SSH access
aws ec2 authorize-security-group-ingress \
    --group-name torchbridge-testing \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

## S3 Bucket Setup

```bash
# Create benchmark results bucket
aws s3 mb s3://torchbridge-benchmarks --region us-west-2

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket torchbridge-benchmarks \
    --versioning-configuration Status=Enabled

# Set lifecycle policy for old results
aws s3api put-bucket-lifecycle-configuration \
    --bucket torchbridge-benchmarks \
    --lifecycle-configuration file://lifecycle.json
```

Example lifecycle.json:
```json
{
    "Rules": [
        {
            "ID": "DeleteOldResults",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "results/"
            },
            "Expiration": {
                "Days": 90
            }
        }
    ]
}
```

## Using the AWS Test Harness

```python
from tests.cloud_testing import AWSTestHarness, AWSInstanceConfig, AWSInstanceType

# Configure instance
config = AWSInstanceConfig(
    instance_type=AWSInstanceType.P4D_24XLARGE,
    ami_id="ami-0123456789abcdef0",  # Your DLAMI ID
    key_name="your-key-pair",
    security_group_ids=["sg-0123456789abcdef0"],
    subnet_id="subnet-0123456789abcdef0",
    region="us-west-2",
    spot_instance=True,  # Use spot for cost savings
    spot_max_price=20.0,  # Max spot price
)

# Create harness
harness = AWSTestHarness(config)

# Run tests
try:
    instance_id = harness.launch_instance()
    result = harness.run_tests(
        test_path="tests/backends/nvidia/",
        pytest_args=["--tb=short", "-v"],
        timeout_seconds=3600,
    )
    print(f"Tests passed: {result.tests_passed}")
    print(f"Tests failed: {result.tests_failed}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Estimated cost: ${result.estimated_cost_usd:.2f}")
finally:
    harness.terminate_instance()
```

## Environment Variables

Set these environment variables for automated testing:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"
export KERNELPYTORCH_S3_BUCKET="torchbridge-benchmarks"
export KERNELPYTORCH_KEY_NAME="your-key-pair"
export KERNELPYTORCH_SECURITY_GROUP="sg-0123456789abcdef0"
export KERNELPYTORCH_SUBNET="subnet-0123456789abcdef0"
```

## CloudWatch Dashboard

Import the provided dashboard configuration:

```bash
aws cloudwatch put-dashboard \
    --dashboard-name TorchBridge-Testing \
    --dashboard-body file://monitoring/cloud_dashboards/aws_cloudwatch_dashboard.json
```

## Spot Instance Best Practices

1. **Use Spot Fleet** for large-scale testing
2. **Set max price** at 60-70% of on-demand for good availability
3. **Enable interruption handling** in your test harness
4. **Use multiple availability zones** for better spot availability
5. **Monitor spot prices** before launching tests

```bash
# Check current spot prices
aws ec2 describe-spot-price-history \
    --instance-types p4d.24xlarge \
    --product-descriptions "Linux/UNIX" \
    --start-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --region us-west-2
```

## Troubleshooting

### Instance Launch Failures

```bash
# Check service quotas
aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-7212CCBC  # P4d instances

# Request quota increase if needed
aws service-quotas request-service-quota-increase \
    --service-code ec2 \
    --quota-code L-7212CCBC \
    --desired-value 8
```

### SSH Connection Issues

```bash
# Verify instance is running
aws ec2 describe-instance-status --instance-ids i-0123456789abcdef0

# Check security group rules
aws ec2 describe-security-groups --group-ids sg-0123456789abcdef0
```

## Next Steps

- [Instance Selection Guide](instance-selection.md)
- [Cost Optimization](cost-optimization.md)
- [Team Workflow](team-workflow.md)
