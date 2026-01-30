# Result Sharing Guide

This guide covers best practices for sharing and collaborating on benchmark results across the team.

## Result Storage Architecture

### Directory Structure

```
cloud-results/
├── s3://torchbridge-benchmarks/     # AWS results
│   ├── daily/                          # Daily test results
│   │   └── YYYY-MM-DD/
│   │       └── {platform}-{suite}-{commit}.json
│   ├── weekly/                         # Weekly benchmarks
│   │   └── YYYY-Www/
│   │       └── full-benchmark.json
│   ├── pr/                             # PR validation results
│   │   └── pr-{number}/
│   │       └── validation.json
│   └── releases/                       # Release benchmarks
│       └── v{version}/
│           └── release-benchmark.json
│
└── gs://torchbridge-benchmarks/     # GCP results (same structure)
```

### Result File Format

```json
{
  "metadata": {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-01-15T10:30:00Z",
    "developer": "alice",
    "branch": "feature/new-optimizer",
    "commit_sha": "abc123def456",
    "commit_message": "Optimize FlashAttention kernel"
  },
  "platform": {
    "cloud": "aws",
    "instance_type": "p4d.24xlarge",
    "region": "us-west-2",
    "gpu_model": "NVIDIA A100",
    "gpu_count": 8
  },
  "execution": {
    "test_path": "tests/backends/nvidia/",
    "pytest_args": ["--tb=short", "-v"],
    "duration_seconds": 1847,
    "spot_instance": true,
    "estimated_cost_usd": 16.82
  },
  "results": {
    "total_tests": 157,
    "passed": 150,
    "failed": 2,
    "skipped": 5,
    "errors": 0,
    "pass_rate": 95.54
  },
  "benchmarks": [
    {
      "name": "flash_attention_forward",
      "latency_ms": 1.52,
      "throughput": 1250000,
      "memory_mb": 2048
    }
  ],
  "failures": [
    {
      "test": "test_fp8_matmul",
      "error": "AssertionError: Expected 0.001, got 0.002",
      "traceback": "..."
    }
  ]
}
```

## Uploading Results

### Using the Result Uploader

```python
from tests.cloud_testing import S3Uploader, GCSUploader

# AWS S3
uploader = S3Uploader(bucket="torchbridge-benchmarks", region="us-west-2")

result_path = uploader.upload_json(
    data=result_dict,
    path=f"daily/{date.today()}/aws-nvidia-{commit_sha}.json",
    metadata={"developer": "alice", "branch": "main"},
)

# GCP GCS
uploader = GCSUploader(bucket="torchbridge-benchmarks", project_id="my-project")

result_path = uploader.upload_json(
    data=result_dict,
    path=f"daily/{date.today()}/gcp-tpu-{commit_sha}.json",
    metadata={"developer": "bob", "branch": "feature/tpu-opt"},
)
```

### Automated Upload in Test Harness

```python
from tests.cloud_testing import AWSTestHarness, S3Uploader

harness = AWSTestHarness(config)
result = harness.run_tests(test_path="tests/backends/nvidia/")

# Auto-upload on completion
if result.success:
    uploader = S3Uploader(bucket="torchbridge-benchmarks")
    uploader.upload_json(
        data=result.to_dict(),
        path=f"daily/{date.today()}/{result.run_id}.json",
    )
```

## Querying Results

### Using the Benchmark Database

```python
from tests.cloud_testing import BenchmarkDatabase

db = BenchmarkDatabase("benchmarks.db")

# Query by platform
aws_results = db.query(cloud_provider="aws", limit=100)

# Query by hardware type
nvidia_results = db.query(hardware_type="nvidia", benchmark_name="flash_attention")

# Query by date range
recent_results = db.query(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
)

# Get statistics
stats = db.get_statistics(
    benchmark_name="flash_attention_forward",
    cloud_provider="aws",
)
print(f"Average latency: {stats['avg_latency_ms']:.2f}ms")
```

### CLI Queries

```bash
# List recent results
python -m tests.cloud_testing.query \
    --platform aws \
    --days 7 \
    --format table

# Compare platforms
python -m tests.cloud_testing.compare \
    --benchmark flash_attention \
    --platform-a aws \
    --platform-b gcp
```

## Cross-Platform Comparison

### Using CrossPlatformComparison

```python
from monitoring.cloud_dashboards import CrossPlatformComparison, PlatformMetrics

comparison = CrossPlatformComparison()

# Add AWS results
comparison.add_from_benchmark_results(
    aws_results,
    platform_name="AWS P4d (A100)",
)

# Add GCP results
comparison.add_from_benchmark_results(
    gcp_results,
    platform_name="GCP A2 (A100)",
)

# Generate report
report = comparison.generate_report("AWS P4d (A100)", "GCP A2 (A100)")

# Output formats
print(report.to_markdown())  # Markdown table
print(report.to_json())      # JSON format

# Save report
with open("comparison_report.md", "w") as f:
    f.write(report.to_markdown())
```

### Comparison Report Example

```markdown
# Cross-Platform Comparison Report

**Generated**: 2024-01-15 10:30:00

## Platforms Compared

| Property | AWS P4d (A100) | GCP A2 (A100) |
|----------|----------------|---------------|
| Cloud | AWS | GCP |
| Instance | p4d.24xlarge | a2-highgpu-8g |
| GPU | A100 40GB | A100 40GB |
| Samples | 50 | 48 |

## Metrics Comparison

| Metric | AWS P4d | GCP A2 | Ratio | Winner |
|--------|---------|--------|-------|--------|
| Latency (ms) | 1.52 | 1.55 | 1.02x | AWS |
| Throughput | 1250000 | 1220000 | 0.98x | AWS |
| Memory (MB) | 2048 | 2048 | 1.00x | tie |
| Cost/Hour ($) | 32.77 | 29.39 | 0.90x | GCP |

## Summary

**Overall Winner**: Tie

- **Cost/Hour ($)**: GCP A2 (A100) is 10.3% better
```

## Regression Detection

### Detecting Performance Regressions

```python
def detect_regression(db: BenchmarkDatabase, benchmark_name: str, threshold: float = 0.10):
    """Detect if recent results show regression vs baseline."""

    # Get baseline (last 30 days average)
    baseline_stats = db.get_statistics(
        benchmark_name=benchmark_name,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now() - timedelta(days=7),
    )

    # Get recent results (last 7 days)
    recent_stats = db.get_statistics(
        benchmark_name=benchmark_name,
        start_date=datetime.now() - timedelta(days=7),
    )

    # Calculate regression
    if baseline_stats["avg_latency_ms"] > 0:
        latency_change = (
            recent_stats["avg_latency_ms"] - baseline_stats["avg_latency_ms"]
        ) / baseline_stats["avg_latency_ms"]

        if latency_change > threshold:
            return {
                "status": "REGRESSION",
                "benchmark": benchmark_name,
                "baseline_latency": baseline_stats["avg_latency_ms"],
                "current_latency": recent_stats["avg_latency_ms"],
                "change_percent": latency_change * 100,
            }

    return {"status": "OK", "benchmark": benchmark_name}
```

### Automated Regression Alerts

```python
# In CI/CD pipeline
def check_regressions_and_alert():
    db = BenchmarkDatabase("benchmarks.db")

    benchmarks = [
        "flash_attention_forward",
        "fused_gelu",
        "transformer_forward",
    ]

    regressions = []
    for bench in benchmarks:
        result = detect_regression(db, bench, threshold=0.10)
        if result["status"] == "REGRESSION":
            regressions.append(result)

    if regressions:
        send_slack_alert(
            channel="#torchbridge-cloud-results",
            message=f"Performance regressions detected: {len(regressions)} benchmarks",
            details=regressions,
        )
        return False

    return True
```

## Sharing Best Practices

### 1. Always Include Context

```python
# Good: Include full context
result = {
    "commit_sha": "abc123",
    "commit_message": "Optimize attention kernel",
    "branch": "feature/attention-opt",
    "base_commit": "def456",  # What we're comparing against
    "pr_number": 123,
}

# Bad: Missing context
result = {
    "latency": 1.5,
}
```

### 2. Use Consistent Naming

```python
# Naming convention
f"{platform}-{hardware}-{suite}-{commit_short}-{timestamp}.json"

# Examples
"aws-nvidia-backend-abc123-20240115T1030.json"
"gcp-tpu-integration-def456-20240115T1100.json"
```

### 3. Version Results

```python
RESULT_SCHEMA_VERSION = "1.0.0"

result = {
    "schema_version": RESULT_SCHEMA_VERSION,
    # ... rest of result
}
```

### 4. Include Reproducibility Info

```python
result["environment"] = {
    "python_version": "3.10.12",
    "pytorch_version": "2.1.0",
    "cuda_version": "12.1",
    "torchbridge_version": "0.3.7",
    "pip_freeze": ["torch==2.1.0", "numpy==1.24.0", ...],
}
```

## Access Control

### Read Access

All team members should have read access to:
- Daily results
- Weekly benchmarks
- PR validation results
- Historical data

### Write Access

Write access should be restricted to:
- CI/CD service accounts
- Team leads
- Designated benchmark owners

### AWS IAM Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::torchbridge-benchmarks",
                "arn:aws:s3:::torchbridge-benchmarks/*"
            ]
        }
    ]
}
```

### GCP IAM Role

```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:developer@example.com" \
    --role="roles/storage.objectViewer"
```

## Visualization

### Dashboard Integration

Results can be visualized using:
- AWS CloudWatch (see `monitoring/cloud_dashboards/aws_cloudwatch_dashboard.json`)
- GCP Cloud Monitoring (see `monitoring/cloud_dashboards/gcp_monitoring_dashboard.json`)
- Custom dashboards using the comparison tool

### Generating Charts

```python
from monitoring.cloud_dashboards import create_comparison_chart

report = comparison.generate_report("AWS P4d", "GCP A2")
chart = create_comparison_chart(report, output_path="comparison_chart.txt")
print(chart)
```

## Next Steps

- [Team Workflow](team-workflow.md)
- [Troubleshooting](troubleshooting.md)
- [Cost Optimization](cost-optimization.md)
