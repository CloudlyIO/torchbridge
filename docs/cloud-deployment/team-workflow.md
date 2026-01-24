# Team Workflow Guide

This guide establishes best practices for multi-developer cloud testing workflows.

## Overview

Coordinating cloud testing across a team requires:
- Clear ownership and scheduling
- Standardized configurations
- Result sharing and comparison
- Cost accountability

## Team Roles

### Test Coordinator

Responsibilities:
- Schedule regular test runs
- Monitor cloud costs
- Maintain test infrastructure
- Review and approve cloud access

### Backend Developers

Responsibilities:
- Run tests for their changes
- Document test configurations
- Report issues and regressions
- Update test suites

### DevOps/Infrastructure

Responsibilities:
- Maintain cloud accounts
- Manage IAM/service accounts
- Set up monitoring and alerts
- Handle quota requests

## Standard Workflows

### 1. Pre-Merge Testing

Before merging significant changes:

```bash
# Step 1: Run local tests first
pytest tests/backends/nvidia/ -v

# Step 2: Request cloud test slot
# (via Slack/Teams channel or scheduling system)

# Step 3: Run cloud tests
python -m tests.cloud_testing.run_tests \
    --platform aws \
    --instance-type p4d.24xlarge \
    --test-path tests/backends/nvidia/ \
    --spot \
    --upload-results

# Step 4: Share results with reviewer
```

### 2. Nightly Integration Tests

Automated nightly runs:

```yaml
# .github/workflows/nightly-cloud-tests.yml
name: Nightly Cloud Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC

jobs:
  aws-nvidia:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run AWS NVIDIA Tests
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          python -m tests.cloud_testing.run_tests \
              --platform aws \
              --instance-type p4d.24xlarge \
              --test-path tests/backends/nvidia/ \
              --spot

  gcp-tpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run GCP TPU Tests
        env:
          GCP_CREDENTIALS: ${{ secrets.GCP_CREDENTIALS }}
        run: |
          python -m tests.cloud_testing.run_tests \
              --platform gcp \
              --tpu-type v5litepod-8 \
              --test-path tests/backends/tpu/ \
              --preemptible
```

### 3. Weekly Performance Benchmarks

```bash
# Run comprehensive benchmarks
python -m tests.cloud_testing.run_benchmarks \
    --platform aws \
    --instance-type p5.48xlarge \
    --benchmark-suite full \
    --compare-baseline \
    --upload-results
```

## Scheduling System

### Shared Calendar

Use a shared calendar for cloud test scheduling:

| Time Slot | Monday | Tuesday | Wednesday | Thursday | Friday |
|-----------|--------|---------|-----------|----------|--------|
| 9-12 | Dev A | Dev B | Nightly Review | Dev C | Release |
| 12-3 | Dev B | Dev C | Dev A | Dev B | Cleanup |
| 3-6 | Open | Open | Open | Open | Open |
| Night | Nightly | Nightly | Nightly | Nightly | Weekly |

### Booking Protocol

1. Check calendar for availability
2. Book slot via team channel
3. Include: name, test type, estimated duration
4. Cancel if not needed (at least 1 hour notice)

## Configuration Management

### Shared Configuration Files

```python
# configs/team_defaults.py
TEAM_AWS_CONFIG = {
    "region": "us-west-2",
    "subnet_id": "subnet-shared-123",
    "security_group_ids": ["sg-team-456"],
    "key_name": "team-key",
    "s3_bucket": "team-kernelpytorch-results",
}

TEAM_GCP_CONFIG = {
    "project_id": "kernelpytorch-team",
    "zone": "us-central1-a",
    "gcs_bucket": "team-kernelpytorch-results",
}
```

### Personal Configuration

```python
# configs/developer_config.py
from configs.team_defaults import TEAM_AWS_CONFIG

MY_CONFIG = {
    **TEAM_AWS_CONFIG,
    "instance_name_prefix": "dev-alice",
    "tags": {"Owner": "alice", "Team": "ml-infra"},
}
```

## Result Sharing

### Standardized Result Format

```python
# All results should include:
result = {
    "run_id": "uuid-here",
    "developer": "alice",
    "branch": "feature/new-optimizer",
    "commit": "abc123",
    "platform": "aws",
    "instance_type": "p4d.24xlarge",
    "test_suite": "nvidia_backend",
    "timestamp": "2024-01-15T10:30:00Z",
    "duration_seconds": 1234,
    "cost_usd": 12.50,
    "results": {
        "passed": 150,
        "failed": 2,
        "skipped": 5,
    },
    "metrics": {
        "avg_latency_ms": 1.5,
        "throughput": 1000,
    },
}
```

### Result Upload Location

```
s3://team-kernelpytorch-results/
├── daily/
│   ├── 2024-01-15/
│   │   ├── aws-nvidia-abc123.json
│   │   └── gcp-tpu-def456.json
├── weekly/
│   └── 2024-w02/
│       └── full-benchmark.json
└── pr/
    └── pr-123/
        └── validation.json
```

## Communication

### Slack/Teams Channel Structure

```
#kernelpytorch-cloud-testing
├── Daily standups on test status
├── Blocking issues
└── Cost alerts

#kernelpytorch-cloud-booking
├── Slot reservations
├── Cancellations
└── Availability questions

#kernelpytorch-cloud-results
├── Automated result notifications
├── Regression alerts
└── Performance comparisons
```

### Notification Bot

```python
# Example Slack notification
def notify_test_complete(result: dict, webhook_url: str):
    status = "PASS" if result["results"]["failed"] == 0 else "FAIL"
    message = {
        "text": f"Cloud Test {status}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{result['developer']}* - {result['test_suite']}\n"
                            f"Branch: `{result['branch']}`\n"
                            f"Passed: {result['results']['passed']} | "
                            f"Failed: {result['results']['failed']}\n"
                            f"Duration: {result['duration_seconds']}s | "
                            f"Cost: ${result['cost_usd']:.2f}"
                }
            }
        ]
    }
    requests.post(webhook_url, json=message)
```

## Cost Accountability

### Per-Developer Tracking

```python
def get_developer_costs(db: BenchmarkDatabase, developer: str, month: str):
    records = db.query(
        start_date=datetime.strptime(f"{month}-01", "%Y-%m-%d"),
        end_date=datetime.strptime(f"{month}-31", "%Y-%m-%d"),
    )

    developer_records = [r for r in records if r.metadata.get("developer") == developer]
    total_cost = sum(r.cost_usd for r in developer_records)

    return {
        "developer": developer,
        "month": month,
        "total_cost": total_cost,
        "test_count": len(developer_records),
        "avg_cost_per_test": total_cost / len(developer_records) if developer_records else 0,
    }
```

### Monthly Budget Allocation

```python
TEAM_MONTHLY_BUDGET = 2000  # $2000/month

DEVELOPER_ALLOCATIONS = {
    "alice": 500,   # Backend lead
    "bob": 300,     # NVIDIA specialist
    "carol": 300,   # TPU specialist
    "dave": 200,    # New team member
    "shared": 700,  # CI/CD, nightly, weekly
}
```

### Budget Alerts

```python
def check_budget_status(developer: str, current_spend: float):
    allocation = DEVELOPER_ALLOCATIONS.get(developer, 0)
    percentage_used = (current_spend / allocation) * 100 if allocation > 0 else 0

    if percentage_used >= 90:
        send_alert(f"CRITICAL: {developer} at {percentage_used:.0f}% of budget")
    elif percentage_used >= 75:
        send_alert(f"WARNING: {developer} at {percentage_used:.0f}% of budget")
```

## Onboarding New Team Members

### Day 1: Access Setup

1. Request cloud account access (AWS IAM user, GCP IAM member)
2. Install required CLIs (aws-cli, gcloud)
3. Configure credentials
4. Verify access with simple commands

### Day 2: First Test Run

1. Run local tests to understand the test suite
2. Shadow experienced developer on cloud test
3. Run first cloud test (g5.xlarge, <$2)
4. Upload results to shared bucket

### Week 1: Independent Testing

1. Complete all documentation reading
2. Run integration tests independently
3. Join cloud-testing channels
4. Book first scheduled slot

### Checklist

- [ ] AWS access configured
- [ ] GCP access configured
- [ ] Local tests passing
- [ ] First cloud test completed
- [ ] Joined communication channels
- [ ] Read all cloud testing docs
- [ ] Understand cost constraints
- [ ] Shadowed experienced developer

## Emergency Procedures

### Runaway Costs

```bash
# Immediately terminate all instances
aws ec2 describe-instances \
    --filters "Name=tag:Team,Values=kernelpytorch" \
    --query "Reservations[].Instances[].InstanceId" | \
    xargs -I {} aws ec2 terminate-instances --instance-ids {}

# GCP
gcloud compute instances list \
    --filter="labels.team=kernelpytorch" \
    --format="value(name,zone)" | \
    while read name zone; do
        gcloud compute instances delete $name --zone=$zone --quiet
    done
```

### Test Stuck/Hanging

1. Check instance status in console
2. Attempt SSH access for debugging
3. Force terminate if unresponsive
4. Document incident for post-mortem

## Next Steps

- [Result Sharing](result-sharing.md)
- [Troubleshooting](troubleshooting.md)
- [Cost Optimization](cost-optimization.md)
