#!/usr/bin/env python3
"""
TorchBridge Cloud Validation Runner

Runs comprehensive validation across AWS and GCP with proper region configuration.

Usage:
    python run_cloud_validation.py --provider aws --instance-type g5.xlarge
    python run_cloud_validation.py --provider gcp --machine-type g2-standard-4
    python run_cloud_validation.py --provider all --tier spot
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "cloud_testing"))

REPORTS_DIR = PROJECT_ROOT / "reports" / "cloud_validation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AWSConfig:
    """AWS configuration with your specific setup."""
    region: str = "us-east-1"
    key_name: str = "shahmod-gpu-key-east1"
    ami_id: str = "ami-069562671a65789b9"  # Deep Learning PyTorch 2.9
    vpc_id: str = "vpc-0f15c7f58720f216f"

    # Instance types to test
    instance_types: dict[str, dict] = None

    def __post_init__(self):
        self.instance_types = {
            "g5.xlarge": {
                "gpu": "A10G",
                "gpu_count": 1,
                "vcpu": 4,
                "memory_gb": 16,
                "spot_price": 0.50,
                "zones": ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d"]
            },
            "g5.2xlarge": {
                "gpu": "A10G",
                "gpu_count": 1,
                "vcpu": 8,
                "memory_gb": 32,
                "spot_price": 0.80,
                "zones": ["us-east-1a", "us-east-1b", "us-east-1d"]
            },
            "p4d.24xlarge": {
                "gpu": "A100",
                "gpu_count": 8,
                "vcpu": 96,
                "memory_gb": 1152,
                "spot_price": 15.00,
                "zones": ["us-east-1a"]
            },
            "p5.48xlarge": {
                "gpu": "H100",
                "gpu_count": 8,
                "vcpu": 192,
                "memory_gb": 2048,
                "spot_price": 40.00,
                "zones": ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1e"]
            },
        }


@dataclass
class GCPConfig:
    """GCP configuration."""
    project_id: str | None = None

    # Zones with GPU availability (T4 is most widely available)
    zones: dict[str, list[str]] = None

    # Machine types to test
    machine_types: dict[str, dict] = None

    def __post_init__(self):
        self.zones = {
            "t4": ["us-central1-a", "us-central1-b", "us-west1-a", "us-east1-b"],
            "l4": ["us-central1-a", "us-west1-a"],
            "a100": ["us-central1-a", "us-west1-b"],
        }

        self.machine_types = {
            "n1-standard-4": {
                "gpu": "T4",
                "gpu_type": "nvidia-tesla-t4",
                "gpu_count": 1,
                "vcpu": 4,
                "memory_gb": 15,
                "zones": self.zones["t4"]
            },
            "g2-standard-4": {
                "gpu": "L4",
                "gpu_type": "nvidia-l4",
                "gpu_count": 1,
                "vcpu": 4,
                "memory_gb": 16,
                "zones": self.zones["l4"]
            },
            "a2-highgpu-1g": {
                "gpu": "A100",
                "gpu_type": "nvidia-tesla-a100",
                "gpu_count": 1,
                "vcpu": 12,
                "memory_gb": 85,
                "zones": self.zones["a100"]
            },
        }


# ============================================================================
# Validation Script Generator
# ============================================================================

def generate_validation_script(backend: str, instance_name: str) -> str:
    """Generate validation script to run on cloud instance."""
    return f'''#!/bin/bash
# TorchBridge v0.4.34 Cloud Validation
# Instance: {instance_name}
# Backend: {backend}
# Generated: {datetime.now().isoformat()}

set -e

echo "=============================================="
echo "TorchBridge Cloud Validation"
echo "Instance: {instance_name}"
echo "Backend: {backend}"
echo "=============================================="

# 1. Environment Setup
echo "[1/7] Setting up environment..."
cd /tmp
git clone https://github.com/CloudlyIO/torchbridge.git torchbridge 2>/dev/null || true
cd torchbridge
pip install -e ".[dev]" -q

# 2. Hardware Detection
echo "[2/7] Detecting hardware..."
python3 -c "
import torch
print(f'PyTorch: {{torch.__version__}}')
print(f'CUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'GPU: {{torch.cuda.get_device_name(0)}}')
    print(f'GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')
"

# 3. Quick Backend Test
echo "[3/7] Running quick backend test..."
python3 -c "
import torchbridge as kpt
print(f'TorchBridge: {{kpt.__version__}}')
manager = kpt.get_manager()
print(f'Manager initialized: {{type(manager).__name__}}')
"

# 4. Unit Tests
echo "[4/7] Running unit tests..."
python3 -m pytest tests/unit/ -v --tb=short -q 2>&1 | tail -30

# 5. Backend Tests
echo "[5/7] Running backend tests..."
python3 -m pytest tests/backends/ -v --tb=short -q 2>&1 | tail -30

# 6. Integration Tests
echo "[6/7] Running integration tests..."
python3 -m pytest tests/integration/ -v --tb=short -q 2>&1 | tail -50

# 7. Benchmark
echo "[7/7] Running quick benchmark..."
python3 -c "
import torch
import time

# Simple benchmark
x = torch.randn(32, 512, 512)
if torch.cuda.is_available():
    x = x.cuda()
    torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    y = torch.nn.functional.softmax(x, dim=-1)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - start
print(f'Benchmark: 100 iterations in {{elapsed:.2f}}s ({{100/elapsed:.1f}} iter/s)')
"

echo "=============================================="
echo "Validation Complete!"
echo "=============================================="
'''


# ============================================================================
# AWS Validation
# ============================================================================

def run_aws_validation(
    instance_type: str = "g5.xlarge",
    spot: bool = True,
    dry_run: bool = False,
) -> dict:
    """Run validation on AWS EC2 instance."""
    config = AWSConfig()

    if instance_type not in config.instance_types:
        print(f"Unknown instance type: {instance_type}")
        print(f"Available: {list(config.instance_types.keys())}")
        return {"status": "error", "message": "Unknown instance type"}

    instance_info = config.instance_types[instance_type]
    zone = instance_info["zones"][0]  # Use first available zone

    print(f"\n{'='*60}")
    print(f"AWS Validation: {instance_type}")
    print(f"{'='*60}")
    print(f"Region: {config.region}")
    print(f"Zone: {zone}")
    print(f"GPU: {instance_info['gpu_count']}x {instance_info['gpu']}")
    print(f"Spot: {spot}")

    if dry_run:
        print("\n[DRY RUN] Would launch instance with:")
        print(f"  AMI: {config.ami_id}")
        print(f"  Key: {config.key_name}")
        return {"status": "dry_run", "instance_type": instance_type}

    # Generate validation script
    script_path = REPORTS_DIR / f"validation_{instance_type.replace('.', '_')}.sh"
    script_content = generate_validation_script("cuda", instance_type)
    script_path.write_text(script_content)
    print(f"\nValidation script: {script_path}")

    # Launch instance
    print("\nLaunching instance...")

    # Create user data to run validation script
    user_data = f'''#!/bin/bash
exec > >(tee /var/log/validation.log) 2>&1
{script_content}
# Upload results
aws s3 cp /var/log/validation.log s3://torchbridge-validation/{instance_type}-$(date +%s).log || true
# Terminate self after validation
shutdown -h now
'''

    # Build launch command
    launch_cmd = [
        "aws", "ec2", "run-instances",
        "--region", config.region,
        "--image-id", config.ami_id,
        "--instance-type", instance_type,
        "--key-name", config.key_name,
        "--placement", f"AvailabilityZone={zone}",
        "--instance-initiated-shutdown-behavior", "terminate",
        "--tag-specifications",
        f'ResourceType=instance,Tags=[{{Key=Name,Value=tb-validation-{instance_type}}},{{Key=Project,Value=TorchBridge}}]',
        "--user-data", user_data,
        "--query", "Instances[0].InstanceId",
        "--output", "text",
    ]

    if spot:
        launch_cmd.extend([
            "--instance-market-options",
            json.dumps({
                "MarketType": "spot",
                "SpotOptions": {
                    "MaxPrice": str(instance_info["spot_price"]),
                    "SpotInstanceType": "one-time",
                }
            })
        ])

    try:
        result = subprocess.run(launch_cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            instance_id = result.stdout.strip()
            print(f"Instance launched: {instance_id}")

            # Save result
            validation_result = {
                "status": "launched",
                "provider": "aws",
                "instance_type": instance_type,
                "instance_id": instance_id,
                "region": config.region,
                "zone": zone,
                "spot": spot,
                "timestamp": datetime.now().isoformat(),
            }

            result_file = REPORTS_DIR / f"aws_{instance_type.replace('.', '_')}_result.json"
            result_file.write_text(json.dumps(validation_result, indent=2))

            print(f"Result saved: {result_file}")
            return validation_result
        else:
            error_msg = result.stderr.strip()
            print(f"Launch failed: {error_msg}")
            return {"status": "failed", "error": error_msg}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "Instance launch timed out"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# GCP Validation
# ============================================================================

def run_gcp_validation(
    machine_type: str = "n1-standard-4",
    preemptible: bool = True,
    dry_run: bool = False,
) -> dict:
    """Run validation on GCP Compute Engine instance."""
    config = GCPConfig()

    if machine_type not in config.machine_types:
        print(f"Unknown machine type: {machine_type}")
        print(f"Available: {list(config.machine_types.keys())}")
        return {"status": "error", "message": "Unknown machine type"}

    machine_info = config.machine_types[machine_type]
    zone = machine_info["zones"][0]  # Use first available zone

    print(f"\n{'='*60}")
    print(f"GCP Validation: {machine_type}")
    print(f"{'='*60}")
    print(f"Zone: {zone}")
    print(f"GPU: {machine_info['gpu_count']}x {machine_info['gpu']}")
    print(f"Preemptible: {preemptible}")

    if dry_run:
        print("\n[DRY RUN] Would launch instance with:")
        print(f"  GPU Type: {machine_info['gpu_type']}")
        return {"status": "dry_run", "machine_type": machine_type}

    # Generate validation script
    script_path = REPORTS_DIR / f"validation_{machine_type.replace('-', '_')}.sh"
    script_content = generate_validation_script("cuda", machine_type)
    script_path.write_text(script_content)
    print(f"\nValidation script: {script_path}")

    # Launch instance
    instance_name = f"tb-val-{int(time.time())}"
    print(f"\nLaunching instance: {instance_name}...")

    launch_cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        "--zone", zone,
        "--machine-type", machine_type,
        "--image-family", "pytorch-latest-gpu",
        "--image-project", "deeplearning-platform-release",
        "--boot-disk-size", "100GB",
        "--boot-disk-type", "pd-ssd",
        "--metadata-from-file", f"startup-script={script_path}",
        "--scopes", "cloud-platform",
        "--format", "json",
    ]

    # Add GPU for non-g2 machine types (g2 has integrated L4)
    if not machine_type.startswith("g2"):
        launch_cmd.extend([
            "--accelerator", f"type={machine_info['gpu_type']},count={machine_info['gpu_count']}",
            "--maintenance-policy", "TERMINATE",
        ])

    if preemptible:
        launch_cmd.append("--preemptible")

    try:
        result = subprocess.run(launch_cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            instance_data = json.loads(result.stdout)
            print(f"Instance launched: {instance_name}")

            validation_result = {
                "status": "launched",
                "provider": "gcp",
                "machine_type": machine_type,
                "instance_name": instance_name,
                "zone": zone,
                "preemptible": preemptible,
                "timestamp": datetime.now().isoformat(),
            }

            result_file = REPORTS_DIR / f"gcp_{machine_type.replace('-', '_')}_result.json"
            result_file.write_text(json.dumps(validation_result, indent=2))

            print(f"Result saved: {result_file}")
            return validation_result
        else:
            error_msg = result.stderr.strip()
            print(f"Launch failed: {error_msg}")
            return {"status": "failed", "error": error_msg}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "Instance launch timed out"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TorchBridge Cloud Validation")
    parser.add_argument("--provider", choices=["aws", "gcp", "all"], default="aws",
                       help="Cloud provider to use")
    parser.add_argument("--instance-type", default="g5.xlarge",
                       help="AWS instance type (for --provider aws)")
    parser.add_argument("--machine-type", default="n1-standard-4",
                       help="GCP machine type (for --provider gcp)")
    parser.add_argument("--tier", choices=["dev", "spot", "full"], default="spot",
                       help="Validation tier: dev (g5.xlarge), spot (multiple), full (all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print what would be done without launching")
    parser.add_argument("--list", action="store_true",
                       help="List available instance/machine types")

    args = parser.parse_args()

    if args.list:
        print("\n=== AWS Instance Types ===")
        for name, info in AWSConfig().instance_types.items():
            print(f"  {name}: {info['gpu_count']}x {info['gpu']} (spot: ${info['spot_price']}/hr)")

        print("\n=== GCP Machine Types ===")
        for name, info in GCPConfig().machine_types.items():
            print(f"  {name}: {info['gpu_count']}x {info['gpu']}")
        return

    results = []

    if args.provider in ["aws", "all"]:
        if args.tier == "dev":
            results.append(run_aws_validation("g5.xlarge", spot=True, dry_run=args.dry_run))
        elif args.tier == "spot":
            for itype in ["g5.xlarge", "g5.2xlarge"]:
                results.append(run_aws_validation(itype, spot=True, dry_run=args.dry_run))
        elif args.tier == "full":
            for itype in ["g5.xlarge", "g5.2xlarge", "p4d.24xlarge"]:
                results.append(run_aws_validation(itype, spot=True, dry_run=args.dry_run))
        else:
            results.append(run_aws_validation(args.instance_type, spot=True, dry_run=args.dry_run))

    if args.provider in ["gcp", "all"]:
        if args.tier == "dev":
            results.append(run_gcp_validation("n1-standard-4", preemptible=True, dry_run=args.dry_run))
        elif args.tier == "spot":
            for mtype in ["n1-standard-4", "g2-standard-4"]:
                results.append(run_gcp_validation(mtype, preemptible=True, dry_run=args.dry_run))
        elif args.tier == "full":
            for mtype in ["n1-standard-4", "g2-standard-4", "a2-highgpu-1g"]:
                results.append(run_gcp_validation(mtype, preemptible=True, dry_run=args.dry_run))
        else:
            results.append(run_gcp_validation(args.machine_type, preemptible=True, dry_run=args.dry_run))

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    launched = sum(1 for r in results if r.get("status") == "launched")
    failed = sum(1 for r in results if r.get("status") == "failed")

    print(f"Launched: {launched}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {REPORTS_DIR}")

    # Save overall summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "provider": args.provider,
        "tier": args.tier,
        "results": results,
    }
    summary_file = REPORTS_DIR / "cloud_validation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
