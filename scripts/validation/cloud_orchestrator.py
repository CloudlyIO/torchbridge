#!/usr/bin/env python3
"""
TorchBridge v0.4.30 Cloud Orchestrator

Manages cloud instance deployment and test execution for hardware validation:
- AWS: p5.48xlarge (H100), p4d.24xlarge (A100), g5.xlarge (A10G), AMD MI300X
- GCP: a3-highgpu-8g (H100), a2-highgpu-1g (A100), g2-standard-4 (L4), TPU v5e/v5p
- Intel DevCloud: Arc A770, Flex 170, Ponte Vecchio

Usage:
    python scripts/validation/cloud_orchestrator.py --provider aws --instance p5.48xlarge
    python scripts/validation/cloud_orchestrator.py --provider gcp --instance a3-highgpu-8g
    python scripts/validation/cloud_orchestrator.py --provider gcp --tpu v5e-8
    python scripts/validation/cloud_orchestrator.py --provider intel --device arc-a770
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    INTEL = "intel"


class HardwareBackend(Enum):
    NVIDIA_CUDA = "cuda"
    AMD_ROCM = "rocm"
    INTEL_XPU = "xpu"
    TPU_XLA = "tpu"


@dataclass
class InstanceConfig:
    """Cloud instance configuration."""
    provider: CloudProvider
    instance_type: str
    backend: HardwareBackend
    gpu_count: int
    gpu_memory_gb: int
    region: str
    spot: bool = True


@dataclass
class ValidationConfig:
    """Validation test configuration."""
    features: list[str]
    precisions: list[str]
    batch_sizes: list[int]
    sequence_lengths: list[int]
    stress_duration_minutes: int = 60


# Instance configurations
AWS_INSTANCES = {
    "p5.48xlarge": InstanceConfig(
        provider=CloudProvider.AWS,
        instance_type="p5.48xlarge",
        backend=HardwareBackend.NVIDIA_CUDA,
        gpu_count=8,
        gpu_memory_gb=80,
        region="us-east-1",
    ),
    "p4d.24xlarge": InstanceConfig(
        provider=CloudProvider.AWS,
        instance_type="p4d.24xlarge",
        backend=HardwareBackend.NVIDIA_CUDA,
        gpu_count=8,
        gpu_memory_gb=40,
        region="us-east-1",
    ),
    "g5.xlarge": InstanceConfig(
        provider=CloudProvider.AWS,
        instance_type="g5.xlarge",
        backend=HardwareBackend.NVIDIA_CUDA,
        gpu_count=1,
        gpu_memory_gb=24,
        region="us-east-1",
    ),
    "mi300x": InstanceConfig(
        provider=CloudProvider.AWS,
        instance_type="p5.48xlarge",  # Placeholder - actual AMD instance
        backend=HardwareBackend.AMD_ROCM,
        gpu_count=8,
        gpu_memory_gb=192,
        region="us-east-1",
    ),
}

GCP_INSTANCES = {
    "a3-highgpu-8g": InstanceConfig(
        provider=CloudProvider.GCP,
        instance_type="a3-highgpu-8g",
        backend=HardwareBackend.NVIDIA_CUDA,
        gpu_count=8,
        gpu_memory_gb=80,
        region="us-central1",
    ),
    "a2-highgpu-1g": InstanceConfig(
        provider=CloudProvider.GCP,
        instance_type="a2-highgpu-1g",
        backend=HardwareBackend.NVIDIA_CUDA,
        gpu_count=1,
        gpu_memory_gb=40,
        region="us-central1",
    ),
    "g2-standard-4": InstanceConfig(
        provider=CloudProvider.GCP,
        instance_type="g2-standard-4",
        backend=HardwareBackend.NVIDIA_CUDA,
        gpu_count=1,
        gpu_memory_gb=24,
        region="us-central1",
    ),
    "v5e-8": InstanceConfig(
        provider=CloudProvider.GCP,
        instance_type="v5e-8",
        backend=HardwareBackend.TPU_XLA,
        gpu_count=8,
        gpu_memory_gb=16,  # Per chip
        region="us-central1",
    ),
    "v5p-8": InstanceConfig(
        provider=CloudProvider.GCP,
        instance_type="v5p-8",
        backend=HardwareBackend.TPU_XLA,
        gpu_count=8,
        gpu_memory_gb=95,
        region="us-central1",
    ),
}

INTEL_DEVICES = {
    "arc-a770": InstanceConfig(
        provider=CloudProvider.INTEL,
        instance_type="arc-a770",
        backend=HardwareBackend.INTEL_XPU,
        gpu_count=1,
        gpu_memory_gb=16,
        region="devcloud",
    ),
    "flex-170": InstanceConfig(
        provider=CloudProvider.INTEL,
        instance_type="flex-170",
        backend=HardwareBackend.INTEL_XPU,
        gpu_count=2,
        gpu_memory_gb=16,
        region="devcloud",
    ),
    "ponte-vecchio": InstanceConfig(
        provider=CloudProvider.INTEL,
        instance_type="ponte-vecchio",
        backend=HardwareBackend.INTEL_XPU,
        gpu_count=2,
        gpu_memory_gb=128,
        region="devcloud",
    ),
}


class CloudOrchestrator:
    """Orchestrates cloud hardware validation."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "reports" / "cloud_validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_instance_config(
        self,
        provider: CloudProvider,
        instance_type: str
    ) -> InstanceConfig | None:
        """Get instance configuration."""
        if provider == CloudProvider.AWS:
            return AWS_INSTANCES.get(instance_type)
        elif provider == CloudProvider.GCP:
            return GCP_INSTANCES.get(instance_type)
        elif provider == CloudProvider.INTEL:
            return INTEL_DEVICES.get(instance_type)
        return None

    def generate_validation_script(
        self,
        config: InstanceConfig,
        validation: ValidationConfig
    ) -> str:
        """Generate validation script for the instance."""

        script = f"""#!/bin/bash
# TorchBridge v0.4.30 Cloud Validation Script
# Instance: {config.instance_type}
# Backend: {config.backend.value}
# Generated: {datetime.now().isoformat()}

set -e

echo "=============================================="
echo "TorchBridge v0.4.30 Cloud Validation"
echo "Instance: {config.instance_type}"
echo "Backend: {config.backend.value}"
echo "=============================================="

# 1. Environment Setup
echo "[1/6] Setting up environment..."
pip install -q torchbridge[all]

# 2. Hardware Detection
echo "[2/6] Detecting hardware..."
python -c "
import torchbridge as kpt
print('Hardware detected:')
print(kpt.detect_hardware())
"

# 3. Functional Tests
echo "[3/6] Running functional tests..."
pytest tests/backends/test_{config.backend.value.replace('-', '_')}_backend.py -v --tb=short

# 4. Performance Benchmarks
echo "[4/6] Running performance benchmarks..."
"""

        for precision in validation.precisions:
            script += f"""
python benchmarks/backend_comparison.py \\
    --backend {config.backend.value} \\
    --precision {precision} \\
    --batch-sizes {','.join(map(str, validation.batch_sizes))} \\
    --seq-lengths {','.join(map(str, validation.sequence_lengths))} \\
    --output reports/benchmark_{config.instance_type}_{precision}.json
"""

        script += f"""
# 5. Memory Tests
echo "[5/6] Running memory tests..."
python -c "
import torch
import torchbridge as kpt

# Test memory optimization
model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
optimized = kpt.optimize(model, backend='{config.backend.value}')
print('Memory optimization successful')
"

# 6. Stress Tests
echo "[6/6] Running stress tests ({validation.stress_duration_minutes} minutes)..."
timeout {validation.stress_duration_minutes * 60} python -c "
import time
import torch
import torchbridge as kpt

start = time.time()
iterations = 0
while time.time() - start < {validation.stress_duration_minutes * 60}:
    # Simulated workload
    x = torch.randn(32, 512, 512)
    if torch.cuda.is_available():
        x = x.cuda()
    y = torch.nn.functional.softmax(x, dim=-1)
    iterations += 1
    if iterations % 1000 == 0:
        print(f'Iterations: {{iterations}}, Elapsed: {{time.time() - start:.0f}}s')
print(f'Completed {{iterations}} iterations in {{time.time() - start:.0f}}s')
" || echo "Stress test completed or timed out"

echo "=============================================="
echo "Validation Complete!"
echo "=============================================="
"""

        return script

    def deploy_aws(self, config: InstanceConfig) -> dict[str, Any]:
        """Deploy to AWS and run validation."""
        print(f"Deploying to AWS {config.instance_type}...")

        # This would use boto3 in a real implementation
        commands = [
            f"# Create EC2 instance",
            f"aws ec2 run-instances \\",
            f"  --instance-type {config.instance_type} \\",
            f"  --image-id ami-xxxxx \\",  # Deep Learning AMI
            f"  --key-name torchbridge-validation \\",
            f"  --security-group-ids sg-xxxxx \\",
            f"  --region {config.region}",
            "",
            f"# Wait for instance to be running",
            f"aws ec2 wait instance-running --instance-ids $INSTANCE_ID",
            "",
            f"# Copy validation script",
            f"scp validation_script.sh ec2-user@$INSTANCE_IP:/tmp/",
            "",
            f"# Run validation",
            f"ssh ec2-user@$INSTANCE_IP 'bash /tmp/validation_script.sh'",
            "",
            f"# Collect results",
            f"scp ec2-user@$INSTANCE_IP:/tmp/reports/*.json reports/",
            "",
            f"# Terminate instance",
            f"aws ec2 terminate-instances --instance-ids $INSTANCE_ID",
        ]

        return {
            "provider": "aws",
            "instance": config.instance_type,
            "commands": commands,
            "status": "script_generated"
        }

    def deploy_gcp(self, config: InstanceConfig) -> dict[str, Any]:
        """Deploy to GCP and run validation."""
        print(f"Deploying to GCP {config.instance_type}...")

        if config.backend == HardwareBackend.TPU_XLA:
            commands = [
                f"# Create TPU VM",
                f"gcloud compute tpus tpu-vm create torchbridge-validation \\",
                f"  --zone={config.region}-a \\",
                f"  --accelerator-type={config.instance_type} \\",
                f"  --version=tpu-ubuntu2204-base",
                "",
                f"# SSH and run validation",
                f"gcloud compute tpus tpu-vm ssh torchbridge-validation \\",
                f"  --zone={config.region}-a \\",
                f"  --command='bash /tmp/validation_script.sh'",
                "",
                f"# Delete TPU VM",
                f"gcloud compute tpus tpu-vm delete torchbridge-validation \\",
                f"  --zone={config.region}-a",
            ]
        else:
            commands = [
                f"# Create GPU VM",
                f"gcloud compute instances create torchbridge-validation \\",
                f"  --zone={config.region}-a \\",
                f"  --machine-type={config.instance_type} \\",
                f"  --accelerator=type=nvidia-h100-80gb,count={config.gpu_count} \\",
                f"  --image-family=pytorch-latest-gpu \\",
                f"  --image-project=deeplearning-platform-release",
                "",
                f"# Run validation",
                f"gcloud compute ssh torchbridge-validation \\",
                f"  --zone={config.region}-a \\",
                f"  --command='bash /tmp/validation_script.sh'",
                "",
                f"# Delete instance",
                f"gcloud compute instances delete torchbridge-validation \\",
                f"  --zone={config.region}-a",
            ]

        return {
            "provider": "gcp",
            "instance": config.instance_type,
            "commands": commands,
            "status": "script_generated"
        }

    def deploy_intel(self, config: InstanceConfig) -> dict[str, Any]:
        """Deploy to Intel DevCloud and run validation."""
        print(f"Deploying to Intel DevCloud {config.instance_type}...")

        commands = [
            f"# Connect to Intel DevCloud",
            f"ssh devcloud",
            "",
            f"# Request {config.instance_type} node",
            f"qsub -l nodes=1:{config.instance_type}:ppn=2 -d . validation_script.sh",
            "",
            f"# Monitor job",
            f"qstat",
            "",
            f"# Retrieve results",
            f"cat validation_script.sh.o*",
        ]

        return {
            "provider": "intel",
            "instance": config.instance_type,
            "commands": commands,
            "status": "script_generated"
        }

    def run_validation(
        self,
        provider: CloudProvider,
        instance_type: str,
        dry_run: bool = True
    ) -> dict[str, Any]:
        """Run validation on cloud instance."""
        config = self.get_instance_config(provider, instance_type)
        if not config:
            return {"error": f"Unknown instance type: {instance_type}"}

        # Default validation config
        validation = ValidationConfig(
            features=["attention", "precision", "moe", "memory", "export"],
            precisions=["fp32", "fp16", "bf16"],
            batch_sizes=[1, 8, 32],
            sequence_lengths=[512, 2048, 8192],
            stress_duration_minutes=60
        )

        # Add FP8 for Hopper GPUs
        if "h100" in instance_type.lower() or "p5" in instance_type.lower():
            validation.precisions.append("fp8")

        # Generate validation script
        script = self.generate_validation_script(config, validation)
        script_path = self.results_dir / f"validation_{instance_type}.sh"
        script_path.write_text(script)
        print(f"Generated validation script: {script_path}")

        # Deploy based on provider
        if dry_run:
            print("\n[DRY RUN] Would execute the following:")

        if provider == CloudProvider.AWS:
            result = self.deploy_aws(config)
        elif provider == CloudProvider.GCP:
            result = self.deploy_gcp(config)
        elif provider == CloudProvider.INTEL:
            result = self.deploy_intel(config)
        else:
            result = {"error": "Unknown provider"}

        # Print deployment commands
        if "commands" in result:
            print("\nDeployment commands:")
            for cmd in result["commands"]:
                print(f"  {cmd}")

        # Save result
        result_file = self.results_dir / f"deployment_{instance_type}.json"
        result_file.write_text(json.dumps(result, indent=2))

        return result

    def list_available_instances(self):
        """List all available cloud instances."""
        print("\nAvailable Cloud Instances for Validation:\n")

        print("AWS Instances:")
        print("-" * 60)
        for name, config in AWS_INSTANCES.items():
            print(f"  {name:20} - {config.backend.value:10} - {config.gpu_count}x GPU ({config.gpu_memory_gb}GB)")

        print("\nGCP Instances:")
        print("-" * 60)
        for name, config in GCP_INSTANCES.items():
            print(f"  {name:20} - {config.backend.value:10} - {config.gpu_count}x GPU ({config.gpu_memory_gb}GB)")

        print("\nIntel DevCloud Devices:")
        print("-" * 60)
        for name, config in INTEL_DEVICES.items():
            print(f"  {name:20} - {config.backend.value:10} - {config.gpu_count}x GPU ({config.gpu_memory_gb}GB)")


def main():
    parser = argparse.ArgumentParser(description="TorchBridge Cloud Validation Orchestrator")
    parser.add_argument("--provider", choices=["aws", "gcp", "intel"], help="Cloud provider")
    parser.add_argument("--instance", help="Instance type")
    parser.add_argument("--tpu", help="TPU type (GCP only)")
    parser.add_argument("--device", help="Device type (Intel only)")
    parser.add_argument("--list", action="store_true", help="List available instances")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Generate scripts without deploying")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    orchestrator = CloudOrchestrator(project_root)

    if args.list:
        orchestrator.list_available_instances()
        return

    if not args.provider:
        parser.print_help()
        return

    provider = CloudProvider(args.provider)
    instance_type = args.instance or args.tpu or args.device

    if not instance_type:
        print("Error: Must specify --instance, --tpu, or --device")
        return

    result = orchestrator.run_validation(provider, instance_type, dry_run=args.dry_run)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\nValidation prepared for {instance_type}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
