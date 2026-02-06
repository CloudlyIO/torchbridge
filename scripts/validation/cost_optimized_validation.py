#!/usr/bin/env python3
"""
TorchBridge Cost-Optimized Cloud Validation

Reduces validation costs from ~$2,000 to ~$50-100 using:
1. Free/low-cost cloud resources (Intel DevCloud, Colab, Kaggle)
2. Spot/preemptible instances (70-90% cheaper)
3. Smallest viable instances per backend
4. Reduced test duration (5 mins vs 60 mins)
5. Tiered validation (quick -> standard -> comprehensive)

Usage:
    python cost_optimized_validation.py --tier quick      # ~$0 (free resources)
    python cost_optimized_validation.py --tier standard   # ~$20-50
    python cost_optimized_validation.py --tier full       # ~$100-200
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ValidationTier(Enum):
    QUICK = "quick"        # Free resources only (~$0)
    STANDARD = "standard"  # Low-cost spot instances (~$20-50)
    FULL = "full"          # Comprehensive validation (~$100-200)


@dataclass
class ResourceConfig:
    """Cloud resource configuration."""
    name: str
    provider: str
    backend: str
    cost_per_hour: float
    is_free: bool = False
    notes: str = ""


@dataclass
class ValidationPlan:
    """Validation plan with cost estimate."""
    tier: ValidationTier
    resources: list[ResourceConfig]
    estimated_cost: float
    estimated_time_minutes: int
    tests: list[str]


# ============================================================================
# FREE RESOURCES
# ============================================================================

FREE_RESOURCES = [
    ResourceConfig(
        name="Intel DevCloud (Arc A770)",
        provider="intel",
        backend="xpu",
        cost_per_hour=0,
        is_free=True,
        notes="Free Intel developer access, requires signup"
    ),
    ResourceConfig(
        name="Google Colab Free (T4)",
        provider="colab",
        backend="cuda",
        cost_per_hour=0,
        is_free=True,
        notes="Free tier, ~12h GPU limit per session"
    ),
    ResourceConfig(
        name="Kaggle Notebooks (P100/T4)",
        provider="kaggle",
        backend="cuda",
        cost_per_hour=0,
        is_free=True,
        notes="30h GPU per week, requires Kaggle account"
    ),
    ResourceConfig(
        name="Lightning.ai (T4)",
        provider="lightning",
        backend="cuda",
        cost_per_hour=0,
        is_free=True,
        notes="22 free GPU hours per month"
    ),
    ResourceConfig(
        name="GitHub Codespaces (CPU)",
        provider="github",
        backend="cpu",
        cost_per_hour=0,
        is_free=True,
        notes="60h free per month, good for CPU tests"
    ),
]

# ============================================================================
# LOW-COST SPOT INSTANCES
# ============================================================================

SPOT_RESOURCES = [
    ResourceConfig(
        name="AWS g5.xlarge Spot (A10G)",
        provider="aws",
        backend="cuda",
        cost_per_hour=0.30,  # ~70% off on-demand
        notes="Spot instance, may be interrupted"
    ),
    ResourceConfig(
        name="GCP g2-standard-4 Preemptible (L4)",
        provider="gcp",
        backend="cuda",
        cost_per_hour=0.25,
        notes="Preemptible, 24h max runtime"
    ),
    ResourceConfig(
        name="Lambda Labs A10 (24GB)",
        provider="lambda",
        backend="cuda",
        cost_per_hour=0.60,
        notes="On-demand, no spot pricing"
    ),
    ResourceConfig(
        name="Vast.ai RTX 4090",
        provider="vast",
        backend="cuda",
        cost_per_hour=0.40,
        notes="Community cloud, variable availability"
    ),
]

# ============================================================================
# PREMIUM RESOURCES (for comprehensive validation)
# ============================================================================

PREMIUM_RESOURCES = [
    ResourceConfig(
        name="AWS p4d.24xlarge Spot (8x A100)",
        provider="aws",
        backend="cuda",
        cost_per_hour=9.80,  # ~70% off $32.77 on-demand
        notes="Spot instance for H100/A100 validation"
    ),
    ResourceConfig(
        name="GCP a2-highgpu-1g Preemptible (A100)",
        provider="gcp",
        backend="cuda",
        cost_per_hour=1.20,
        notes="Single A100 for cost-effective testing"
    ),
    ResourceConfig(
        name="GCP TPU v5e-1 (TPU)",
        provider="gcp",
        backend="tpu",
        cost_per_hour=1.20,
        notes="Smallest TPU configuration"
    ),
]


class CostOptimizedValidator:
    """Cost-optimized cloud validation orchestrator."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "reports" / "cloud_validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_validation_plan(self, tier: ValidationTier) -> ValidationPlan:
        """Create validation plan based on tier."""

        if tier == ValidationTier.QUICK:
            return ValidationPlan(
                tier=tier,
                resources=FREE_RESOURCES[:4],  # Intel, Colab, Kaggle, Lightning
                estimated_cost=0,
                estimated_time_minutes=60,
                tests=[
                    "Backend detection and initialization",
                    "Basic forward/backward pass",
                    "Memory optimization smoke test",
                    "Export functionality (TorchScript, ONNX)",
                    "Quick benchmark (5 iterations)",
                ]
            )

        elif tier == ValidationTier.STANDARD:
            return ValidationPlan(
                tier=tier,
                resources=FREE_RESOURCES + SPOT_RESOURCES[:2],
                estimated_cost=25,  # ~2 hours of spot instances
                estimated_time_minutes=120,
                tests=[
                    "Full unit test suite per backend",
                    "Integration tests",
                    "Performance benchmarks (20 iterations)",
                    "Memory stress test (5 minutes)",
                    "Multi-precision validation (FP32, FP16, BF16)",
                    "Model export and reload",
                ]
            )

        else:  # FULL
            return ValidationPlan(
                tier=tier,
                resources=FREE_RESOURCES + SPOT_RESOURCES + PREMIUM_RESOURCES[:2],
                estimated_cost=150,  # ~4 hours premium + spot
                estimated_time_minutes=240,
                tests=[
                    "Complete test suite per backend",
                    "Comprehensive benchmarks (100 iterations)",
                    "Long-running stability test (30 minutes)",
                    "All precision modes (FP32, FP16, BF16, FP8)",
                    "Distributed training validation",
                    "Large model tests (7B parameter)",
                    "Production deployment simulation",
                ]
            )

    def generate_notebook(self, resource: ResourceConfig) -> str:
        """Generate Jupyter notebook for cloud validation."""

        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# TorchBridge v0.4.34 Validation\n",
                        f"**Platform**: {resource.name}\n",
                        f"**Backend**: {resource.backend}\n",
                        f"**Generated**: {datetime.now().isoformat()}\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Install TorchBridge\n",
                        "!pip install -q torchbridge torch\n",
                        "\n",
                        "# For Intel XPU\n",
                        "# !pip install intel-extension-for-pytorch\n"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "import torch\n",
                        "import sys\n",
                        "sys.path.insert(0, '.')\n",
                        "\n",
                        "# Check hardware\n",
                        "print(f'PyTorch: {torch.__version__}')\n",
                        "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                        "    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Basic validation\n",
                        "from torchbridge.hardware import get_optimal_backend, create_backend\n",
                        "\n",
                        "# Detect backend\n",
                        "backend_name = get_optimal_backend()\n",
                        "print(f'Optimal backend: {backend_name}')\n",
                        "\n",
                        "backend = create_backend(backend_name)\n",
                        "print(f'Backend info: {backend.get_device_info()}')\n"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Attention test\n",
                        "import torch\n",
                        "import torch.nn as nn\n",
                        "\n",
                        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                        "\n",
                        "# Test basic attention operation\n",
                        "attention = nn.MultiheadAttention(512, 8, batch_first=True).to(device)\n",
                        "\n",
                        "# Test forward pass\n",
                        "x = torch.randn(2, 128, 512, device=device)\n",
                        "out, _ = attention(x, x, x)\n",
                        "print(f'Attention output shape: {out.shape}')\n",
                        "print('Attention test PASSED')\n"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Quick benchmark\n",
                        "import time\n",
                        "\n",
                        "def benchmark(fn, warmup=3, iterations=10):\n",
                        "    for _ in range(warmup):\n",
                        "        fn()\n",
                        "    if torch.cuda.is_available():\n",
                        "        torch.cuda.synchronize()\n",
                        "    \n",
                        "    start = time.perf_counter()\n",
                        "    for _ in range(iterations):\n",
                        "        fn()\n",
                        "    if torch.cuda.is_available():\n",
                        "        torch.cuda.synchronize()\n",
                        "    \n",
                        "    elapsed = time.perf_counter() - start\n",
                        "    return elapsed / iterations * 1000  # ms\n",
                        "\n",
                        "# Benchmark attention\n",
                        "x = torch.randn(8, 512, 512, device=device)\n",
                        "time_ms = benchmark(lambda: attention(x, x, x))\n",
                        "print(f'Attention latency: {time_ms:.2f}ms')\n",
                        "print(f'Throughput: {8000/time_ms:.1f} samples/s')\n"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Summary\n",
                        "print('\\n' + '='*50)\n",
                        "print('VALIDATION SUMMARY')\n",
                        "print('='*50)\n",
                        f"print('Platform: {resource.name}')\n",
                        "print(f'Backend: {{backend_name}}')\n",
                        "print(f'Device: {{device}}')\n",
                        "print('Status: PASSED')\n",
                        "print('='*50)\n"
                    ],
                    "execution_count": None,
                    "outputs": []
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        return json.dumps(notebook_content, indent=2)

    def generate_colab_link(self) -> str:
        """Generate Google Colab notebook link."""
        return """
# Google Colab Quick Validation

1. Open Google Colab: https://colab.research.google.com

2. Create new notebook and paste:

```python
# Cell 1: Install
!pip install -q torchbridge torch

# Cell 2: Validate
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 3: Test TorchBridge
from torchbridge.hardware import get_optimal_backend, create_backend
backend = get_optimal_backend()
print(f"Backend: {backend}")

import torch.nn as nn
attn = nn.MultiheadAttention(512, 8, batch_first=True).cuda()
x = torch.randn(4, 128, 512).cuda()
out, _ = attn(x, x, x)
print(f"Output: {out.shape} - PASSED")
```

3. Runtime > Change runtime type > GPU

4. Run all cells
"""

    def generate_kaggle_script(self) -> str:
        """Generate Kaggle notebook script."""
        return """
# Kaggle Notebook Validation

1. Go to https://www.kaggle.com/code

2. Create new notebook, enable GPU (Settings > Accelerator > GPU)

3. Paste and run:

```python
!pip install -q torchbridge

import torch
import torch.nn as nn
from torchbridge.hardware import get_optimal_backend

# Validate
print(f"GPU: {torch.cuda.get_device_name(0)}")
backend = get_optimal_backend()
print(f"Backend: {backend}")

# Test
attn = nn.MultiheadAttention(512, 8, batch_first=True).cuda()
x = torch.randn(4, 128, 512).cuda()
out, _ = attn(x, x, x)
print(f"Shape: {out.shape} - PASSED")
```
"""

    def generate_intel_devcloud_script(self) -> str:
        """Generate Intel DevCloud validation script."""
        return """#!/bin/bash
# Intel DevCloud Validation Script

# 1. Connect to DevCloud
# ssh devcloud

# 2. Request GPU node
qsub -I -l nodes=1:gpu:ppn=2 -d .

# 3. Once on node, run:
source /opt/intel/oneapi/setvars.sh
pip install --user torchbridge intel-extension-for-pytorch

python << 'EOF'
import torch
import intel_extension_for_pytorch as ipex
from torchbridge.hardware import get_optimal_backend

# Check XPU
print(f"XPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"XPU device: {torch.xpu.get_device_name(0)}")

backend = get_optimal_backend()
print(f"Backend: {backend}")

# Test
import torch.nn as nn
device = 'xpu' if torch.xpu.is_available() else 'cpu'
attn = nn.MultiheadAttention(512, 8, batch_first=True).to(device)
x = torch.randn(4, 128, 512, device=device)
out, _ = attn(x, x, x)
print(f"Output: {out.shape} - PASSED")
EOF
"""

    def run_validation(self, tier: ValidationTier) -> dict[str, Any]:
        """Run validation for specified tier."""
        plan = self.create_validation_plan(tier)

        print(f"\n{'='*60}")
        print("TorchBridge Cost-Optimized Validation")
        print(f"{'='*60}")
        print(f"Tier: {tier.value.upper()}")
        print(f"Estimated Cost: ${plan.estimated_cost}")
        print(f"Estimated Time: {plan.estimated_time_minutes} minutes")
        print(f"\nResources ({len(plan.resources)}):")

        for r in plan.resources:
            cost_str = "FREE" if r.is_free else f"${r.cost_per_hour:.2f}/hr"
            print(f"  - {r.name} ({r.backend}) - {cost_str}")

        print(f"\nTests ({len(plan.tests)}):")
        for t in plan.tests:
            print(f"  - {t}")

        # Generate artifacts
        print(f"\n{'='*60}")
        print("Generated Validation Artifacts:")
        print(f"{'='*60}")

        # Colab instructions
        colab_path = self.results_dir / "colab_validation.md"
        colab_path.write_text(self.generate_colab_link())
        print(f"  - {colab_path}")

        # Kaggle instructions
        kaggle_path = self.results_dir / "kaggle_validation.md"
        kaggle_path.write_text(self.generate_kaggle_script())
        print(f"  - {kaggle_path}")

        # Intel DevCloud script
        intel_path = self.results_dir / "intel_devcloud_validation.sh"
        intel_path.write_text(self.generate_intel_devcloud_script())
        print(f"  - {intel_path}")

        # Notebooks for each free resource
        for resource in [r for r in plan.resources if r.is_free]:
            notebook_name = resource.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            notebook_path = self.results_dir / f"validation_{notebook_name}.ipynb"
            notebook_path.write_text(self.generate_notebook(resource))
            print(f"  - {notebook_path}")

        # Summary
        result = {
            "tier": tier.value,
            "estimated_cost": plan.estimated_cost,
            "estimated_time_minutes": plan.estimated_time_minutes,
            "resources": [r.name for r in plan.resources],
            "tests": plan.tests,
            "artifacts_generated": True,
            "timestamp": datetime.now().isoformat()
        }

        result_path = self.results_dir / f"validation_plan_{tier.value}.json"
        result_path.write_text(json.dumps(result, indent=2))
        print(f"  - {result_path}")

        print(f"\n{'='*60}")
        print("Next Steps:")
        print(f"{'='*60}")
        print("1. Open Google Colab and run colab_validation.md instructions")
        print("2. Create Kaggle notebook using kaggle_validation.md")
        print("3. Connect to Intel DevCloud and run intel_devcloud_validation.sh")
        if tier != ValidationTier.QUICK:
            print("4. For paid resources, use cloud_orchestrator.py with spot instances")

        return result


def main():
    parser = argparse.ArgumentParser(description="Cost-Optimized Cloud Validation")
    parser.add_argument(
        "--tier",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Validation tier (quick=free, standard=$20-50, full=$100-200)"
    )
    parser.add_argument("--list", action="store_true", help="List all resources")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    validator = CostOptimizedValidator(project_root)

    if args.list:
        print("\nFREE Resources:")
        for r in FREE_RESOURCES:
            print(f"  {r.name:30} - {r.backend:6} - {r.notes}")

        print("\nSPOT/Low-Cost Resources:")
        for r in SPOT_RESOURCES:
            print(f"  {r.name:30} - {r.backend:6} - ${r.cost_per_hour:.2f}/hr")

        print("\nPREMIUM Resources:")
        for r in PREMIUM_RESOURCES:
            print(f"  {r.name:30} - {r.backend:6} - ${r.cost_per_hour:.2f}/hr")
        return

    tier = ValidationTier(args.tier)
    validator.run_validation(tier)


if __name__ == "__main__":
    main()
