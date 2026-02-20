"""
Cross-Backend Checkpoint Example

Demonstrates TorchBridge checkpoint management: async save/load,
cross-backend portability, frequency advisor, and health triggers.
"""

import torch

from torchbridge.checkpoint import (
    CheckpointConfig,
    CheckpointFrequencyAdvisor,
    CheckpointHealthTrigger,
    CheckpointManager,
    StorageBackendType,
)
from torchbridge.core.config import HardwareBackend


def scenario_1_basic_checkpoint():
    """Basic checkpoint save with metadata and rotation."""
    print("=" * 60)
    print("Scenario 1: Basic Checkpoint Configuration")
    print("=" * 60)

    config = CheckpointConfig(
        storage_backend=StorageBackendType.LOCAL,
        storage_path="./demo_checkpoints",
        async_save=True,
        plan_caching=True,
        pinned_memory_staging=True,
        max_checkpoints_to_keep=3,
        normalize_on_save=True,
        include_metadata=True,
    )

    manager = CheckpointManager(
        config=config,
        backend=HardwareBackend.CUDA,
    )

    info = manager.get_info()
    print(f"  Backend:         {info['backend']}")
    print(f"  Async save:      {info['config']['async_save']}")
    print(f"  Plan caching:    {info['config']['plan_caching']}")
    print(f"  Max kept:        {info['config']['max_checkpoints_to_keep']}")
    print(f"  Normalize:       {info['config']['normalize_on_save']}")
    print()


def scenario_2_cross_backend():
    """Cross-backend checkpoint: save as CUDA, load as AMD."""
    print("=" * 60)
    print("Scenario 2: Cross-Backend Checkpoint Portability")
    print("=" * 60)

    # CUDA checkpoint config
    cuda_config = CheckpointConfig(
        normalize_on_save=True,
        include_metadata=True,
    )
    cuda_manager = CheckpointManager(
        config=cuda_config,
        backend=HardwareBackend.CUDA,
    )

    # AMD checkpoint config (for loading)
    amd_manager = CheckpointManager(
        config=CheckpointConfig(),
        backend=HardwareBackend.AMD,
    )

    print("  Save backend:    CUDA (NVIDIA)")
    print("  Load backend:    AMD (ROCm)")
    print("  Normalization:   FP8→FP16 on save, restore if target supports")
    print("  Device:          GPU→CPU on save, target device on load")
    print()


def scenario_3_frequency_advisor():
    """Checkpoint frequency advisor for different cluster sizes."""
    print("=" * 60)
    print("Scenario 3: Checkpoint Frequency Advisor")
    print("=" * 60)

    advisor = CheckpointFrequencyAdvisor()

    clusters = [
        (4, "Small (4 GPUs, single node)"),
        (32, "Medium (32 GPUs, 4 nodes)"),
        (128, "Large (128 GPUs, 16 nodes)"),
        (512, "XL (512 GPUs, 64 nodes)"),
    ]

    for world_size, description in clusters:
        rec = advisor.recommend(
            world_size=world_size,
            checkpoint_time_seconds=60.0,
            step_time_seconds=0.5,
        )
        print(f"\n  {description}")
        print(f"    Interval:    {rec.interval_minutes} min (~{rec.interval_steps} steps)")
        print(f"    Risk:        {rec.risk_level}")
        print(f"    Overhead:    {rec.optimal_overhead_pct:.1f}%")

    print()


def scenario_4_health_triggers():
    """Health-triggered checkpoint decisions."""
    print("=" * 60)
    print("Scenario 4: Health-Triggered Checkpointing")
    print("=" * 60)

    trigger = CheckpointHealthTrigger(
        health_threshold_temp_c=85.0,
        trigger_on_degrading=True,
        utilization_drop_threshold=0.5,
    )

    scenarios = [
        ("Healthy GPU", {
            "device_id": "gpu:0",
            "temperature_c": 65.0,
            "health_trend": "stable",
            "memory_errors": 0,
            "utilization": 0.92,
            "avg_utilization": 0.90,
        }),
        ("Overheating GPU", {
            "device_id": "gpu:1",
            "temperature_c": 91.0,
            "health_trend": "degrading",
            "memory_errors": 0,
            "utilization": 0.88,
            "avg_utilization": 0.90,
        }),
        ("Memory errors", {
            "device_id": "gpu:2",
            "temperature_c": 72.0,
            "health_trend": "stable",
            "memory_errors": 3,
            "utilization": 0.85,
            "avg_utilization": 0.90,
        }),
    ]

    for name, health in scenarios:
        should_save, reason = trigger.should_checkpoint(health)
        status = "TRIGGER" if should_save else "OK"
        print(f"\n  {name}: [{status}]")
        if reason:
            print(f"    Reason: {reason}")

    print()


def main():
    """Run all checkpoint scenarios."""
    print("\nTorchBridge Checkpoint Cross-Backend Demo")
    print("=" * 60)
    print()

    scenario_1_basic_checkpoint()
    scenario_2_cross_backend()
    scenario_3_frequency_advisor()
    scenario_4_health_triggers()

    print("All scenarios completed.")


if __name__ == "__main__":
    main()
