"""
Distributed Training Cross-Backend Example

Demonstrates TorchBridge's distributed training configuration:
- Auto-detect hardware and recommend parallelism
- Configure FSDP2, pipeline schedules, and communication backends
- Export reproducible TOML configs
"""

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TrainiumArchitecture,
    TPUVersion,
)
from torchbridge.distributed import (
    CollectiveBackendMatrix,
    DistributedConfig,
    FSDPManager,
    PipelineScheduleFactory,
    TopologyDetector,
)
from torchbridge.distributed.config import recommend_parallelism


def main():
    print("=" * 60)
    print("TorchBridge Distributed Training Configuration")
    print("=" * 60)

    # ── Scenario 1: Single-node H100 training ──────────────────
    print("\n--- Scenario 1: 7B model on 8x H100 (single node) ---\n")

    config = DistributedConfig.auto(
        model_params=int(7e9),
        backend=HardwareBackend.CUDA,
        architecture=NVIDIAArchitecture.HOPPER,
        world_size=8,
    )

    print(f"FSDP strategy:    {config.fsdp.sharding_strategy.value}")
    print(f"Mixed precision:   {config.fsdp.mixed_precision.value}")
    print(f"Float8 all-gather: {config.fsdp.float8_all_gather}")
    print(f"Pipeline schedule: {config.pipeline.schedule.value}")
    print(f"Collective:        {config.collective.backend.value}")

    # ── Scenario 2: Multi-node training ────────────────────────
    print("\n--- Scenario 2: 70B model on 16x H100 (2 nodes) ---\n")

    config_mn = DistributedConfig.auto(
        model_params=int(70e9),
        backend=HardwareBackend.CUDA,
        architecture=NVIDIAArchitecture.HOPPER,
        world_size=16,
        gpus_per_node=8,
    )

    print(f"FSDP strategy:    {config_mn.fsdp.sharding_strategy.value}")
    print(f"Multi-node:        {config_mn.mesh.is_multi_node()}")
    print(f"Mesh shape:        {list(config_mn.mesh.mesh_shape)}")

    # ── Scenario 3: AMD MI300X ─────────────────────────────────
    print("\n--- Scenario 3: 7B model on 4x MI300X ---\n")

    config_amd = DistributedConfig.auto(
        model_params=int(7e9),
        backend=HardwareBackend.AMD,
        architecture=AMDArchitecture.CDNA3,
        world_size=4,
    )

    print(f"Collective:        {config_amd.collective.backend.value}")
    print(f"Symmetric memory:  {config_amd.collective.symmetric_memory}")
    print(f"Mixed precision:   {config_amd.fsdp.mixed_precision.value}")

    # ── Scenario 4: Parallelism recommendation ─────────────────
    print("\n--- Scenario 4: Parallelism Advisor ---\n")

    rec = recommend_parallelism(
        model_params=int(70e9),
        backend=HardwareBackend.CUDA,
        architecture=NVIDIAArchitecture.HOPPER,
        world_size=16,
        gpus_per_node=8,
    )

    print(f"TP degree:         {rec.tensor_parallel_degree}")
    print(f"PP stages:         {rec.pipeline_parallel_stages}")
    print(f"FSDP strategy:     {rec.fsdp_strategy}")
    print(f"Memory/rank:       {rec.estimated_memory_per_rank_gb:.2f} GB")
    print(f"Comm volume:       {rec.estimated_communication_volume_gb:.2f} GB/step")

    for note in rec.notes:
        print(f"  Note: {note}")

    # ── Scenario 5: TOML export ────────────────────────────────
    print("\n--- Scenario 5: TOML Configuration Export ---\n")
    print(config.to_toml())

    # ── Pipeline schedule comparison ───────────────────────────
    print("--- Pipeline Schedule Support ---\n")

    backends = [
        ("NVIDIA Hopper", HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
        ("NVIDIA Ampere", HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
        ("AMD CDNA3", HardwareBackend.AMD, AMDArchitecture.CDNA3),
        ("Trainium TRN2", HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
        ("TPU v5e", HardwareBackend.TPU, TPUVersion.V5E),
    ]

    for name, backend, arch in backends:
        schedules = PipelineScheduleFactory.get_supported_schedules(backend, arch)
        schedule_names = [s.value for s in schedules]
        print(f"  {name:16s}: {', '.join(schedule_names)}")

    print()


if __name__ == "__main__":
    main()
