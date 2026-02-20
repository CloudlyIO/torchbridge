"""
Unified Distributed Training Configuration

Single config for all distributed settings: FSDP2, pipeline parallelism,
collective backend, and topology. Includes auto-configuration from model
size, backend, and world size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.distributed.collective_backend import CollectiveConfig
from torchbridge.distributed.fsdp2 import FSDP2Config, FSDP2Manager, ShardingStrategy
from torchbridge.distributed.pipeline_schedules import (
    PipelineConfig,
    PipelineScheduleFactory,
)
from torchbridge.distributed.topology import MeshConfig, TopologyDetector

logger = logging.getLogger(__name__)

Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)

# ── Heuristics for auto-config ──────────────────────────────────────────────

# Bytes per parameter in BF16/FP16
_BYTES_PER_PARAM_MIXED = 2
# Overhead factor for optimizer states + gradients + activations
_MEMORY_OVERHEAD_FACTOR = 4.0
# Default GPU memory (GB) if unknown
_DEFAULT_GPU_MEMORY_GB = 80.0

# GPU memory per backend/architecture (GB)
_GPU_MEMORY_MAP: dict[HardwareBackend, dict[Architecture, float]] = {
    HardwareBackend.CUDA: {
        NVIDIAArchitecture.BLACKWELL_DC: 192.0,
        NVIDIAArchitecture.BLACKWELL_CONSUMER: 32.0,
        NVIDIAArchitecture.HOPPER: 80.0,
        NVIDIAArchitecture.ADA: 48.0,
        NVIDIAArchitecture.AMPERE: 80.0,
        NVIDIAArchitecture.TURING: 24.0,
        None: 80.0,
    },
    HardwareBackend.AMD: {
        AMDArchitecture.CDNA4: 288.0,
        AMDArchitecture.CDNA3: 192.0,
        AMDArchitecture.CDNA2: 128.0,
        None: 192.0,
    },
    HardwareBackend.TRAINIUM: {
        TrainiumArchitecture.TRN3: 128.0,
        TrainiumArchitecture.TRN2: 64.0,
        None: 64.0,
    },
    HardwareBackend.TPU: {
        TPUVersion.V7: 128.0,
        TPUVersion.V5E: 16.0,
        None: 32.0,
    },
    HardwareBackend.CPU: {
        None: 256.0,
    },
}


def _estimate_gpu_memory_gb(
    backend: HardwareBackend, architecture: Architecture = None
) -> float:
    """Estimate GPU memory in GB for the given hardware."""
    backend_map = _GPU_MEMORY_MAP.get(backend, {})
    if architecture in backend_map:
        return backend_map[architecture]
    return backend_map.get(None, _DEFAULT_GPU_MEMORY_GB)


@dataclass
class ParallelismRecommendation:
    """Recommended parallelism configuration.

    Produced by DistributedConfig.auto() or the advisor CLI.
    """

    tensor_parallel_degree: int
    pipeline_parallel_stages: int
    fsdp_strategy: str
    estimated_memory_per_rank_gb: float
    estimated_communication_volume_gb: float
    mixed_precision: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tensor_parallel_degree": self.tensor_parallel_degree,
            "pipeline_parallel_stages": self.pipeline_parallel_stages,
            "fsdp_strategy": self.fsdp_strategy,
            "estimated_memory_per_rank_gb": round(
                self.estimated_memory_per_rank_gb, 2
            ),
            "estimated_communication_volume_gb": round(
                self.estimated_communication_volume_gb, 2
            ),
            "mixed_precision": self.mixed_precision,
            "notes": self.notes,
        }


@dataclass
class DistributedConfig:
    """Unified configuration for distributed training.

    Combines FSDP2, pipeline, collective, and topology settings.
    Use auto() for automatic configuration from model and hardware.
    """

    fsdp2: FSDP2Config = field(default_factory=FSDP2Config)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    collective: CollectiveConfig = field(default_factory=CollectiveConfig)
    mesh: MeshConfig | None = None

    @classmethod
    def auto(
        cls,
        model_params: int,
        backend: HardwareBackend = HardwareBackend.CPU,
        architecture: Architecture = None,
        world_size: int = 1,
        gpus_per_node: int | None = None,
    ) -> DistributedConfig:
        """Auto-configure distributed training from model and hardware.

        Args:
            model_params: Total model parameters.
            backend: Hardware backend.
            architecture: Specific architecture.
            world_size: Total number of ranks.
            gpus_per_node: GPUs per node (None = all on one node).

        Returns:
            Fully configured DistributedConfig.
        """
        if gpus_per_node is None:
            gpus_per_node = world_size
        num_nodes = max(1, world_size // gpus_per_node)
        multi_node = num_nodes > 1

        # Estimate memory requirements
        gpu_memory_gb = _estimate_gpu_memory_gb(backend, architecture)
        model_memory_gb = (
            model_params * _BYTES_PER_PARAM_MIXED * _MEMORY_OVERHEAD_FACTOR
        ) / (1024**3)

        # Determine parallelism strategy
        recommendation = _recommend_parallelism(
            model_params=model_params,
            model_memory_gb=model_memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            num_nodes=num_nodes,
            backend=backend,
            architecture=architecture,
        )

        # Build FSDP2 config
        fsdp2_manager = FSDP2Manager(
            config=FSDP2Config(
                sharding_strategy=(
                    ShardingStrategy.HYBRID_SHARD
                    if multi_node
                    else ShardingStrategy.FULL_SHARD
                ),
            ),
            backend=backend,
            architecture=architecture,
            multi_node=multi_node,
        )

        # Build pipeline config
        pipeline = PipelineConfig(
            schedule=PipelineScheduleFactory.get_optimal_schedule(
                backend,
                architecture,
                num_stages=recommendation.pipeline_parallel_stages,
            ),
            num_stages=recommendation.pipeline_parallel_stages,
        )

        # Build collective config
        collective = CollectiveConfig().resolve(backend, architecture)

        # Build mesh config
        if multi_node:
            mesh = MeshConfig(
                world_size=world_size,
                local_world_size=gpus_per_node,
                num_nodes=num_nodes,
                interconnect_intra=TopologyDetector.detect_interconnect_intra(backend),
                interconnect_inter=TopologyDetector.detect_interconnect_inter(),
                mesh_shape=(num_nodes, gpus_per_node),
                mesh_dim_names=("inter", "intra"),
            )
        else:
            mesh = MeshConfig(
                world_size=world_size,
                local_world_size=gpus_per_node,
                num_nodes=1,
                interconnect_intra=TopologyDetector.detect_interconnect_intra(backend),
                interconnect_inter=TopologyDetector.detect_interconnect_inter(),
                mesh_shape=(gpus_per_node,),
                mesh_dim_names=("intra",),
            )

        return cls(
            fsdp2=fsdp2_manager.resolved_config,
            pipeline=pipeline,
            collective=collective,
            mesh=mesh,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fsdp2": self.fsdp2.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "collective": self.collective.to_dict(),
            "mesh": self.mesh.to_dict() if self.mesh else None,
        }

    def to_toml(self) -> str:
        """Export as TOML string for reproducible configs."""
        lines = ["# TorchBridge Distributed Training Configuration", ""]

        # FSDP2
        lines.append("[fsdp2]")
        fsdp = self.fsdp2.to_dict()
        for key, value in fsdp.items():
            lines.append(f'{key} = {_toml_value(value)}')
        lines.append("")

        # Pipeline
        lines.append("[pipeline]")
        pipe = self.pipeline.to_dict()
        for key, value in pipe.items():
            lines.append(f'{key} = {_toml_value(value)}')
        lines.append("")

        # Collective
        lines.append("[collective]")
        coll = self.collective.to_dict()
        for key, value in coll.items():
            lines.append(f'{key} = {_toml_value(value)}')
        lines.append("")

        # Mesh
        if self.mesh:
            lines.append("[mesh]")
            mesh = self.mesh.to_dict()
            for key, value in mesh.items():
                lines.append(f'{key} = {_toml_value(value)}')
            lines.append("")

        return "\n".join(lines)

def _toml_value(value: Any) -> str:
    """Format a Python value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        items = ", ".join(_toml_value(v) for v in value)
        return f"[{items}]"
    if value is None:
        return '"auto"'
    return f'"{value}"'


def _recommend_parallelism(
    *,
    model_params: int,
    model_memory_gb: float,
    gpu_memory_gb: float,
    world_size: int,
    gpus_per_node: int,
    num_nodes: int,
    backend: HardwareBackend,
    architecture: Architecture,
) -> ParallelismRecommendation:
    """Compute parallelism recommendation from model and hardware specs."""
    notes: list[str] = []

    # Memory per rank with full FSDP sharding
    fsdp_memory_per_rank = model_memory_gb / world_size
    memory_per_rank = fsdp_memory_per_rank

    # Tensor parallelism: needed when a single layer doesn't fit in GPU memory
    # Heuristic: TP needed if model > 10B params and enough GPUs per node
    tp_degree = 1
    if model_params > 10e9 and gpus_per_node >= 4:
        tp_degree = min(8, gpus_per_node)
        notes.append(
            f"TP={tp_degree} recommended for {model_params/1e9:.0f}B model "
            f"(large layers benefit from tensor parallelism)"
        )
    elif model_params > 3e9 and gpus_per_node >= 2:
        tp_degree = min(4, gpus_per_node)
        notes.append(
            f"TP={tp_degree} for efficient large-model training"
        )

    # Pipeline parallelism: needed when model is very large
    pp_stages = 1
    if model_params > 70e9 and world_size >= 16:
        pp_stages = min(8, max(1, world_size // tp_degree))
        notes.append(
            f"PP={pp_stages} stages for {model_params/1e9:.0f}B model "
            f"across {world_size} ranks"
        )
    elif model_params > 30e9 and world_size >= 8:
        pp_stages = min(4, num_nodes) if num_nodes > 1 else 2
        notes.append(
            f"PP={pp_stages} stages to distribute {model_params/1e9:.0f}B "
            f"model layers across ranks"
        )

    # FSDP strategy
    if num_nodes > 1:
        fsdp_strategy = "hybrid_shard"
        notes.append(
            "Hybrid sharding: full shard within node, replicate across nodes"
        )
    else:
        fsdp_strategy = "full_shard"
        notes.append("Full sharding within single node")

    # Adjusted memory per rank
    effective_shards = world_size // (tp_degree * pp_stages)
    if effective_shards > 0:
        memory_per_rank = model_memory_gb / effective_shards

    # Communication volume estimate (simplified)
    # All-gather + reduce-scatter per FSDP step ≈ 2 × model_params × bytes
    comm_per_step_gb = (
        2 * model_params * _BYTES_PER_PARAM_MIXED / (1024**3)
    )
    # TP adds all-reduce per layer
    if tp_degree > 1:
        comm_per_step_gb *= 1.5
        notes.append("TP all-reduce adds ~50% communication overhead")

    # Mixed precision
    fsdp2_manager = FSDP2Manager(backend=backend, architecture=architecture)
    mp = fsdp2_manager.mixed_precision.value

    # Memory warning
    if memory_per_rank > gpu_memory_gb * 0.9:
        notes.append(
            f"WARNING: estimated {memory_per_rank:.1f} GB/rank exceeds "
            f"90% of {gpu_memory_gb:.0f} GB GPU memory. "
            f"Consider increasing world_size or enabling CPU offload."
        )

    return ParallelismRecommendation(
        tensor_parallel_degree=tp_degree,
        pipeline_parallel_stages=pp_stages,
        fsdp_strategy=fsdp_strategy,
        estimated_memory_per_rank_gb=memory_per_rank,
        estimated_communication_volume_gb=comm_per_step_gb,
        mixed_precision=mp,
        notes=notes,
    )


def recommend_parallelism(
    model_params: int,
    backend: HardwareBackend = HardwareBackend.CPU,
    architecture: Architecture = None,
    world_size: int = 1,
    gpus_per_node: int | None = None,
) -> ParallelismRecommendation:
    """Public API: compute parallelism recommendation.

    Args:
        model_params: Total model parameters.
        backend: Hardware backend.
        architecture: Specific architecture.
        world_size: Total number of ranks.
        gpus_per_node: GPUs per node (None = all on one node).

    Returns:
        ParallelismRecommendation with strategy and estimates.
    """
    if gpus_per_node is None:
        gpus_per_node = world_size
    num_nodes = max(1, world_size // gpus_per_node)

    gpu_memory_gb = _estimate_gpu_memory_gb(backend, architecture)
    model_memory_gb = (
        model_params * _BYTES_PER_PARAM_MIXED * _MEMORY_OVERHEAD_FACTOR
    ) / (1024**3)

    return _recommend_parallelism(
        model_params=model_params,
        model_memory_gb=model_memory_gb,
        gpu_memory_gb=gpu_memory_gb,
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        num_nodes=num_nodes,
        backend=backend,
        architecture=architecture,
    )
