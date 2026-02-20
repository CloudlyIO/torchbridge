"""
TorchBridge Distributed Training

Backend-aware FSDP2, topology detection, pipeline scheduling,
communication backend selection, and parallelism advisor.
"""

from torchbridge.distributed.collective_backend import (
    CollectiveBackendMatrix,
    CollectiveBackendType,
    CollectiveConfig,
)
from torchbridge.distributed.config import (
    DistributedConfig,
    ParallelismRecommendation,
)
from torchbridge.distributed.fsdp2 import (
    FSDP2Config,
    FSDP2Manager,
    MixedPrecisionChoice,
    ShardingStrategy,
)
from torchbridge.distributed.pipeline_schedules import (
    PipelineConfig,
    PipelineScheduleFactory,
    PipelineScheduleType,
)
from torchbridge.distributed.topology import (
    InterconnectType,
    MeshConfig,
    TopologyDetector,
)

__all__ = [
    # FSDP2
    "FSDP2Config",
    "FSDP2Manager",
    "MixedPrecisionChoice",
    "ShardingStrategy",
    # Topology
    "InterconnectType",
    "MeshConfig",
    "TopologyDetector",
    # Pipeline
    "PipelineConfig",
    "PipelineScheduleFactory",
    "PipelineScheduleType",
    # Collective
    "CollectiveBackendMatrix",
    "CollectiveBackendType",
    "CollectiveConfig",
    # Config
    "DistributedConfig",
    "ParallelismRecommendation",
]
