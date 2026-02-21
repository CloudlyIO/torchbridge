"""
TorchBridge Distributed Training — Configuration Advisor

Generates configurations for PyTorch's distributed APIs (FSDP2,
pipeline parallelism, collective backends) based on detected hardware.
Does not implement distributed training itself — it produces config
objects that users pass to PyTorch's native distributed primitives.
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
