"""
TorchBridge Distributed Training — Configuration Advisor

Generates configurations for PyTorch's distributed APIs (FSDP,
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
from torchbridge.distributed.fsdp import (
    FSDPConfig,
    FSDPManager,
    MixedPrecisionChoice,
    ShardingStrategy,
)
from torchbridge.distributed.hetero import (
    HeterogeneousClusterAdvisor,
    HeterogeneousClusterConfig,
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

# Backward compatibility alias (FSDP2Config was a misnomer; kept for import compat)
FSDP2Config = FSDPConfig

__all__ = [
    # FSDP
    "FSDPConfig",
    "FSDPManager",
    # Backward compat
    "FSDP2Config",
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
    # Heterogeneous cluster
    "HeterogeneousClusterAdvisor",
    "HeterogeneousClusterConfig",
]
