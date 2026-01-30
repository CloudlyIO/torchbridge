"""
Job Management and Specifications

Core job management components for distributed training orchestration:
- Job state tracking and lifecycle management
- Resource requirement specifications
- Training job configurations and status monitoring
- Resource types and failure classifications
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class JobState(Enum):
    """Training job states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCALING = "scaling"


class ResourceType(Enum):
    """Resource types for scheduling"""
    GPU_MEMORY = "gpu_memory"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE_IOPS = "storage_iops"


class FailureType(Enum):
    """Types of failures that can occur"""
    NODE_FAILURE = "node_failure"
    GPU_FAILURE = "gpu_failure"
    NETWORK_FAILURE = "network_failure"
    SOFTWARE_FAILURE = "software_failure"
    OOM_FAILURE = "oom_failure"
    TIMEOUT_FAILURE = "timeout_failure"


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: int
    memory_gb: int
    storage_gb: int = 100
    network_bandwidth_gbps: int = 10
    required_gpu_types: list[str] | None = None
    node_selector: dict[str, str] | None = None


@dataclass
class TrainingJobSpec:
    """Training job specification"""
    job_id: str
    name: str
    image: str
    command: list[str]
    resources: ResourceRequirement

    # Training-specific parameters
    world_size: int
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int | None = None

    # Job management
    priority: int = 0
    max_runtime_hours: int = 24
    restart_policy: str = "OnFailure"
    checkpoint_interval_minutes: int = 30

    # Environment
    env_vars: dict[str, str] = field(default_factory=dict)
    volumes: list[dict[str, Any]] = field(default_factory=list)

    # Fault tolerance
    max_retries: int = 3
    enable_auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int | None = None


@dataclass
class JobStatus:
    """Current status of a training job"""
    job_id: str
    state: JobState
    start_time: datetime | None = None
    end_time: datetime | None = None
    allocated_nodes: list[str] = field(default_factory=list)
    allocated_gpus: list[int] = field(default_factory=list)
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float | None = None
    throughput_samples_per_sec: float | None = None
    error_message: str | None = None
    restart_count: int = 0
    last_checkpoint: str | None = None


@dataclass
class ClusterNode:
    """Cluster node information"""
    node_id: str
    hostname: str
    total_gpus: int
    available_gpus: int
    total_memory_gb: int
    available_memory_gb: int
    total_cpu_cores: int
    available_cpu_cores: int
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[dict[str, Any]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
