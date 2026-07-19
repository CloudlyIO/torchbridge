# SPDX-License-Identifier: Apache-2.0
"""
Topology-Aware DeviceMesh Auto-Configuration

Detects cluster topology from environment variables (SLURM, Kubernetes,
torch.distributed) and constructs optimal DeviceMesh configurations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

from torchbridge.core.config import HardwareBackend

logger = logging.getLogger(__name__)


class InterconnectType(Enum):
    """Interconnect fabric type between devices or nodes."""

    NVLINK = "nvlink"
    PCIE = "pcie"
    INFINIBAND = "infiniband"
    EFA = "efa"
    ROCE = "roce"
    GCE_NETWORK = "gce_network"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InterconnectSpec:
    """Bandwidth and latency characteristics of an interconnect."""

    interconnect: InterconnectType
    bandwidth_gbps: float
    latency_us: float
    description: str


INTERCONNECT_SPECS: dict[InterconnectType, InterconnectSpec] = {
    InterconnectType.NVLINK: InterconnectSpec(
        interconnect=InterconnectType.NVLINK,
        bandwidth_gbps=900.0,
        latency_us=1.0,
        description="NVIDIA NVLink (intra-node, 900 GB/s on Blackwell)",
    ),
    InterconnectType.PCIE: InterconnectSpec(
        interconnect=InterconnectType.PCIE,
        bandwidth_gbps=64.0,
        latency_us=5.0,
        description="PCIe Gen5 x16 (64 GB/s)",
    ),
    InterconnectType.INFINIBAND: InterconnectSpec(
        interconnect=InterconnectType.INFINIBAND,
        bandwidth_gbps=400.0,
        latency_us=2.0,
        description="InfiniBand NDR (inter-node, 400 Gb/s)",
    ),
    InterconnectType.EFA: InterconnectSpec(
        interconnect=InterconnectType.EFA,
        bandwidth_gbps=100.0,
        latency_us=10.0,
        description="AWS Elastic Fabric Adapter (100 Gbps)",
    ),
    InterconnectType.ROCE: InterconnectSpec(
        interconnect=InterconnectType.ROCE,
        bandwidth_gbps=200.0,
        latency_us=5.0,
        description="RDMA over Converged Ethernet (AMD ROCm)",
    ),
    InterconnectType.GCE_NETWORK: InterconnectSpec(
        interconnect=InterconnectType.GCE_NETWORK,
        bandwidth_gbps=100.0,
        latency_us=15.0,
        description="Google Cloud inter-node network",
    ),
    InterconnectType.UNKNOWN: InterconnectSpec(
        interconnect=InterconnectType.UNKNOWN,
        bandwidth_gbps=10.0,
        latency_us=50.0,
        description="Unknown interconnect (conservative estimates)",
    ),
}


@dataclass
class MeshConfig:
    """DeviceMesh configuration for distributed training.

    Describes the cluster layout: how many nodes, GPUs per node,
    interconnect type, and the resulting mesh shape.
    """

    world_size: int
    local_world_size: int
    num_nodes: int
    interconnect_intra: InterconnectType
    interconnect_inter: InterconnectType
    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]

    def is_multi_node(self) -> bool:
        """Return True if the mesh spans multiple nodes."""
        return self.num_nodes > 1

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "num_nodes": self.num_nodes,
            "interconnect_intra": self.interconnect_intra.value,
            "interconnect_inter": self.interconnect_inter.value,
            "mesh_shape": list(self.mesh_shape),
            "mesh_dim_names": list(self.mesh_dim_names),
        }


class TopologyDetector:
    """Auto-detect cluster topology from environment.

    Reads SLURM, Kubernetes, and torch.distributed environment variables
    to construct a MeshConfig without manual configuration.
    """

    @staticmethod
    def detect_mesh_from_environment() -> MeshConfig:
        """Detect topology from the current environment.

        Priority: SLURM > Kubernetes > torch.distributed env > single-node.

        Returns:
            MeshConfig with detected topology.
        """
        # SLURM
        if os.environ.get("SLURM_JOB_ID"):
            return TopologyDetector._detect_slurm()

        # Kubernetes
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            return TopologyDetector._detect_kubernetes()

        # torch.distributed env vars
        world_size = int(os.environ.get("WORLD_SIZE", "0"))
        if world_size > 0:
            return TopologyDetector._detect_torch_env()

        # Single-node fallback
        return TopologyDetector._detect_single_node()

    @staticmethod
    def detect_interconnect_intra(backend: HardwareBackend) -> InterconnectType:
        """Detect intra-node interconnect for the given backend."""
        if backend == HardwareBackend.CUDA:
            # Check for NVLink via nvidia-smi or assume based on GPU class
            return InterconnectType.NVLINK
        if backend == HardwareBackend.AMD:
            return InterconnectType.PCIE
        if backend == HardwareBackend.TPU:
            # TPU pods have custom interconnect, but from torch perspective
            return InterconnectType.PCIE
        return InterconnectType.UNKNOWN

    @staticmethod
    def detect_interconnect_inter() -> InterconnectType:
        """Detect inter-node interconnect from environment."""
        # AWS EFA
        if (
            os.environ.get("FI_EFA_USE_DEVICE_RDMA")
            or os.environ.get("FI_PROVIDER") == "efa"
        ):
            return InterconnectType.EFA

        # InfiniBand (common on HPC, NVIDIA DGX)
        if os.path.exists("/sys/class/infiniband"):
            return InterconnectType.INFINIBAND

        # GCP
        if os.environ.get("GCE_METADATA_HOST") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        ):
            return InterconnectType.GCE_NETWORK

        # AMD ROCe
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION") or os.environ.get("ROCM_PATH"):
            return InterconnectType.ROCE

        return InterconnectType.UNKNOWN

    @staticmethod
    def _detect_slurm() -> MeshConfig:
        """Detect topology from SLURM environment."""
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "0"))

        if gpus_per_node == 0:
            # Try tasks per node
            gpus_per_node = max(1, world_size // num_nodes)

        local_world_size = gpus_per_node
        interconnect_inter = TopologyDetector.detect_interconnect_inter()

        mesh_shape: tuple[int, ...]
        mesh_dim_names: tuple[str, ...]
        if num_nodes > 1:
            mesh_shape = (num_nodes, local_world_size)
            mesh_dim_names = ("inter", "intra")
        else:
            mesh_shape = (local_world_size,)
            mesh_dim_names = ("intra",)

        return MeshConfig(
            world_size=world_size,
            local_world_size=local_world_size,
            num_nodes=num_nodes,
            interconnect_intra=InterconnectType.NVLINK,
            interconnect_inter=interconnect_inter,
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

    @staticmethod
    def _detect_kubernetes() -> MeshConfig:
        """Detect topology from Kubernetes environment."""
        return TopologyDetector._detect_from_world_env()

    @staticmethod
    def _detect_torch_env() -> MeshConfig:
        """Detect topology from torch.distributed env vars."""
        return TopologyDetector._detect_from_world_env()

    @staticmethod
    def _detect_from_world_env() -> MeshConfig:
        """Build MeshConfig from WORLD_SIZE/LOCAL_WORLD_SIZE env vars."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        num_nodes = (
            max(1, world_size // local_world_size) if local_world_size > 0 else 1
        )
        interconnect_inter = TopologyDetector.detect_interconnect_inter()

        mesh_shape: tuple[int, ...]
        mesh_dim_names: tuple[str, ...]
        if num_nodes > 1:
            mesh_shape = (num_nodes, local_world_size)
            mesh_dim_names = ("inter", "intra")
        else:
            mesh_shape = (local_world_size,)
            mesh_dim_names = ("intra",)

        return MeshConfig(
            world_size=world_size,
            local_world_size=local_world_size,
            num_nodes=num_nodes,
            interconnect_intra=InterconnectType.NVLINK,
            interconnect_inter=interconnect_inter,
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

    @staticmethod
    def _detect_single_node() -> MeshConfig:
        """Fallback: single node with local GPU count."""
        import torch

        if torch.cuda.is_available():
            local_world_size = torch.cuda.device_count()
        else:
            local_world_size = 1

        return MeshConfig(
            world_size=local_world_size,
            local_world_size=local_world_size,
            num_nodes=1,
            interconnect_intra=InterconnectType.PCIE,
            interconnect_inter=InterconnectType.UNKNOWN,
            mesh_shape=(local_world_size,),
            mesh_dim_names=("intra",),
        )
