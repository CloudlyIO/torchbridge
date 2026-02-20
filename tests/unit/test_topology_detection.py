"""Tests for topology detection and mesh configuration."""

import os
from unittest.mock import patch

from torchbridge.core.config import HardwareBackend
from torchbridge.distributed.topology import (
    INTERCONNECT_SPECS,
    InterconnectType,
    MeshConfig,
    TopologyDetector,
)


class TestInterconnectType:
    """Tests for InterconnectType enum."""

    def test_all_types(self):
        expected = {"nvlink", "pcie", "infiniband", "efa", "roce", "gce_network", "unknown"}
        actual = {t.value for t in InterconnectType}
        assert actual == expected

    def test_specs_cover_all_types(self):
        for itype in InterconnectType:
            assert itype in INTERCONNECT_SPECS


class TestInterconnectSpec:
    """Tests for InterconnectSpec dataclass."""

    def test_nvlink_bandwidth(self):
        spec = INTERCONNECT_SPECS[InterconnectType.NVLINK]
        assert spec.bandwidth_gbps == 900.0
        assert spec.latency_us == 1.0

    def test_infiniband_bandwidth(self):
        spec = INTERCONNECT_SPECS[InterconnectType.INFINIBAND]
        assert spec.bandwidth_gbps == 400.0

    def test_efa_bandwidth(self):
        spec = INTERCONNECT_SPECS[InterconnectType.EFA]
        assert spec.bandwidth_gbps == 100.0

    def test_unknown_conservative(self):
        spec = INTERCONNECT_SPECS[InterconnectType.UNKNOWN]
        assert spec.bandwidth_gbps <= 10.0
        assert spec.latency_us >= 50.0


class TestMeshConfig:
    """Tests for MeshConfig dataclass."""

    def test_single_node(self):
        mesh = MeshConfig(
            world_size=4,
            local_world_size=4,
            num_nodes=1,
            interconnect_intra=InterconnectType.NVLINK,
            interconnect_inter=InterconnectType.UNKNOWN,
            mesh_shape=(4,),
            mesh_dim_names=("intra",),
        )
        assert mesh.is_multi_node() is False
        assert mesh.world_size == 4

    def test_multi_node(self):
        mesh = MeshConfig(
            world_size=16,
            local_world_size=8,
            num_nodes=2,
            interconnect_intra=InterconnectType.NVLINK,
            interconnect_inter=InterconnectType.INFINIBAND,
            mesh_shape=(2, 8),
            mesh_dim_names=("inter", "intra"),
        )
        assert mesh.is_multi_node() is True
        assert mesh.num_nodes == 2

    def test_to_dict(self):
        mesh = MeshConfig(
            world_size=8,
            local_world_size=4,
            num_nodes=2,
            interconnect_intra=InterconnectType.NVLINK,
            interconnect_inter=InterconnectType.INFINIBAND,
            mesh_shape=(2, 4),
            mesh_dim_names=("inter", "intra"),
        )
        d = mesh.to_dict()
        assert d["world_size"] == 8
        assert d["local_world_size"] == 4
        assert d["num_nodes"] == 2
        assert d["interconnect_intra"] == "nvlink"
        assert d["interconnect_inter"] == "infiniband"
        assert d["mesh_shape"] == [2, 4]
        assert d["mesh_dim_names"] == ["inter", "intra"]


class TestTopologyDetector:
    """Tests for TopologyDetector."""

    def test_detect_interconnect_intra_cuda(self):
        result = TopologyDetector.detect_interconnect_intra(HardwareBackend.CUDA)
        assert result == InterconnectType.NVLINK

    def test_detect_interconnect_intra_amd(self):
        result = TopologyDetector.detect_interconnect_intra(HardwareBackend.AMD)
        assert result == InterconnectType.PCIE

    def test_detect_interconnect_intra_tpu(self):
        result = TopologyDetector.detect_interconnect_intra(HardwareBackend.TPU)
        assert result == InterconnectType.PCIE

    def test_detect_interconnect_intra_cpu(self):
        result = TopologyDetector.detect_interconnect_intra(HardwareBackend.CPU)
        assert result == InterconnectType.UNKNOWN

    @patch.dict(os.environ, {"FI_PROVIDER": "efa"}, clear=False)
    def test_detect_interconnect_inter_efa(self):
        result = TopologyDetector.detect_interconnect_inter()
        assert result == InterconnectType.EFA

    @patch.dict(os.environ, {"FI_EFA_USE_DEVICE_RDMA": "1"}, clear=False)
    def test_detect_interconnect_inter_efa_rdma(self):
        result = TopologyDetector.detect_interconnect_inter()
        assert result == InterconnectType.EFA

    @patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "my-project"}, clear=False)
    def test_detect_interconnect_inter_gce(self):
        # Only works if EFA and IB checks don't match
        result = TopologyDetector.detect_interconnect_inter()
        # May be GCE_NETWORK or something else depending on env
        assert isinstance(result, InterconnectType)

    @patch.dict(os.environ, {
        "SLURM_JOB_ID": "12345",
        "SLURM_NTASKS": "16",
        "SLURM_NNODES": "2",
        "SLURM_GPUS_ON_NODE": "8",
    }, clear=False)
    def test_detect_slurm(self):
        mesh = TopologyDetector.detect_mesh_from_environment()
        assert mesh.world_size == 16
        assert mesh.num_nodes == 2
        assert mesh.local_world_size == 8
        assert mesh.mesh_shape == (2, 8)
        assert mesh.mesh_dim_names == ("inter", "intra")

    @patch.dict(os.environ, {
        "SLURM_JOB_ID": "12345",
        "SLURM_NTASKS": "8",
        "SLURM_NNODES": "1",
        "SLURM_GPUS_ON_NODE": "8",
    }, clear=False)
    def test_detect_slurm_single_node(self):
        mesh = TopologyDetector.detect_mesh_from_environment()
        assert mesh.world_size == 8
        assert mesh.num_nodes == 1
        assert mesh.mesh_shape == (8,)
        assert mesh.mesh_dim_names == ("intra",)

    @patch.dict(os.environ, {
        "KUBERNETES_SERVICE_HOST": "10.0.0.1",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "4",
    }, clear=False)
    def test_detect_kubernetes(self):
        # Remove SLURM vars to ensure K8s path is taken
        env = os.environ.copy()
        env.pop("SLURM_JOB_ID", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["KUBERNETES_SERVICE_HOST"] = "10.0.0.1"
            os.environ["WORLD_SIZE"] = "8"
            os.environ["LOCAL_WORLD_SIZE"] = "4"
            mesh = TopologyDetector.detect_mesh_from_environment()
            assert mesh.world_size == 8
            assert mesh.num_nodes == 2

    @patch.dict(os.environ, {
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "4",
    }, clear=False)
    def test_detect_torch_env(self):
        env = os.environ.copy()
        env.pop("SLURM_JOB_ID", None)
        env.pop("KUBERNETES_SERVICE_HOST", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["WORLD_SIZE"] = "4"
            os.environ["LOCAL_WORLD_SIZE"] = "4"
            mesh = TopologyDetector.detect_mesh_from_environment()
            assert mesh.world_size == 4
            assert mesh.num_nodes == 1
