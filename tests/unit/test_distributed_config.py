"""Tests for unified distributed config and parallelism recommendation."""

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TrainiumArchitecture,
)
from torchbridge.distributed.collective_backend import CollectiveConfig
from torchbridge.distributed.config import (
    DistributedConfig,
    ParallelismRecommendation,
    recommend_parallelism,
)
from torchbridge.distributed.fsdp import (
    FSDPConfig,
    MixedPrecisionChoice,
    ShardingStrategy,
)
from torchbridge.distributed.pipeline_schedules import PipelineConfig


class TestParallelismRecommendation:
    """Tests for ParallelismRecommendation dataclass."""

    def test_to_dict(self):
        rec = ParallelismRecommendation(
            tensor_parallel_degree=4,
            pipeline_parallel_stages=2,
            fsdp_strategy="hybrid_shard",
            estimated_memory_per_rank_gb=12.5,
            estimated_communication_volume_gb=5.3,
            mixed_precision="bf16",
            notes=["TP=4 recommended"],
        )
        d = rec.to_dict()
        assert d["tensor_parallel_degree"] == 4
        assert d["pipeline_parallel_stages"] == 2
        assert d["fsdp_strategy"] == "hybrid_shard"
        assert d["estimated_memory_per_rank_gb"] == 12.5
        assert d["mixed_precision"] == "bf16"
        assert len(d["notes"]) == 1

    def test_default_notes(self):
        rec = ParallelismRecommendation(
            tensor_parallel_degree=1,
            pipeline_parallel_stages=1,
            fsdp_strategy="full_shard",
            estimated_memory_per_rank_gb=1.0,
            estimated_communication_volume_gb=0.5,
            mixed_precision="fp32",
        )
        assert rec.notes == []


class TestRecommendParallelism:
    """Tests for recommend_parallelism function."""

    def test_small_model_single_gpu(self):
        rec = recommend_parallelism(
            model_params=int(1e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=1,
        )
        assert rec.tensor_parallel_degree == 1
        assert rec.pipeline_parallel_stages == 1
        assert rec.fsdp_strategy == "full_shard"

    def test_large_model_multi_gpu(self):
        rec = recommend_parallelism(
            model_params=int(70e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=8,
        )
        assert rec.tensor_parallel_degree > 1
        assert rec.mixed_precision == "bf16"

    def test_very_large_model_multi_node(self):
        rec = recommend_parallelism(
            model_params=int(70e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=16,
            gpus_per_node=8,
        )
        assert rec.fsdp_strategy == "hybrid_shard"
        assert rec.tensor_parallel_degree > 1
        assert len(rec.notes) > 0

    def test_cpu_defaults_fp32(self):
        rec = recommend_parallelism(
            model_params=int(1e9),
            backend=HardwareBackend.CPU,
            world_size=1,
        )
        assert rec.mixed_precision == "fp32"

    def test_amd_backend(self):
        rec = recommend_parallelism(
            model_params=int(7e9),
            backend=HardwareBackend.AMD,
            architecture=AMDArchitecture.CDNA3,
            world_size=4,
        )
        assert rec.mixed_precision == "bf16"

    def test_memory_warning_for_oversubscription(self):
        # Very large model, small world size → should warn
        rec = recommend_parallelism(
            model_params=int(175e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
            world_size=1,
        )
        warning_notes = [n for n in rec.notes if "WARNING" in n]
        assert len(warning_notes) > 0

    def test_communication_volume_positive(self):
        rec = recommend_parallelism(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            world_size=8,
        )
        assert rec.estimated_communication_volume_gb > 0


class TestDistributedConfig:
    """Tests for DistributedConfig dataclass."""

    def test_defaults(self):
        config = DistributedConfig()
        assert isinstance(config.fsdp, FSDPConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.collective, CollectiveConfig)
        assert config.mesh is None

    def test_to_dict(self):
        config = DistributedConfig()
        d = config.to_dict()
        assert "fsdp" in d
        assert "pipeline" in d
        assert "collective" in d
        assert d["mesh"] is None

    def test_auto_single_gpu(self):
        config = DistributedConfig.auto(
            model_params=int(1e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=1,
        )
        assert config.mesh is not None
        assert config.mesh.world_size == 1
        assert config.fsdp.sharding_strategy == ShardingStrategy.FULL_SHARD

    def test_auto_multi_gpu(self):
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=8,
        )
        assert config.mesh is not None
        assert config.mesh.world_size == 8

    def test_auto_multi_node(self):
        config = DistributedConfig.auto(
            model_params=int(70e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=16,
            gpus_per_node=8,
        )
        assert config.mesh.is_multi_node() is True
        assert config.fsdp.sharding_strategy == ShardingStrategy.HYBRID_SHARD

    def test_to_toml(self):
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=8,
        )
        toml = config.to_toml()
        assert "[fsdp]" in toml
        assert "[pipeline]" in toml
        assert "[collective]" in toml
        assert "[mesh]" in toml
        assert "sharding_strategy" in toml

    def test_to_toml_no_mesh(self):
        config = DistributedConfig()
        toml = config.to_toml()
        assert "[fsdp]" in toml
        assert "[mesh]" not in toml

    def test_auto_cpu(self):
        config = DistributedConfig.auto(
            model_params=int(1e9),
            backend=HardwareBackend.CPU,
            world_size=1,
        )
        assert config.fsdp.mixed_precision == MixedPrecisionChoice.FP32

    def test_auto_amd(self):
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.AMD,
            architecture=AMDArchitecture.CDNA3,
            world_size=4,
        )
        assert config.fsdp.mixed_precision == MixedPrecisionChoice.BF16

    def test_auto_trainium(self):
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.TRAINIUM,
            architecture=TrainiumArchitecture.TRN2,
            world_size=8,
        )
        assert config.fsdp.mixed_precision == MixedPrecisionChoice.BF16
