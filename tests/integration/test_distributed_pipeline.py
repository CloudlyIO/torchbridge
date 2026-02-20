"""Integration tests for distributed config → recommendation pipeline."""


import pytest

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.distributed import (
    CollectiveBackendType,
    DistributedConfig,
    MixedPrecisionChoice,
    ShardingStrategy,
)
from torchbridge.distributed.config import recommend_parallelism


class TestEndToEndConfigGeneration:
    """Test the full config → recommendation → export pipeline."""

    def test_hopper_8gpu_7b(self):
        """Typical single-node H100 training of a 7B model."""
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=8,
        )
        assert config.fsdp2.mixed_precision == MixedPrecisionChoice.BF16
        assert config.fsdp2.float8_all_gather is True
        assert config.fsdp2.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert config.mesh.world_size == 8
        assert config.mesh.is_multi_node() is False

    def test_hopper_16gpu_70b_multi_node(self):
        """Multi-node H100 training of a 70B model."""
        config = DistributedConfig.auto(
            model_params=int(70e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=16,
            gpus_per_node=8,
        )
        assert config.fsdp2.sharding_strategy == ShardingStrategy.HYBRID_SHARD
        assert config.mesh.is_multi_node() is True
        assert config.mesh.num_nodes == 2

    def test_amd_mi300x_4gpu(self):
        """AMD MI300X single-node training."""
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.AMD,
            architecture=AMDArchitecture.CDNA3,
            world_size=4,
        )
        assert config.fsdp2.mixed_precision == MixedPrecisionChoice.BF16
        assert config.fsdp2.float8_all_gather is False
        assert config.collective.backend == CollectiveBackendType.RCCL

    def test_cpu_fallback(self):
        """CPU fallback — simplest config."""
        config = DistributedConfig.auto(
            model_params=int(1e9),
            backend=HardwareBackend.CPU,
            world_size=1,
        )
        assert config.fsdp2.mixed_precision == MixedPrecisionChoice.FP32
        assert config.fsdp2.float8_all_gather is False
        assert config.collective.backend == CollectiveBackendType.GLOO

    def test_trainium_8gpu(self):
        """Trainium training setup."""
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.TRAINIUM,
            architecture=TrainiumArchitecture.TRN2,
            world_size=8,
        )
        assert config.fsdp2.mixed_precision == MixedPrecisionChoice.BF16
        assert config.collective.backend == CollectiveBackendType.NEURON_CC

    def test_tpu_v5e(self):
        """TPU v5e training setup."""
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.TPU,
            architecture=TPUVersion.V5E,
            world_size=8,
        )
        assert config.fsdp2.mixed_precision == MixedPrecisionChoice.BF16
        assert config.collective.backend == CollectiveBackendType.XLA_COLLECTIVES


class TestTomlRoundTrip:
    """Test TOML export produces valid output."""

    def test_toml_has_all_sections(self):
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            world_size=8,
        )
        toml = config.to_toml()
        assert "[fsdp2]" in toml
        assert "[pipeline]" in toml
        assert "[collective]" in toml
        assert "[mesh]" in toml

    def test_toml_values_parseable(self):
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            world_size=4,
        )
        toml = config.to_toml()
        # Check key values are present
        assert "sharding_strategy" in toml
        assert "schedule" in toml
        assert "world_size" in toml


class TestRecommendationConsistency:
    """Verify recommendations are consistent with configs."""

    @pytest.mark.parametrize("backend,arch,world_size", [
        (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER, 8),
        (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE, 4),
        (HardwareBackend.AMD, AMDArchitecture.CDNA3, 4),
        (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2, 8),
        (HardwareBackend.TPU, TPUVersion.V5E, 8),
        (HardwareBackend.CPU, None, 1),
    ])
    def test_recommendation_matches_config(self, backend, arch, world_size):
        """Recommendation and auto-config should agree on strategy."""
        rec = recommend_parallelism(
            model_params=int(7e9),
            backend=backend,
            architecture=arch,
            world_size=world_size,
        )
        config = DistributedConfig.auto(
            model_params=int(7e9),
            backend=backend,
            architecture=arch,
            world_size=world_size,
        )
        # Both should agree on FSDP strategy
        assert config.fsdp2.sharding_strategy.value == rec.fsdp_strategy
        # Both should agree on mixed precision
        assert config.fsdp2.mixed_precision.value == rec.mixed_precision


class TestPackageImports:
    """Verify all public API is accessible from the distributed package."""

    def test_all_exports(self):
        from torchbridge.distributed import __all__
        expected_names = [
            "FSDP2Config", "FSDP2Manager", "MixedPrecisionChoice", "ShardingStrategy",
            "InterconnectType", "MeshConfig", "TopologyDetector",
            "PipelineConfig", "PipelineScheduleFactory", "PipelineScheduleType",
            "CollectiveBackendMatrix", "CollectiveBackendType", "CollectiveConfig",
            "DistributedConfig", "ParallelismRecommendation",
        ]
        for name in expected_names:
            assert name in __all__, f"{name} not in __all__"

    def test_imports_work(self):
        """All __all__ names should be importable."""
        import torchbridge.distributed as dist
        for name in dist.__all__:
            assert hasattr(dist, name), f"Cannot access distributed.{name}"
