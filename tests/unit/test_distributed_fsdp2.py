"""Tests for FSDP configuration and manager."""


from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.distributed.fsdp2 import (
    FSDPConfig,
    FSDPManager,
    MixedPrecisionChoice,
    ShardingStrategy,
)


class TestFSDPConfig:
    """Tests for FSDPConfig dataclass."""

    def test_defaults(self):
        config = FSDPConfig()
        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert config.cpu_offload is False
        assert config.mixed_precision is None
        assert config.backward_prefetch is True
        assert config.forward_prefetch is False
        assert config.float8_all_gather is False
        assert config.reshard_after_forward is True
        assert config.limit_all_gathers is True

    def test_to_dict(self):
        config = FSDPConfig(mixed_precision=MixedPrecisionChoice.BF16)
        d = config.to_dict()
        assert d["sharding_strategy"] == "full_shard"
        assert d["mixed_precision"] == "bf16"
        assert d["cpu_offload"] is False

    def test_to_dict_auto_mixed_precision(self):
        config = FSDPConfig()
        d = config.to_dict()
        assert d["mixed_precision"] == "auto"

    def test_custom_config(self):
        config = FSDPConfig(
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            cpu_offload=True,
            mixed_precision=MixedPrecisionChoice.FP8,
            float8_all_gather=True,
        )
        assert config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
        assert config.cpu_offload is True
        assert config.mixed_precision == MixedPrecisionChoice.FP8
        assert config.float8_all_gather is True


class TestShardingStrategy:
    """Tests for ShardingStrategy enum."""

    def test_values(self):
        assert ShardingStrategy.FULL_SHARD.value == "full_shard"
        assert ShardingStrategy.SHARD_GRAD_OP.value == "shard_grad_op"
        assert ShardingStrategy.HYBRID_SHARD.value == "hybrid_shard"
        assert ShardingStrategy.NO_SHARD.value == "no_shard"

    def test_all_strategies(self):
        assert len(ShardingStrategy) == 4


class TestMixedPrecisionChoice:
    """Tests for MixedPrecisionChoice enum."""

    def test_values(self):
        assert MixedPrecisionChoice.FP32.value == "fp32"
        assert MixedPrecisionChoice.FP16.value == "fp16"
        assert MixedPrecisionChoice.BF16.value == "bf16"
        assert MixedPrecisionChoice.FP8.value == "fp8"


class TestFSDPManager:
    """Tests for FSDPManager."""

    def test_default_cpu_backend(self):
        manager = FSDPManager()
        assert manager.mixed_precision == MixedPrecisionChoice.FP32
        assert manager.sharding_strategy == ShardingStrategy.FULL_SHARD

    def test_cuda_ampere_auto_precision(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.BF16

    def test_cuda_hopper_auto_precision(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.BF16

    def test_cuda_blackwell_dc_fp8(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.BLACKWELL_DC,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.FP8

    def test_cuda_turing_fp16(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.TURING,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.FP16

    def test_amd_cdna3_bf16(self):
        manager = FSDPManager(
            backend=HardwareBackend.AMD,
            architecture=AMDArchitecture.CDNA3,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.BF16

    def test_trainium_bf16(self):
        manager = FSDPManager(
            backend=HardwareBackend.TRAINIUM,
            architecture=TrainiumArchitecture.TRN2,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.BF16

    def test_tpu_bf16(self):
        manager = FSDPManager(
            backend=HardwareBackend.TPU,
            architecture=TPUVersion.V7,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.BF16

    def test_multi_node_hybrid_shard(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            multi_node=True,
        )
        assert manager.sharding_strategy == ShardingStrategy.HYBRID_SHARD

    def test_single_node_full_shard(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            multi_node=False,
        )
        assert manager.sharding_strategy == ShardingStrategy.FULL_SHARD

    def test_explicit_strategy_not_overridden_single_node(self):
        config = FSDPConfig(sharding_strategy=ShardingStrategy.NO_SHARD)
        manager = FSDPManager(config=config, backend=HardwareBackend.CUDA)
        assert manager.sharding_strategy == ShardingStrategy.NO_SHARD

    def test_float8_all_gather_hopper(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        resolved = manager.resolved_config
        assert resolved.float8_all_gather is True

    def test_float8_all_gather_blackwell(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.BLACKWELL_DC,
        )
        assert manager.resolved_config.float8_all_gather is True

    def test_no_float8_all_gather_ampere(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        assert manager.resolved_config.float8_all_gather is False

    def test_no_float8_all_gather_amd(self):
        manager = FSDPManager(
            backend=HardwareBackend.AMD,
            architecture=AMDArchitecture.CDNA3,
        )
        assert manager.resolved_config.float8_all_gather is False

    def test_get_info(self):
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        info = manager.get_info()
        assert info["backend"] == "cuda"
        assert info["architecture"] == "hopper"
        assert info["multi_node"] is False
        assert "resolved_config" in info
        assert isinstance(info["resolved_config"], dict)
        assert info["float8_all_gather_supported"] is True

    def test_get_info_no_architecture(self):
        manager = FSDPManager(backend=HardwareBackend.CPU)
        info = manager.get_info()
        assert info["architecture"] is None

    def test_resolved_config_is_fsdp2config(self):
        manager = FSDPManager(backend=HardwareBackend.CUDA)
        assert isinstance(manager.resolved_config, FSDPConfig)

    def test_explicit_mixed_precision_preserved(self):
        config = FSDPConfig(mixed_precision=MixedPrecisionChoice.FP16)
        manager = FSDPManager(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        assert manager.mixed_precision == MixedPrecisionChoice.FP16

    def test_cuda_no_architecture_defaults_bf16(self):
        manager = FSDPManager(backend=HardwareBackend.CUDA)
        assert manager.mixed_precision == MixedPrecisionChoice.BF16

    def test_amd_no_architecture_defaults_bf16(self):
        manager = FSDPManager(backend=HardwareBackend.AMD)
        assert manager.mixed_precision == MixedPrecisionChoice.BF16
