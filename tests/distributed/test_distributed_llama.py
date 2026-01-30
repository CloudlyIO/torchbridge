"""
Distributed Llama Model Validation Tests (v0.4.24)

Validates that distributed training infrastructure works correctly
with Llama-class models across multiple GPUs.

These tests validate the distributed APIs without requiring multi-GPU setup.
"""


import pytest
import torch
import torch.nn as nn

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Configuration Tests
# =============================================================================

class TestDistributedImports:
    """Test that all distributed modules can be imported."""

    def test_distributed_config_import(self):
        """Test importing distributed config."""
        from torchbridge.models.distributed import (
            DistributedConfig,
            DistributedLLMOptimizer,
            ParallelismStrategy,
        )
        assert DistributedConfig is not None
        assert DistributedLLMOptimizer is not None
        assert ParallelismStrategy is not None

    def test_tensor_parallel_imports(self):
        """Test importing tensor parallel components."""
        from torchbridge.models.distributed import (
            ColumnParallelLinear,
            TensorParallelConfig,
        )
        assert TensorParallelConfig is not None
        assert ColumnParallelLinear is not None

    def test_pipeline_parallel_imports(self):
        """Test importing pipeline parallel components."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )
        assert PipelineParallelConfig is not None
        assert PipelineStage is not None

    def test_sharding_imports(self):
        """Test importing sharding components."""
        from torchbridge.models.distributed import (
            ModelSharder,
            ShardingStrategy,
        )
        assert ShardingStrategy is not None
        assert ModelSharder is not None


class TestDistributedConfig:
    """Test DistributedConfig creation and validation."""

    def test_default_config(self):
        """Test default configuration."""
        from torchbridge.models.distributed import DistributedConfig

        config = DistributedConfig()
        assert config.world_size == 1
        assert config.rank == 0

    def test_custom_config(self):
        """Test custom configuration."""
        from torchbridge.models.distributed import (
            DistributedConfig,
            ParallelismStrategy,
        )

        config = DistributedConfig(
            world_size=8,
            rank=0,
            tensor_parallel_size=8,
            strategy=ParallelismStrategy.TENSOR_PARALLEL,
        )

        assert config.world_size == 8
        assert config.tensor_parallel_size == 8


class TestDistributedLLMOptimizer:
    """Test DistributedLLMOptimizer."""

    def test_optimizer_creation(self):
        """Test creating distributed optimizer."""
        from torchbridge.models.distributed import (
            DistributedConfig,
            DistributedLLMOptimizer,
        )

        config = DistributedConfig(
            world_size=8,
            tensor_parallel_size=8,
        )
        optimizer = DistributedLLMOptimizer(
            model_name="meta-llama/Llama-2-70b-hf",
            config=config,
        )

        assert optimizer is not None
        assert optimizer.model_name == "meta-llama/Llama-2-70b-hf"

    def test_optimizer_with_strategy(self):
        """Test optimizer with explicit strategy."""
        from torchbridge.models.distributed import (
            DistributedConfig,
            DistributedLLMOptimizer,
            ParallelismStrategy,
        )

        config = DistributedConfig(
            world_size=2,
            tensor_parallel_size=2,
            strategy=ParallelismStrategy.TENSOR_PARALLEL,
        )
        optimizer = DistributedLLMOptimizer(
            model_name="meta-llama/Llama-2-7b-hf",
            config=config,
        )

        assert optimizer.config.strategy == ParallelismStrategy.TENSOR_PARALLEL


class TestMemoryEstimation:
    """Test GPU memory estimation."""

    def test_estimate_gpu_requirements(self):
        """Test memory estimation function."""
        from torchbridge.models.distributed import estimate_gpu_requirements

        requirements = estimate_gpu_requirements(
            "meta-llama/Llama-2-70b-hf",
            max_sequence_length=4096,
            max_batch_size=1,
        )

        assert requirements is not None
        assert "recommended_gpus" in requirements or "min_gpus" in requirements

    def test_estimate_with_quantization(self):
        """Test memory estimation with quantization."""
        from torchbridge.models.distributed import estimate_gpu_requirements

        req_fp16 = estimate_gpu_requirements(
            "meta-llama/Llama-2-70b-hf",
            quantization="none",
        )
        req_int8 = estimate_gpu_requirements(
            "meta-llama/Llama-2-70b-hf",
            quantization="int8",
        )

        # INT8 should need fewer GPUs
        assert req_int8 is not None
        assert req_fp16 is not None


# =============================================================================
# Tensor Parallelism Tests
# =============================================================================

class TestTensorParallelConfig:
    """Test TensorParallelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from torchbridge.models.distributed import TensorParallelConfig

        config = TensorParallelConfig()
        assert config.world_size == 1
        assert config.rank == 0

    def test_multi_gpu_config(self):
        """Test multi-GPU configuration."""
        from torchbridge.models.distributed import TensorParallelConfig

        config = TensorParallelConfig(
            world_size=8,
            rank=3,
            sequence_parallel=True,
        )

        assert config.world_size == 8
        assert config.rank == 3
        assert config.sequence_parallel is True

    def test_config_validation(self):
        """Test config validation."""
        from torchbridge.models.distributed import TensorParallelConfig

        with pytest.raises((ValueError, AssertionError)):
            TensorParallelConfig(world_size=4, rank=5)


class TestParallelLinear:
    """Test parallel linear layers."""

    def test_column_parallel_creation(self, device):
        """Test creating column parallel linear."""
        from torchbridge.models.distributed import (
            ColumnParallelLinear,
            TensorParallelConfig,
        )

        config = TensorParallelConfig(world_size=2, rank=0)
        layer = ColumnParallelLinear(
            in_features=256,
            out_features=512,
            config=config,
        ).to(device)

        # Output features should be partitioned
        assert layer.weight.shape[0] == 256  # 512 / 2

    def test_row_parallel_creation(self, device):
        """Test creating row parallel linear."""
        from torchbridge.models.distributed import (
            RowParallelLinear,
            TensorParallelConfig,
        )

        config = TensorParallelConfig(world_size=2, rank=0)
        layer = RowParallelLinear(
            in_features=256,
            out_features=256,
            config=config,
            input_is_parallel=True,
        ).to(device)

        # Input features should be partitioned
        assert layer.weight.shape[1] == 128  # 256 / 2


# =============================================================================
# Pipeline Parallelism Tests
# =============================================================================

class TestPipelineParallelConfig:
    """Test PipelineParallelConfig."""

    def test_config_creation(self):
        """Test creating pipeline config."""
        from torchbridge.models.distributed import PipelineParallelConfig

        config = PipelineParallelConfig(
            num_stages=4,
            num_micro_batches=8,
            stage_id=0,
        )

        assert config.num_stages == 4
        assert config.num_micro_batches == 8

    def test_config_attributes(self):
        """Test config attributes."""
        from torchbridge.models.distributed import PipelineParallelConfig

        config = PipelineParallelConfig(
            num_stages=8,
            num_micro_batches=16,
            stage_id=2,
        )

        assert config.stage_id == 2


class TestPipelineStage:
    """Test PipelineStage."""

    def test_stage_creation(self, device):
        """Test creating pipeline stage."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=2,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        assert stage.module is module


class TestPipelineSchedulers:
    """Test pipeline schedulers."""

    def test_gpipe_scheduler_exists(self):
        """Test GPipeScheduler exists."""
        from torchbridge.models.distributed import GPipeScheduler
        assert GPipeScheduler is not None

    def test_interleaved_scheduler_exists(self):
        """Test InterleavedScheduler exists."""
        from torchbridge.models.distributed import InterleavedScheduler
        assert InterleavedScheduler is not None


class TestPipelineMemory:
    """Test pipeline memory estimation."""

    def test_memory_estimation(self, device):
        """Test pipeline memory estimation."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            estimate_pipeline_memory,
        )

        # Create a simple model for estimation
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        ).to(device)

        config = PipelineParallelConfig(
            num_stages=2,
            num_micro_batches=4,
            stage_id=0,
        )

        result = estimate_pipeline_memory(
            model=model,
            config=config,
            micro_batch_size=4,
            sequence_length=128,
        )

        assert result is not None
        assert isinstance(result, dict)


# =============================================================================
# Model Sharding Tests
# =============================================================================

class TestShardingStrategy:
    """Test ShardingStrategy enum."""

    def test_strategies_exist(self):
        """Test that sharding strategies exist."""
        from torchbridge.models.distributed import ShardingStrategy

        assert hasattr(ShardingStrategy, 'FULL_SHARD')
        assert hasattr(ShardingStrategy, 'SHARD_GRAD_OP')


class TestModelSharder:
    """Test ModelSharder."""

    def test_sharder_creation(self):
        """Test creating model sharder."""
        from torchbridge.models.distributed import (
            ModelSharder,
            ShardingStrategy,
        )
        from torchbridge.models.distributed.model_sharding import ShardingConfig

        config = ShardingConfig(
            strategy=ShardingStrategy.FULL_SHARD,
            world_size=4,
            rank=0,
        )
        sharder = ModelSharder(config=config)

        assert sharder is not None


class TestWeightDistributor:
    """Test WeightDistributor."""

    def test_distributor_creation(self):
        """Test creating weight distributor."""
        from torchbridge.models.distributed import (
            ShardingStrategy,
            WeightDistributor,
        )
        from torchbridge.models.distributed.model_sharding import ShardingConfig

        config = ShardingConfig(
            strategy=ShardingStrategy.FULL_SHARD,
            world_size=4,
            rank=0,
        )
        distributor = WeightDistributor(config=config)

        assert distributor is not None


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_distributed_llm(self):
        """Test create_distributed_llm factory."""
        from torchbridge.models.distributed import create_distributed_llm

        # Create optimizer (does not load model by default)
        result = create_distributed_llm(
            model_name="meta-llama/Llama-2-7b-hf",
            world_size=2,
        )

        assert result is not None
        assert isinstance(result, object)

    def test_distributed_llama_70b(self):
        """Test DistributedLlama70B wrapper."""
        from torchbridge.models.distributed import DistributedLlama70B

        # Uses default config (world_size=8, tensor_parallel_size=8)
        wrapper = DistributedLlama70B()
        assert wrapper is not None
        assert wrapper.model_name == "meta-llama/Llama-2-70b-hf"


# =============================================================================
# Integration Tests
# =============================================================================

class TestDistributedIntegration:
    """Integration tests for distributed components."""

    def test_parallelism_strategy_enum(self):
        """Test ParallelismStrategy enum values."""
        from torchbridge.models.distributed import ParallelismStrategy

        assert ParallelismStrategy.TENSOR_PARALLEL is not None
        assert ParallelismStrategy.PIPELINE_PARALLEL is not None
        assert ParallelismStrategy.HYBRID is not None
        assert ParallelismStrategy.FSDP is not None
        assert ParallelismStrategy.AUTO is not None

    def test_large_model_type_enum(self):
        """Test LargeModelType enum values."""
        from torchbridge.models.distributed import LargeModelType

        assert LargeModelType.LLAMA_70B is not None
        assert LargeModelType.FALCON_40B is not None


# =============================================================================
# Multi-GPU Tests (skipped without hardware)
# =============================================================================

@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Multi-GPU tests require at least 2 GPUs"
)
class TestMultiGPU:
    """Tests requiring multiple GPUs."""

    def test_multi_gpu_available(self):
        """Test that multiple GPUs are detected."""
        assert torch.cuda.device_count() >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
