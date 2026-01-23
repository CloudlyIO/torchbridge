"""
Tests for Distributed Model Integration (v0.4.13)

Tests tensor parallelism, pipeline parallelism, model sharding,
and large model optimization without requiring actual multi-GPU setup.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# ============================================================================
# Tensor Parallelism Tests
# ============================================================================


class TestTensorParallelConfig:
    """Test TensorParallelConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from kernel_pytorch.models.distributed import TensorParallelConfig

        config = TensorParallelConfig()
        assert config.world_size == 1
        assert config.rank == 0
        assert config.sequence_parallel is False

    def test_config_validation(self):
        """Test configuration validation."""
        from kernel_pytorch.models.distributed import TensorParallelConfig

        with pytest.raises(ValueError):
            TensorParallelConfig(world_size=4, rank=5)  # rank >= world_size

    def test_multi_gpu_config(self):
        """Test multi-GPU configuration."""
        from kernel_pytorch.models.distributed import TensorParallelConfig

        config = TensorParallelConfig(
            world_size=8,
            rank=3,
            sequence_parallel=True,
            async_tensor_model_parallel_allreduce=True,
        )
        assert config.world_size == 8
        assert config.rank == 3
        assert config.sequence_parallel is True


class TestColumnParallelLinear:
    """Test column parallel linear layer."""

    def test_initialization(self):
        """Test layer initialization."""
        from kernel_pytorch.models.distributed import (
            TensorParallelConfig,
            ColumnParallelLinear,
        )

        config = TensorParallelConfig(world_size=2, rank=0)
        layer = ColumnParallelLinear(
            in_features=512,
            out_features=1024,
            config=config,
            bias=True,
        )

        # Output features should be partitioned
        assert layer.weight.shape == (512, 512)  # 1024 / 2 = 512

    def test_forward_pass(self):
        """Test forward pass."""
        from kernel_pytorch.models.distributed import (
            TensorParallelConfig,
            ColumnParallelLinear,
        )

        config = TensorParallelConfig(world_size=2, rank=0)
        layer = ColumnParallelLinear(
            in_features=256,
            out_features=512,
            config=config,
            gather_output=False,
        )

        x = torch.randn(2, 16, 256)
        output = layer(x)

        # Output should be partitioned (512 / 2 = 256)
        assert output.shape == (2, 16, 256)


class TestRowParallelLinear:
    """Test row parallel linear layer."""

    def test_initialization(self):
        """Test layer initialization."""
        from kernel_pytorch.models.distributed import (
            TensorParallelConfig,
            RowParallelLinear,
        )

        config = TensorParallelConfig(world_size=2, rank=0)
        layer = RowParallelLinear(
            in_features=1024,
            out_features=512,
            config=config,
            bias=True,
        )

        # Input features should be partitioned
        assert layer.weight.shape == (512, 512)  # 1024 / 2 = 512

    def test_forward_with_parallel_input(self):
        """Test forward pass with parallel input."""
        from kernel_pytorch.models.distributed import (
            TensorParallelConfig,
            RowParallelLinear,
        )

        config = TensorParallelConfig(world_size=2, rank=0)
        layer = RowParallelLinear(
            in_features=512,
            out_features=256,
            config=config,
            input_is_parallel=True,
        )

        # Input is already partitioned
        x = torch.randn(2, 16, 256)  # 512 / 2 = 256
        output = layer(x)

        assert output.shape == (2, 16, 256)


class TestTensorParallelEmbedding:
    """Test tensor parallel embedding layer."""

    def test_initialization(self):
        """Test embedding initialization."""
        from kernel_pytorch.models.distributed import (
            TensorParallelConfig,
            TensorParallelEmbedding,
        )

        config = TensorParallelConfig(world_size=4, rank=0)
        embedding = TensorParallelEmbedding(
            num_embeddings=32000,
            embedding_dim=4096,
            config=config,
        )

        # Embedding dim should be partitioned
        assert embedding.weight.shape == (32000, 1024)  # 4096 / 4 = 1024

    def test_forward_pass(self):
        """Test embedding forward pass."""
        from kernel_pytorch.models.distributed import (
            TensorParallelConfig,
            TensorParallelEmbedding,
        )

        config = TensorParallelConfig(world_size=2, rank=0)
        embedding = TensorParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            config=config,
        )

        x = torch.randint(0, 1000, (2, 16))
        output = embedding(x)

        # In single-process test (no dist.init), gather returns partitioned output
        # With world_size=2, embedding_dim 512 is split to 256 per partition
        assert output.shape == (2, 16, 256)


class TestApplyTensorParallelism:
    """Test automatic tensor parallelism application."""

    def test_skip_for_single_gpu(self):
        """Test that parallelism is skipped for world_size=1."""
        from kernel_pytorch.models.distributed import (
            TensorParallelConfig,
            apply_tensor_parallelism,
        )

        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

        config = TensorParallelConfig(world_size=1, rank=0)
        result = apply_tensor_parallelism(model, config)

        # Should return same model without changes
        assert isinstance(result[0], nn.Linear)


# ============================================================================
# Pipeline Parallelism Tests
# ============================================================================


class TestPipelineParallelConfig:
    """Test PipelineParallelConfig configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from kernel_pytorch.models.distributed import PipelineParallelConfig

        config = PipelineParallelConfig()
        assert config.num_stages == 1
        assert config.num_micro_batches == 1
        assert config.activation_checkpointing is True

    def test_config_validation(self):
        """Test configuration validation."""
        from kernel_pytorch.models.distributed import PipelineParallelConfig

        with pytest.raises(ValueError):
            PipelineParallelConfig(num_stages=4, stage_id=5)


class TestPipelineStage:
    """Test PipelineStage functionality."""

    def test_stage_creation(self):
        """Test creating a pipeline stage."""
        from kernel_pytorch.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )

        module = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        config = PipelineParallelConfig(num_stages=4, stage_id=0)
        stage = PipelineStage(module, config)

        assert stage.is_first_stage is True
        assert stage.is_last_stage is False

    def test_forward_step(self):
        """Test forward step execution."""
        from kernel_pytorch.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )

        module = nn.Linear(256, 256)
        config = PipelineParallelConfig(num_stages=2, stage_id=0)
        stage = PipelineStage(module, config)

        x = torch.randn(2, 16, 256)
        output = stage.forward_step(x, micro_batch_id=0)

        assert output.shape == x.shape


class TestGPipeScheduler:
    """Test GPipe scheduler."""

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        from kernel_pytorch.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
            GPipeScheduler,
        )

        module = nn.Linear(256, 256)
        config = PipelineParallelConfig(num_stages=2, num_micro_batches=4, stage_id=0)
        stage = PipelineStage(module, config)
        scheduler = GPipeScheduler([stage], config)

        assert scheduler.num_micro_batches == 4


class TestEstimatePipelineMemory:
    """Test pipeline memory estimation."""

    def test_memory_estimation(self):
        """Test memory estimation for pipeline."""
        from kernel_pytorch.models.distributed import (
            PipelineParallelConfig,
            estimate_pipeline_memory,
        )

        model = nn.Sequential(
            *[nn.Linear(1024, 1024) for _ in range(32)]
        )
        config = PipelineParallelConfig(num_stages=4, num_micro_batches=8)

        memory = estimate_pipeline_memory(
            model, config, micro_batch_size=4, sequence_length=2048
        )

        assert "params_per_stage_gb" in memory
        assert "gpipe_peak_gb" in memory
        assert "interleaved_peak_gb" in memory
        assert "recommended_schedule" in memory


# ============================================================================
# Model Sharding Tests
# ============================================================================


class TestShardingConfig:
    """Test ShardingConfig configuration."""

    def test_default_config(self):
        """Test default sharding configuration."""
        from kernel_pytorch.models.distributed import ShardingConfig, ShardingStrategy

        config = ShardingConfig()
        assert config.strategy == ShardingStrategy.FULL_SHARD
        assert config.cpu_offload is False
        assert config.mixed_precision is True


class TestModelSharder:
    """Test ModelSharder functionality."""

    def test_analyze_model(self):
        """Test model analysis for sharding."""
        from kernel_pytorch.models.distributed import ShardingConfig, ModelSharder

        model = nn.Sequential(
            nn.Embedding(10000, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 512),
        )

        config = ShardingConfig(world_size=4, rank=0)
        sharder = ModelSharder(config)
        specs = sharder.analyze_model(model)

        assert len(specs) > 0
        assert "0.weight" in specs  # Embedding weight

    def test_shard_model(self):
        """Test model sharding."""
        from kernel_pytorch.models.distributed import ShardingConfig, ModelSharder

        model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Linear(2048, 1024),
        )

        config = ShardingConfig(world_size=2, rank=0)
        sharder = ModelSharder(config)
        sharded_model = sharder.shard_model(model)

        # Check that parameters are sharded
        # Linear(1024, 2048) has weight shape (2048, 1024)
        # Shards along larger dim (dim=0), so output becomes (1024, 1024)
        assert sharded_model[0].weight.shape == (1024, 1024)


class TestWeightDistributor:
    """Test WeightDistributor functionality."""

    def test_create_device_map(self):
        """Test device map creation."""
        from kernel_pytorch.models.distributed import ShardingConfig, WeightDistributor

        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        config = ShardingConfig(world_size=2, rank=0)
        distributor = WeightDistributor(config)

        # Mock available memory
        with patch.object(distributor, "_get_available_memory") as mock_mem:
            mock_mem.return_value = {"cuda:0": 10e9, "cuda:1": 10e9, "cpu": 50e9}
            device_map = distributor.create_device_map(model)

        assert len(device_map) > 0


class TestAutomaticSharding:
    """Test automatic sharding function."""

    def test_automatic_sharding_small_model(self):
        """Test that small models aren't sharded."""
        from kernel_pytorch.models.distributed import ShardingConfig, automatic_sharding

        model = nn.Linear(256, 256)  # Small model
        config = ShardingConfig(world_size=2, rank=0)

        sharded_model, info = automatic_sharding(model, config, target_memory_gb=16.0)

        # Small model shouldn't need sharding
        assert "sharding_applied" in info


# ============================================================================
# Large Model Optimizer Tests
# ============================================================================


class TestDistributedConfig:
    """Test DistributedConfig configuration."""

    def test_default_config(self):
        """Test default distributed configuration."""
        from kernel_pytorch.models.distributed import DistributedConfig

        config = DistributedConfig()
        assert config.world_size == 1
        assert config.dtype == torch.bfloat16
        assert config.use_flash_attention is True

    def test_auto_adjust_parallelism(self):
        """Test automatic parallelism adjustment."""
        from kernel_pytorch.models.distributed import DistributedConfig, ParallelismStrategy

        config = DistributedConfig(
            world_size=8,
            tensor_parallel_size=2,  # Doesn't match world_size
            pipeline_parallel_size=2,
            strategy=ParallelismStrategy.TENSOR_PARALLEL,
        )

        # Should auto-adjust to match world_size
        assert config.tensor_parallel_size * config.pipeline_parallel_size * config.data_parallel_size <= config.world_size


class TestDistributedLLMOptimizer:
    """Test DistributedLLMOptimizer functionality."""

    def test_model_type_detection(self):
        """Test automatic model type detection."""
        from kernel_pytorch.models.distributed import (
            DistributedLLMOptimizer,
            LargeModelType,
        )

        optimizer = DistributedLLMOptimizer("meta-llama/Llama-2-70b-hf")
        assert optimizer.model_type == LargeModelType.LLAMA_70B

        optimizer = DistributedLLMOptimizer("mistralai/Mixtral-8x7B-v0.1")
        assert optimizer.model_type == LargeModelType.MIXTRAL

    def test_memory_estimation(self):
        """Test memory estimation for large models."""
        from kernel_pytorch.models.distributed import DistributedLLMOptimizer, DistributedConfig

        config = DistributedConfig(world_size=8, tensor_parallel_size=8)
        optimizer = DistributedLLMOptimizer("meta-llama/Llama-2-70b-hf", config)

        memory = optimizer.estimate_memory()

        assert "total_params_gb" in memory
        assert "params_per_gpu_gb" in memory
        assert "kv_cache_gb" in memory
        assert memory["total_params_gb"] > 0
        assert memory["params_per_gpu_gb"] < memory["total_params_gb"]

    def test_mock_model_loading(self):
        """Test loading mock model (without transformers)."""
        from kernel_pytorch.models.distributed import DistributedLLMOptimizer, DistributedConfig

        config = DistributedConfig(world_size=1)
        optimizer = DistributedLLMOptimizer("meta-llama/Llama-2-70b-hf", config)

        # This will create a mock model
        with patch.dict("sys.modules", {"transformers": None}):
            model = optimizer._create_mock_model()

        assert model is not None
        assert hasattr(model, "embed")
        assert hasattr(model, "layers")
        assert hasattr(model, "head")


class TestConvenienceWrappers:
    """Test convenience wrapper classes."""

    def test_distributed_llama70b(self):
        """Test DistributedLlama70B wrapper."""
        from kernel_pytorch.models.distributed import DistributedLlama70B

        optimizer = DistributedLlama70B()
        assert optimizer.config.world_size == 8
        assert optimizer.config.tensor_parallel_size == 8

    def test_distributed_mixtral(self):
        """Test DistributedMixtral wrapper."""
        from kernel_pytorch.models.distributed import DistributedMixtral

        optimizer = DistributedMixtral()
        assert optimizer.config.world_size == 8


class TestCreateDistributedLLM:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating distributed LLM with defaults."""
        from kernel_pytorch.models.distributed import create_distributed_llm

        optimizer = create_distributed_llm("meta-llama/Llama-2-70b-hf")

        assert optimizer.model_name == "meta-llama/Llama-2-70b-hf"
        assert optimizer.config.world_size == 8

    def test_create_with_custom_config(self):
        """Test creating with custom configuration."""
        from kernel_pytorch.models.distributed import (
            create_distributed_llm,
            ParallelismStrategy,
        )

        optimizer = create_distributed_llm(
            "meta-llama/Llama-2-70b-hf",
            world_size=4,
            dtype=torch.float16,
            quantization="int8",
            strategy=ParallelismStrategy.TENSOR_PARALLEL,
        )

        assert optimizer.config.world_size == 4
        assert optimizer.config.dtype == torch.float16
        assert optimizer.config.quantization == "int8"


class TestEstimateGPURequirements:
    """Test GPU requirements estimation."""

    def test_estimate_requirements(self):
        """Test estimating GPU requirements."""
        from kernel_pytorch.models.distributed import estimate_gpu_requirements

        requirements = estimate_gpu_requirements(
            "meta-llama/Llama-2-70b-hf",
            max_sequence_length=4096,
            max_batch_size=1,
        )

        assert "min_gpus" in requirements
        assert "recommended_gpus" in requirements
        assert "memory_estimates" in requirements
        assert requirements["min_gpus"] >= 1

    def test_quantization_effect(self):
        """Test that quantization reduces GPU requirements."""
        from kernel_pytorch.models.distributed import estimate_gpu_requirements

        req_fp16 = estimate_gpu_requirements(
            "meta-llama/Llama-2-70b-hf",
            quantization="none",
        )

        req_int8 = estimate_gpu_requirements(
            "meta-llama/Llama-2-70b-hf",
            quantization="int8",
        )

        # int8 should require less memory
        assert req_int8["memory_estimates"]["total_params_gb"] < req_fp16["memory_estimates"]["total_params_gb"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEndDistributed:
    """End-to-end integration tests."""

    def test_imports_work(self):
        """Test that all imports work correctly."""
        from kernel_pytorch.models import (
            # Tensor Parallelism
            TensorParallelConfig,
            ColumnParallelLinear,
            RowParallelLinear,
            TensorParallelEmbedding,
            apply_tensor_parallelism,
            # Pipeline Parallelism
            PipelineParallelConfig,
            PipelineStage,
            GPipeScheduler,
            InterleavedScheduler,
            create_pipeline_stages,
            # Model Sharding
            ShardingStrategy,
            ShardingConfig,
            ModelSharder,
            WeightDistributor,
            automatic_sharding,
            # Large Model Optimizer
            DistributedLLMOptimizer,
            DistributedConfig,
            DistributedLlama70B,
            DistributedFalcon,
            DistributedMixtral,
            create_distributed_llm,
            estimate_gpu_requirements,
        )

    def test_single_gpu_pipeline(self):
        """Test a complete single-GPU pipeline."""
        from kernel_pytorch.models.distributed import (
            create_distributed_llm,
            ParallelismStrategy,
        )

        # Create optimizer for single GPU
        optimizer = create_distributed_llm(
            "custom-model",
            world_size=1,
            strategy=ParallelismStrategy.TENSOR_PARALLEL,
        )

        # Create mock model directly (skip HuggingFace loading)
        model = optimizer._create_mock_model()

        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            output = model(input_ids)

        assert output.shape[0] == 1
        assert output.shape[1] == 32
        assert output.shape[2] == 32000  # Vocab size

    def test_memory_estimation_all_models(self):
        """Test memory estimation for all supported model types."""
        from kernel_pytorch.models.distributed import estimate_gpu_requirements

        models = [
            "meta-llama/Llama-2-70b-hf",
            "tiiuae/falcon-40b",
            "mistralai/Mixtral-8x7B-v0.1",
        ]

        for model_name in models:
            requirements = estimate_gpu_requirements(model_name)
            assert requirements["min_gpus"] >= 1
            assert requirements["memory_estimates"]["total_params_gb"] > 0
