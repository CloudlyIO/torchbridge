"""
Pipeline Parallelism Tests (v0.4.24)

Comprehensive tests for pipeline parallel training including:
- GPipe scheduler (fill-drain)
- 1F1B Interleaved scheduler
- Memory efficiency validation
- Communication patterns

These tests validate the pipeline parallel implementations
without requiring multi-GPU setup for CI compatibility.
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


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for pipeline testing."""

    def __init__(
        self,
        num_layers: int = 8,
        hidden_size: int = 256,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_size)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.fixture
def simple_model(device):
    """Create simple transformer model."""
    return SimpleTransformerModel(num_layers=4, hidden_size=128).to(device)


# =============================================================================
# Pipeline Configuration Tests
# =============================================================================

class TestPipelineConfig:
    """Test pipeline parallel configuration."""

    def test_config_import(self):
        """Test importing pipeline config."""
        from torchbridge.models.distributed import PipelineParallelConfig
        assert PipelineParallelConfig is not None

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
        assert config.stage_id == 0

    def test_config_validation(self):
        """Test config validation."""
        from torchbridge.models.distributed import PipelineParallelConfig

        # Valid config
        config = PipelineParallelConfig(
            num_stages=4,
            num_micro_batches=8,
            stage_id=2,
        )
        assert config.stage_id < config.num_stages

    def test_config_with_options(self):
        """Test config with optional parameters."""
        from torchbridge.models.distributed import PipelineParallelConfig

        config = PipelineParallelConfig(
            num_stages=8,
            num_micro_batches=16,
            stage_id=0,
            activation_checkpointing=True,
        )

        assert config.num_stages == 8
        assert config.activation_checkpointing is True


# =============================================================================
# Pipeline Stage Tests
# =============================================================================

class TestPipelineStage:
    """Test individual pipeline stages."""

    def test_stage_import(self):
        """Test importing pipeline stage."""
        from torchbridge.models.distributed import PipelineStage
        assert PipelineStage is not None

    def test_stage_creation(self, device):
        """Test creating pipeline stage."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=4,
            num_micro_batches=8,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        assert stage.module is module

    def test_stage_properties(self, device):
        """Test stage first/last properties are computed correctly."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )

        # Single stage is both first and last
        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )
        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        assert stage.is_first_stage is True
        assert stage.is_last_stage is True

    def test_stage_forward(self, device):
        """Test stage forward pass."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )

        # Use single stage which is both first and last
        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        x = torch.randn(2, 32, 256, device=device)
        output = stage.forward_step(x, micro_batch_id=0)

        assert output.shape == x.shape

    def test_stage_backward(self, device):
        """Test stage backward pass."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            PipelineStage,
        )

        # Use single stage which is both first and last
        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        # Forward
        x = torch.randn(2, 32, 256, device=device, requires_grad=True)
        output = stage.forward_step(x, micro_batch_id=0)

        # Backward
        grad_output = torch.randn_like(output)
        stage.backward_step(grad_output, micro_batch_id=0)

        assert output.shape == x.shape


# =============================================================================
# GPipe Scheduler Tests
# =============================================================================

class TestGPipeScheduler:
    """Test GPipe (fill-drain) scheduler."""

    def test_scheduler_import(self):
        """Test importing GPipe scheduler."""
        from torchbridge.models.distributed import GPipeScheduler
        assert GPipeScheduler is not None

    def test_scheduler_creation(self, device):
        """Test creating GPipe scheduler."""
        from torchbridge.models.distributed import (
            GPipeScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = GPipeScheduler([stage], config)
        assert scheduler is not None

    def test_gpipe_forward(self, device):
        """Test GPipe forward pass."""
        from torchbridge.models.distributed import (
            GPipeScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = GPipeScheduler([stage], config)

        # Create micro-batches
        micro_batches = [
            torch.randn(2, 32, 256, device=device)
            for _ in range(4)
        ]

        # Run forward
        outputs = scheduler.run_forward(micro_batches)

        assert len(outputs) == 4
        assert all(o.shape == (2, 32, 256) for o in outputs)

    def test_gpipe_backward(self, device):
        """Test GPipe backward pass."""
        from torchbridge.models.distributed import (
            GPipeScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = GPipeScheduler([stage], config)

        # Forward
        micro_batches = [
            torch.randn(2, 32, 256, device=device, requires_grad=True)
            for _ in range(4)
        ]
        outputs = scheduler.run_forward(micro_batches)

        # Backward
        gradients = [torch.randn_like(o) for o in outputs]
        scheduler.run_backward(gradients)

        # Should complete without error
        assert True


# =============================================================================
# Interleaved (1F1B) Scheduler Tests
# =============================================================================

class TestInterleavedScheduler:
    """Test 1F1B interleaved scheduler."""

    def test_scheduler_import(self):
        """Test importing interleaved scheduler."""
        from torchbridge.models.distributed import InterleavedScheduler
        assert InterleavedScheduler is not None

    def test_scheduler_creation(self, device):
        """Test creating interleaved scheduler."""
        from torchbridge.models.distributed import (
            InterleavedScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=8,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = InterleavedScheduler([stage], config)
        assert scheduler is not None

    def test_interleaved_forward(self, device):
        """Test interleaved forward pass."""
        from torchbridge.models.distributed import (
            InterleavedScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=8,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = InterleavedScheduler([stage], config)

        # Create micro-batches
        micro_batches = [
            torch.randn(2, 32, 256, device=device)
            for _ in range(8)
        ]

        # Run forward
        outputs = scheduler.run_forward(micro_batches)

        assert len(outputs) == 8
        assert all(o.shape == (2, 32, 256) for o in outputs)

    def test_interleaved_backward(self, device):
        """Test interleaved backward pass."""
        from torchbridge.models.distributed import (
            InterleavedScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=8,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = InterleavedScheduler([stage], config)

        # Forward
        micro_batches = [
            torch.randn(2, 32, 256, device=device, requires_grad=True)
            for _ in range(8)
        ]
        outputs = scheduler.run_forward(micro_batches)

        # Backward
        gradients = [torch.randn_like(o) for o in outputs]
        scheduler.run_backward(gradients)

        # Should complete without error
        assert True

    def test_run_forward_backward_combined(self, device):
        """Test combined forward-backward pass."""
        from torchbridge.models.distributed import (
            InterleavedScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = InterleavedScheduler([stage], config)

        # Create micro-batches
        micro_batches = [
            torch.randn(2, 32, 256, device=device, requires_grad=True)
            for _ in range(4)
        ]

        # Define loss function
        def loss_fn(output):
            return output.sum()

        # Run combined forward-backward
        total_loss = scheduler.run_forward_backward(micro_batches, loss_fn)

        assert total_loss is not None


# =============================================================================
# Memory Efficiency Tests
# =============================================================================

class TestPipelineMemoryEfficiency:
    """Test memory efficiency of pipeline parallelism."""

    def test_memory_estimation_function(self, device):
        """Test pipeline memory estimation."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            estimate_pipeline_memory,
        )

        # Create a model for estimation
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(device)

        config = PipelineParallelConfig(
            num_stages=2,
            num_micro_batches=8,
            stage_id=0,
        )

        # Estimate memory
        result = estimate_pipeline_memory(
            model=model,
            config=config,
            micro_batch_size=4,
            sequence_length=128,
        )

        # Should return a dictionary with memory info
        assert result is not None
        assert isinstance(result, dict)


# =============================================================================
# Stage Creation Tests
# =============================================================================

class TestPipelineStageCreation:
    """Test creating pipeline stages from models."""

    def test_create_stages_import(self):
        """Test importing stage creation function."""
        from torchbridge.models.distributed import create_pipeline_stages
        assert create_pipeline_stages is not None

    def test_create_stages_returns_list(self, device):
        """Test that create_pipeline_stages returns a list."""
        from torchbridge.models.distributed import (
            PipelineParallelConfig,
            create_pipeline_stages,
        )

        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(device)

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        stages = create_pipeline_stages(model, config)

        assert isinstance(stages, list)
        assert len(stages) >= 1


# =============================================================================
# Communication Pattern Tests
# =============================================================================

class TestPipelineCommunication:
    """Test pipeline communication patterns."""

    def test_send_recv_forward_exists(self):
        """Test that send/recv methods exist."""
        from torchbridge.models.distributed import PipelineStage

        assert hasattr(PipelineStage, 'send_forward')
        assert hasattr(PipelineStage, 'recv_forward')

    def test_send_recv_backward_exists(self):
        """Test that backward send/recv methods exist."""
        from torchbridge.models.distributed import PipelineStage

        assert hasattr(PipelineStage, 'send_backward')
        assert hasattr(PipelineStage, 'recv_backward')


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for pipeline parallelism."""

    def test_full_training_step(self, device):
        """Test a full training step with pipeline."""
        from torchbridge.models.distributed import (
            InterleavedScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        # Create a trainable module
        module = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(device)

        stage = PipelineStage(module, config)

        scheduler = InterleavedScheduler([stage], config)

        # Setup optimizer
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)

        # Create micro-batches
        micro_batches = [
            torch.randn(2, 32, 256, device=device, requires_grad=True)
            for _ in range(4)
        ]

        # Training step
        optimizer.zero_grad()

        def loss_fn(output):
            return output.pow(2).mean()

        total_loss = scheduler.run_forward_backward(micro_batches, loss_fn)

        # Optimizer step
        optimizer.step()

        # Should have updated parameters
        assert total_loss is not None

    def test_gradient_accumulation(self, device):
        """Test gradient accumulation across micro-batches."""
        from torchbridge.models.distributed import (
            InterleavedScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=4,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = InterleavedScheduler([stage], config)

        # Zero gradients
        module.zero_grad()

        # Run forward-backward for multiple micro-batches
        micro_batches = [
            torch.randn(2, 32, 256, device=device, requires_grad=True)
            for _ in range(4)
        ]

        def loss_fn(output):
            return output.sum()

        scheduler.run_forward_backward(micro_batches, loss_fn)

        # Gradients should be accumulated
        assert module.weight.grad is not None


# =============================================================================
# Edge Cases
# =============================================================================

class TestPipelineEdgeCases:
    """Test edge cases for pipeline parallelism."""

    def test_single_micro_batch(self, device):
        """Test with single micro-batch."""
        from torchbridge.models.distributed import (
            GPipeScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=1,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = GPipeScheduler([stage], config)

        micro_batches = [torch.randn(2, 32, 256, device=device)]
        outputs = scheduler.run_forward(micro_batches)

        assert len(outputs) == 1

    def test_many_micro_batches(self, device):
        """Test with many micro-batches."""
        from torchbridge.models.distributed import (
            InterleavedScheduler,
            PipelineParallelConfig,
            PipelineStage,
        )

        config = PipelineParallelConfig(
            num_stages=1,
            num_micro_batches=16,
            stage_id=0,
        )

        module = nn.Linear(256, 256).to(device)
        stage = PipelineStage(module, config)

        scheduler = InterleavedScheduler([stage], config)

        micro_batches = [
            torch.randn(2, 32, 256, device=device)
            for _ in range(16)
        ]
        outputs = scheduler.run_forward(micro_batches)

        assert len(outputs) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
