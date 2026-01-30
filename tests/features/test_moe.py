"""
Comprehensive Test Suite for Mixture of Experts (MoE)

Tests the MoE implementation including:
- MoE layer creation and configuration
- Different MoE types (standard, sparse, switch, glam, adaptive)
- Routing mechanisms (TopK, Switch, Hash, Learned)
- Expert networks (FeedForward, Convolutional, Attention)
- Load balancing and capacity management
- Training mode with auxiliary losses
- Memory and performance characteristics
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default values for import flags
MOE_AVAILABLE = False
MOE_EXTENDED_AVAILABLE = False

# Import MoE modules
try:
    from kernel_pytorch import (
        FeedForwardExpert,
        GLaMStyleMoE,
        LoadBalancer,
        MoEConfig,
        MoELayer,
        SparseMoELayer,
        SwitchRouter,
        SwitchTransformerMoE,
        TopKRouter,
        create_moe,
        create_moe_layer,
    )
    MOE_AVAILABLE = True
except ImportError:
    pass

# Extended imports - some may not be exported from main __init__
try:
    from kernel_pytorch.mixture_of_experts.expert_networks import (
        AttentionExpert,
        ConvolutionalExpert,
        ParameterEfficientExpert,
    )
    from kernel_pytorch.mixture_of_experts.moe_layers import AdaptiveMoELayer
    from kernel_pytorch.mixture_of_experts.optimization import (
        ExpertParallelism,
        ExpertScheduler,
        MemoryEfficientSwitching,
    )
    from kernel_pytorch.mixture_of_experts.routing import (
        DynamicCapacityRouter,
        HashRouter,
        LearnedRouter,  # noqa: F401
    )
    MOE_EXTENDED_AVAILABLE = True
except ImportError:
    pass


@pytest.fixture
def device():
    """Get test device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def moe_config():
    """Create standard MoE config"""
    return MoEConfig(
        num_experts=8,
        top_k=2,
        capacity_factor=1.25,
        expert_dropout=0.1,
        router_dropout=0.1,
        load_balance_loss_weight=0.01,
    )


@pytest.fixture
def sample_input(device):
    """Create sample input tensor"""
    batch_size = 4
    seq_len = 32
    hidden_size = 256
    return torch.randn(batch_size, seq_len, hidden_size, device=device)


# =============================================================================
# MoE Configuration Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestMoEConfig:
    """Test MoE configuration"""

    def test_config_creation(self):
        """Test basic config creation"""
        config = MoEConfig()

        assert config.num_experts == 8
        assert config.top_k == 2
        assert config.capacity_factor == 1.25

    def test_config_customization(self):
        """Test custom config values"""
        config = MoEConfig(
            num_experts=16,
            top_k=4,
            capacity_factor=1.5,
            expert_dropout=0.2,
        )

        assert config.num_experts == 16
        assert config.top_k == 4
        assert config.capacity_factor == 1.5
        assert config.expert_dropout == 0.2


# =============================================================================
# MoE Layer Creation Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestMoELayerCreation:
    """Test MoE layer creation"""

    def test_standard_moe_creation(self, moe_config, device):
        """Test standard MoE layer creation"""
        layer = MoELayer(moe_config, hidden_size=256, device=device)

        assert layer.num_experts == 8
        assert layer.top_k == 2
        assert len(layer.experts) == 8

    def test_sparse_moe_creation(self, moe_config, device):
        """Test sparse MoE layer creation"""
        layer = SparseMoELayer(moe_config, hidden_size=256, sparsity_level=0.25, device=device)

        assert layer.num_experts == 8
        assert layer.sparsity_level == 0.25

    def test_switch_transformer_creation(self, moe_config, device):
        """Test Switch Transformer MoE creation"""
        layer = SwitchTransformerMoE(moe_config, hidden_size=256, device=device)

        assert layer.num_experts == 8
        assert layer.top_k == 1  # Switch uses top-1 routing

    def test_glam_style_creation(self, moe_config, device):
        """Test GLaM-style MoE creation"""
        layer = GLaMStyleMoE(moe_config, hidden_size=256, device=device)

        assert layer.num_experts == 8

    @pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
    def test_adaptive_moe_creation(self, moe_config, device):
        """Test adaptive MoE layer creation"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        layer = AdaptiveMoELayer(moe_config, hidden_size=256, device=device)

        assert layer.num_experts == 8
        assert hasattr(layer, 'adaptation_rate')

    def test_factory_function(self, device):
        """Test create_moe_layer factory function"""
        for moe_type in ["standard", "sparse", "switch", "glam", "adaptive"]:
            layer = create_moe_layer(
                moe_type=moe_type,
                hidden_size=256,
                num_experts=4,
                top_k=2 if moe_type != "switch" else 1,
            )
            layer = layer.to(device)
            assert isinstance(layer, MoELayer)

    def test_create_moe_convenience(self, device):
        """Test create_moe convenience function"""
        layer = create_moe(hidden_size=256, num_experts=4, top_k=2)
        layer = layer.to(device)

        assert isinstance(layer, MoELayer)
        assert layer.num_experts == 4


# =============================================================================
# MoE Forward Pass Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestMoEForwardPass:
    """Test MoE forward pass"""

    def test_standard_forward(self, moe_config, sample_input, device):
        """Test standard MoE forward pass"""
        layer = MoELayer(moe_config, hidden_size=256, device=device)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()

    def test_forward_with_aux_losses(self, moe_config, sample_input, device):
        """Test forward pass with auxiliary losses"""
        layer = MoELayer(moe_config, hidden_size=256, device=device)
        output, aux_losses = layer(sample_input, return_router_logits=True)

        assert output.shape == sample_input.shape
        assert 'load_balance_loss' in aux_losses
        assert 'router_z_loss' in aux_losses
        assert 'expert_utilization' in aux_losses

    def test_sparse_forward(self, moe_config, sample_input, device):
        """Test sparse MoE forward pass"""
        layer = SparseMoELayer(moe_config, hidden_size=256, sparsity_level=0.25, device=device)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()

    def test_switch_forward(self, moe_config, sample_input, device):
        """Test Switch Transformer forward pass"""
        layer = SwitchTransformerMoE(moe_config, hidden_size=256, device=device)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()

    def test_glam_forward(self, moe_config, sample_input, device):
        """Test GLaM-style forward pass"""
        layer = GLaMStyleMoE(moe_config, hidden_size=256, device=device)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
    def test_adaptive_forward(self, moe_config, sample_input, device):
        """Test adaptive MoE forward pass"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        layer = AdaptiveMoELayer(moe_config, hidden_size=256, device=device)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()


# =============================================================================
# Router Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestRouters:
    """Test routing mechanisms"""

    def test_topk_router(self, device):
        """Test TopK router"""
        router = TopKRouter(
            hidden_size=256,
            num_experts=8,
            top_k=2,
            device=device
        )

        x = torch.randn(32, 256, device=device)  # [num_tokens, hidden_size]
        output = router(x)

        assert 'probs' in output
        assert 'logits' in output
        assert 'expert_indices' in output
        assert 'expert_weights' in output

        assert output['expert_indices'].shape == (32, 2)  # [num_tokens, top_k]
        assert output['expert_weights'].shape == (32, 2)

    def test_switch_router(self, device):
        """Test Switch router (top-1)"""
        router = SwitchRouter(
            hidden_size=256,
            num_experts=8,
            top_k=1,
            device=device
        )

        x = torch.randn(32, 256, device=device)
        output = router(x)

        assert output['expert_indices'].shape == (32, 1)  # Top-1

    @pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
    def test_hash_router(self, device):
        """Test Hash router"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        router = HashRouter(
            hidden_size=256,
            num_experts=8,
            top_k=2,
            device=device
        )

        x = torch.randn(32, 256, device=device)
        output = router(x)

        assert 'expert_indices' in output

    @pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
    def test_dynamic_capacity_router(self, device):
        """Test Dynamic Capacity router"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        router = DynamicCapacityRouter(
            hidden_size=256,
            num_experts=8,
            top_k=2,
            device=device
        )

        x = torch.randn(32, 256, device=device)
        output = router(x)

        assert 'expert_indices' in output


# =============================================================================
# Expert Network Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestExpertNetworks:
    """Test expert network implementations"""

    def test_feedforward_expert(self, device):
        """Test FeedForward expert"""
        expert = FeedForwardExpert(
            input_size=256,
            hidden_size=1024,
            output_size=256,
            device=device
        )

        x = torch.randn(32, 256, device=device)
        output = expert(x)

        assert output.shape == (32, 256)
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
    def test_convolutional_expert(self, device):
        """Test Convolutional expert"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        expert = ConvolutionalExpert(
            input_size=256,
            hidden_size=1024,
            output_size=256,
            device=device
        )

        x = torch.randn(32, 256, device=device)
        output = expert(x)

        assert output.shape == (32, 256)

    @pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
    def test_attention_expert(self, device):
        """Test Attention expert"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        expert = AttentionExpert(
            input_size=256,
            hidden_size=1024,
            output_size=256,
            device=device
        )

        x = torch.randn(32, 256, device=device)
        output = expert(x)

        assert output.shape == (32, 256)

    @pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
    def test_parameter_efficient_expert(self, device):
        """Test Parameter-efficient expert"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        expert = ParameterEfficientExpert(
            input_size=256,
            hidden_size=1024,
            output_size=256,
            device=device
        )

        x = torch.randn(32, 256, device=device)
        output = expert(x)

        assert output.shape == (32, 256)


# =============================================================================
# Load Balancing Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestLoadBalancing:
    """Test load balancing mechanisms"""

    def test_load_balancer_creation(self):
        """Test LoadBalancer creation"""
        balancer = LoadBalancer(
            num_experts=8,
            capacity_factor=1.25,
            min_capacity=4
        )

        assert balancer.num_experts == 8

    def test_capacity_info(self, device):
        """Test capacity info computation"""
        balancer = LoadBalancer(
            num_experts=8,
            capacity_factor=1.25,
            min_capacity=4
        )

        router_probs = torch.randn(128, 8, device=device).softmax(dim=-1)
        num_tokens = 128

        capacity_info = balancer.get_capacity_info(router_probs, num_tokens)

        assert 'expert_capacities' in capacity_info
        assert 'expert_utilization' in capacity_info

    def test_load_balance_loss(self, device):
        """Test load balance loss computation"""
        balancer = LoadBalancer(num_experts=8)

        router_probs = torch.randn(128, 8, device=device).softmax(dim=-1)
        expert_indices = torch.randint(0, 8, (128, 2), device=device)

        loss = balancer.compute_load_balance_loss(router_probs, expert_indices)

        assert torch.isfinite(loss)
        assert loss >= 0


# =============================================================================
# Training Mode Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestMoETraining:
    """Test MoE training functionality"""

    def test_gradient_flow(self, moe_config, sample_input, device):
        """Test gradient flow through MoE layer"""
        layer = MoELayer(moe_config, hidden_size=256, device=device)
        sample_input = sample_input.requires_grad_(True)

        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None
        assert torch.isfinite(sample_input.grad).all()

    def test_training_vs_eval(self, moe_config, sample_input, device):
        """Test training vs eval mode differences"""
        layer = MoELayer(moe_config, hidden_size=256, device=device)

        # Training mode
        layer.train()
        output_train = layer(sample_input)

        # Eval mode
        layer.eval()
        output_eval = layer(sample_input)

        # Both should produce valid outputs
        assert torch.isfinite(output_train).all()
        assert torch.isfinite(output_eval).all()

    def test_expert_utilization_tracking(self, moe_config, sample_input, device):
        """Test expert utilization statistics tracking"""
        layer = MoELayer(moe_config, hidden_size=256, device=device)
        layer.train()

        # Reset statistics
        layer.reset_statistics()

        # Forward pass
        _ = layer(sample_input)

        # Check statistics
        stats = layer.get_expert_utilization_stats()

        assert 'expert_balance' in stats
        assert 'expert_efficiency' in stats
        assert 'usage_rates' in stats

    def test_auxiliary_losses_gradient(self, moe_config, sample_input, device):
        """Test that auxiliary losses have gradients"""
        layer = MoELayer(moe_config, hidden_size=256, device=device)
        layer.train()

        output, aux_losses = layer(sample_input, return_router_logits=True)

        # Combine losses
        total_loss = output.sum() + aux_losses['load_balance_loss'] + aux_losses['router_z_loss']
        total_loss.backward()

        # Check gradients exist
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None or param.numel() == 0


# =============================================================================
# Expert Parallelism Tests
# =============================================================================

@pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
class TestExpertParallelism:
    """Test expert parallelism features"""

    def test_parallelism_creation(self, device):
        """Test ExpertParallelism creation"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        experts = nn.ModuleList([
            FeedForwardExpert(256, 1024, 256, device=device)
            for _ in range(8)
        ])

        parallelism = ExpertParallelism(num_experts=8, experts=experts)

        assert parallelism.num_experts == 8


# =============================================================================
# Memory Efficiency Tests
# =============================================================================

@pytest.mark.skipif(not MOE_EXTENDED_AVAILABLE, reason="Extended MoE imports not available")
class TestMoEMemoryEfficiency:
    """Test MoE memory efficiency"""

    def test_memory_efficient_switching(self):
        """Test MemoryEfficientSwitching"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        switching = MemoryEfficientSwitching(
            use_gradient_checkpointing=True,
            expert_offloading=True
        )

        assert switching.use_gradient_checkpointing is True
        assert switching.expert_offloading is True

    def test_expert_scheduler(self):
        """Test ExpertScheduler"""
        if not MOE_EXTENDED_AVAILABLE:
            pytest.skip("Extended MoE not available")
        scheduler = ExpertScheduler(
            num_experts=8,
            initial_capacity_factor=1.25,
            target_expert_utilization=0.8
        )

        assert scheduler.num_experts == 8
        assert scheduler.initial_capacity_factor == 1.25


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
class TestMoEIntegration:
    """Integration tests for MoE"""

    def test_moe_in_transformer(self, moe_config, device):
        """Test MoE layer in a transformer-like architecture"""
        class SimpleMoETransformerBlock(nn.Module):
            def __init__(self, hidden_size, num_heads, moe_config):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
                self.moe = MoELayer(moe_config, hidden_size)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)

            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)

                # MoE FFN
                moe_out = self.moe(x)
                x = self.norm2(x + moe_out)

                return x

        block = SimpleMoETransformerBlock(256, 4, moe_config).to(device)
        x = torch.randn(2, 16, 256, device=device)

        output = block(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_moe_multiple_layers(self, moe_config, device):
        """Test multiple MoE layers stacked"""
        layers = nn.Sequential(*[
            MoELayer(moe_config, hidden_size=256, device=device)
            for _ in range(3)
        ])

        x = torch.randn(2, 16, 256, device=device)
        output = layers(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_end_to_end_training(self, moe_config, device):
        """Test end-to-end training with MoE"""
        model = nn.Sequential(
            nn.Linear(256, 256),
            MoELayer(moe_config, hidden_size=256, device=device),
            nn.Linear(256, 10)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        for _ in range(5):
            x = torch.randn(4, 16, 256, device=device)
            targets = torch.randint(0, 10, (4, 16), device=device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = F.cross_entropy(outputs.view(-1, 10), targets.view(-1))
            loss.backward()
            optimizer.step()

        assert torch.isfinite(loss)


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
@pytest.mark.parametrize("moe_type", ["standard", "sparse", "switch", "glam", "adaptive"])
def test_moe_types(moe_type, device):
    """Test all MoE types"""
    layer = create_moe_layer(
        moe_type=moe_type,
        hidden_size=256,
        num_experts=4,
        top_k=2 if moe_type != "switch" else 1,
    ).to(device)

    x = torch.randn(2, 16, 256, device=device)
    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
@pytest.mark.parametrize("num_experts", [2, 4, 8, 16])
def test_different_expert_counts(num_experts, device):
    """Test MoE with different numbers of experts"""
    config = MoEConfig(num_experts=num_experts, top_k=min(2, num_experts))
    layer = MoELayer(config, hidden_size=256, device=device)

    x = torch.randn(2, 16, 256, device=device)
    output = layer(x)

    assert output.shape == x.shape
    assert layer.num_experts == num_experts


@pytest.mark.skipif(not MOE_AVAILABLE, reason="MoE not available")
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_different_top_k(top_k, device):
    """Test MoE with different top-k values"""
    config = MoEConfig(num_experts=8, top_k=top_k)
    layer = MoELayer(config, hidden_size=256, device=device)

    x = torch.randn(2, 16, 256, device=device)
    output = layer(x)

    assert output.shape == x.shape
    assert layer.top_k == top_k


if __name__ == "__main__":
    # Run basic smoke test
    if MOE_AVAILABLE:
        print("Running MoE smoke test...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Test basic MoE
        layer = create_moe(hidden_size=256, num_experts=4, top_k=2).to(device)
        x = torch.randn(2, 16, 256, device=device)
        output = layer(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output finite: {torch.isfinite(output).all()}")

        # Test with auxiliary losses
        output, aux = layer(x, return_router_logits=True)
        print(f"Auxiliary losses: {list(aux.keys())}")

        # Expert statistics
        stats = layer.get_expert_utilization_stats()
        print(f"Expert stats: {stats}")

        print("\nMoE smoke test PASSED!")
    else:
        print("MoE not available - skipping smoke test")
