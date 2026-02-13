"""
Mixture of Experts (MoE) Unit Tests

Tests for routing, expert networks, MoE layers, and load balancing.
Addresses the critical test gap: 4 source files, 2,645 LOC, 0 tests.
"""

import torch
import torch.nn as nn

from torchbridge.mixture_of_experts import (
    FeedForwardExpert,
    GLaMStyleMoE,
    LoadBalancer,
    MoEConfig,
    MoELayer,
    SwitchRouter,
    SwitchTransformerMoE,
    TopKRouter,
    create_moe_layer,
)

# =============================================================================
# Router Tests
# =============================================================================


class TestTopKRouter:
    """Tests for TopKRouter."""

    def test_creation(self):
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)
        assert isinstance(router, nn.Module)

    def test_forward_shape(self):
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)
        x = torch.randn(16, 64)  # [num_tokens, hidden_size]
        result = router(x)
        assert "logits" in result
        assert "probs" in result
        assert "expert_indices" in result
        assert "expert_weights" in result
        assert result["logits"].shape == (16, 8)
        assert result["probs"].shape == (16, 8)
        assert result["expert_indices"].shape == (16, 2)
        assert result["expert_weights"].shape == (16, 2)

    def test_top_k_selection(self):
        router = TopKRouter(hidden_size=32, num_experts=4, top_k=2)
        x = torch.randn(8, 32)
        result = router(x)
        # Expert indices should be in valid range
        assert result["expert_indices"].min() >= 0
        assert result["expert_indices"].max() < 4

    def test_weights_sum(self):
        router = TopKRouter(hidden_size=32, num_experts=4, top_k=2)
        x = torch.randn(8, 32)
        result = router(x)
        # Weights should be non-negative
        assert (result["expert_weights"] >= 0).all()

    def test_expert_mask(self):
        router = TopKRouter(hidden_size=32, num_experts=4, top_k=2)
        x = torch.randn(8, 32)
        # Mask out expert 0 and 1
        mask = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
        result = router(x, expert_mask=mask)
        # Only experts 2 and 3 should be selected
        assert (result["expert_indices"] >= 2).all()


class TestSwitchRouter:
    """Tests for SwitchRouter (top-1 routing)."""

    def test_creation(self):
        router = SwitchRouter(hidden_size=64, num_experts=8)
        assert isinstance(router, nn.Module)

    def test_forward_shape(self):
        router = SwitchRouter(hidden_size=64, num_experts=8)
        x = torch.randn(16, 64)
        result = router(x)
        # Switch router uses top_k=1
        assert result["expert_indices"].shape == (16, 1)
        assert result["expert_weights"].shape == (16, 1)

    def test_capacity_factor(self):
        router = SwitchRouter(hidden_size=32, num_experts=4, capacity_factor=1.5)
        router.set_capacity_factor(2.0)
        x = torch.randn(8, 32)
        result = router(x)
        assert result["expert_indices"].shape[0] == 8


# =============================================================================
# Expert Network Tests
# =============================================================================


class TestFeedForwardExpert:
    """Tests for FeedForwardExpert."""

    def test_creation(self):
        expert = FeedForwardExpert(input_size=64, hidden_size=256)
        assert isinstance(expert, nn.Module)

    def test_forward_shape(self):
        expert = FeedForwardExpert(input_size=64, hidden_size=256)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)  # output_size defaults to input_size

    def test_custom_output_size(self):
        expert = FeedForwardExpert(input_size=64, hidden_size=256, output_size=128)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 128)

    def test_activation_functions(self):
        for act in ["relu", "gelu", "silu"]:
            expert = FeedForwardExpert(
                input_size=32, hidden_size=64, activation_fn=act
            )
            x = torch.randn(4, 32)
            out = expert(x)
            assert out.shape == (4, 32)

    def test_gradient_flow(self):
        expert = FeedForwardExpert(input_size=32, hidden_size=64)
        x = torch.randn(4, 32, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 32)


# =============================================================================
# MoE Layer Tests
# =============================================================================


class TestMoEConfig:
    """Tests for MoEConfig dataclass."""

    def test_defaults(self):
        config = MoEConfig()
        assert config.num_experts == 8
        assert config.top_k == 2
        assert config.capacity_factor == 1.25

    def test_custom_values(self):
        config = MoEConfig(num_experts=16, top_k=4, capacity_factor=2.0)
        assert config.num_experts == 16
        assert config.top_k == 4
        assert config.capacity_factor == 2.0


class TestMoELayer:
    """Tests for MoELayer."""

    def test_creation(self):
        config = MoEConfig(num_experts=4, top_k=2)
        layer = MoELayer(config=config, hidden_size=64)
        assert isinstance(layer, nn.Module)

    def test_forward_shape(self):
        config = MoEConfig(num_experts=4, top_k=2)
        layer = MoELayer(config=config, hidden_size=64)
        x = torch.randn(2, 8, 64)  # [batch, seq_len, hidden]
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_forward_with_router_logits(self):
        config = MoEConfig(num_experts=4, top_k=2)
        layer = MoELayer(config=config, hidden_size=64)
        x = torch.randn(2, 8, 64)
        out, aux = layer(x, return_router_logits=True)
        assert out.shape == (2, 8, 64)
        assert isinstance(aux, dict)

    def test_gradient_flow(self):
        config = MoEConfig(num_experts=4, top_k=2)
        layer = MoELayer(config=config, hidden_size=32)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_expert_utilization_stats(self):
        config = MoEConfig(num_experts=4, top_k=2)
        layer = MoELayer(config=config, hidden_size=32)
        x = torch.randn(2, 8, 32)
        layer(x)
        stats = layer.get_expert_utilization_stats()
        assert isinstance(stats, dict)


class TestSwitchTransformerMoE:
    """Tests for SwitchTransformerMoE (top-1)."""

    def test_creation(self):
        config = MoEConfig(num_experts=4)
        layer = SwitchTransformerMoE(config=config, hidden_size=64)
        assert isinstance(layer, nn.Module)

    def test_forward_shape(self):
        config = MoEConfig(num_experts=4)
        layer = SwitchTransformerMoE(config=config, hidden_size=64)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)


class TestGLaMStyleMoE:
    """Tests for GLaMStyleMoE."""

    def test_creation(self):
        config = MoEConfig(num_experts=4, top_k=2)
        layer = GLaMStyleMoE(config=config, hidden_size=64)
        assert isinstance(layer, nn.Module)

    def test_forward_shape(self):
        config = MoEConfig(num_experts=4, top_k=2)
        layer = GLaMStyleMoE(config=config, hidden_size=64)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)


class TestCreateMoELayer:
    """Tests for the create_moe_layer factory function."""

    def test_standard(self):
        layer = create_moe_layer("standard", hidden_size=64, num_experts=4, top_k=2)
        assert isinstance(layer, MoELayer)

    def test_switch(self):
        layer = create_moe_layer("switch", hidden_size=64, num_experts=4)
        assert isinstance(layer, SwitchTransformerMoE)

    def test_glam(self):
        layer = create_moe_layer("glam", hidden_size=64, num_experts=4, top_k=2)
        assert isinstance(layer, GLaMStyleMoE)


# =============================================================================
# Load Balancer Tests
# =============================================================================


class TestLoadBalancer:
    """Tests for LoadBalancer."""

    def test_creation(self):
        lb = LoadBalancer(num_experts=8)
        assert isinstance(lb, LoadBalancer)

    def test_compute_load_balance_loss(self):
        lb = LoadBalancer(num_experts=4)
        router_probs = torch.softmax(torch.randn(16, 4), dim=-1)
        expert_indices = torch.randint(0, 4, (16, 2))
        loss = lb.compute_load_balance_loss(router_probs, expert_indices)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_capacity_info(self):
        lb = LoadBalancer(num_experts=4, capacity_factor=1.25)
        router_probs = torch.softmax(torch.randn(16, 4), dim=-1)
        info = lb.get_capacity_info(router_probs, num_tokens=16)
        assert "expert_capacities" in info

    def test_balance_metrics(self):
        lb = LoadBalancer(num_experts=4)
        router_probs = torch.softmax(torch.randn(32, 4), dim=-1)
        expert_indices = torch.randint(0, 4, (32, 2))
        metrics = lb.get_expert_balance_metrics(router_probs, expert_indices)
        assert isinstance(metrics, dict)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMoEIntegration:
    """End-to-end MoE integration tests."""

    def test_moe_in_sequential(self):
        config = MoEConfig(num_experts=4, top_k=2)
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        moe = MoELayer(config=config, hidden_size=64)

        x = torch.randn(2, 8, 32)
        hidden = model(x)
        out = moe(hidden)
        assert out.shape == (2, 8, 64)

    def test_moe_training_step(self):
        config = MoEConfig(num_experts=4, top_k=2)
        moe = MoELayer(config=config, hidden_size=32)
        optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)

        x = torch.randn(2, 4, 32)
        target = torch.randn(2, 4, 32)

        # Training step
        out, aux = moe(x, return_router_logits=True)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss.item() > 0

    def test_different_batch_sizes(self):
        config = MoEConfig(num_experts=4, top_k=2)
        moe = MoELayer(config=config, hidden_size=32)

        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, 4, 32)
            out = moe(x)
            assert out.shape == (batch_size, 4, 32)
