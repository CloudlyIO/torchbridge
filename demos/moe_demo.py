"""
Mixture of Experts (MoE) Demo

Demonstrates the MoE implementation including:
- Different MoE layer types (Standard, Sparse, Switch, GLaM, Adaptive)
- Various routing strategies (TopK, Switch, Hash, Learned, Dynamic)
- Expert networks and architectures
- Load balancing and capacity management
- Training with auxiliary losses
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def demo_basic_moe():
    """Demo 1: Basic MoE Layer Creation and Forward Pass"""
    print("\n" + "=" * 60)
    print("Demo 1: Basic MoE Layer")
    print("=" * 60)

    from torchbridge import MoEConfig, MoELayer, create_moe

    # Create configuration
    config = MoEConfig(
        num_experts=8,
        top_k=2,
        capacity_factor=1.25,
        load_balance_loss_weight=0.01,
    )
    print(f"Config: {config.num_experts} experts, top-{config.top_k} routing")

    # Create MoE layer
    hidden_size = 256
    layer = MoELayer(config, hidden_size=hidden_size)
    print(f"Created MoE layer with {len(layer.experts)} experts")

    # Forward pass
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, hidden_size)

    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Using convenience function
    layer2 = create_moe(hidden_size=256, num_experts=8, top_k=2)
    print(f"\nUsing create_moe convenience function: {len(layer2.experts)} experts")


def demo_moe_types():
    """Demo 2: Different MoE Layer Types"""
    print("\n" + "=" * 60)
    print("Demo 2: Different MoE Layer Types")
    print("=" * 60)

    from torchbridge import (
        GLaMStyleMoE,
        MoEConfig,
        MoELayer,
        SparseMoELayer,
        SwitchTransformerMoE,
        create_moe_layer,
    )
    from torchbridge.mixture_of_experts.moe_layers import AdaptiveMoELayer

    config = MoEConfig(num_experts=8, top_k=2)
    hidden_size = 256
    x = torch.randn(2, 16, hidden_size)

    # Standard MoE
    standard_moe = MoELayer(config, hidden_size)
    out = standard_moe(x)
    print(f"Standard MoE output: {out.shape}")

    # Sparse MoE (activates fewer experts)
    sparse_moe = SparseMoELayer(config, hidden_size, sparsity_level=0.25)
    out = sparse_moe(x)
    print(f"Sparse MoE (25% sparsity) output: {out.shape}")

    # Switch Transformer (top-1 routing)
    switch_config = MoEConfig(num_experts=8)
    switch_moe = SwitchTransformerMoE(switch_config, hidden_size)
    out = switch_moe(x)
    print(f"Switch Transformer (top-1) output: {out.shape}")

    # GLaM-style (parameter-efficient experts)
    glam_moe = GLaMStyleMoE(config, hidden_size)
    out = glam_moe(x)
    print(f"GLaM-style MoE output: {out.shape}")

    # Adaptive MoE (dynamic expert selection)
    adaptive_moe = AdaptiveMoELayer(config, hidden_size)
    out = adaptive_moe(x)
    print(f"Adaptive MoE output: {out.shape}")

    # Using factory function
    for moe_type in ["standard", "sparse", "switch", "glam", "adaptive"]:
        layer = create_moe_layer(
            moe_type=moe_type,
            hidden_size=hidden_size,
            num_experts=8,
            top_k=2
        )
        print(f"  Factory '{moe_type}': {type(layer).__name__}")


def demo_routing_strategies():
    """Demo 3: Different Routing Strategies"""
    print("\n" + "=" * 60)
    print("Demo 3: Routing Strategies")
    print("=" * 60)

    from torchbridge import SwitchRouter, TopKRouter
    from torchbridge.mixture_of_experts.routing import (
        DynamicCapacityRouter,
        HashRouter,
        LearnedRouter,
        create_router,
    )

    hidden_size = 256
    num_experts = 8
    num_tokens = 100
    x = torch.randn(num_tokens, hidden_size)

    # TopK Router (standard)
    topk_router = TopKRouter(hidden_size, num_experts, top_k=2)
    output = topk_router(x)
    print("TopK Router:")
    print(f"  Expert indices shape: {output['expert_indices'].shape}")
    print(f"  Expert weights shape: {output['expert_weights'].shape}")

    # Switch Router (top-1)
    switch_router = SwitchRouter(hidden_size, num_experts)
    output = switch_router(x)
    print("Switch Router (top-1):")
    print(f"  Expert indices shape: {output['expert_indices'].shape}")

    # Hash Router (deterministic)
    hash_router = HashRouter(hidden_size, num_experts, top_k=2)
    output = hash_router(x)
    print("Hash Router:")
    print(f"  Deterministic expert assignment: {output['expert_indices'][:5].tolist()}")

    # Learned Router (neural network gating)
    learned_router = LearnedRouter(
        hidden_size, num_experts, top_k=2,
        router_hidden_size=128, num_router_layers=2
    )
    output = learned_router(x)
    print("Learned Router:")
    print(f"  Router features shape: {output['router_features'].shape}")

    # Dynamic Capacity Router
    dynamic_router = DynamicCapacityRouter(
        hidden_size, num_experts, top_k=2,
        initial_capacity_factor=1.25
    )
    output = dynamic_router(x)
    print("Dynamic Capacity Router:")
    print(f"  Complexity scores shape: {output['complexity_scores'].shape}")

    # Using factory
    for router_type in ["topk", "switch", "hash", "learned", "dynamic"]:
        router = create_router(
            router_type, hidden_size, num_experts,
            top_k=2 if router_type != "switch" else 1
        )
        print(f"  Factory '{router_type}': {type(router).__name__}")


def demo_expert_networks():
    """Demo 4: Different Expert Network Architectures"""
    print("\n" + "=" * 60)
    print("Demo 4: Expert Network Architectures")
    print("=" * 60)

    from torchbridge import FeedForwardExpert
    from torchbridge.mixture_of_experts.expert_networks import (
        AttentionExpert,
        ConvolutionalExpert,
        ParameterEfficientExpert,
    )

    hidden_size = 256
    x = torch.randn(100, hidden_size)

    # Standard FeedForward Expert
    ff_expert = FeedForwardExpert(
        input_size=hidden_size,
        hidden_size=hidden_size * 4,
        output_size=hidden_size,
        activation_fn="gelu"
    )
    out = ff_expert(x)
    params = sum(p.numel() for p in ff_expert.parameters())
    print(f"FeedForward Expert: {params:,} params, output shape: {out.shape}")

    # Convolutional Expert
    conv_expert = ConvolutionalExpert(
        input_size=hidden_size,
        hidden_size=hidden_size * 4,
        output_size=hidden_size,
        kernel_size=3
    )
    out = conv_expert(x)
    params = sum(p.numel() for p in conv_expert.parameters())
    print(f"Convolutional Expert: {params:,} params, output shape: {out.shape}")

    # Attention Expert
    attn_expert = AttentionExpert(
        input_size=hidden_size,
        hidden_size=hidden_size * 4,
        output_size=hidden_size,
        num_heads=4
    )
    out = attn_expert(x)
    params = sum(p.numel() for p in attn_expert.parameters())
    print(f"Attention Expert: {params:,} params, output shape: {out.shape}")

    # Parameter-Efficient Expert (smaller)
    pe_expert = ParameterEfficientExpert(
        input_size=hidden_size,
        hidden_size=hidden_size * 4,
        output_size=hidden_size,
        rank=32  # Low-rank approximation
    )
    out = pe_expert(x)
    params = sum(p.numel() for p in pe_expert.parameters())
    print(f"Parameter-Efficient Expert: {params:,} params, output shape: {out.shape}")


def demo_load_balancing():
    """Demo 5: Load Balancing and Capacity Management"""
    print("\n" + "=" * 60)
    print("Demo 5: Load Balancing")
    print("=" * 60)

    from torchbridge import LoadBalancer, MoEConfig, MoELayer

    # Create load balancer
    num_experts = 8
    load_balancer = LoadBalancer(
        num_experts=num_experts,
        capacity_factor=1.25,
        load_balance_loss_type="switch"  # Can be "switch", "gshard", "entropy"
    )

    # Simulate router output
    num_tokens = 1000
    router_probs = torch.softmax(torch.randn(num_tokens, num_experts), dim=-1)
    expert_indices = torch.randint(0, num_experts, (num_tokens, 2))

    # Get capacity info
    capacity_info = load_balancer.get_capacity_info(router_probs, num_tokens)
    print(f"Expert capacities: {capacity_info['expert_capacities'].tolist()}")
    print(f"Tokens dropped: {capacity_info['tokens_dropped'].item():.0f}")

    # Compute load balance loss
    loss = load_balancer.compute_load_balance_loss(router_probs, expert_indices)
    print(f"Load balance loss (switch): {loss.item():.4f}")

    # Get balance metrics
    metrics = load_balancer.get_expert_balance_metrics(router_probs, expert_indices)
    print(f"Assignment entropy: {metrics['assignment_entropy']:.3f}")
    print(f"Max/Min assignment ratio: {metrics['max_assignment_ratio']:.2f}/{metrics['min_assignment_ratio']:.2f}")

    # Using MoE with auxiliary losses
    config = MoEConfig(num_experts=8, load_balance_loss_weight=0.01)
    moe = MoELayer(config, hidden_size=256)
    x = torch.randn(4, 32, 256)

    output, aux_losses = moe(x, return_router_logits=True)
    print("\nMoE with auxiliary losses:")
    for name, loss in aux_losses.items():
        if loss.numel() == 1:  # Only scalar losses
            print(f"  {name}: {loss.item():.6f}")


def demo_training():
    """Demo 6: Training MoE Models"""
    print("\n" + "=" * 60)
    print("Demo 6: Training MoE Models")
    print("=" * 60)

    from torchbridge import MoEConfig, MoELayer

    # Simple MoE model
    class MoEModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_experts=8):
            super().__init__()
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.moe = MoELayer(
                MoEConfig(num_experts=num_experts, load_balance_loss_weight=0.01),
                hidden_size
            )
            self.output_proj = nn.Linear(hidden_size, output_size)

        def forward(self, x, return_aux_loss=False):
            x = self.input_proj(x)
            if return_aux_loss:
                x, aux_losses = self.moe(x, return_router_logits=True)
                x = self.output_proj(x)
                return x, aux_losses
            else:
                x = self.moe(x)
                x = self.output_proj(x)
                return x

    # Create model
    model = MoEModel(input_size=64, hidden_size=256, output_size=10, num_experts=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    print("Training for 5 steps...")

    for step in range(5):
        # Random batch
        x = torch.randn(8, 16, 64)
        targets = torch.randint(0, 10, (8, 16))

        # Forward with auxiliary losses
        logits, aux_losses = model(x, return_aux_loss=True)

        # Compute main loss
        main_loss = F.cross_entropy(logits.view(-1, 10), targets.view(-1))

        # Total loss includes auxiliary losses (only scalar losses)
        total_loss = main_loss
        aux_loss_sum = 0.0
        for name, aux_loss in aux_losses.items():
            if aux_loss.numel() == 1:  # Only scalar losses
                total_loss = total_loss + aux_loss
                aux_loss_sum += aux_loss.item()

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}: main_loss={main_loss.item():.4f}, "
              f"aux_loss={aux_loss_sum:.6f}")

    # Check expert utilization
    stats = model.moe.get_expert_utilization_stats()
    print("\nExpert utilization after training:")
    print(f"  Balance: {stats['expert_balance']:.3f}")
    print(f"  Efficiency: {stats['expert_efficiency']:.3f}")


def demo_transformer_integration():
    """Demo 7: MoE in Transformer Architecture"""
    print("\n" + "=" * 60)
    print("Demo 7: MoE in Transformer")
    print("=" * 60)

    from torchbridge import MoEConfig, MoELayer

    class MoETransformerBlock(nn.Module):
        """Transformer block with MoE FFN"""
        def __init__(self, hidden_size, num_heads, num_experts=8):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads, batch_first=True
            )
            self.moe = MoELayer(
                MoEConfig(num_experts=num_experts, top_k=2),
                hidden_size
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

        def forward(self, x, return_aux_loss=False):
            # Self-attention with residual
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # MoE FFN with residual
            if return_aux_loss:
                moe_out, aux_losses = self.moe(x, return_router_logits=True)
                x = self.norm2(x + moe_out)
                return x, aux_losses
            else:
                moe_out = self.moe(x)
                x = self.norm2(x + moe_out)
                return x

    class MoETransformer(nn.Module):
        """Simple MoE Transformer"""
        def __init__(self, vocab_size, hidden_size, num_heads, num_layers,
                     num_experts=8, max_seq_len=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
            self.layers = nn.ModuleList([
                MoETransformerBlock(hidden_size, num_heads, num_experts)
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, return_aux_loss=False):
            seq_len = input_ids.size(1)
            positions = torch.arange(seq_len, device=input_ids.device)

            x = self.embedding(input_ids) + self.pos_embedding(positions)

            all_aux_losses = {}
            for i, layer in enumerate(self.layers):
                if return_aux_loss:
                    x, aux_losses = layer(x, return_aux_loss=True)
                    for name, loss in aux_losses.items():
                        all_aux_losses[f"layer_{i}_{name}"] = loss
                else:
                    x = layer(x)

            logits = self.output(x)

            if return_aux_loss:
                return logits, all_aux_losses
            return logits

    # Create model
    model = MoETransformer(
        vocab_size=1000,
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        num_experts=8
    )

    # Forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    logits, aux_losses = model(input_ids, return_aux_loss=True)

    print("MoE Transformer:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Auxiliary losses: {len(aux_losses)} items")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    moe_params = sum(
        sum(p.numel() for p in layer.moe.parameters())
        for layer in model.layers
    )
    print(f"  Total params: {total_params:,}")
    print(f"  MoE params: {moe_params:,} ({100*moe_params/total_params:.1f}%)")


def demo_performance():
    """Demo 8: Performance Comparison"""
    print("\n" + "=" * 60)
    print("Demo 8: Performance Comparison")
    print("=" * 60)

    from torchbridge import MoEConfig, MoELayer

    hidden_size = 512
    batch_size = 8
    seq_len = 128

    # Standard FFN
    class StandardFFN(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
            self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    # MoE layer
    config = MoEConfig(num_experts=8, top_k=2)
    moe_layer = MoELayer(config, hidden_size)
    ffn_layer = StandardFFN(hidden_size)

    x = torch.randn(batch_size, seq_len, hidden_size)

    # Warm-up
    for _ in range(3):
        _ = moe_layer(x)
        _ = ffn_layer(x)

    # Benchmark
    num_iterations = 50

    start = time.time()
    for _ in range(num_iterations):
        _ = moe_layer(x)
    moe_time = (time.time() - start) / num_iterations * 1000

    start = time.time()
    for _ in range(num_iterations):
        _ = ffn_layer(x)
    ffn_time = (time.time() - start) / num_iterations * 1000

    # Parameter counts
    moe_params = sum(p.numel() for p in moe_layer.parameters())
    ffn_params = sum(p.numel() for p in ffn_layer.parameters())

    print(f"Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print("\nStandard FFN:")
    print(f"  Parameters: {ffn_params:,}")
    print(f"  Time: {ffn_time:.2f} ms")

    print("\nMoE (8 experts, top-2):")
    print(f"  Parameters: {moe_params:,}")
    print(f"  Time: {moe_time:.2f} ms")
    print(f"  Parameter ratio: {moe_params/ffn_params:.1f}x")
    print(f"  Time ratio: {moe_time/ffn_time:.1f}x")

    # Note: MoE has more parameters but similar compute per token
    # due to sparse expert activation


def main():
    """Run all demos"""
    print("=" * 60)
    print("Mixture of Experts (MoE) Demonstrations")
    print("torchbridge v0.5.0")
    print("=" * 60)

    demos = [
        ("Basic MoE", demo_basic_moe),
        ("MoE Types", demo_moe_types),
        ("Routing Strategies", demo_routing_strategies),
        ("Expert Networks", demo_expert_networks),
        ("Load Balancing", demo_load_balancing),
        ("Training", demo_training),
        ("Transformer Integration", demo_transformer_integration),
        ("Performance", demo_performance),
    ]

    for name, demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\n[ERROR] {name} demo failed: {e}")

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
