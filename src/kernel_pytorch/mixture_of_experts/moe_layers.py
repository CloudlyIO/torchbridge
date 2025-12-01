"""
Mixture of Experts Layer Implementations

Advanced MoE layer implementations based on 2025-2026 research:
- Sparse MoE with configurable expert routing
- Switch Transformer style routing
- GLaM-style expert parallelism
- Memory-efficient expert switching
- Load balancing and capacity factor optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union, List
import math
from dataclasses import dataclass

from .expert_networks import FeedForwardExpert, ParameterEfficientExpert
from .routing import TopKRouter, SwitchRouter, DynamicCapacityRouter
from .optimization import LoadBalancer, ExpertParallelism


@dataclass
class MoEConfig:
    """Configuration for MoE layers"""
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    min_capacity: int = 4
    expert_dropout: float = 0.1
    router_dropout: float = 0.1
    load_balance_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    expert_parallelism: bool = False
    use_bias: bool = True
    activation_fn: str = "relu"
    normalize_router_probs: bool = True


class MoELayer(nn.Module):
    """
    Base Mixture of Experts layer with configurable routing and expert types

    This implementation supports:
    - Multiple routing strategies (Top-K, Switch, Hash)
    - Various expert architectures
    - Load balancing and capacity management
    - Expert parallelism for distributed training
    """

    def __init__(
        self,
        config: MoEConfig,
        hidden_size: int,
        expert_hidden_size: Optional[int] = None,
        expert_class: type = FeedForwardExpert,
        router_class: type = TopKRouter,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.config = config
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size or hidden_size * 4
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        factory_kwargs = {'device': device, 'dtype': dtype}

        # Create experts
        self.experts = nn.ModuleList([
            expert_class(
                input_size=hidden_size,
                hidden_size=self.expert_hidden_size,
                output_size=hidden_size,
                dropout=config.expert_dropout,
                activation_fn=config.activation_fn,
                use_bias=config.use_bias,
                **factory_kwargs
            )
            for _ in range(self.num_experts)
        ])

        # Create router
        self.router = router_class(
            hidden_size=hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            dropout=config.router_dropout,
            normalize_probs=config.normalize_router_probs,
            **factory_kwargs
        )

        # Load balancer for expert utilization
        self.load_balancer = LoadBalancer(
            num_experts=config.num_experts,
            capacity_factor=config.capacity_factor,
            min_capacity=config.min_capacity
        )

        # Expert parallelism handler
        if config.expert_parallelism:
            self.expert_parallelism = ExpertParallelism(
                num_experts=config.num_experts,
                experts=self.experts
            )
        else:
            self.expert_parallelism = None

        # Loss weights
        self.load_balance_loss_weight = config.load_balance_loss_weight
        self.router_z_loss_weight = config.router_z_loss_weight

        # Statistics for monitoring
        self.register_buffer('expert_usage_count', torch.zeros(self.num_experts))
        self.register_buffer('total_tokens_processed', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        return_router_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through MoE layer

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            expert_mask: Optional mask for expert availability
            return_router_logits: Whether to return router information

        Returns:
            output: Processed tensor
            aux_losses: Dictionary of auxiliary losses if return_router_logits=True
        """
        original_shape = x.shape
        batch_size, seq_len, hidden_size = original_shape

        # Flatten input for easier processing
        x_flat = x.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        num_tokens = x_flat.size(0)

        # Route tokens to experts
        router_output = self.router(x_flat, expert_mask)
        router_probs = router_output['probs']  # [num_tokens, num_experts]
        router_logits = router_output['logits']  # [num_tokens, num_experts]
        expert_indices = router_output['expert_indices']  # [num_tokens, top_k]
        expert_weights = router_output['expert_weights']  # [num_tokens, top_k]

        # Apply load balancing
        capacity_info = self.load_balancer.get_capacity_info(
            router_probs, num_tokens, expert_mask
        )

        # Process tokens through experts
        if self.expert_parallelism is not None:
            # Distributed expert processing
            expert_outputs = self.expert_parallelism.forward(
                x_flat, expert_indices, expert_weights, capacity_info
            )
        else:
            # Local expert processing
            expert_outputs = self._process_experts_locally(
                x_flat, expert_indices, expert_weights, capacity_info
            )

        # Reshape output
        output = expert_outputs.view(original_shape)

        # Update usage statistics
        if self.training:
            self._update_statistics(expert_indices, num_tokens)

        if return_router_logits:
            aux_losses = self._compute_auxiliary_losses(
                router_logits, router_probs, expert_indices, capacity_info
            )
            return output, aux_losses

        return output

    def _process_experts_locally(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process tokens through experts locally (non-distributed)"""
        num_tokens, hidden_size = x.shape
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)
            if not expert_mask.any():
                continue

            # Get tokens for this expert (considering capacity)
            token_indices, expert_positions = torch.where(expert_mask)

            # Apply capacity constraints
            max_capacity = capacity_info['expert_capacities'][expert_idx].item()
            if len(token_indices) > max_capacity:
                # Randomly select tokens within capacity (can be improved with better selection)
                perm = torch.randperm(len(token_indices))[:max_capacity]
                token_indices = token_indices[perm]
                expert_positions = expert_positions[perm]

            if len(token_indices) == 0:
                continue

            # Get input tokens and weights for this expert
            expert_input = x[token_indices]  # [num_selected_tokens, hidden_size]
            weights = expert_weights[token_indices, expert_positions]  # [num_selected_tokens]

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Apply routing weights and accumulate
            weighted_output = expert_output * weights.unsqueeze(-1)
            output[token_indices] += weighted_output

        return output

    def _compute_auxiliary_losses(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        capacity_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training"""
        aux_losses = {}

        # Load balancing loss
        if self.load_balance_loss_weight > 0:
            load_balance_loss = self.load_balancer.compute_load_balance_loss(
                router_probs, expert_indices
            )
            aux_losses['load_balance_loss'] = load_balance_loss * self.load_balance_loss_weight

        # Router z-loss (encourages router to be confident)
        if self.router_z_loss_weight > 0:
            router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
            aux_losses['router_z_loss'] = router_z_loss * self.router_z_loss_weight

        # Expert utilization statistics
        aux_losses['expert_utilization'] = capacity_info['expert_utilization']
        aux_losses['tokens_dropped'] = capacity_info['tokens_dropped']

        return aux_losses

    def _update_statistics(self, expert_indices: torch.Tensor, num_tokens: int):
        """Update expert usage statistics"""
        with torch.no_grad():
            # Count expert usage
            for expert_idx in range(self.num_experts):
                count = (expert_indices == expert_idx).sum()
                self.expert_usage_count[expert_idx] += count

            self.total_tokens_processed += num_tokens

    def get_expert_utilization_stats(self) -> Dict[str, float]:
        """Get expert utilization statistics"""
        if self.total_tokens_processed == 0:
            return {'expert_balance': 1.0, 'expert_efficiency': 1.0}

        # Calculate expert balance (lower variance = better balance)
        usage_rates = self.expert_usage_count / self.total_tokens_processed.float()
        expert_balance = 1.0 / (1.0 + usage_rates.var().item())

        # Calculate expert efficiency (higher usage = better efficiency)
        expert_efficiency = usage_rates.mean().item()

        return {
            'expert_balance': expert_balance,
            'expert_efficiency': expert_efficiency,
            'usage_rates': usage_rates.cpu().tolist()
        }

    def reset_statistics(self):
        """Reset expert usage statistics"""
        self.expert_usage_count.zero_()
        self.total_tokens_processed.zero_()


class SparseMoELayer(MoELayer):
    """
    Sparse MoE layer with advanced sparsity patterns

    Features:
    - Configurable sparsity levels
    - Dynamic expert selection
    - Memory-efficient sparse routing
    """

    def __init__(
        self,
        config: MoEConfig,
        hidden_size: int,
        sparsity_level: float = 0.1,  # Fraction of experts to activate
        **kwargs
    ):
        # Adjust top_k based on sparsity level
        effective_top_k = max(1, int(config.num_experts * sparsity_level))
        config.top_k = min(config.top_k, effective_top_k)

        super().__init__(config, hidden_size, **kwargs)
        self.sparsity_level = sparsity_level

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with sparse expert activation"""
        # Add sparsity-specific logic here if needed
        return super().forward(x, **kwargs)


class SwitchTransformerMoE(MoELayer):
    """
    Switch Transformer style MoE (top-1 routing)

    Based on "Switch Transformer: Scaling to Trillion Parameter Models"
    - Uses top-1 routing for simplicity
    - Implements capacity factor and load balancing
    - Optimized for large-scale training
    """

    def __init__(
        self,
        config: MoEConfig,
        hidden_size: int,
        **kwargs
    ):
        # Force top-1 routing for Switch Transformer
        config.top_k = 1
        config.capacity_factor = kwargs.get('capacity_factor', 1.0)

        super().__init__(
            config,
            hidden_size,
            router_class=SwitchRouter,
            **kwargs
        )


class GLaMStyleMoE(MoELayer):
    """
    GLaM-style MoE with parameter-efficient experts

    Based on "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"
    - Uses parameter-efficient experts
    - Implements expert parallelism
    - Optimized for memory efficiency
    """

    def __init__(
        self,
        config: MoEConfig,
        hidden_size: int,
        **kwargs
    ):
        # Use parameter-efficient experts
        config.expert_parallelism = True

        super().__init__(
            config,
            hidden_size,
            expert_class=ParameterEfficientExpert,
            **kwargs
        )


class AdaptiveMoELayer(MoELayer):
    """
    Adaptive MoE layer that adjusts routing based on input complexity

    Features:
    - Dynamic capacity factor based on input statistics
    - Adaptive expert selection
    - Complexity-aware routing
    """

    def __init__(
        self,
        config: MoEConfig,
        hidden_size: int,
        adaptation_rate: float = 0.01,
        **kwargs
    ):
        super().__init__(
            config,
            hidden_size,
            router_class=DynamicCapacityRouter,
            **kwargs
        )

        self.adaptation_rate = adaptation_rate
        self.register_buffer('input_complexity_ema', torch.tensor(1.0))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with adaptive routing"""
        # Compute input complexity (variance as proxy)
        input_complexity = x.var(dim=-1).mean()

        # Update EMA of input complexity
        if self.training:
            self.input_complexity_ema = (
                (1 - self.adaptation_rate) * self.input_complexity_ema +
                self.adaptation_rate * input_complexity
            )

        # Adjust capacity factor based on complexity
        complexity_factor = torch.clamp(
            self.input_complexity_ema / 1.0,  # Normalize to some baseline
            0.5, 2.0
        )

        # Update router with adaptive capacity
        if hasattr(self.router, 'set_capacity_factor'):
            self.router.set_capacity_factor(
                self.config.capacity_factor * complexity_factor.item()
            )

        return super().forward(x, **kwargs)


def create_moe_layer(
    moe_type: str,
    hidden_size: int,
    num_experts: int = 8,
    top_k: int = 2,
    **kwargs
) -> MoELayer:
    """
    Factory function to create MoE layers

    Args:
        moe_type: Type of MoE ("standard", "sparse", "switch", "glam", "adaptive")
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        top_k: Number of experts to route to
        **kwargs: Additional configuration options

    Returns:
        MoE layer instance
    """
    config = MoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        **{k: v for k, v in kwargs.items() if hasattr(MoEConfig, k)}
    )

    if moe_type == "standard":
        return MoELayer(config, hidden_size)
    elif moe_type == "sparse":
        return SparseMoELayer(config, hidden_size, **kwargs)
    elif moe_type == "switch":
        return SwitchTransformerMoE(config, hidden_size, **kwargs)
    elif moe_type == "glam":
        return GLaMStyleMoE(config, hidden_size, **kwargs)
    elif moe_type == "adaptive":
        return AdaptiveMoELayer(config, hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown MoE type: {moe_type}")


if __name__ == "__main__":
    # Example usage
    hidden_size = 768
    batch_size = 4
    seq_len = 512

    # Test different MoE types
    moe_types = ["standard", "sparse", "switch", "glam", "adaptive"]

    for moe_type in moe_types:
        print(f"\nTesting {moe_type} MoE:")

        moe = create_moe_layer(
            moe_type=moe_type,
            hidden_size=hidden_size,
            num_experts=8,
            top_k=2 if moe_type != "switch" else 1
        )

        x = torch.randn(batch_size, seq_len, hidden_size)

        if torch.cuda.is_available():
            x = x.cuda()
            moe = moe.cuda()

        # Forward pass
        if moe_type in ["standard", "sparse"]:
            output, aux_losses = moe(x, return_router_logits=True)
            print(f"  Output shape: {output.shape}")
            print(f"  Aux losses: {list(aux_losses.keys())}")
        else:
            output = moe(x)
            print(f"  Output shape: {output.shape}")

        # Check expert utilization
        stats = moe.get_expert_utilization_stats()
        print(f"  Expert balance: {stats['expert_balance']:.3f}")
        print(f"  Expert efficiency: {stats['expert_efficiency']:.3f}")