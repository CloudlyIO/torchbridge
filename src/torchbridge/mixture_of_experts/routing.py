"""
MoE Routing Strategies

Various routing mechanisms for Mixture of Experts:
- Top-K routing with learned gating
- Switch routing (top-1 with capacity)
- Hash routing for deterministic assignment
- Learned routing with neural networks
- Dynamic capacity routing
"""

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRouter(nn.Module):
    """Base class for MoE routers"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
        normalize_probs: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.dropout = dropout
        self.normalize_probs = normalize_probs

        factory_kwargs = {'device': device, 'dtype': dtype}

        # Routing network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, **factory_kwargs)

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize gate weights
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Base forward method - should be overridden by subclasses

        Args:
            x: Input tensor [num_tokens, hidden_size]
            expert_mask: Optional mask for expert availability

        Returns:
            Dictionary containing routing information
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def _apply_expert_mask(
        self,
        logits: torch.Tensor,
        expert_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply expert availability mask to logits"""
        if expert_mask is not None:
            # expert_mask should be [num_experts] or [batch_size, num_experts]
            if expert_mask.dim() == 1:
                expert_mask = expert_mask.unsqueeze(0)

            # Expand mask to match logits shape
            mask = expert_mask.expand_as(logits)
            logits = logits.masked_fill(~mask, float('-inf'))

        return logits


class TopKRouter(BaseRouter):
    """
    Top-K routing with learnable gating

    Routes each token to the top-k experts based on a learned gating function.
    This is the standard approach used in most MoE implementations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
        normalize_probs: bool = True,
        noise_epsilon: float = 1e-2,
        **kwargs
    ):
        # Only keep device/dtype in kwargs for BaseRouter
        base_kwargs = {k: v for k, v in kwargs.items() if k in ('device', 'dtype')}
        super().__init__(
            hidden_size, num_experts, top_k, dropout, normalize_probs, **base_kwargs
        )

        self.noise_epsilon = noise_epsilon

        # Noise generation for training (helps with load balancing)
        self.register_buffer(
            'noise_std',
            torch.ones(num_experts) * noise_epsilon
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Top-K routing forward pass

        Args:
            x: Input tensor [num_tokens, hidden_size]
            expert_mask: Optional expert availability mask

        Returns:
            Dictionary with routing information
        """
        x.size(0)

        # Compute routing logits
        if self.dropout_layer is not None and self.training:
            x_dropout = self.dropout_layer(x)
            logits = self.gate(x_dropout)
        else:
            logits = self.gate(x)  # [num_tokens, num_experts]

        # Apply expert mask
        logits = self._apply_expert_mask(logits, expert_mask)

        # Add noise during training for load balancing
        if self.training and self.noise_epsilon > 0:
            noise_std = self.noise_std.to(logits.device)
            noise = torch.randn_like(logits) * noise_std.unsqueeze(0)
            logits = logits + noise

        # Compute routing probabilities
        if self.normalize_probs:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize top-k probabilities
        if self.normalize_probs:
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return {
            'logits': logits,
            'probs': probs,
            'expert_indices': top_k_indices,  # [num_tokens, top_k]
            'expert_weights': top_k_probs,    # [num_tokens, top_k]
            'top_k_probs': top_k_probs,
            'top_k_indices': top_k_indices
        }


class SwitchRouter(BaseRouter):
    """
    Switch Transformer router (top-1 routing)

    Based on "Switch Transformer: Scaling to Trillion Parameter Models"
    Uses top-1 routing with capacity constraints and load balancing.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        jitter_noise: float = 1e-2,
        **kwargs
    ):
        # Remove top_k from kwargs if present (Switch always uses top_k=1)
        kwargs.pop('top_k', None)
        kwargs.pop('normalize_probs', None)
        super().__init__(
            hidden_size, num_experts, top_k=1, dropout=dropout, **kwargs
        )

        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Switch routing with capacity constraints"""
        x.size(0)

        # Compute routing logits
        logits = self.gate(x)  # [num_tokens, num_experts]

        # Apply expert mask
        logits = self._apply_expert_mask(logits, expert_mask)

        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.jitter_noise,
                1.0 + self.jitter_noise
            )
            logits = logits * noise

        # Compute probabilities and select top-1
        probs = F.softmax(logits, dim=-1)
        top1_probs, top1_indices = torch.max(probs, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'expert_indices': top1_indices.unsqueeze(-1),  # [num_tokens, 1]
            'expert_weights': top1_probs.unsqueeze(-1),    # [num_tokens, 1]
            'top_k_probs': top1_probs.unsqueeze(-1),
            'top_k_indices': top1_indices.unsqueeze(-1)
        }

    def set_capacity_factor(self, capacity_factor: float):
        """Dynamically adjust capacity factor"""
        self.capacity_factor = capacity_factor


class HashRouter(BaseRouter):
    """
    Hash-based routing for deterministic expert assignment

    Uses hash functions to deterministically assign tokens to experts.
    Useful for reproducible routing and avoiding learned routing biases.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 1,
        hash_function: str = "md5",
        use_token_position: bool = True,
        **kwargs
    ):
        # Extract device/dtype before removing other params
        factory_kwargs = {k: kwargs.get(k) for k in ('device', 'dtype') if k in kwargs}

        # Remove params that might conflict with BaseRouter
        kwargs.pop('dropout', None)
        kwargs.pop('normalize_probs', None)
        super().__init__(
            hidden_size, num_experts, top_k, **kwargs
        )

        self.hash_function = hash_function
        self.use_token_position = use_token_position

        # Note: Gate is not used for hash routing, but we keep it for interface compatibility
        # Instead, we use a simple projection for consistent interface
        self.token_projection = nn.Linear(hidden_size, 32, **factory_kwargs)  # Project to smaller space for hashing

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Hash-based routing"""
        num_tokens = x.size(0)

        # Project tokens to a smaller space for hashing
        projected = self.token_projection(x)

        # Generate hash-based expert assignments
        expert_indices = []
        expert_weights = []

        for i in range(num_tokens):
            token_repr = projected[i].detach().cpu().numpy().tobytes()

            if self.use_token_position:
                token_repr += str(i).encode()

            # Compute hash and map to expert
            if self.hash_function == "md5":
                hash_obj = hashlib.md5(token_repr)
            elif self.hash_function == "sha256":
                hash_obj = hashlib.sha256(token_repr)
            else:
                raise ValueError(f"Unsupported hash function: {self.hash_function}")

            hash_int = int(hash_obj.hexdigest(), 16)
            expert_idx = hash_int % self.num_experts

            # For top-k > 1, use additional hash iterations
            if self.top_k > 1:
                selected_experts = []
                for k in range(self.top_k):
                    hash_input = token_repr + str(k).encode()
                    hash_obj = hashlib.md5(hash_input)
                    hash_int = int(hash_obj.hexdigest(), 16)
                    expert_idx = hash_int % self.num_experts

                    # Avoid duplicates
                    while expert_idx in selected_experts and len(selected_experts) < self.num_experts:
                        hash_input = hash_input + b"_retry"
                        hash_obj = hashlib.md5(hash_input)
                        hash_int = int(hash_obj.hexdigest(), 16)
                        expert_idx = hash_int % self.num_experts

                    selected_experts.append(expert_idx)

                expert_indices.append(selected_experts)
                # Equal weights for hash routing
                expert_weights.append([1.0 / self.top_k] * self.top_k)
            else:
                expert_indices.append([expert_idx])
                expert_weights.append([1.0])

        # Convert to tensors
        expert_indices = torch.tensor(expert_indices, device=x.device)  # [num_tokens, top_k]
        expert_weights = torch.tensor(expert_weights, device=x.device)  # [num_tokens, top_k]

        # Create dummy logits and probs for interface compatibility
        logits = torch.zeros(num_tokens, self.num_experts, device=x.device)
        probs = torch.zeros(num_tokens, self.num_experts, device=x.device)

        # Set probabilities based on hash assignments
        for i in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = expert_indices[i, k].item()
                probs[i, expert_idx] = expert_weights[i, k].item()

        return {
            'logits': logits,
            'probs': probs,
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'top_k_probs': expert_weights,
            'top_k_indices': expert_indices
        }


class LearnedRouter(BaseRouter):
    """
    Advanced learned router with neural network gating

    Uses a more sophisticated neural network for routing decisions,
    potentially with multiple layers and attention mechanisms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_hidden_size: int = 512,
        num_router_layers: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
        **kwargs
    ):
        # Remove params already handled from kwargs to avoid conflicts
        kwargs.pop('normalize_probs', None)
        super().__init__(
            hidden_size, num_experts, top_k, dropout, **kwargs
        )

        self.router_hidden_size = router_hidden_size
        self.num_router_layers = num_router_layers
        self.use_attention = use_attention

        # Build routing network
        router_layers = []

        # Input layer
        router_layers.append(nn.Linear(hidden_size, router_hidden_size))
        router_layers.append(nn.ReLU())

        if dropout > 0:
            router_layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_router_layers - 1):
            router_layers.append(nn.Linear(router_hidden_size, router_hidden_size))
            router_layers.append(nn.ReLU())
            if dropout > 0:
                router_layers.append(nn.Dropout(dropout))

        self.router_network = nn.Sequential(*router_layers)

        # Attention mechanism for routing (optional)
        if use_attention:
            self.router_attention = nn.MultiheadAttention(
                embed_dim=router_hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )

        # Final routing projection (replace the simple gate)
        self.gate = nn.Linear(router_hidden_size, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Advanced learned routing"""
        x.size(0)

        # Process through routing network
        router_features = self.router_network(x)  # [num_tokens, router_hidden_size]

        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over router features
            # Add batch dimension for attention
            router_features_batched = router_features.unsqueeze(0)  # [1, num_tokens, hidden_size]

            attended_features, _ = self.router_attention(
                router_features_batched,
                router_features_batched,
                router_features_batched
            )

            router_features = attended_features.squeeze(0)  # [num_tokens, hidden_size]

        # Compute routing logits
        logits = self.gate(router_features)  # [num_tokens, num_experts]

        # Apply expert mask
        logits = self._apply_expert_mask(logits, expert_mask)

        # Compute probabilities and top-k selection
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return {
            'logits': logits,
            'probs': probs,
            'expert_indices': top_k_indices,
            'expert_weights': top_k_probs,
            'top_k_probs': top_k_probs,
            'top_k_indices': top_k_indices,
            'router_features': router_features  # Additional info
        }


class DynamicCapacityRouter(BaseRouter):
    """
    Dynamic capacity router that adapts based on input characteristics

    Automatically adjusts capacity factors and routing strategies based
    on input complexity and expert utilization patterns.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        initial_capacity_factor: float = 1.25,
        min_capacity_factor: float = 0.5,
        max_capacity_factor: float = 2.0,
        adaptation_rate: float = 0.01,
        complexity_threshold: float = 1.0,
        **kwargs
    ):
        # Extract device/dtype before removing other params
        factory_kwargs = {k: kwargs.get(k) for k in ('device', 'dtype') if k in kwargs}

        # Remove params that might conflict with BaseRouter
        kwargs.pop('dropout', None)
        kwargs.pop('normalize_probs', None)
        super().__init__(
            hidden_size, num_experts, top_k, **kwargs
        )

        self.initial_capacity_factor = initial_capacity_factor
        self.min_capacity_factor = min_capacity_factor
        self.max_capacity_factor = max_capacity_factor
        self.adaptation_rate = adaptation_rate
        self.complexity_threshold = complexity_threshold

        # Learnable capacity factors per expert
        self.register_buffer(
            'capacity_factors',
            torch.full((num_experts,), initial_capacity_factor)
        )

        # Input complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1, **factory_kwargs),
            nn.Sigmoid()
        )

        # Expert utilization tracker
        self.register_buffer(
            'expert_utilization_ema',
            torch.zeros(num_experts)
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Dynamic capacity routing"""
        num_tokens = x.size(0)

        # Analyze input complexity
        complexity_scores = self.complexity_analyzer(x).squeeze(-1)  # [num_tokens]
        avg_complexity = complexity_scores.mean()

        # Adjust capacity factors based on complexity
        if self.training:
            complexity_adjustment = torch.clamp(
                avg_complexity / self.complexity_threshold,
                self.min_capacity_factor / self.initial_capacity_factor,
                self.max_capacity_factor / self.initial_capacity_factor
            )

            # Update capacity factors with EMA (ensure device compatibility)
            capacity_on_device = self.capacity_factors.to(x.device)
            updated_capacity = (
                (1 - self.adaptation_rate) * capacity_on_device +
                self.adaptation_rate * self.initial_capacity_factor * complexity_adjustment
            )
            self.capacity_factors.data.copy_(updated_capacity.cpu())

        # Standard routing computation
        logits = self.gate(x)
        logits = self._apply_expert_mask(logits, expert_mask)

        # Apply complexity-based routing adjustments
        # Boost routing to less-utilized experts for complex inputs
        if self.training:
            expert_utilization = self.expert_utilization_ema.to(logits.device)
            utilization_penalty = expert_utilization.unsqueeze(0) * 0.1
            complexity_boost = complexity_scores.unsqueeze(-1) * (1.0 - expert_utilization.unsqueeze(0))
            logits = logits - utilization_penalty + complexity_boost

        # Compute probabilities and top-k selection
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Update expert utilization statistics
        if self.training:
            with torch.no_grad():
                current_utilization = torch.zeros(self.num_experts, device=x.device)
                for expert_idx in range(self.num_experts):
                    usage_mask = (top_k_indices == expert_idx)
                    current_utilization[expert_idx] = usage_mask.sum().float() / num_tokens

                # Update EMA (ensure device compatibility)
                ema_on_device = self.expert_utilization_ema.to(x.device)
                updated_ema = (
                    (1 - self.adaptation_rate) * ema_on_device +
                    self.adaptation_rate * current_utilization
                )
                self.expert_utilization_ema.data.copy_(updated_ema.cpu())

        return {
            'logits': logits,
            'probs': probs,
            'expert_indices': top_k_indices,
            'expert_weights': top_k_probs,
            'top_k_probs': top_k_probs,
            'top_k_indices': top_k_indices,
            'complexity_scores': complexity_scores,
            'capacity_factors': self.capacity_factors.clone(),
            'expert_utilization': self.expert_utilization_ema.clone()
        }

    def set_capacity_factor(self, capacity_factor: float):
        """Set capacity factor for all experts"""
        self.capacity_factors.fill_(capacity_factor)

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics"""
        return {
            'capacity_factors': self.capacity_factors.cpu().tolist(),
            'expert_utilization': self.expert_utilization_ema.cpu().tolist(),
            'avg_capacity_factor': self.capacity_factors.mean().item(),
            'capacity_variance': self.capacity_factors.var().item(),
            'utilization_variance': self.expert_utilization_ema.var().item()
        }


def create_router(
    router_type: str,
    hidden_size: int,
    num_experts: int,
    top_k: int = 2,
    **kwargs
) -> BaseRouter:
    """
    Factory function to create routers

    Args:
        router_type: Type of router ("topk", "switch", "hash", "learned", "dynamic")
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        top_k: Number of experts to route to
        **kwargs: Additional router-specific arguments

    Returns:
        Router instance
    """
    if router_type == "topk":
        return TopKRouter(hidden_size, num_experts, top_k, **kwargs)
    elif router_type == "switch":
        return SwitchRouter(hidden_size, num_experts, **kwargs)
    elif router_type == "hash":
        return HashRouter(hidden_size, num_experts, top_k, **kwargs)
    elif router_type == "learned":
        return LearnedRouter(hidden_size, num_experts, top_k, **kwargs)
    elif router_type == "dynamic":
        return DynamicCapacityRouter(hidden_size, num_experts, top_k, **kwargs)
    else:
        raise ValueError(f"Unknown router type: {router_type}")


if __name__ == "__main__":
    # Test different router types
    hidden_size = 768
    num_experts = 8
    top_k = 2
    num_tokens = 100

    router_types = ["topk", "switch", "hash", "learned", "dynamic"]

    x = torch.randn(num_tokens, hidden_size)

    for router_type in router_types:
        print(f"\nTesting {router_type} router:")

        router = create_router(
            router_type=router_type,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k if router_type != "switch" else 1
        )

        if torch.cuda.is_available():
            x_test = x.cuda()
            router = router.cuda()
        else:
            x_test = x

        # Test routing
        router_output = router(x_test)

        print(f"  Logits shape: {router_output['logits'].shape}")
        print(f"  Probs shape: {router_output['probs'].shape}")
        print(f"  Expert indices shape: {router_output['expert_indices'].shape}")
        print(f"  Expert weights shape: {router_output['expert_weights'].shape}")

        # Check probability normalization
        if router_type in ["topk", "switch", "learned"]:
            total_prob = router_output['probs'].sum(dim=-1)
            print(f"  Prob normalization check (should be ~1.0): {total_prob.mean().item():.4f}")

        # Special checks for dynamic router
        if router_type == "dynamic":
            stats = router.get_routing_stats()
            print(f"  Avg capacity factor: {stats['avg_capacity_factor']:.3f}")
            print(f"  Utilization variance: {stats['utilization_variance']:.6f}")

    print("\nAll router types tested successfully!")
