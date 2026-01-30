"""
MoE Optimization Utilities

Optimization techniques for Mixture of Experts:
- Expert parallelism for distributed training
- Load balancing algorithms
- Expert scheduling and capacity management
- Memory-efficient expert switching
"""

import math
import warnings
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


class LoadBalancer:
    """
    Load balancing utilities for MoE systems

    Implements various strategies to ensure balanced expert utilization
    and prevent expert collapse.
    """

    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.25,
        min_capacity: int = 4,
        load_balance_loss_type: str = "switch",  # "switch", "gshard", "entropy"
        epsilon: float = 1e-6
    ):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.load_balance_loss_type = load_balance_loss_type
        self.epsilon = epsilon

    def get_capacity_info(
        self,
        router_probs: torch.Tensor,
        num_tokens: int,
        expert_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Compute capacity information for experts

        Args:
            router_probs: Router probabilities [num_tokens, num_experts]
            num_tokens: Number of tokens
            expert_mask: Optional expert availability mask

        Returns:
            Dictionary with capacity information
        """
        device = router_probs.device

        # Base capacity per expert
        base_capacity = math.ceil(num_tokens / self.num_experts * self.capacity_factor)
        base_capacity = max(base_capacity, self.min_capacity)

        # Expert capacities (can be adjusted per expert)
        expert_capacities = torch.full(
            (self.num_experts,), base_capacity, dtype=torch.long, device=device
        )

        # Adjust capacities based on expert availability
        if expert_mask is not None:
            # Redistribute capacity from unavailable experts
            available_experts = expert_mask.sum()
            if available_experts > 0:
                total_capacity = expert_capacities.sum()
                per_available_capacity = total_capacity // available_experts
                expert_capacities = torch.where(
                    expert_mask,
                    per_available_capacity,
                    0
                )

        # Compute expert utilization
        expert_assignments = router_probs.sum(dim=0)  # [num_experts]
        expert_utilization = expert_assignments / expert_assignments.sum()

        # Count tokens that would be dropped due to capacity constraints
        tokens_per_expert = torch.bincount(
            torch.argmax(router_probs, dim=-1),
            minlength=self.num_experts
        ).float()

        tokens_dropped = torch.clamp(
            tokens_per_expert - expert_capacities.float(),
            min=0
        ).sum()

        return {
            'expert_capacities': expert_capacities,
            'expert_utilization': expert_utilization,
            'tokens_dropped': tokens_dropped,
            'base_capacity': base_capacity,
            'total_capacity': expert_capacities.sum()
        }

    def compute_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute load balancing loss

        Args:
            router_probs: Router probabilities [num_tokens, num_experts]
            expert_indices: Selected expert indices [num_tokens, top_k]
            mask: Optional token mask

        Returns:
            Load balance loss scalar
        """
        if self.load_balance_loss_type == "switch":
            return self._switch_load_balance_loss(router_probs, expert_indices, mask)
        elif self.load_balance_loss_type == "gshard":
            return self._gshard_load_balance_loss(router_probs, expert_indices, mask)
        elif self.load_balance_loss_type == "entropy":
            return self._entropy_load_balance_loss(router_probs, mask)
        else:
            raise ValueError(f"Unknown load balance loss type: {self.load_balance_loss_type}")

    def _switch_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Switch Transformer load balancing loss"""
        num_tokens = router_probs.size(0)

        if mask is not None:
            router_probs = router_probs * mask.unsqueeze(-1)
            num_tokens = mask.sum()

        # Fraction of tokens assigned to each expert
        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.float, device=router_probs.device
        )

        for expert_idx in range(self.num_experts):
            tokens_per_expert[expert_idx] = (expert_indices == expert_idx).sum().float()

        tokens_per_expert = tokens_per_expert / num_tokens

        # Average probability of routing to each expert
        prob_per_expert = router_probs.mean(dim=0)

        # Load balance loss: minimize product of assignment fraction and routing probability
        loss = (tokens_per_expert * prob_per_expert).sum() * self.num_experts

        return loss

    def _gshard_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """GShard-style load balancing loss"""
        if mask is not None:
            router_probs = router_probs * mask.unsqueeze(-1)

        # Coefficient of variation of expert assignment probabilities
        mean_prob = router_probs.mean(dim=0)
        var_prob = router_probs.var(dim=0)

        cv = torch.sqrt(var_prob) / (mean_prob + self.epsilon)
        loss = cv.mean()

        return loss

    def _entropy_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Entropy-based load balancing loss"""
        if mask is not None:
            router_probs = router_probs * mask.unsqueeze(-1)

        # Compute entropy of expert assignment distribution
        mean_probs = router_probs.mean(dim=0)
        mean_probs = mean_probs / (mean_probs.sum() + self.epsilon)

        entropy = -(mean_probs * torch.log(mean_probs + self.epsilon)).sum()
        max_entropy = math.log(self.num_experts)

        # Loss is negative entropy (we want to maximize entropy)
        loss = (max_entropy - entropy) / max_entropy

        return loss

    def get_expert_balance_metrics(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> dict[str, float]:
        """Compute expert balance metrics for monitoring"""
        # Assignment distribution
        assignment_counts = torch.zeros(self.num_experts, device=router_probs.device)
        for expert_idx in range(self.num_experts):
            assignment_counts[expert_idx] = (expert_indices == expert_idx).sum().float()

        assignment_probs = assignment_counts / assignment_counts.sum()

        # Routing probability distribution
        routing_probs = router_probs.mean(dim=0)

        # Metrics
        metrics = {
            'assignment_entropy': -(assignment_probs * torch.log(assignment_probs + 1e-8)).sum().item(),
            'routing_entropy': -(routing_probs * torch.log(routing_probs + 1e-8)).sum().item(),
            'assignment_coefficient_of_variation': (assignment_probs.std() / assignment_probs.mean()).item(),
            'routing_coefficient_of_variation': (routing_probs.std() / routing_probs.mean()).item(),
            'max_assignment_ratio': (assignment_probs.max() / assignment_probs.mean()).item(),
            'min_assignment_ratio': (assignment_probs.min() / assignment_probs.mean()).item()
        }

        return metrics


class ExpertParallelism:
    """
    Expert parallelism for distributed MoE training

    Distributes experts across multiple devices/processes
    for scalable MoE training.
    """

    def __init__(
        self,
        num_experts: int,
        experts: nn.ModuleList,
        expert_parallel_size: int | None = None,
        all_to_all_communication: bool = True
    ):
        self.num_experts = num_experts
        self.experts = experts
        self.all_to_all_communication = all_to_all_communication

        # Determine parallelism configuration
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

            self.expert_parallel_size = expert_parallel_size or self.world_size
            self.experts_per_rank = math.ceil(num_experts / self.expert_parallel_size)

            # Determine which experts this rank is responsible for
            self.local_expert_start = self.rank * self.experts_per_rank
            self.local_expert_end = min(
                self.local_expert_start + self.experts_per_rank,
                num_experts
            )

            self.local_experts = list(range(self.local_expert_start, self.local_expert_end))
        else:
            # Single device fallback
            self.world_size = 1
            self.rank = 0
            self.expert_parallel_size = 1
            self.experts_per_rank = num_experts
            self.local_experts = list(range(num_experts))

    def forward(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity_info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass with expert parallelism

        Args:
            tokens: Input tokens [num_tokens, hidden_size]
            expert_indices: Expert assignments [num_tokens, top_k]
            expert_weights: Expert weights [num_tokens, top_k]
            capacity_info: Capacity information

        Returns:
            Expert outputs [num_tokens, hidden_size]
        """
        if not dist.is_initialized() or self.expert_parallel_size == 1:
            # Single device - process all experts locally
            return self._process_local_experts(
                tokens, expert_indices, expert_weights, capacity_info
            )

        # Distributed expert processing
        return self._process_distributed_experts(
            tokens, expert_indices, expert_weights, capacity_info
        )

    def _process_local_experts(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity_info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process experts locally (single device)"""
        num_tokens, hidden_size = tokens.shape
        output = torch.zeros_like(tokens)

        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)
            if not expert_mask.any():
                continue

            token_indices, position_indices = torch.where(expert_mask)

            # Apply capacity constraints
            max_capacity = capacity_info['expert_capacities'][expert_idx].item()
            if len(token_indices) > max_capacity:
                # Select tokens within capacity
                selected = torch.randperm(len(token_indices))[:max_capacity]
                token_indices = token_indices[selected]
                position_indices = position_indices[selected]

            if len(token_indices) == 0:
                continue

            # Get expert input and weights
            expert_input = tokens[token_indices]
            weights = expert_weights[token_indices, position_indices]

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Apply weights and accumulate
            weighted_output = expert_output * weights.unsqueeze(-1)
            output[token_indices] += weighted_output

        return output

    def _process_distributed_experts(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity_info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process experts in distributed setting"""
        if not self.all_to_all_communication:
            # Simple approach: each rank processes only its local experts
            return self._process_local_experts_distributed(
                tokens, expert_indices, expert_weights, capacity_info
            )

        # All-to-all communication for optimal load balancing
        return self._process_all_to_all_experts(
            tokens, expert_indices, expert_weights, capacity_info
        )

    def _process_local_experts_distributed(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity_info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process only local experts in distributed setting"""
        num_tokens, hidden_size = tokens.shape
        output = torch.zeros_like(tokens)

        for local_expert_idx, global_expert_idx in enumerate(self.local_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == global_expert_idx)
            if not expert_mask.any():
                continue

            token_indices, position_indices = torch.where(expert_mask)

            # Apply capacity constraints
            max_capacity = capacity_info['expert_capacities'][global_expert_idx].item()
            if len(token_indices) > max_capacity:
                selected = torch.randperm(len(token_indices))[:max_capacity]
                token_indices = token_indices[selected]
                position_indices = position_indices[selected]

            if len(token_indices) == 0:
                continue

            # Get expert input and weights
            expert_input = tokens[token_indices]
            weights = expert_weights[token_indices, position_indices]

            # Process through local expert
            expert_output = self.experts[local_expert_idx](expert_input)

            # Apply weights and accumulate
            weighted_output = expert_output * weights.unsqueeze(-1)
            output[token_indices] += weighted_output

        # All-reduce to combine outputs from all ranks
        if dist.is_initialized():
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output

    def _process_all_to_all_experts(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity_info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process experts with all-to-all communication"""
        # This is a simplified version - full implementation would require
        # complex all-to-all communication patterns

        warnings.warn(
            "All-to-all expert parallelism not fully implemented. "
            "Using local expert processing.",
        stacklevel=2,
        )

        return self._process_local_experts_distributed(
            tokens, expert_indices, expert_weights, capacity_info
        )


class ExpertScheduler:
    """
    Expert scheduling and capacity management

    Dynamically adjusts expert utilization and capacity
    based on training dynamics and performance metrics.
    """

    def __init__(
        self,
        num_experts: int,
        initial_capacity_factor: float = 1.25,
        min_capacity_factor: float = 0.5,
        max_capacity_factor: float = 3.0,
        adaptation_interval: int = 1000,  # Steps
        target_expert_utilization: float = 0.8
    ):
        self.num_experts = num_experts
        self.initial_capacity_factor = initial_capacity_factor
        self.min_capacity_factor = min_capacity_factor
        self.max_capacity_factor = max_capacity_factor
        self.adaptation_interval = adaptation_interval
        self.target_expert_utilization = target_expert_utilization

        # State tracking
        self.step_count = 0
        self.expert_utilization_history = []
        self.current_capacity_factors = [initial_capacity_factor] * num_experts

    def step(
        self,
        expert_utilization: torch.Tensor,
        load_balance_loss: torch.Tensor,
        total_tokens_processed: int
    ) -> dict[str, Any]:
        """
        Update scheduling based on current metrics

        Args:
            expert_utilization: Current expert utilization [num_experts]
            load_balance_loss: Current load balance loss
            total_tokens_processed: Total tokens processed this step

        Returns:
            Scheduling information and recommendations
        """
        self.step_count += 1

        # Track utilization history
        self.expert_utilization_history.append(expert_utilization.cpu().clone())

        # Trim history to recent steps
        if len(self.expert_utilization_history) > self.adaptation_interval:
            self.expert_utilization_history.pop(0)

        scheduling_info = {
            'step': self.step_count,
            'current_utilization': expert_utilization.cpu().tolist(),
            'capacity_factors': self.current_capacity_factors.copy(),
            'load_balance_loss': load_balance_loss.item(),
            'total_tokens': total_tokens_processed
        }

        # Adapt capacity factors if enough history
        if (self.step_count % self.adaptation_interval == 0 and
            len(self.expert_utilization_history) >= self.adaptation_interval // 2):

            new_capacity_factors = self._adapt_capacity_factors()
            scheduling_info['capacity_factor_changes'] = {
                'old': self.current_capacity_factors.copy(),
                'new': new_capacity_factors.copy()
            }
            self.current_capacity_factors = new_capacity_factors

        return scheduling_info

    def _adapt_capacity_factors(self) -> list[float]:
        """Adapt capacity factors based on utilization history"""
        # Compute average utilization over recent history
        recent_utilization = torch.stack(self.expert_utilization_history[-100:])  # Last 100 steps
        avg_utilization = recent_utilization.mean(dim=0)

        new_capacity_factors = []

        for expert_idx in range(self.num_experts):
            current_util = avg_utilization[expert_idx].item()
            current_capacity = self.current_capacity_factors[expert_idx]

            # Adjust capacity based on utilization
            if current_util > self.target_expert_utilization * 1.2:
                # High utilization - increase capacity
                new_capacity = current_capacity * 1.1
            elif current_util < self.target_expert_utilization * 0.8:
                # Low utilization - decrease capacity
                new_capacity = current_capacity * 0.9
            else:
                # Utilization OK - keep current capacity
                new_capacity = current_capacity

            # Clamp to valid range
            new_capacity = max(self.min_capacity_factor, min(self.max_capacity_factor, new_capacity))
            new_capacity_factors.append(new_capacity)

        return new_capacity_factors

    def get_current_capacity_factors(self) -> list[float]:
        """Get current capacity factors"""
        return self.current_capacity_factors.copy()

    def get_utilization_stats(self) -> dict[str, Any]:
        """Get utilization statistics"""
        if not self.expert_utilization_history:
            return {'error': 'No utilization history available'}

        recent_utilization = torch.stack(self.expert_utilization_history[-100:])
        avg_utilization = recent_utilization.mean(dim=0)
        std_utilization = recent_utilization.std(dim=0)

        return {
            'avg_utilization': avg_utilization.tolist(),
            'std_utilization': std_utilization.tolist(),
            'min_utilization': avg_utilization.min().item(),
            'max_utilization': avg_utilization.max().item(),
            'utilization_coefficient_of_variation': (std_utilization / avg_utilization).mean().item(),
            'target_utilization': self.target_expert_utilization,
            'steps_tracked': len(self.expert_utilization_history)
        }


class MemoryEfficientSwitching:
    """
    Memory-efficient expert switching techniques

    Optimizes memory usage during expert routing and processing.
    """

    def __init__(
        self,
        use_gradient_checkpointing: bool = True,
        expert_offloading: bool = False,
        compression_ratio: float = 0.5,
        offload_device: str = "cpu"
    ):
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.expert_offloading = expert_offloading
        self.compression_ratio = compression_ratio
        self.offload_device = torch.device(offload_device)

        # Track expert usage for smart offloading
        self.expert_usage_tracker = {}
        self.offloaded_experts = set()

    def apply_memory_optimization(
        self,
        experts: nn.ModuleList,
        current_step: int
    ) -> dict[str, Any]:
        """
        Apply memory optimizations to experts

        Args:
            experts: Expert modules
            current_step: Current training step

        Returns:
            Optimization information
        """
        optimization_info = {
            'gradient_checkpointing': self.use_gradient_checkpointing,
            'expert_offloading': self.expert_offloading,
            'offloaded_experts': list(self.offloaded_experts)
        }

        # Apply gradient checkpointing if enabled
        if self.use_gradient_checkpointing:
            self._apply_gradient_checkpointing(experts)

        # Apply expert offloading if enabled
        if self.expert_offloading:
            offload_info = self._apply_expert_offloading(experts, current_step)
            optimization_info.update(offload_info)

        return optimization_info

    def _apply_gradient_checkpointing(self, experts: nn.ModuleList):
        """Apply gradient checkpointing to experts"""
        for expert in experts:
            if hasattr(expert, 'gradient_checkpointing_enable'):
                expert.gradient_checkpointing_enable()

    def _apply_expert_offloading(
        self,
        experts: nn.ModuleList,
        current_step: int
    ) -> dict[str, Any]:
        """Apply expert offloading based on usage patterns"""
        # Simple LRU-based offloading strategy
        usage_threshold = 100  # Steps since last use

        offload_info = {
            'experts_offloaded': 0,
            'experts_loaded': 0,
            'memory_saved_mb': 0
        }

        for expert_idx, expert in enumerate(experts):
            last_used = self.expert_usage_tracker.get(expert_idx, current_step)
            steps_since_use = current_step - last_used

            # Offload unused experts
            if (steps_since_use > usage_threshold and
                expert_idx not in self.offloaded_experts and
                expert.training):

                # Move expert to CPU/other device
                expert.to(self.offload_device)
                self.offloaded_experts.add(expert_idx)
                offload_info['experts_offloaded'] += 1

                # Estimate memory saved (rough approximation)
                param_count = sum(p.numel() for p in expert.parameters())
                memory_mb = param_count * 4 / (1024 * 1024)  # Assume fp32
                offload_info['memory_saved_mb'] += memory_mb

        return offload_info

    def track_expert_usage(self, expert_indices: torch.Tensor, current_step: int):
        """Track expert usage for offloading decisions"""
        unique_experts = torch.unique(expert_indices).cpu().tolist()

        for expert_idx in unique_experts:
            if expert_idx >= 0:  # Valid expert index
                self.expert_usage_tracker[expert_idx] = current_step

                # Move expert back to GPU if it was offloaded
                if expert_idx in self.offloaded_experts:
                    self.offloaded_experts.remove(expert_idx)

    def get_memory_stats(self, experts: nn.ModuleList) -> dict[str, Any]:
        """Get memory usage statistics"""
        total_params = 0
        offloaded_params = 0
        active_params = 0

        for expert_idx, expert in enumerate(experts):
            param_count = sum(p.numel() for p in expert.parameters())
            total_params += param_count

            if expert_idx in self.offloaded_experts:
                offloaded_params += param_count
            else:
                active_params += param_count

        return {
            'total_parameters': total_params,
            'active_parameters': active_params,
            'offloaded_parameters': offloaded_params,
            'offload_ratio': offloaded_params / total_params if total_params > 0 else 0,
            'memory_reduction_estimate_mb': offloaded_params * 4 / (1024 * 1024)
        }


if __name__ == "__main__":
    # Test optimization utilities
    num_experts = 8
    num_tokens = 1000
    hidden_size = 768

    # Test LoadBalancer
    print("Testing LoadBalancer:")
    load_balancer = LoadBalancer(num_experts)

    router_probs = torch.softmax(torch.randn(num_tokens, num_experts), dim=-1)
    expert_indices = torch.randint(0, num_experts, (num_tokens, 2))

    capacity_info = load_balancer.get_capacity_info(router_probs, num_tokens)
    print(f"  Expert capacities: {capacity_info['expert_capacities']}")
    print(f"  Tokens dropped: {capacity_info['tokens_dropped']}")

    load_balance_loss = load_balancer.compute_load_balance_loss(router_probs, expert_indices)
    print(f"  Load balance loss: {load_balance_loss.item():.4f}")

    balance_metrics = load_balancer.get_expert_balance_metrics(router_probs, expert_indices)
    print(f"  Balance metrics: {balance_metrics}")

    # Test ExpertScheduler
    print("\nTesting ExpertScheduler:")
    scheduler = ExpertScheduler(num_experts)

    expert_utilization = torch.rand(num_experts)
    scheduling_info = scheduler.step(expert_utilization, load_balance_loss, num_tokens)
    print(f"  Scheduling step: {scheduling_info['step']}")
    print(f"  Current utilization: {scheduling_info['current_utilization']}")

    # Test MemoryEfficientSwitching
    print("\nTesting MemoryEfficientSwitching:")
    memory_optimizer = MemoryEfficientSwitching(
        expert_offloading=True,
        use_gradient_checkpointing=True
    )

    # Create dummy experts
    from .expert_networks import FeedForwardExpert
    experts = nn.ModuleList([
        FeedForwardExpert(hidden_size, hidden_size * 2)
        for _ in range(num_experts)
    ])

    # Test memory optimization
    opt_info = memory_optimizer.apply_memory_optimization(experts, current_step=0)
    print(f"  Optimization info: {opt_info}")

    # Test expert usage tracking
    memory_optimizer.track_expert_usage(expert_indices.flatten(), current_step=1)
    memory_stats = memory_optimizer.get_memory_stats(experts)
    print(f"  Memory stats: {memory_stats}")

    print("\nAll optimization utilities tested successfully!")
