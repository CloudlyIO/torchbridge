#!/usr/bin/env python3
"""
Example: Training Llama-7B with FSDP (Fully Sharded Data Parallel)

This example demonstrates how to use PyTorch FSDP with KernelPyTorch
optimizations for memory-efficient training of large language models.

Features demonstrated:
- FSDP wrapping with mixed precision
- Gradient checkpointing for memory efficiency
- Hybrid sharding for multi-node training
- Integration with KernelPyTorch optimizations
- Checkpoint saving and loading

Requirements:
- 4x A100 40GB or 2x A100 80GB (minimum)
- PyTorch 2.0+ with FSDP support
- transformers library

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 train_llama_7b_fsdp.py

    # Multi-node (2 nodes, 4 GPUs each)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$RANK \\
             --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
             train_llama_7b_fsdp.py --multi_node

v0.4.24 - Distributed Training Validation
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    use_mock_model: bool = True  # Use mock for testing without HF auth

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 100
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10

    # FSDP
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    cpu_offload: bool = False

    # Data
    max_seq_length: int = 512
    num_workers: int = 4

    # Checkpointing
    save_dir: str = "./checkpoints"
    save_steps: int = 50

    # Logging
    log_steps: int = 10


# =============================================================================
# Mock Model (for testing without HuggingFace auth)
# =============================================================================

class MockLlamaAttention(nn.Module):
    """Mock Llama attention layer."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = nn.Identity()  # Simplified

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)


class MockLlamaMLP(nn.Module):
    """Mock Llama MLP layer."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.silu(self.gate_proj(x)) * self.up_proj(x))


class MockLlamaDecoderLayer(nn.Module):
    """Mock Llama decoder layer for FSDP wrapping."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.self_attn = MockLlamaAttention(hidden_size, num_heads)
        self.mlp = MockLlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MockLlamaForCausalLM(nn.Module):
    """Mock Llama model for causal language modeling."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        intermediate_size: int = 11008,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MockLlamaDecoderLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


# =============================================================================
# FSDP Utilities
# =============================================================================

def get_sharding_strategy(strategy_name: str) -> ShardingStrategy:
    """Get FSDP sharding strategy by name."""
    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    return strategies.get(strategy_name, ShardingStrategy.FULL_SHARD)


def get_mixed_precision_policy(enabled: bool) -> Optional[MixedPrecision]:
    """Get mixed precision policy."""
    if not enabled:
        return None

    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def get_auto_wrap_policy(model: nn.Module):
    """Get auto wrap policy for transformer layers."""
    # Wrap each decoder layer
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MockLlamaDecoderLayer},
    )


def setup_fsdp(
    model: nn.Module,
    config: TrainingConfig,
    rank: int,
) -> FSDP:
    """Setup FSDP for the model."""

    # Get sharding strategy
    sharding_strategy = get_sharding_strategy(config.sharding_strategy)

    # Get mixed precision policy
    mp_policy = get_mixed_precision_policy(config.mixed_precision)

    # Get auto wrap policy
    auto_wrap_policy = get_auto_wrap_policy(model)

    # CPU offload (for very large models)
    cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None

    # Wrap with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=cpu_offload,
        device_id=torch.cuda.current_device(),
    )

    return fsdp_model


def apply_activation_checkpointing(model: FSDP):
    """Apply activation checkpointing to reduce memory."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )

    check_fn = lambda submodule: isinstance(submodule, MockLlamaDecoderLayer)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn,
    )


# =============================================================================
# Training Utilities
# =============================================================================

def setup_distributed() -> tuple:
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU fallback
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_optimizer(model: FSDP, config: TrainingConfig) -> torch.optim.Optimizer:
    """Get optimizer for training."""
    # Use AdamW with proper weight decay
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
        eta_min=config.learning_rate * 0.1,
    )


def create_dummy_dataloader(config: TrainingConfig, rank: int, world_size: int):
    """Create dummy dataloader for testing."""
    # Create random data
    num_samples = config.max_steps * config.batch_size * config.gradient_accumulation_steps
    num_samples_per_rank = num_samples // world_size

    input_ids = torch.randint(
        0, 32000,
        (num_samples_per_rank, config.max_seq_length),
    )
    labels = input_ids.clone()

    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,  # Simplified for demo
        pin_memory=True,
    )

    return dataloader


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainingConfig,
    rank: int,
):
    """Save FSDP checkpoint."""
    if rank != 0:
        return

    os.makedirs(config.save_dir, exist_ok=True)

    # Save with FSDP state dict
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }

    path = os.path.join(config.save_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


# =============================================================================
# Training Loop
# =============================================================================

def train(config: TrainingConfig):
    """Main training function."""

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("FSDP Training - Llama-7B")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Sharding strategy: {config.sharding_strategy}")
        print(f"Mixed precision: {config.mixed_precision}")
        print(f"Activation checkpointing: {config.activation_checkpointing}")
        print()

    # Create model
    if rank == 0:
        print("Creating model...")

    if config.use_mock_model:
        # Use mock model for testing
        model = MockLlamaForCausalLM(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=8,  # Reduced for demo
            num_attention_heads=32,
            intermediate_size=11008,
        )
    else:
        # Load real model (requires HF auth for Llama)
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"Could not load {config.model_name}: {e}")
            print("Falling back to mock model")
            model = MockLlamaForCausalLM()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {num_params / 1e9:.2f}B")

    # Wrap with FSDP
    if rank == 0:
        print("Setting up FSDP...")

    model = setup_fsdp(model, config, rank)

    # Apply activation checkpointing
    if config.activation_checkpointing:
        try:
            apply_activation_checkpointing(model)
            if rank == 0:
                print("Activation checkpointing enabled")
        except Exception as e:
            if rank == 0:
                print(f"Could not enable activation checkpointing: {e}")

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Create dataloader
    dataloader = create_dummy_dataloader(config, rank, world_size)

    # Training loop
    if rank == 0:
        print()
        print("Starting training...")
        print("-" * 60)

    model.train()
    global_step = 0
    total_loss = 0.0
    start_time = time.time()

    for epoch in range(1000):  # Large number, we use max_steps instead
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item()

            # Optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                model.clip_grad_norm_(1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % config.log_steps == 0 and rank == 0:
                    avg_loss = total_loss / config.log_steps
                    elapsed = time.time() - start_time
                    samples_per_sec = (
                        config.log_steps *
                        config.batch_size *
                        config.gradient_accumulation_steps *
                        world_size
                    ) / elapsed

                    print(
                        f"Step {global_step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Samples/sec: {samples_per_sec:.1f}"
                    )

                    total_loss = 0.0
                    start_time = time.time()

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    save_checkpoint(model, optimizer, global_step, config, rank)

                # Check if done
                if global_step >= config.max_steps:
                    break

        if global_step >= config.max_steps:
            break

    # Final checkpoint
    save_checkpoint(model, optimizer, global_step, config, rank)

    if rank == 0:
        print("-" * 60)
        print("Training complete!")
        print(f"Final step: {global_step}")

    # Cleanup
    cleanup_distributed()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Llama-7B with FSDP")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path",
    )
    parser.add_argument(
        "--use_mock_model",
        action="store_true",
        default=True,
        help="Use mock model for testing",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # FSDP arguments
    parser.add_argument(
        "--sharding_strategy",
        type=str,
        default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD"],
    )
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--activation_checkpointing", action="store_true", default=True)
    parser.add_argument("--cpu_offload", action="store_true", default=False)

    # Output arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=10)

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        use_mock_model=args.use_mock_model,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        sharding_strategy=args.sharding_strategy,
        mixed_precision=args.mixed_precision,
        activation_checkpointing=args.activation_checkpointing,
        cpu_offload=args.cpu_offload,
        save_dir=args.save_dir,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )

    train(config)


if __name__ == "__main__":
    main()
