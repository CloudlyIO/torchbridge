#!/usr/bin/env python3
"""
Example: Distributed Llama-70B Optimization

This example demonstrates how to use KernelPyTorch's distributed module
to run Llama-70B and other large models across multiple GPUs.

Features demonstrated:
- Automatic parallelism strategy selection
- Tensor parallelism for layer distribution
- Memory estimation for hardware planning
- Optimized generation with KV-cache

Requirements:
- 8x A100 80GB GPUs (recommended) or 4x H100 80GB GPUs
- PyTorch 2.0+
- transformers library (optional, for real model loading)

Usage:
    # Single node, 8 GPUs
    torchrun --nproc_per_node=8 llama_70b_distributed.py

    # Multi-node (2 nodes, 8 GPUs each)
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$RANK \
             --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
             llama_70b_distributed.py
"""

import os
import time
import torch
import torch.distributed as dist

from kernel_pytorch.models.distributed import (
    # Main optimizer classes
    DistributedLLMOptimizer,
    DistributedConfig,
    DistributedLlama70B,
    create_distributed_llm,
    estimate_gpu_requirements,
    # Parallelism strategies
    ParallelismStrategy,
    # Tensor parallelism components
    TensorParallelConfig,
    apply_tensor_parallelism,
    # Pipeline parallelism components
    PipelineParallelConfig,
    create_pipeline_stages,
)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ:
        # Launched with torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def example_memory_estimation():
    """Example: Estimate GPU requirements before loading."""
    print("\n" + "=" * 60)
    print("GPU Requirements Estimation")
    print("=" * 60)

    # Estimate for different models
    models = [
        ("meta-llama/Llama-2-70b-hf", "Llama-2 70B"),
        ("meta-llama/Llama-3-70B", "Llama-3 70B"),
        ("tiiuae/falcon-40b", "Falcon 40B"),
        ("mistralai/Mixtral-8x7B-v0.1", "Mixtral 8x7B"),
    ]

    for model_name, display_name in models:
        print(f"\n{display_name}:")

        # Without quantization
        req_fp16 = estimate_gpu_requirements(
            model_name,
            max_sequence_length=4096,
            max_batch_size=1,
            quantization="none",
        )

        # With INT8 quantization
        req_int8 = estimate_gpu_requirements(
            model_name,
            max_sequence_length=4096,
            max_batch_size=1,
            quantization="int8",
        )

        print(f"  FP16: {req_fp16['memory_estimates']['total_params_gb']:.1f} GB params, "
              f"{req_fp16['min_gpus']} GPUs minimum")
        print(f"  INT8: {req_int8['memory_estimates']['total_params_gb']:.1f} GB params, "
              f"{req_int8['min_gpus']} GPUs minimum")
        print(f"  Recommended: TP={req_fp16['recommended_tensor_parallel']}, "
              f"PP={req_fp16['recommended_pipeline_parallel']}")


def example_basic_usage(rank, world_size):
    """Example: Basic distributed model usage."""
    print("\n" + "=" * 60)
    print("Basic Distributed Model Usage")
    print("=" * 60)

    # Create distributed optimizer with automatic configuration
    optimizer = create_distributed_llm(
        "meta-llama/Llama-2-70b-hf",
        world_size=world_size,
        dtype=torch.bfloat16,
        quantization="none",
        strategy=ParallelismStrategy.AUTO,
    )

    print(f"\nConfiguration:")
    print(f"  World size: {optimizer.config.world_size}")
    print(f"  Tensor parallel: {optimizer.config.tensor_parallel_size}")
    print(f"  Pipeline parallel: {optimizer.config.pipeline_parallel_size}")
    print(f"  Strategy: {optimizer.config.strategy.value}")

    # Estimate memory per GPU
    memory = optimizer.estimate_memory()
    print(f"\nMemory Estimates (per GPU):")
    print(f"  Parameters: {memory['params_per_gpu_gb']:.2f} GB")
    print(f"  KV-cache: {memory['kv_cache_gb']:.2f} GB")
    print(f"  Total: {memory['total_per_gpu_gb']:.2f} GB")

    # Load model (uses mock model if transformers not available)
    print("\nLoading model...")
    try:
        model = optimizer.load_model(trust_remote_code=True)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Note: Using mock model ({e})")
        model = optimizer._create_mock_model()
        if torch.cuda.is_available():
            model = model.cuda()

    return optimizer, model


def example_tensor_parallelism():
    """Example: Manual tensor parallelism configuration."""
    print("\n" + "=" * 60)
    print("Manual Tensor Parallelism")
    print("=" * 60)

    # Create a simple model to demonstrate tensor parallelism
    class SimpleTransformerBlock(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.o_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.gate_proj = torch.nn.Linear(hidden_size, hidden_size * 4)
            self.up_proj = torch.nn.Linear(hidden_size, hidden_size * 4)
            self.down_proj = torch.nn.Linear(hidden_size * 4, hidden_size)

        def forward(self, x):
            # Simplified attention
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            attn = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
            x = x + self.o_proj(attn)

            # Simplified MLP
            gate = torch.nn.functional.silu(self.gate_proj(x))
            up = self.up_proj(x)
            x = x + self.down_proj(gate * up)
            return x

    # Create model
    hidden_size = 4096
    model = SimpleTransformerBlock(hidden_size)

    print(f"Original model:")
    print(f"  q_proj: {model.q_proj.weight.shape}")
    print(f"  gate_proj: {model.gate_proj.weight.shape}")

    # Configure tensor parallelism for 4 GPUs (simulated)
    tp_config = TensorParallelConfig(
        world_size=4,
        rank=0,  # This would be different for each GPU
        sequence_parallel=True,
    )

    # Apply tensor parallelism
    model = apply_tensor_parallelism(
        model,
        tp_config,
        linear_layer_names=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    print(f"\nAfter tensor parallelism (world_size=4):")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")


def example_pipeline_parallelism():
    """Example: Pipeline parallelism configuration."""
    print("\n" + "=" * 60)
    print("Pipeline Parallelism")
    print("=" * 60)

    # Create a multi-layer model
    layers = torch.nn.Sequential(
        *[torch.nn.Linear(1024, 1024) for _ in range(8)]
    )

    print(f"Original model: {len(layers)} layers")

    # Configure pipeline parallelism
    pp_config = PipelineParallelConfig(
        num_stages=4,
        num_micro_batches=8,
        stage_id=0,  # This would be different for each GPU
        activation_checkpointing=True,
    )

    # Create pipeline stages
    stages = create_pipeline_stages(layers, pp_config)

    print(f"\nPipeline configuration:")
    print(f"  Stages: {pp_config.num_stages}")
    print(f"  Micro-batches: {pp_config.num_micro_batches}")
    print(f"  Activation checkpointing: {pp_config.activation_checkpointing}")
    print(f"  Created {len(stages)} stage(s) for this rank")


def example_generation(optimizer, model):
    """Example: Text generation with distributed model."""
    print("\n" + "=" * 60)
    print("Text Generation")
    print("=" * 60)

    # Create sample input
    batch_size = 1
    seq_length = 32
    vocab_size = 32000

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    print(f"Input shape: {input_ids.shape}")

    # Generate tokens
    print("\nGenerating...")
    start_time = time.time()

    with torch.no_grad():
        if hasattr(model, "generate"):
            output_ids = optimizer.generate(
                input_ids,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        else:
            # Manual generation for mock model
            output_ids = optimizer._manual_generate(
                input_ids,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
            )

    elapsed = time.time() - start_time
    new_tokens = output_ids.shape[1] - input_ids.shape[1]

    print(f"Output shape: {output_ids.shape}")
    print(f"Generated {new_tokens} tokens in {elapsed:.2f}s")
    print(f"Throughput: {new_tokens / elapsed:.1f} tokens/sec")


def example_different_models():
    """Example: Using different model wrappers."""
    print("\n" + "=" * 60)
    print("Different Model Wrappers")
    print("=" * 60)

    # Llama-70B with default settings
    llama = DistributedLlama70B()
    print(f"\nLlama-70B defaults:")
    print(f"  World size: {llama.config.world_size}")
    print(f"  Tensor parallel: {llama.config.tensor_parallel_size}")

    # Custom configuration for smaller setup
    from kernel_pytorch.models.distributed import DistributedConfig

    config = DistributedConfig(
        world_size=4,
        tensor_parallel_size=4,
        dtype=torch.float16,
        quantization="int8",
        use_flash_attention=True,
        max_sequence_length=8192,
    )

    custom_optimizer = DistributedLLMOptimizer(
        "meta-llama/Llama-2-70b-hf",
        config=config,
    )

    memory = custom_optimizer.estimate_memory()
    print(f"\nCustom config (4 GPUs, INT8):")
    print(f"  Memory per GPU: {memory['total_per_gpu_gb']:.2f} GB")


def main():
    """Run all examples."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("KernelPyTorch Distributed Model Examples")
        print(f"Running on {world_size} GPU(s)")
        print("=" * 60)

    try:
        # Example 1: Memory estimation (runs on all ranks)
        if rank == 0:
            example_memory_estimation()

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Example 2: Basic usage
        if rank == 0:
            optimizer, model = example_basic_usage(rank, world_size)

        # Example 3: Tensor parallelism details
        if rank == 0:
            example_tensor_parallelism()

        # Example 4: Pipeline parallelism details
        if rank == 0:
            example_pipeline_parallelism()

        # Example 5: Different model wrappers
        if rank == 0:
            example_different_models()

        # Example 6: Text generation (only if model loaded)
        if rank == 0 and "optimizer" in dir() and "model" in dir():
            example_generation(optimizer, model)

    finally:
        cleanup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
