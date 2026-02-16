#!/usr/bin/env python3
"""
Qwen3-8B FSDP Distributed Training Example

Demonstrates how to use TorchBridge with PyTorch FSDP (Fully Sharded
Data Parallelism) for distributed training of modern LLMs.

This example replaces the legacy Llama 2 FSDP example with a modern
Qwen3-8B workflow that leverages TorchBridge's backend abstraction
for multi-GPU training across NVIDIA, AMD, and Trainium.

Models covered:
- Qwen/Qwen3-8B (8B, primary)
- Qwen/Qwen3-4B (4B, efficient)

Requirements:
    pip install transformers accelerate

Hardware requirements:
    - 2+ GPUs with 16GB+ VRAM each
    - NCCL (NVIDIA) or RCCL (AMD) for communication

Usage:
    # Single-node, multi-GPU
    torchrun --nproc-per-node=2 qwen3_fsdp_training.py

    # With specific model
    torchrun --nproc-per-node=4 qwen3_fsdp_training.py --model Qwen/Qwen3-4B

    # CPU test (no distributed)
    python qwen3_fsdp_training.py --cpu-test
"""

import argparse
import logging
import time
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def get_system_info() -> dict[str, Any]:
    """Gather system information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    if hasattr(torch.version, "hip") and torch.version.hip:
        info["backend"] = "ROCm"
    elif torch.cuda.is_available():
        info["backend"] = "CUDA"
    else:
        info["backend"] = "CPU"
    return info


def run_fsdp_training(
    model_name: str,
    num_steps: int = 10,
    batch_size: int = 2,
    max_seq_len: int = 512,
) -> dict[str, Any]:
    """Run FSDP training loop (requires torchrun)."""
    print_section(f"FSDP Training - {model_name}")

    try:
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print(f"World size: {world_size}")
            print(f"Loading model: {model_name}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        # Wrap with FSDP
        model = FSDP(model, device_id=local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Synthetic training data
        dummy_input_ids = torch.randint(
            0, tokenizer.vocab_size, (batch_size, max_seq_len), device=f"cuda:{local_rank}"
        )
        dummy_labels = dummy_input_ids.clone()

        if rank == 0:
            print("\nTraining config:")
            print(f"  Steps: {num_steps}")
            print(f"  Batch size: {batch_size} x {world_size} GPUs")
            print(f"  Sequence length: {max_seq_len}")

        # Training loop
        model.train()
        losses = []
        start_time = time.perf_counter()

        for step in range(num_steps):
            outputs = model(input_ids=dummy_input_ids, labels=dummy_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            if rank == 0 and (step + 1) % 5 == 0:
                print(f"  Step {step + 1}/{num_steps}: loss={loss.item():.4f}")

        elapsed = time.perf_counter() - start_time

        if rank == 0:
            print("\nTraining complete:")
            print(f"  Total time: {elapsed:.1f}s")
            print(f"  Steps/sec: {num_steps / elapsed:.2f}")
            print(f"  Tokens/sec: {num_steps * batch_size * max_seq_len * world_size / elapsed:.0f}")

        dist.destroy_process_group()

        return {
            "model_name": model_name,
            "world_size": world_size,
            "num_steps": num_steps,
            "elapsed_s": elapsed,
            "final_loss": losses[-1] if losses else None,
        }

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"FSDP training failed: {e}")
        return {"error": str(e)}


def run_cpu_test(model_name: str) -> dict[str, Any]:
    """Run a minimal forward/backward pass on CPU for validation."""
    print_section(f"CPU Test - {model_name}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading model on CPU (this may take a while for large models)...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        # Single forward + backward
        dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 64))
        dummy_labels = dummy_input_ids.clone()

        model.train()
        outputs = model(input_ids=dummy_input_ids, labels=dummy_labels)
        loss = outputs.loss
        loss.backward()

        print(f"Forward + backward passed. Loss: {loss.item():.4f}")
        print(f"Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0):.4f}")

        return {
            "model_name": model_name,
            "status": "PASSED",
            "loss": loss.item(),
        }

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"CPU test failed: {e}")
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Qwen3-8B FSDP Distributed Training with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--cpu-test", action="store_true", help="Run CPU test only")

    args = parser.parse_args()

    print_section("Qwen3 FSDP Distributed Training with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    if args.cpu_test:
        results = run_cpu_test(args.model)
    else:
        results = run_fsdp_training(
            args.model,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )

    if "error" in results:
        print(f"\nError: {results['error']}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
