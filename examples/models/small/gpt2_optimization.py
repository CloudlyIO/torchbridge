#!/usr/bin/env python3
"""
GPT-2 Optimization Example

Demonstrates how to use KernelPyTorch to optimize GPT-2 for text generation.
Shows optimized inference with KV-cache and autoregressive generation.

Models covered:
- gpt2 (124M params, ~500MB) - Small
- gpt2-medium (355M params, ~1.4GB) - Medium
- gpt2-large (774M params, ~3.1GB) - Large (requires more VRAM)

Usage:
    python gpt2_optimization.py [--model gpt2] [--prompt "Hello, world"]

Requirements:
    pip install transformers

Version: 0.4.11
"""

import argparse
import logging
import time
from typing import Dict, Any, Optional

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import transformers
        logger.info(f"transformers version: {transformers.__version__}")
        return True
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        return False


def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    num_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark text generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    total_tokens = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            total_tokens += outputs.shape[1] - inputs["input_ids"].shape[1]

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    tokens_per_sec = total_tokens / total_time

    return {
        "total_time_s": total_time,
        "avg_latency_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_sec,
        "total_tokens_generated": total_tokens,
    }


def run_baseline(model_name: str, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    """Run baseline PyTorch model without optimization."""
    print_section(f"Baseline PyTorch - {model_name}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Benchmark
    results = benchmark_generation(model, tokenizer, prompt, device, max_new_tokens)

    print(f"Average latency: {results['avg_latency_ms']:.2f} ms")
    print(f"Tokens/second: {results['tokens_per_second']:.1f}")

    # Generate sample
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nSample generation:\n  '{generated_text[:200]}...'")

    return {
        "model_name": model_name,
        "device": str(device),
        "max_new_tokens": max_new_tokens,
        **results
    }


def run_optimized(model_name: str, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    """Run KernelPyTorch optimized model."""
    print_section(f"KernelPyTorch Optimized - {model_name}")

    from transformers import AutoTokenizer

    # Import KernelPyTorch
    try:
        from kernel_pytorch.models.text import (
            TextModelOptimizer,
            TextModelConfig,
            OptimizationMode,
            TextModelType
        )

        # Create optimized config
        config = TextModelConfig(
            model_name=model_name,
            model_type=TextModelType.GPT2,
            optimization_mode=OptimizationMode.INFERENCE,
            use_torch_compile=True,
            compile_mode="reduce-overhead",
        )

        # Create optimizer and optimize model
        optimizer = TextModelOptimizer(config)
        model = optimizer.optimize(model_name, task="causal-lm")

        device = optimizer.device
        optimization_info = optimizer.get_optimization_info()

        print(f"Device: {device}")
        print(f"Backend: {optimization_info['backend']}")
        print(f"Dtype: {optimization_info['dtype']}")
        print(f"Optimization mode: {optimization_info['optimization_mode']}")
        print(f"torch.compile: {optimization_info['torch_compile']}")

    except ImportError as e:
        logger.warning(f"KernelPyTorch not available: {e}")
        logger.info("Falling back to torch.compile only")

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Apply torch.compile
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")

        optimization_info = {"backend": "pytorch", "torch_compile": True}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Benchmark
    results = benchmark_generation(model, tokenizer, prompt, device, max_new_tokens)

    print(f"Average latency: {results['avg_latency_ms']:.2f} ms")
    print(f"Tokens/second: {results['tokens_per_second']:.1f}")

    # Generate sample
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nSample generation:\n  '{generated_text[:200]}...'")

    return {
        "model_name": model_name,
        "device": str(device),
        "max_new_tokens": max_new_tokens,
        "optimization_info": optimization_info,
        **results
    }


def run_interactive_demo(model_name: str):
    """Run an interactive text generation demo."""
    print_section("Interactive Text Generation Demo")

    from transformers import AutoTokenizer

    try:
        from kernel_pytorch.models.text import OptimizedGPT2

        # Create optimized model
        model = OptimizedGPT2(model_name=model_name, task="causal-lm")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Optimized GPT-2 is ready for text generation!")
        print(f"Device: {model.device}")
        print(f"Optimization: {model.get_optimization_info()}")
        print()

        # Demo prompts
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a land far away",
            "Machine learning has revolutionized the way we",
        ]

        for prompt in prompts:
            print(f"Prompt: '{prompt}'")

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            generation_time = time.perf_counter() - start_time

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]

            print(f"Generated: '{generated_text}'")
            print(f"Time: {generation_time*1000:.1f}ms, Tokens: {tokens_generated}, "
                  f"Speed: {tokens_generated/generation_time:.1f} tokens/sec")
            print()

    except ImportError as e:
        logger.warning(f"Demo skipped: {e}")


def run_batch_generation_demo(model_name: str):
    """Demonstrate batch text generation."""
    print_section("Batch Generation Demo")

    from transformers import AutoTokenizer

    try:
        from kernel_pytorch.models.text import OptimizedGPT2

        model = OptimizedGPT2(model_name=model_name, task="causal-lm")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Batch of prompts
        prompts = [
            "The capital of France is",
            "Python is a programming language that",
            "Deep learning neural networks are used for",
            "The quick brown fox",
        ]

        print(f"Generating text for {len(prompts)} prompts in batch...")

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        batch_time = time.perf_counter() - start_time

        print(f"\nBatch generation completed in {batch_time*1000:.1f}ms")
        print(f"Average per sample: {batch_time*1000/len(prompts):.1f}ms")
        print()

        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            generated = tokenizer.decode(output, skip_special_tokens=True)
            print(f"[{i+1}] {generated[:100]}...")

    except ImportError as e:
        logger.warning(f"Batch demo skipped: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPT-2 Optimization Example")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large"],
        help="Model to optimize"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence",
        help="Prompt for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline benchmark"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo"
    )

    args = parser.parse_args()

    print_section("GPT-2 Optimization with KernelPyTorch")

    # Check dependencies
    if not check_dependencies():
        return

    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run baseline
    if not args.skip_baseline:
        baseline_results = run_baseline(args.model, args.prompt, args.max_new_tokens)
    else:
        baseline_results = None

    # Run optimized
    optimized_results = run_optimized(args.model, args.prompt, args.max_new_tokens)

    # Print comparison
    if baseline_results:
        print_section("Performance Comparison")

        speedup = baseline_results["avg_latency_ms"] / optimized_results["avg_latency_ms"]
        token_speedup = optimized_results["tokens_per_second"] / baseline_results["tokens_per_second"]

        print(f"Baseline latency:  {baseline_results['avg_latency_ms']:.2f} ms")
        print(f"Optimized latency: {optimized_results['avg_latency_ms']:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print()
        print(f"Baseline tokens/sec:  {baseline_results['tokens_per_second']:.1f}")
        print(f"Optimized tokens/sec: {optimized_results['tokens_per_second']:.1f}")
        print(f"Token generation speedup: {token_speedup:.2f}x")

    # Run demos
    if args.interactive:
        run_interactive_demo(args.model)

    run_batch_generation_demo(args.model)

    print_section("Complete!")


if __name__ == "__main__":
    main()
