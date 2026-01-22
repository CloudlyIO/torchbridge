#!/usr/bin/env python3
"""
Llama-7B Optimization Example

Demonstrates how to use KernelPyTorch to optimize Llama-2-7B and similar
7B parameter LLMs for production inference.

Models covered:
- meta-llama/Llama-2-7b-hf (7B params)
- meta-llama/Llama-2-7b-chat-hf (7B params, chat-tuned)
- meta-llama/Meta-Llama-3-8B (8B params)

Requirements:
    pip install transformers accelerate
    pip install bitsandbytes  # For quantization (optional)

Hardware requirements:
    - FP16: ~14GB VRAM (RTX 3090, A10G, L4)
    - INT8: ~7GB VRAM
    - INT4: ~4GB VRAM

Usage:
    python llama_optimization.py [--model meta-llama/Llama-2-7b-hf]

Version: 0.4.12
"""

import argparse
import logging
import time
from typing import Dict, Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    deps = {}

    try:
        import transformers
        deps["transformers"] = True
        logger.info(f"transformers version: {transformers.__version__}")
    except ImportError:
        deps["transformers"] = False
        logger.error("transformers not installed")

    try:
        import accelerate
        deps["accelerate"] = True
        logger.info(f"accelerate version: {accelerate.__version__}")
    except ImportError:
        deps["accelerate"] = False
        logger.warning("accelerate not installed (optional)")

    try:
        import bitsandbytes
        deps["bitsandbytes"] = True
        logger.info(f"bitsandbytes available for quantization")
    except ImportError:
        deps["bitsandbytes"] = False
        logger.warning("bitsandbytes not installed (optional, for quantization)")

    return deps


def estimate_memory(model_name: str, quantization: str = "none") -> Dict[str, float]:
    """Estimate memory requirements."""
    try:
        from kernel_pytorch.models.llm import LLMOptimizer, LLMConfig, QuantizationMode

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.INT4,
        }

        config = LLMConfig(
            model_name=model_name,
            quantization=quant_map.get(quantization, QuantizationMode.NONE)
        )
        optimizer = LLMOptimizer(config)
        return optimizer.estimate_memory(model_name)
    except ImportError:
        # Fallback estimates
        base_7b = 14.0  # GB for FP16
        if "70b" in model_name.lower():
            base = 140.0
        elif "13b" in model_name.lower():
            base = 26.0
        elif "8b" in model_name.lower():
            base = 16.0
        else:
            base = base_7b

        if quantization == "int8":
            base /= 2
        elif quantization == "int4":
            base /= 4

        return {"model_memory_gb": base, "total_gb": base * 1.1}


def run_with_kernelpytorch(
    model_name: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int
) -> Dict[str, Any]:
    """Run optimized inference with KernelPyTorch."""
    print_section(f"KernelPyTorch Optimized - {model_name}")

    try:
        from kernel_pytorch.models.llm import (
            LLMOptimizer,
            LLMConfig,
            QuantizationMode
        )

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.BNBT4,
        }

        config = LLMConfig(
            model_name=model_name,
            quantization=quant_map.get(quantization, QuantizationMode.NONE),
            use_flash_attention=True,
            use_torch_compile=True,
            compile_mode="reduce-overhead",
        )

        optimizer = LLMOptimizer(config)

        # Print memory estimate
        memory_est = optimizer.estimate_memory(model_name)
        print(f"Estimated memory: {memory_est['total_gb']:.1f} GB")

        # Check if we have enough memory
        if torch.cuda.is_available():
            available = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available VRAM: {available:.1f} GB")

            if memory_est['total_gb'] > available * 0.9:
                print(f"WARNING: Model may not fit in memory!")
                if quantization == "none":
                    print("Consider using --quantization int8 or int4")

        print("\nLoading model (this may take a while)...")
        model, tokenizer = optimizer.optimize(model_name)

        opt_info = optimizer.get_optimization_info()
        print(f"\nOptimization applied:")
        print(f"  Device: {opt_info['device']}")
        print(f"  Dtype: {opt_info['dtype']}")
        print(f"  Backend: {opt_info['backend']}")
        print(f"  Quantization: {opt_info['quantization']}")
        print(f"  Flash Attention: {opt_info['flash_attention']}")
        print(f"  torch.compile: {opt_info['torch_compile']}")

        # Generate
        print(f"\nPrompt: '{prompt}'")
        print("Generating...")

        inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generation_time = time.perf_counter() - start_time
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nGenerated ({tokens_generated} tokens in {generation_time:.2f}s):")
        print(f"  {generated_text[:500]}...")
        print(f"\nPerformance:")
        print(f"  Time: {generation_time:.2f}s")
        print(f"  Tokens/sec: {tokens_generated/generation_time:.1f}")

        return {
            "model_name": model_name,
            "quantization": quantization,
            "generation_time_s": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_generated / generation_time,
            "optimization_info": opt_info,
        }

    except ImportError as e:
        logger.error(f"KernelPyTorch import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"error": str(e)}


def run_memory_demo():
    """Demonstrate memory estimation for different configurations."""
    print_section("Memory Estimation")

    models = [
        ("Llama-2-7B", "llama-7b"),
        ("Llama-2-13B", "llama-13b"),
        ("Llama-2-70B", "llama-70b"),
    ]

    quantizations = ["none", "int8", "int4"]

    print(f"{'Model':<15} {'Quant':<8} {'Memory (GB)':<12}")
    print("-" * 40)

    for name, model_id in models:
        for quant in quantizations:
            est = estimate_memory(model_id, quant)
            print(f"{name:<15} {quant:<8} {est['total_gb']:<12.1f}")


def run_interactive_demo(model, tokenizer, device):
    """Run interactive chat demo."""
    print_section("Interactive Chat Demo")

    print("Enter prompts to generate text (type 'quit' to exit)")
    print("-" * 50)

    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == 'quit':
            break

        if not prompt:
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        gen_time = time.perf_counter() - start_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        print(f"\nAssistant: {response}")
        print(f"({tokens} tokens, {gen_time:.2f}s, {tokens/gen_time:.1f} tok/s)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Llama Optimization Example")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4"],
        help="Quantization mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence will",
        help="Prompt for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only show memory estimation"
    )

    args = parser.parse_args()

    print_section("Llama Optimization with KernelPyTorch v0.4.12")

    # Check dependencies
    deps = check_dependencies()
    if not deps.get("transformers"):
        print("ERROR: transformers is required. Install with: pip install transformers")
        return

    # Print system info
    print(f"\nSystem Info:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {total_mem:.1f} GB")

    # Memory demo
    if args.memory_only:
        run_memory_demo()
        return

    # Run optimization
    result = run_with_kernelpytorch(
        args.model,
        args.quantization,
        args.prompt,
        args.max_new_tokens
    )

    if "error" in result:
        print(f"\nNote: Full demo requires model access. Error: {result['error']}")
        print("\nShowing memory estimation instead:")
        run_memory_demo()

    print_section("Complete!")


if __name__ == "__main__":
    main()
