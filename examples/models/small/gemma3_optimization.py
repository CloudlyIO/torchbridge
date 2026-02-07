#!/usr/bin/env python3
"""
Gemma 3 12B Optimization Example

Demonstrates how to use TorchBridge to optimize Google DeepMind's Gemma 3
for efficient inference across CUDA, ROCm, and CPU backends.

Gemma 3 is multimodal (text + image), supports 128K context window,
140+ languages, and runs efficiently on a single GPU.

Models covered:
- google/gemma-3-12b-it (12B, instruction-tuned, multimodal)
- google/gemma-3-4b-it (4B, efficient, multimodal)
- google/gemma-3-1b-it (1B, edge-ready)
- google/gemma-3-27b-it (27B, highest quality)

Requirements:
    pip install transformers accelerate

Hardware requirements (12B):
    - FP16: ~24GB VRAM (A10G, L4)
    - INT8: ~12GB VRAM
    - INT4: ~7GB VRAM

Usage:
    python gemma3_optimization.py
    python gemma3_optimization.py --model google/gemma-3-4b-it
    python gemma3_optimization.py --quantization int4
    python gemma3_optimization.py --benchmark
"""

import argparse
import json
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


def check_dependencies() -> dict[str, bool]:
    """Check if required dependencies are installed."""
    deps = {}
    try:
        import transformers

        deps["transformers"] = True
        logger.info(f"transformers version: {transformers.__version__}")
    except ImportError:
        deps["transformers"] = False
    try:
        import accelerate  # noqa: F401

        deps["accelerate"] = True
    except ImportError:
        deps["accelerate"] = False
    return deps


def get_system_info() -> dict[str, Any]:
    """Gather system information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
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


def estimate_memory(model_name: str, quantization: str = "none") -> dict[str, float]:
    """Estimate memory requirements for Gemma 3."""
    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.INT4,
        }
        config = LLMConfig(
            model_name=model_name,
            quantization=quant_map.get(quantization, QuantizationMode.NONE),
        )
        optimizer = LLMOptimizer(config)
        return optimizer.estimate_memory(model_name)
    except ImportError:
        # Fallback estimates
        if "1b" in model_name.lower():
            base_gb = 2.0
        elif "4b" in model_name.lower():
            base_gb = 8.0
        elif "27b" in model_name.lower():
            base_gb = 54.0
        else:
            base_gb = 24.0  # 12B default
        multipliers = {"none": 1.0, "int8": 0.5, "int4": 0.3}
        mem = base_gb * multipliers.get(quantization, 1.0)
        return {"model_memory_gb": mem, "total_gb": mem * 1.15}


def run_optimized_inference(
    model_name: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run optimized inference with TorchBridge on Gemma 3."""
    print_section(f"TorchBridge Optimized Inference - {model_name}")

    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

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
            max_sequence_length=8192,
        )
        optimizer = LLMOptimizer(config)

        memory_est = optimizer.estimate_memory(model_name)
        print(f"Estimated memory: {memory_est['total_gb']:.1f} GB")

        if torch.cuda.is_available():
            available = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available VRAM: {available:.1f} GB")

        print("\nLoading model...")
        model, tokenizer = optimizer.optimize(model_name)

        opt_info = optimizer.get_optimization_info()
        print("\nOptimization applied:")
        for key in ["device", "dtype", "backend", "quantization", "flash_attention"]:
            print(f"  {key}: {opt_info.get(key, 'N/A')}")

        print(f"\nPrompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed generation
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generation_time = time.perf_counter() - start_time
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nGenerated ({tokens_generated} tokens in {generation_time:.2f}s):")
        print(f"  {generated_text[:500]}")
        print("\nPerformance:")
        print(f"  Latency: {generation_time:.2f}s")
        print(f"  Tokens/sec: {tokens_generated / generation_time:.1f}")

        return {
            "model_name": model_name,
            "quantization": quantization,
            "generation_time_s": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_generated / generation_time,
            "optimization_info": opt_info,
        }

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {"error": str(e)}


def run_size_comparison() -> dict[str, Any]:
    """Compare Gemma 3 model sizes for deployment planning."""
    print_section("Gemma 3 Size Comparison")

    models = [
        ("Gemma 3 1B", "google/gemma-3-1b-it", "Edge / mobile"),
        ("Gemma 3 4B", "google/gemma-3-4b-it", "Laptop / light GPU"),
        ("Gemma 3 12B", "google/gemma-3-12b-it", "Single GPU (A10G/L4)"),
        ("Gemma 3 27B", "google/gemma-3-27b-it", "High-end GPU (H100)"),
    ]

    quants = ["none", "int8", "int4"]

    print(f"{'Model':<18} {'Quant':<8} {'Memory (GB)':<12} {'Use Case':<25}")
    print("-" * 65)

    for name, model_id, use_case in models:
        for quant in quants:
            est = estimate_memory(model_id, quant)
            label = f"{quant.upper()}" if quant != "none" else "FP16"
            uc = use_case if quant == "none" else ""
            print(f"{name:<18} {label:<8} {est['total_gb']:<12.1f} {uc:<25}")
        print()

    print("Notes:")
    print("  - All Gemma 3 models support 128K context window")
    print("  - Multimodal (text + image) across all sizes")
    print("  - 140+ language support")
    print("  - 1B and 4B suitable for edge deployment via ExecuTorch")

    return {"comparison": "displayed"}


def run_benchmark(
    model_name: str,
    quantization: str,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Run structured benchmark for Gemma 3."""
    print_section(f"Benchmark - {model_name} ({quantization})")

    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

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
        model, tokenizer = optimizer.optimize(model_name)

        prompts = [
            "Explain the concept of attention in neural networks.",
            "Write a Fibonacci function in Python.",
            "Summarize the benefits of renewable energy.",
        ]

        latencies = []
        throughputs = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for _ in range(num_runs):
                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                latencies.append(elapsed)
                throughputs.append(tokens / elapsed)

        latencies.sort()

        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                "current_memory_gb": torch.cuda.memory_allocated() / 1e9,
            }

        results = {
            "model": model_name,
            "quantization": quantization,
            "num_runs": num_runs * len(prompts),
            "latency_p50_s": latencies[len(latencies) // 2],
            "latency_p95_s": latencies[int(len(latencies) * 0.95)],
            "throughput_avg_tok_s": sum(throughputs) / len(throughputs),
            **memory_stats,
            "system_info": get_system_info(),
        }

        print(f"Results ({results['num_runs']} runs):")
        print(f"  Latency p50: {results['latency_p50_s']:.3f}s")
        print(f"  Latency p95: {results['latency_p95_s']:.3f}s")
        print(f"  Throughput avg: {results['throughput_avg_tok_s']:.1f} tok/s")
        if memory_stats:
            print(f"  Peak memory: {memory_stats['peak_memory_gb']:.2f} GB")

        return results

    except ImportError as e:
        logger.error(f"Benchmark requires TorchBridge and transformers: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gemma 3 Optimization with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-12b-it",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how gradient descent works in machine learning.",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max new tokens"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--compare-sizes",
        action="store_true",
        help="Compare Gemma 3 model sizes",
    )
    parser.add_argument("--output-json", type=str, help="Save results to JSON")

    args = parser.parse_args()

    print_section("Gemma 3 Optimization with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    deps = check_dependencies()
    if not deps.get("transformers"):
        print("\nERROR: transformers required. Install: pip install transformers")
        return

    results = {}

    if args.compare_sizes:
        results = run_size_comparison()
    elif args.benchmark:
        results = run_benchmark(args.model, args.quantization)
    else:
        results = run_optimized_inference(
            args.model, args.quantization, args.prompt, args.max_new_tokens
        )

    if "error" in results:
        print(f"\nFull demo requires model access. Error: {results['error']}")

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
