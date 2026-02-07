#!/usr/bin/env python3
"""
Qwen 3 8B Optimization Example

Demonstrates how to use TorchBridge to optimize Alibaba's Qwen 3 models
for multilingual enterprise inference across CUDA, ROCm, and CPU backends.

Qwen 3 supports hybrid thinking modes (thinking + non-thinking) and
140+ languages, making it ideal for global enterprise deployments.

Models covered:
- Qwen/Qwen3-8B (8B dense, base)
- Qwen/Qwen3-8B-Instruct (8B dense, instruction-tuned)
- Qwen/Qwen3-4B (4B dense, efficient)
- Qwen/Qwen3-14B (14B dense, higher quality)

Requirements:
    pip install transformers accelerate

Hardware requirements:
    - FP16: ~16GB VRAM (A10G, L4, MI300X)
    - INT8: ~8GB VRAM
    - INT4: ~5GB VRAM

Usage:
    python qwen3_optimization.py
    python qwen3_optimization.py --model Qwen/Qwen3-4B
    python qwen3_optimization.py --quantization int4
    python qwen3_optimization.py --multilingual
    python qwen3_optimization.py --benchmark
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
    """Estimate memory requirements for Qwen 3."""
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
        # Fallback: estimate based on model name
        if "4b" in model_name.lower():
            base_gb = 8.0
        elif "14b" in model_name.lower():
            base_gb = 28.0
        elif "32b" in model_name.lower():
            base_gb = 64.0
        else:
            base_gb = 16.0  # 8B default
        multipliers = {"none": 1.0, "int8": 0.5, "int4": 0.3}
        mem = base_gb * multipliers.get(quantization, 1.0)
        return {"model_memory_gb": mem, "total_gb": mem * 1.15}


def run_optimized_inference(
    model_name: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run optimized inference with TorchBridge on Qwen 3."""
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


def run_multilingual_demo(model_name: str, quantization: str) -> dict[str, Any]:
    """Demonstrate Qwen 3's multilingual capabilities across backends."""
    print_section("Multilingual Demo - Qwen 3 (140+ languages)")

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
        )
        optimizer = LLMOptimizer(config)
        model, tokenizer = optimizer.optimize(model_name)

        # Multilingual prompts
        prompts = {
            "English": "What is the capital of France?",
            "Chinese": "\u6cd5\u56fd\u7684\u9996\u90fd\u662f\u54ea\u91cc\uff1f",
            "Japanese": "\u30d5\u30e9\u30f3\u30b9\u306e\u9996\u90fd\u306f\u3069\u3053\u3067\u3059\u304b\uff1f",
            "Arabic": "\u0645\u0627 \u0647\u064a \u0639\u0627\u0635\u0645\u0629 \u0641\u0631\u0646\u0633\u0627\u061f",
            "Spanish": "\u00bfCu\u00e1l es la capital de Francia?",
        }

        results = {}

        for lang, prompt in prompts.items():
            print(f"\n[{lang}] Prompt: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

            print(f"[{lang}] Response: {response[:200]}")
            print(f"[{lang}] {tokens} tokens, {elapsed:.2f}s, {tokens/elapsed:.1f} tok/s")

            results[lang] = {
                "tokens": tokens,
                "latency_s": elapsed,
                "tokens_per_sec": tokens / elapsed,
            }

        return {"multilingual_results": results}

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Multilingual demo failed: {e}")
        return {"error": str(e)}


def run_benchmark(
    model_name: str,
    quantization: str,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Run structured benchmark for Qwen 3."""
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
            "Explain quantum computing in simple terms.",
            "Write a SQL query to find duplicate rows in a table.",
            "What are the main differences between Python and Rust?",
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
        description="Qwen 3 Optimization with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
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
        default="Explain how transformers work in deep learning.",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max new tokens"
    )
    parser.add_argument(
        "--multilingual", action="store_true", help="Run multilingual demo"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--memory-only", action="store_true", help="Show memory only")
    parser.add_argument("--output-json", type=str, help="Save results to JSON")

    args = parser.parse_args()

    print_section("Qwen 3 Optimization with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    deps = check_dependencies()
    if not deps.get("transformers"):
        print("\nERROR: transformers required. Install: pip install transformers")
        return

    results = {}

    if args.memory_only:
        print_section("Memory Comparison - Qwen 3 8B")
        configs = [("FP16", "none"), ("INT8", "int8"), ("INT4", "int4")]
        print(f"{'Config':<15} {'Model (GB)':<12} {'Total (GB)':<12}")
        print("-" * 40)
        for label, quant in configs:
            est = estimate_memory(args.model, quant)
            print(f"{label:<15} {est['model_memory_gb']:<12.1f} {est['total_gb']:<12.1f}")
        return

    if args.multilingual:
        results = run_multilingual_demo(args.model, args.quantization)
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
