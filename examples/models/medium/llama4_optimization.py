#!/usr/bin/env python3
"""
Llama 4 Scout Optimization Example

Demonstrates how to use TorchBridge to optimize Meta Llama 4 Scout for
production inference across CUDA, ROCm, and CPU backends. Llama 4 Scout
uses a Mixture-of-Experts (MoE) architecture with 17B active parameters
and 16 experts, fitting on a single H100/A100 GPU.

Models covered:
- meta-llama/Llama-4-Scout-17B-16E (base, 109B total / 17B active)
- meta-llama/Llama-4-Scout-17B-16E-Instruct (instruction-tuned)

Key features:
- Native multimodal (text + image input)
- 10M token context window
- MoE with 16 experts (17B active per forward pass)
- Outperforms Gemma 3, Gemini 2.0 Flash-Lite, Mistral 3.1

Requirements:
    pip install transformers accelerate

Hardware requirements:
    - FP16: ~35GB VRAM (H100, MI300X) — fits single GPU
    - FP8:  ~18GB VRAM (H100, B200, MI350X)
    - INT4: ~14GB VRAM (A10G, L4)

Usage:
    python llama4_optimization.py
    python llama4_optimization.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct
    python llama4_optimization.py --quantization int4
    python llama4_optimization.py --benchmark
    python llama4_optimization.py --export safetensors
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
        logger.error("transformers not installed")

    try:
        import accelerate  # noqa: F401

        deps["accelerate"] = True
    except ImportError:
        deps["accelerate"] = False
        logger.warning("accelerate not installed (recommended for Llama 4)")

    return deps


def get_system_info() -> dict[str, Any]:
    """Gather system information for benchmarking context."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        info["cuda_version"] = torch.version.cuda or "N/A"
    if hasattr(torch.version, "hip") and torch.version.hip:
        info["rocm_version"] = torch.version.hip
        info["backend"] = "ROCm"
    elif torch.cuda.is_available():
        info["backend"] = "CUDA"
    else:
        info["backend"] = "CPU"
    return info


def estimate_memory(model_name: str, quantization: str = "none") -> dict[str, float]:
    """Estimate memory requirements for Llama 4 Scout."""
    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.INT4,
            "fp8": QuantizationMode.FP8,
        }
        config = LLMConfig(
            model_name=model_name,
            quantization=quant_map.get(quantization, QuantizationMode.NONE),
        )
        optimizer = LLMOptimizer(config)
        return optimizer.estimate_memory(model_name)
    except ImportError:
        # Fallback estimates for Llama 4 Scout (109B total, 17B active)
        # Active parameters dominate VRAM during inference
        base_gb = 35.0  # FP16 for active params + expert routing
        multipliers = {"none": 1.0, "int8": 0.55, "int4": 0.4, "fp8": 0.5}
        mem = base_gb * multipliers.get(quantization, 1.0)
        return {"model_memory_gb": mem, "total_gb": mem * 1.15}


def run_optimized_inference(
    model_name: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run optimized inference with TorchBridge on Llama 4 Scout."""
    print_section("TorchBridge Optimized Inference - Llama 4 Scout")

    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.BNBT4,
            "fp8": QuantizationMode.FP8,
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
        print("Architecture: MoE (109B total, 17B active, 16 experts)")

        if torch.cuda.is_available():
            available = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available VRAM: {available:.1f} GB")
            if memory_est["total_gb"] > available * 0.9:
                print("WARNING: Model may not fit — try --quantization int4 or fp8")

        print("\nLoading model (MoE models take longer to load)...")
        model, tokenizer = optimizer.optimize(model_name)

        opt_info = optimizer.get_optimization_info()
        print("\nOptimization applied:")
        for key in ["device", "dtype", "backend", "quantization", "flash_attention"]:
            print(f"  {key}: {opt_info.get(key, 'N/A')}")

        # Generation
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
        print(f"  Time-to-first-token: ~{generation_time / max(tokens_generated, 1) * 1000:.1f}ms")

        return {
            "model_name": model_name,
            "architecture": "MoE (17B active / 109B total, 16 experts)",
            "quantization": quantization,
            "generation_time_s": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_generated / generation_time,
            "optimization_info": opt_info,
        }

    except ImportError as e:
        logger.error(f"TorchBridge import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"error": str(e)}


def run_benchmark(
    model_name: str,
    quantization: str,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Run structured benchmark across multiple iterations."""
    print_section(f"Benchmark - Llama 4 Scout ({quantization})")

    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.BNBT4,
            "fp8": QuantizationMode.FP8,
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
            "Explain the theory of relativity in simple terms.",
            "Write a Python function to sort a list of integers.",
            "What are the key differences between TCP and UDP?",
        ]

        latencies = []
        throughputs = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)

            # Warmup
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
        throughputs.sort()

        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                "current_memory_gb": torch.cuda.memory_allocated() / 1e9,
            }

        results = {
            "model": model_name,
            "architecture": "MoE (17B active / 109B total, 16 experts)",
            "quantization": quantization,
            "num_runs": num_runs * len(prompts),
            "latency_p50_s": latencies[len(latencies) // 2],
            "latency_p95_s": latencies[int(len(latencies) * 0.95)],
            "latency_p99_s": latencies[int(len(latencies) * 0.99)],
            "throughput_avg_tok_s": sum(throughputs) / len(throughputs),
            "throughput_max_tok_s": max(throughputs),
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


def run_export(model_name: str, export_format: str) -> dict[str, Any]:
    """Export optimized model to production format."""
    print_section(f"Export - Llama 4 Scout to {export_format}")

    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer

        config = LLMConfig(model_name=model_name, use_flash_attention=True)
        optimizer = LLMOptimizer(config)
        model, tokenizer = optimizer.optimize(model_name)

        export_path = f"./exports/llama4_scout_{export_format}"

        if export_format == "safetensors":
            print("Exporting to SafeTensors...")
            model.save_pretrained(export_path, safe_serialization=True)
            tokenizer.save_pretrained(export_path)
            print(f"Saved to {export_path}/")

        elif export_format == "onnx":
            print("Exporting to ONNX...")
            example_input = tokenizer(
                "Hello world", return_tensors="pt"
            ).to(optimizer.device)
            torch.onnx.export(
                model,
                (example_input["input_ids"],),
                f"{export_path}.onnx",
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=17,
            )
            print(f"Saved to {export_path}.onnx")

        else:
            return {"error": f"Unknown format: {export_format}"}

        return {"format": export_format, "path": export_path, "status": "success"}

    except ImportError as e:
        logger.error(f"Export requires TorchBridge and transformers: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return {"error": str(e)}


def run_memory_comparison():
    """Compare memory usage across quantization modes."""
    print_section("Memory Comparison - Llama 4 Scout (17B active / 109B total)")

    configs = [
        ("FP16 (baseline)", "none"),
        ("INT8 (dynamic)", "int8"),
        ("INT4 (BnB/GPTQ)", "int4"),
        ("FP8 (H100+)", "fp8"),
    ]

    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    print(f"{'Configuration':<25} {'Model (GB)':<12} {'Total (GB)':<12}")
    print("-" * 50)

    for label, quant in configs:
        est = estimate_memory(model_name, quant)
        print(f"{label:<25} {est['model_memory_gb']:<12.1f} {est['total_gb']:<12.1f}")

    print()
    print("Notes:")
    print("  - Llama 4 Scout fits on a single H100 (80GB) in FP16")
    print("  - INT4 enables running on A10G (24GB) or L4 (24GB)")
    print("  - FP8 requires H100, B200, or MI350X hardware")
    print("  - MoE: only 17B of 109B total params active per forward pass")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Llama 4 Scout Optimization with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4", "fp8"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how neural networks learn from data in simple terms.",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--export",
        type=str,
        choices=["onnx", "safetensors"],
        help="Export model to format",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only show memory comparison",
    )
    parser.add_argument("--output-json", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    print_section("Llama 4 Scout Optimization with TorchBridge")

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
        run_memory_comparison()
        return

    if args.benchmark:
        results = run_benchmark(args.model, args.quantization)
    elif args.export:
        results = run_export(args.model, args.export)
    else:
        results = run_optimized_inference(
            args.model, args.quantization, args.prompt, args.max_new_tokens
        )

    if "error" in results:
        print(f"\nFull demo requires model access. Error: {results['error']}")
        print("\nShowing memory comparison instead:")
        run_memory_comparison()

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
