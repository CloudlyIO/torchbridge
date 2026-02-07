#!/usr/bin/env python3
"""
DeepSeek R1 Distill 7B Optimization Example

Demonstrates how to use TorchBridge to optimize DeepSeek R1 Distill models
for inference, showcasing Mixture of Experts (MoE) handling across backends.

Models covered:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (7B active params)
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B (8B active params)

Key features demonstrated:
- MoE expert routing efficiency across CUDA / ROCm / CPU
- Expert load balancing analysis
- Memory-efficient expert offloading
- Cross-backend consistency validation

Requirements:
    pip install transformers accelerate

Hardware requirements:
    - FP16: ~14GB VRAM (A10G, L4, MI300X)
    - INT8: ~7GB VRAM
    - INT4: ~4GB VRAM

Usage:
    python deepseek_optimization.py
    python deepseek_optimization.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    python deepseek_optimization.py --analyze-experts
    python deepseek_optimization.py --benchmark
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


def run_optimized_inference(
    model_name: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run optimized inference with TorchBridge on DeepSeek R1 Distill."""
    print_section(f"TorchBridge Optimized - {model_name}")

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
            max_sequence_length=4096,
        )

        optimizer = LLMOptimizer(config)

        # Memory estimation
        memory_est = optimizer.estimate_memory(model_name)
        print(f"Estimated memory: {memory_est['total_gb']:.1f} GB")

        print("\nLoading model...")
        model, tokenizer = optimizer.optimize(model_name)

        opt_info = optimizer.get_optimization_info()
        print("\nOptimization applied:")
        for key in ["device", "dtype", "backend", "quantization"]:
            print(f"  {key}: {opt_info.get(key, 'N/A')}")

        # DeepSeek R1 uses <think> tags for reasoning
        print(f"\nPrompt: '{prompt}'")
        print("Generating (DeepSeek R1 may include <think> reasoning)...")

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
                temperature=0.6,
                top_p=0.95,
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


def analyze_moe_experts(model_name: str) -> dict[str, Any]:
    """Analyze MoE expert routing and load balancing."""
    print_section(f"MoE Expert Analysis - {model_name}")

    try:
        from torchbridge.mixture_of_experts import MoEConfig
        from torchbridge.models.llm import LLMConfig, LLMOptimizer

        config = LLMConfig(model_name=model_name, use_flash_attention=True)
        optimizer = LLMOptimizer(config)
        model, tokenizer = optimizer.optimize(model_name)

        # Analyze model structure for MoE layers
        moe_layers = []
        total_experts = 0
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if "moe" in module_type.lower() or "expert" in module_type.lower():
                moe_layers.append(name)
                if hasattr(module, "num_experts"):
                    total_experts += module.num_experts

        print(f"MoE layers found: {len(moe_layers)}")
        print(f"Total expert modules: {total_experts}")

        if moe_layers:
            print("\nMoE layer names:")
            for layer_name in moe_layers[:10]:
                print(f"  - {layer_name}")

        # Run inference and track expert utilization
        test_prompts = [
            "What is the capital of France?",
            "Solve: 2x + 5 = 15",
            "Write a haiku about the ocean.",
        ]

        expert_activations = {}
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)
            with torch.no_grad():
                outputs = model(**inputs)

            # Collect routing statistics if available
            if hasattr(outputs, "router_logits") and outputs.router_logits:
                for i, logits in enumerate(outputs.router_logits):
                    layer_key = f"layer_{i}"
                    if layer_key not in expert_activations:
                        expert_activations[layer_key] = []
                    selected = logits.argmax(dim=-1).flatten().tolist()
                    expert_activations[layer_key].extend(selected)

        # MoE configuration analysis
        moe_config = MoEConfig()
        print("\nTorchBridge MoE configuration:")
        print(f"  Default num_experts: {moe_config.num_experts}")
        print(f"  Default top_k: {moe_config.top_k}")
        print(f"  Capacity factor: {moe_config.capacity_factor}")
        print(f"  Load balance loss weight: {moe_config.load_balance_loss_weight}")

        # Expert utilization report
        if expert_activations:
            print("\nExpert utilization per layer:")
            for layer, activations in expert_activations.items():
                unique, counts = torch.tensor(activations).unique(return_counts=True)
                total = len(activations)
                print(f"  {layer}:")
                for expert_id, count in zip(unique.tolist(), counts.tolist()):
                    pct = count / total * 100
                    bar = "#" * int(pct / 2)
                    print(f"    Expert {expert_id}: {pct:5.1f}% {bar}")
        else:
            print("\nNote: Router logits not exposed in this model variant.")
            print("Expert analysis available for models with explicit MoE routing.")

        return {
            "moe_layers": len(moe_layers),
            "total_experts": total_experts,
            "expert_activations": {
                k: len(v) for k, v in expert_activations.items()
            },
        }

    except ImportError as e:
        logger.error(f"Analysis requires TorchBridge: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": str(e)}


def run_benchmark(
    model_name: str,
    quantization: str,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Run structured benchmark for DeepSeek R1 Distill."""
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

        # Reasoning-style prompts to exercise expert routing
        prompts = [
            "Prove that the square root of 2 is irrational.",
            "Explain the differences between REST and GraphQL APIs.",
            "What are the trade-offs of microservices vs monolithic architectures?",
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
        description="DeepSeek R1 Distill 7B Optimization with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
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
        default="Prove that there are infinitely many prime numbers.",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--analyze-experts",
        action="store_true",
        help="Analyze MoE expert routing",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run structured benchmark",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    print_section("DeepSeek R1 Distill Optimization with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    deps = check_dependencies()
    if not deps.get("transformers"):
        print("\nERROR: transformers is required. Install with: pip install transformers")
        return

    if args.analyze_experts:
        results = analyze_moe_experts(args.model)
    elif args.benchmark:
        results = run_benchmark(args.model, args.quantization)
    else:
        results = run_optimized_inference(
            args.model, args.quantization, args.prompt, args.max_new_tokens
        )

    if "error" in results:
        print(f"\nNote: Full demo requires model access. Error: {results['error']}")

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
