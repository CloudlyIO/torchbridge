#!/usr/bin/env python3
"""
Mixture of Experts (MoE) Cross-Backend Example

Demonstrates how TorchBridge handles MoE architectures across CUDA,
ROCm, Trainium, TPU, and CPU backends. MoE models activate only a subset of parameters
per token, offering better quality per FLOP than dense models.

As of Feb 2026, MoE dominates the model landscape — 9 of the top 15
models use MoE architectures (DeepSeek-V3, Llama 4, Qwen3 MoE variants).

Models covered:
- Qwen/Qwen3-30B-A3B (30B total / 3B active, primary — fits single GPU)
- Qwen/Qwen3-235B-A22B (235B total / 22B active, multi-GPU)
- deepseek-ai/DeepSeek-V3-0324 (685B total / 37B active, 256 experts, multi-node)

Key features demonstrated:
- MoE expert routing efficiency across backends
- Expert load balancing analysis
- Active parameter utilization vs total parameter count
- Cross-backend consistency for sparse computation

Requirements:
    pip install transformers accelerate

Hardware requirements (Qwen3-30B-A3B):
    - FP16: ~60GB VRAM (total params loaded, only 3B active per forward)
    - INT4: ~15GB VRAM (fits on A10G 24GB with quantization)
    - CPU: ~60GB RAM (slow but functional)

Usage:
    python moe_cross_backend.py
    python moe_cross_backend.py --model Qwen/Qwen3-235B-A22B
    python moe_cross_backend.py --quantization int4
    python moe_cross_backend.py --analyze-experts
    python moe_cross_backend.py --benchmark
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


# MoE model catalog with architecture details
MOE_MODELS = {
    "Qwen/Qwen3-30B-A3B": {
        "total_params": "30B",
        "active_params": "3B",
        "num_experts": 128,
        "top_k": 8,
        "vram_fp16_gb": 60,
        "vram_int4_gb": 15,
    },
    "Qwen/Qwen3-235B-A22B": {
        "total_params": "235B",
        "active_params": "22B",
        "num_experts": 128,
        "top_k": 8,
        "vram_fp16_gb": 470,
        "vram_int4_gb": 120,
    },
    "deepseek-ai/DeepSeek-V3-0324": {
        "total_params": "685B",
        "active_params": "37B",
        "num_experts": 256,
        "top_k": 8,
        "vram_fp16_gb": 1370,
        "vram_int4_gb": 340,
    },
}


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
        info["gpu_count"] = torch.cuda.device_count()
    if hasattr(torch.version, "hip") and torch.version.hip:
        info["backend"] = "ROCm"
    elif torch.cuda.is_available():
        info["backend"] = "CUDA"
    else:
        info["backend"] = "CPU"
    return info


def print_moe_info(model_name: str) -> None:
    """Print MoE architecture details for the selected model."""
    if model_name in MOE_MODELS:
        info = MOE_MODELS[model_name]
        print("MoE Architecture:")
        print(f"  Total parameters: {info['total_params']}")
        print(f"  Active per token: {info['active_params']}")
        print(f"  Number of experts: {info['num_experts']}")
        print(f"  Top-K routing: {info['top_k']}")
        print(f"  VRAM (FP16): ~{info['vram_fp16_gb']}GB")
        print(f"  VRAM (INT4): ~{info['vram_int4_gb']}GB")
    else:
        print(f"  Model: {model_name} (architecture details not cataloged)")


def run_optimized_inference(
    model_name: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run optimized MoE inference with TorchBridge."""
    print_section(f"TorchBridge MoE Inference - {model_name}")
    print_moe_info(model_name)

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
            use_flash_attention=True,
            use_torch_compile=True,
            compile_mode="reduce-overhead",
            max_sequence_length=4096,
        )

        optimizer = LLMOptimizer(config)

        memory_est = optimizer.estimate_memory(model_name)
        print(f"\nEstimated memory: {memory_est['total_gb']:.1f} GB")

        print("\nLoading model...")
        model, tokenizer = optimizer.optimize(model_name)

        opt_info = optimizer.get_optimization_info()
        print("\nOptimization applied:")
        for key in ["device", "dtype", "backend", "quantization"]:
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
            "architecture": "MoE",
            "quantization": quantization,
            "generation_time_s": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_generated / generation_time,
            "moe_info": MOE_MODELS.get(model_name, {}),
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
    print_moe_info(model_name)

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

        print(f"\nMoE layers found: {len(moe_layers)}")
        print(f"Total expert modules: {total_experts}")

        if moe_layers:
            print("\nMoE layer names:")
            for layer_name in moe_layers[:10]:
                print(f"  - {layer_name}")
            if len(moe_layers) > 10:
                print(f"  ... and {len(moe_layers) - 10} more")

        # Run inference and track expert utilization
        test_prompts = [
            "What is the capital of France?",
            "Solve: 2x + 5 = 15",
            "Write a haiku about the ocean.",
            "Explain the difference between TCP and UDP.",
        ]

        expert_activations = {}
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)
            with torch.no_grad():
                outputs = model(**inputs)

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
            print("\nExpert utilization per layer (first 3 layers):")
            for layer in list(expert_activations.keys())[:3]:
                activations = expert_activations[layer]
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
    """Run structured benchmark for MoE model."""
    print_section(f"MoE Benchmark - {model_name} ({quantization})")
    print_moe_info(model_name)

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
            use_flash_attention=True,
            use_torch_compile=True,
            compile_mode="reduce-overhead",
        )

        optimizer = LLMOptimizer(config)
        model, tokenizer = optimizer.optimize(model_name)

        prompts = [
            "Explain how mixture of experts routing works in transformer models.",
            "Compare the efficiency of dense vs sparse models for inference.",
            "What are the trade-offs of having more experts with fewer active?",
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
            "architecture": "MoE",
            "quantization": quantization,
            "num_runs": num_runs * len(prompts),
            "latency_p50_s": latencies[len(latencies) // 2],
            "latency_p95_s": latencies[int(len(latencies) * 0.95)],
            "throughput_avg_tok_s": sum(throughputs) / len(throughputs),
            **memory_stats,
            "moe_info": MOE_MODELS.get(model_name, {}),
            "system_info": get_system_info(),
        }

        print(f"\nResults ({results['num_runs']} runs):")
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
        description="MoE Cross-Backend Inference with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model name (MoE model)",
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
        default="Explain how mixture of experts improves model efficiency.",
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
        help="Analyze MoE expert routing and load balancing",
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

    print_section("MoE Cross-Backend Inference with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    print()
    print_moe_info(args.model)

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
