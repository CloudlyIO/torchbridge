#!/usr/bin/env python3
"""
TorchBridge Cross-Backend Benchmark Suite

Standard benchmark suite for evaluating LLM inference performance across
hardware backends with real model weights.

Models benchmarked:
- Llama 4 Scout 17B (MoE, 16 experts)
- DeepSeek R1 Distill 7B (reasoning)
- Qwen 3 8B (multilingual)

Metrics collected:
- Latency: p50, p95, p99
- Throughput: tokens/sec (average and peak)
- Memory: peak VRAM allocation
- Time-to-first-token (TTFT)

Backends: CUDA (H100, A10G, L4), ROCm (MI300X, MI350X), CPU

Output: JSON results + markdown report

Requirements:
    pip install transformers accelerate torch

Usage:
    python scripts/benchmark_suite.py
    python scripts/benchmark_suite.py --models llama4 qwen3
    python scripts/benchmark_suite.py --quantization int4
    python scripts/benchmark_suite.py --output-dir ./benchmark_results
    python scripts/benchmark_suite.py --quick  # Fewer runs for faster results
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Model Registry ───

MODEL_REGISTRY = {
    "llama4": {
        "name": "Llama 4 Scout",
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "architecture": "MoE (17B active / 109B total, 16 experts)",
        "params_b": 17,
        "category": "llm",
    },
    "deepseek": {
        "name": "DeepSeek R1 Distill 7B",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "architecture": "Dense (7B)",
        "params_b": 7,
        "category": "llm",
    },
    "qwen3": {
        "name": "Qwen 3 8B",
        "model_id": "Qwen/Qwen3-8B",
        "architecture": "Dense (8B)",
        "params_b": 8,
        "category": "llm",
    },
}

BENCHMARK_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function that implements binary search.",
    "What are the main differences between microservices and monolithic architectures?",
    "Describe how transformers work in deep learning.",
    "Write a SQL query to find the top 10 customers by revenue.",
]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    models: list[str] = field(default_factory=lambda: list(MODEL_REGISTRY.keys()))
    quantization: str = "none"
    num_warmup: int = 3
    num_runs: int = 10
    max_new_tokens: int = 100
    output_dir: str = "./benchmark_results"
    quick: bool = False


@dataclass
class BenchmarkResult:
    """Results from a single model benchmark."""

    model_name: str
    model_id: str
    architecture: str
    quantization: str
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    throughput_avg_tok_s: float
    throughput_peak_tok_s: float
    ttft_avg_ms: float
    peak_memory_gb: float
    current_memory_gb: float
    num_runs: int
    system_info: dict = field(default_factory=dict)
    timestamp: str = ""
    error: str | None = None


def get_system_info() -> dict[str, Any]:
    """Gather comprehensive system information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["num_gpus"] = torch.cuda.device_count()

    if hasattr(torch.version, "hip") and torch.version.hip:
        info["rocm_version"] = torch.version.hip
        info["backend"] = "ROCm"
    elif torch.cuda.is_available():
        info["backend"] = "CUDA"
    else:
        info["backend"] = "CPU"

    return info


def benchmark_model(
    model_key: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Run benchmark for a single model."""
    model_info = MODEL_REGISTRY[model_key]
    model_id = model_info["model_id"]
    model_name = model_info["name"]
    sys_info = get_system_info()

    logger.info(f"Benchmarking {model_name} ({config.quantization})...")

    try:
        from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

        quant_map = {
            "none": QuantizationMode.NONE,
            "int8": QuantizationMode.INT8,
            "int4": QuantizationMode.BNBT4,
            "fp8": QuantizationMode.FP8,
        }

        llm_config = LLMConfig(
            model_name=model_id,
            quantization=quant_map.get(config.quantization, QuantizationMode.NONE),
            use_flash_attention=True,
            use_torch_compile=True,
            compile_mode="reduce-overhead",
        )

        optimizer = LLMOptimizer(llm_config)
        model, tokenizer = optimizer.optimize(model_id)

        prompts = BENCHMARK_PROMPTS
        if config.quick:
            prompts = prompts[:2]

        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        throughputs = []
        ttfts = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(optimizer.device)

            # Warmup
            for _ in range(config.num_warmup):
                with torch.no_grad():
                    _ = model.generate(
                        **inputs, max_new_tokens=10, do_sample=False
                    )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark runs
            for _ in range(config.num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Measure TTFT (first token latency)
                ttft_start = time.perf_counter()
                with torch.no_grad():
                    model.generate(
                        **inputs, max_new_tokens=1, do_sample=False
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft = (time.perf_counter() - ttft_start) * 1000  # ms
                ttfts.append(ttft)

                # Full generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=False,
                        pad_token_id=(
                            tokenizer.pad_token_id or tokenizer.eos_token_id
                        ),
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                latencies.append(elapsed)
                throughputs.append(tokens / elapsed if elapsed > 0 else 0)

        latencies.sort()
        throughputs.sort()
        ttfts.sort()

        # Memory stats
        peak_mem = 0.0
        current_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            current_mem = torch.cuda.memory_allocated() / 1e9

        n = len(latencies)
        result = BenchmarkResult(
            model_name=model_name,
            model_id=model_id,
            architecture=model_info["architecture"],
            quantization=config.quantization,
            latency_p50_s=latencies[n // 2],
            latency_p95_s=latencies[int(n * 0.95)],
            latency_p99_s=latencies[int(n * 0.99)],
            throughput_avg_tok_s=sum(throughputs) / len(throughputs),
            throughput_peak_tok_s=max(throughputs),
            ttft_avg_ms=sum(ttfts) / len(ttfts),
            peak_memory_gb=peak_mem,
            current_memory_gb=current_mem,
            num_runs=len(latencies),
            system_info=sys_info,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Clean up
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except ImportError as e:
        logger.error(f"Import failed for {model_name}: {e}")
        return BenchmarkResult(
            model_name=model_name,
            model_id=model_id,
            architecture=model_info["architecture"],
            quantization=config.quantization,
            latency_p50_s=0, latency_p95_s=0, latency_p99_s=0,
            throughput_avg_tok_s=0, throughput_peak_tok_s=0,
            ttft_avg_ms=0, peak_memory_gb=0, current_memory_gb=0,
            num_runs=0, system_info=sys_info,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Benchmark failed for {model_name}: {e}")
        return BenchmarkResult(
            model_name=model_name,
            model_id=model_id,
            architecture=model_info["architecture"],
            quantization=config.quantization,
            latency_p50_s=0, latency_p95_s=0, latency_p99_s=0,
            throughput_avg_tok_s=0, throughput_peak_tok_s=0,
            ttft_avg_ms=0, peak_memory_gb=0, current_memory_gb=0,
            num_runs=0, system_info=sys_info,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


def generate_markdown_report(
    results: list[BenchmarkResult],
    sys_info: dict[str, Any],
) -> str:
    """Generate a markdown benchmark report."""
    lines = [
        "# TorchBridge Cross-Backend Benchmark Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Backend:** {sys_info.get('backend', 'Unknown')}",
    ]

    if sys_info.get("gpu_name"):
        lines.append(f"**GPU:** {sys_info['gpu_name']} ({sys_info.get('gpu_memory_gb', '?')} GB)")
    if sys_info.get("cuda_version"):
        lines.append(f"**CUDA:** {sys_info['cuda_version']}")
    if sys_info.get("rocm_version"):
        lines.append(f"**ROCm:** {sys_info['rocm_version']}")
    lines.append(f"**PyTorch:** {sys_info.get('pytorch_version', '?')}")
    lines.append("")

    # Summary table
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Model | Quant | p50 (s) | p95 (s) | Tok/s | TTFT (ms) | Memory (GB) |")
    lines.append("|-------|-------|---------|---------|-------|-----------|-------------|")

    for r in results:
        if r.error:
            lines.append(f"| {r.model_name} | {r.quantization} | ERROR | - | - | - | - |")
        else:
            lines.append(
                f"| {r.model_name} | {r.quantization} | "
                f"{r.latency_p50_s:.3f} | {r.latency_p95_s:.3f} | "
                f"{r.throughput_avg_tok_s:.1f} | {r.ttft_avg_ms:.1f} | "
                f"{r.peak_memory_gb:.2f} |"
            )

    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for r in results:
        lines.append(f"### {r.model_name}")
        lines.append("")
        if r.error:
            lines.append(f"**Error:** {r.error}")
        else:
            lines.append(f"- **Architecture:** {r.architecture}")
            lines.append(f"- **Model ID:** `{r.model_id}`")
            lines.append(f"- **Quantization:** {r.quantization}")
            lines.append(f"- **Runs:** {r.num_runs}")
            lines.append(f"- **Latency p50:** {r.latency_p50_s:.3f}s")
            lines.append(f"- **Latency p95:** {r.latency_p95_s:.3f}s")
            lines.append(f"- **Latency p99:** {r.latency_p99_s:.3f}s")
            lines.append(f"- **Throughput avg:** {r.throughput_avg_tok_s:.1f} tok/s")
            lines.append(f"- **Throughput peak:** {r.throughput_peak_tok_s:.1f} tok/s")
            lines.append(f"- **TTFT avg:** {r.ttft_avg_ms:.1f}ms")
            lines.append(f"- **Peak memory:** {r.peak_memory_gb:.2f} GB")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by TorchBridge Benchmark Suite*")
    lines.append("")

    return "\n".join(lines)


def run_suite(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run the full benchmark suite."""
    print(f"\n{'=' * 70}")
    print("  TorchBridge Cross-Backend Benchmark Suite")
    print(f"{'=' * 70}\n")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    print(f"\nModels: {', '.join(config.models)}")
    print(f"Quantization: {config.quantization}")
    print(f"Runs per prompt: {config.num_runs}")
    print(f"Max new tokens: {config.max_new_tokens}")
    print()

    results = []
    for model_key in config.models:
        if model_key not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_key}, skipping")
            continue

        result = benchmark_model(model_key, config)
        results.append(result)

        if result.error:
            print(f"  {result.model_name}: FAILED ({result.error})")
        else:
            print(f"  {result.model_name}: {result.throughput_avg_tok_s:.1f} tok/s, "
                  f"p50={result.latency_p50_s:.3f}s, "
                  f"mem={result.peak_memory_gb:.2f}GB")

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backend = sys_info.get("backend", "unknown").lower()

    # JSON output
    json_path = os.path.join(config.output_dir, f"benchmark_{backend}_{timestamp}.json")
    json_data = {
        "system_info": sys_info,
        "config": {
            "models": config.models,
            "quantization": config.quantization,
            "num_runs": config.num_runs,
            "max_new_tokens": config.max_new_tokens,
        },
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nJSON results: {json_path}")

    # Markdown report
    md_path = os.path.join(config.output_dir, f"benchmark_{backend}_{timestamp}.md")
    report = generate_markdown_report(results, sys_info)
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Markdown report: {md_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TorchBridge Cross-Backend Benchmark Suite"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to benchmark",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4", "fp8"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs per prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max new tokens per generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer prompts and runs)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for key, info in MODEL_REGISTRY.items():
            print(f"  {key:<12} {info['name']:<30} {info['architecture']}")
        return

    config = BenchmarkConfig(
        models=args.models,
        quantization=args.quantization,
        num_runs=3 if args.quick else args.num_runs,
        max_new_tokens=50 if args.quick else args.max_new_tokens,
        output_dir=args.output_dir,
        quick=args.quick,
    )

    run_suite(config)

    print(f"\n{'=' * 70}")
    print("  Benchmark Complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
