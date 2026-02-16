#!/usr/bin/env python3
"""
DINOv2 Cross-Backend Example

Demonstrates how to use TorchBridge to run Meta's DINOv2 vision encoder
for universal feature extraction across CUDA, ROCm, Trainium, TPU, and CPU backends.

DINOv2 is a self-supervised ViT that produces universal visual features
without needing fine-tuning. It serves as a backbone for classification,
segmentation, depth estimation, and multimodal pipelines.

Models covered:
- facebook/dinov2-base (86M, 768-dim features)
- facebook/dinov2-small (22M, 384-dim features)
- facebook/dinov2-large (300M, 1024-dim features)
- facebook/dinov2-giant (1.1B, 1536-dim features)

Requirements:
    pip install transformers Pillow

Hardware requirements (base):
    - FP16: ~0.3GB VRAM
    - FP32: ~0.6GB VRAM

Usage:
    python dinov2_cross_backend.py
    python dinov2_cross_backend.py --model facebook/dinov2-small
    python dinov2_cross_backend.py --benchmark
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


def run_feature_extraction(
    model_name: str,
) -> dict[str, Any]:
    """Run DINOv2 feature extraction with cross-backend comparison."""
    print_section(f"DINOv2 Feature Extraction - {model_name}")

    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Create synthetic image input (batch of 4, 224x224 RGB)
        dummy_images = torch.randn(4, 3, 224, 224)
        inputs = processor(images=dummy_images, return_tensors="pt", do_rescale=False)

        # CPU forward pass
        with torch.no_grad():
            cpu_out = model(**inputs)
        cpu_features = cpu_out.last_hidden_state
        print(f"CPU feature shape: {cpu_features.shape}")
        print(f"CPU feature norm: {cpu_features.norm(dim=-1).mean():.4f}")

        # GPU forward pass (if available)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_gpu = model.to(device)
            inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gpu_out = model_gpu(**inputs_gpu)
            gpu_features = gpu_out.last_hidden_state

            # Cross-backend comparison
            max_diff = torch.abs(cpu_features - gpu_features.cpu()).max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                cpu_features.flatten().unsqueeze(0),
                gpu_features.cpu().flatten().unsqueeze(0),
            ).item()

            print("\nCross-backend comparison:")
            print(f"  Max diff: {max_diff:.2e}")
            print(f"  Cosine sim: {cos_sim:.6f}")
            print(f"  Status: {'PASSED' if max_diff < 1e-4 else 'REVIEW'}")

            return {
                "model_name": model_name,
                "feature_shape": list(cpu_features.shape),
                "max_diff": max_diff,
                "cosine_sim": cos_sim,
                "status": "PASSED" if max_diff < 1e-4 else "REVIEW",
            }

        return {
            "model_name": model_name,
            "feature_shape": list(cpu_features.shape),
            "status": "CPU_ONLY",
        }

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return {"error": str(e)}


def run_benchmark(
    model_name: str,
    num_runs: int = 10,
) -> dict[str, Any]:
    """Run structured benchmark for DINOv2."""
    print_section(f"Benchmark - {model_name}")

    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        dummy_images = torch.randn(4, 3, 224, 224)
        inputs = processor(images=dummy_images, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        results = {
            "model": model_name,
            "device": str(device),
            "batch_size": 4,
            "num_runs": num_runs,
            "latency_p50_ms": latencies[len(latencies) // 2],
            "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
            "throughput_images_per_sec": 4000.0 / latencies[len(latencies) // 2],
            "system_info": get_system_info(),
        }

        print(f"Results ({num_runs} runs, batch=4):")
        print(f"  Latency p50: {results['latency_p50_ms']:.1f} ms")
        print(f"  Latency p95: {results['latency_p95_ms']:.1f} ms")
        print(f"  Throughput: {results['throughput_images_per_sec']:.1f} img/s")

        return results

    except ImportError as e:
        logger.error(f"Benchmark requires transformers: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DINOv2 Cross-Backend Feature Extraction with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/dinov2-base",
        help="HuggingFace model name",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output-json", type=str, help="Save results to JSON")

    args = parser.parse_args()

    print_section("DINOv2 Cross-Backend Feature Extraction with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    deps = check_dependencies()
    if not deps.get("transformers"):
        print("\nERROR: transformers required. Install: pip install transformers")
        return

    if args.benchmark:
        results = run_benchmark(args.model)
    else:
        results = run_feature_extraction(args.model)

    if "error" in results:
        print(f"\nFull demo requires model access. Error: {results['error']}")

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
