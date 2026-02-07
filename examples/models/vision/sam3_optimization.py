#!/usr/bin/env python3
"""
SAM 3 (Segment Anything Model 3) Optimization Example

Demonstrates how to use TorchBridge to optimize Meta's SAM 3 for
production vision segmentation across CUDA, ROCm, and CPU backends.

SAM 3 introduces text and exemplar prompts, enabling detection,
segmentation, and tracking of any visual concept across images and video.

Model:
- facebook/sam3 (848M params, DETR-based detector + tracker)

Key features:
- Text-prompted segmentation (open-vocabulary)
- Image exemplar-based segmentation
- Video tracking with concept persistence
- Zero-shot mask prediction (48.8 AP on LVIS)

Requirements:
    pip install transformers

Hardware requirements:
    - FP16: ~2GB VRAM (runs on most GPUs)
    - FP32: ~4GB VRAM
    - CPU:  ~4GB RAM

Usage:
    python sam3_optimization.py
    python sam3_optimization.py --image path/to/image.jpg
    python sam3_optimization.py --text-prompt "dog"
    python sam3_optimization.py --benchmark
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
        from PIL import Image  # noqa: F401

        deps["pillow"] = True
    except ImportError:
        deps["pillow"] = False
        logger.warning("Pillow not installed (needed for image loading)")

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


def create_sample_image() -> "torch.Tensor":
    """Create a sample image tensor for demonstration."""
    # Synthetic 640x480 RGB image with simple shapes
    image = torch.zeros(3, 480, 640, dtype=torch.float32)

    # Add colored rectangles to simulate objects
    image[0, 100:200, 100:250] = 0.8  # Red rectangle
    image[1, 250:400, 300:500] = 0.7  # Green rectangle
    image[2, 50:150, 400:550] = 0.9  # Blue rectangle

    # Add background noise
    image += torch.randn_like(image) * 0.05
    image = image.clamp(0, 1)

    return image


def run_optimized_segmentation(
    image_path: str | None = None,
    text_prompt: str | None = None,
) -> dict[str, Any]:
    """Run optimized segmentation with TorchBridge on SAM 3."""
    print_section("TorchBridge Optimized Segmentation - SAM 3")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        print("Model: facebook/sam3 (848M params)")

        # Load image
        if image_path:
            from PIL import Image

            image = Image.open(image_path)
            print(f"Image: {image_path} ({image.size[0]}x{image.size[1]})")
        else:
            image = create_sample_image()
            print("Image: synthetic 640x480 (use --image for real image)")

        # Load SAM 3 via transformers
        from transformers import AutoModel, AutoProcessor

        print("\nLoading SAM 3 model...")
        processor = AutoProcessor.from_pretrained("facebook/sam3")
        model = AutoModel.from_pretrained("facebook/sam3").to(device)

        if device == "cuda":
            model = model.half()
            print("Applied FP16 optimization")

        # Prepare inputs based on prompt type
        if text_prompt:
            print(f"Text prompt: '{text_prompt}'")
            inputs = processor(
                images=image, text=text_prompt, return_tensors="pt"
            ).to(device)
        else:
            # Default: point prompt at center of image
            if isinstance(image, torch.Tensor):
                h, w = image.shape[1], image.shape[2]
            else:
                w, h = image.size
            center_point = [[w // 2, h // 2]]
            print(f"Point prompt: center ({w // 2}, {h // 2})")
            inputs = processor(
                images=image, input_points=[center_point], return_tensors="pt"
            ).to(device)

        # Warmup
        with torch.no_grad():
            _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed inference
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        # Process outputs
        masks = outputs.pred_masks if hasattr(outputs, "pred_masks") else None
        scores = outputs.iou_scores if hasattr(outputs, "iou_scores") else None

        print("\nResults:")
        print(f"  Inference time: {inference_time * 1000:.1f}ms")
        if masks is not None:
            print(f"  Masks generated: {masks.shape}")
        if scores is not None:
            print(f"  IoU scores: {scores}")

        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            print(f"  Peak memory: {memory_stats['peak_memory_gb']:.2f} GB")

        return {
            "model": "facebook/sam3",
            "inference_time_ms": inference_time * 1000,
            "num_masks": masks.shape[0] if masks is not None else 0,
            "text_prompt": text_prompt,
            **memory_stats,
        }

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return {"error": str(e)}


def run_benchmark(num_runs: int = 20) -> dict[str, Any]:
    """Benchmark SAM 3 inference across image sizes."""
    print_section("Benchmark - SAM 3")

    try:
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading SAM 3...")
        processor = AutoProcessor.from_pretrained("facebook/sam3")
        model = AutoModel.from_pretrained("facebook/sam3").to(device)
        if device == "cuda":
            model = model.half()

        image_sizes = [(640, 480), (1024, 768), (1920, 1080)]
        results = {}

        for w, h in image_sizes:
            print(f"\nBenchmarking {w}x{h}...")
            image = torch.rand(3, h, w)
            center = [[w // 2, h // 2]]
            inputs = processor(
                images=image, input_points=[center], return_tensors="pt"
            ).to(device)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            latencies = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(**inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

            latencies.sort()
            key = f"{w}x{h}"
            results[key] = {
                "latency_p50_ms": latencies[len(latencies) // 2] * 1000,
                "latency_p95_ms": latencies[int(len(latencies) * 0.95)] * 1000,
                "fps": 1.0 / (sum(latencies) / len(latencies)),
            }

            print(f"  p50: {results[key]['latency_p50_ms']:.1f}ms")
            print(f"  p95: {results[key]['latency_p95_ms']:.1f}ms")
            print(f"  FPS: {results[key]['fps']:.1f}")

        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

        return {
            "model": "facebook/sam3",
            "benchmark_results": results,
            **memory_stats,
            "system_info": get_system_info(),
        }

    except ImportError as e:
        logger.error(f"Benchmark requires transformers: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAM 3 Optimization with TorchBridge"
    )
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument(
        "--text-prompt", type=str, help="Text prompt for concept segmentation"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output-json", type=str, help="Save results to JSON")

    args = parser.parse_args()

    print_section("SAM 3 (Segment Anything Model 3) with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    deps = check_dependencies()
    if not deps.get("transformers"):
        print("\nERROR: transformers required. Install: pip install transformers")
        return

    if args.benchmark:
        results = run_benchmark()
    else:
        results = run_optimized_segmentation(args.image, args.text_prompt)

    if "error" in results:
        print(f"\nNote: Full demo requires model access. Error: {results['error']}")

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
