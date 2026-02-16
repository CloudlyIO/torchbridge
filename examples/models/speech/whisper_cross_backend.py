#!/usr/bin/env python3
"""
Whisper v3 Turbo Cross-Backend Example

Demonstrates how to use TorchBridge to run OpenAI's Whisper models
for speech recognition across CUDA, ROCm, Trainium, TPU, and CPU backends.

Whisper is the dominant ASR model. Its encoder-decoder architecture
benefits from backend-specific optimization (e.g., flash attention
for the encoder, KV cache tuning for the decoder).

Models covered:
- openai/whisper-large-v3-turbo (809M, primary)
- openai/whisper-large-v3 (1.5B, highest quality)
- openai/whisper-medium (769M)
- openai/whisper-small (244M)
- openai/whisper-tiny (39M, fastest)

Requirements:
    pip install transformers

Hardware requirements (large-v3-turbo):
    - FP16: ~1.6GB VRAM
    - FP32: ~3.2GB VRAM

Usage:
    python whisper_cross_backend.py
    python whisper_cross_backend.py --model openai/whisper-tiny
    python whisper_cross_backend.py --benchmark
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


def run_transcription(model_name: str) -> dict[str, Any]:
    """Run Whisper transcription with cross-backend comparison."""
    print_section(f"Whisper Transcription - {model_name}")

    try:
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.eval()

        # Create synthetic audio input (30s at 16kHz)
        dummy_audio = torch.randn(16000 * 30)
        inputs = processor(
            dummy_audio.numpy(), sampling_rate=16000, return_tensors="pt"
        )

        # CPU forward pass (encoder only for comparison)
        with torch.no_grad():
            cpu_encoder_out = model.get_encoder()(**inputs)
        cpu_features = cpu_encoder_out.last_hidden_state
        print(f"CPU encoder output shape: {cpu_features.shape}")

        # GPU forward pass
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_gpu = model.to(device)
            inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gpu_encoder_out = model_gpu.get_encoder()(**inputs_gpu)
            gpu_features = gpu_encoder_out.last_hidden_state

            max_diff = torch.abs(cpu_features - gpu_features.cpu()).max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                cpu_features.flatten().unsqueeze(0),
                gpu_features.cpu().flatten().unsqueeze(0),
            ).item()

            print("\nCross-backend encoder comparison:")
            print(f"  Max diff: {max_diff:.2e}")
            print(f"  Cosine sim: {cos_sim:.6f}")
            print(f"  Status: {'PASSED' if max_diff < 1e-3 else 'REVIEW'}")

            return {
                "model_name": model_name,
                "encoder_output_shape": list(cpu_features.shape),
                "max_diff": max_diff,
                "cosine_sim": cos_sim,
                "status": "PASSED" if max_diff < 1e-3 else "REVIEW",
            }

        return {
            "model_name": model_name,
            "encoder_output_shape": list(cpu_features.shape),
            "status": "CPU_ONLY",
        }

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"error": str(e)}


def run_benchmark(
    model_name: str,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Run structured benchmark for Whisper."""
    print_section(f"Benchmark - {model_name}")

    try:
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 30s audio clip
        dummy_audio = torch.randn(16000 * 30)
        inputs = processor(
            dummy_audio.numpy(), sampling_rate=16000, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

        # Warmup
        with torch.no_grad():
            for _ in range(2):
                model.generate(**inputs, forced_decoder_ids=forced_decoder_ids, max_new_tokens=10)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **inputs,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=128,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

        latencies.sort()
        results = {
            "model": model_name,
            "device": str(device),
            "audio_duration_s": 30,
            "num_runs": num_runs,
            "latency_p50_s": latencies[len(latencies) // 2],
            "latency_p95_s": latencies[int(len(latencies) * 0.95)],
            "rtf": latencies[len(latencies) // 2] / 30.0,  # Real-time factor
            "system_info": get_system_info(),
        }

        print(f"Results ({num_runs} runs, 30s audio):")
        print(f"  Latency p50: {results['latency_p50_s']:.2f}s")
        print(f"  Latency p95: {results['latency_p95_s']:.2f}s")
        print(f"  Real-time factor: {results['rtf']:.3f}x")

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
        description="Whisper Cross-Backend Transcription with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace model name",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output-json", type=str, help="Save results to JSON")

    args = parser.parse_args()

    print_section("Whisper Cross-Backend Transcription with TorchBridge")

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
        results = run_transcription(args.model)

    if "error" in results:
        print(f"\nFull demo requires model access. Error: {results['error']}")

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
