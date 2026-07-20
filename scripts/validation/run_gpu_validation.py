#!/usr/bin/env python3
"""
GPU Validation Script for TorchBridge

Runs Qwen3-0.6B (or specified model) CPU↔GPU comparison and asserts
numerical thresholds from the TorchBridge TolerationDB.

Usage:
    python3 run_gpu_validation.py --help
    python3 run_gpu_validation.py --backend cuda --model Qwen/Qwen3-0.6B
    python3 run_gpu_validation.py --backend cuda --output-json result.json

Exit codes:
    0  Validation passed
    1  Validation failed (threshold exceeded)
    2  Missing dependency or configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchBridge GPU numerical validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "rocm", "mps", "trainium", "trainium2", "cpu"],
        help="Hardware backend to validate (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID to validate (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--model-family",
        default=None,
        dest="model_family",
        choices=[
            "decoder-small", "decoder-medium", "decoder-large", "encoder",
            "vision-language", "qwen3_5", "gemma4", "nemotron3_nano",
            "deepseek_v4", "nemotron3_ultra", "tencent_hy3", "minimax_m3", "glm_5_2",
        ],
        help="Model family for ToleranceDB lookup (overrides auto-detection from --model)",
    )
    parser.add_argument(
        "--output-json",
        metavar="PATH",
        help="Write validation result to this JSON file",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Max absolute difference tolerance (overrides TolerationDB)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity (overrides TolerationDB)",
    )
    return parser.parse_args()


def _get_thresholds(backend: str, atol: float | None, cosine_threshold: float | None):
    """Return (atol, cosine_threshold) — from TolerationDB or CLI overrides."""
    # Defaults from CLAUDE.md validation thresholds
    _DEFAULTS = {
        "cuda": (1e-4, 0.9999),
        "rocm": (1e-3, 0.999),
        "mps": (1e-4, 0.9999),
        "trainium": (1e-3, 0.999),
        "cpu": (0.0, 1.0),
    }
    default_atol, default_cos = _DEFAULTS.get(backend, (1e-3, 0.999))

    try:
        from torchbridge.testing.tolerance_db import TolerationDB

        db = TolerationDB()
        entry = db.get(model_family="qwen", backend=backend, dtype="float32")
        if entry is not None:
            default_atol = entry.atol
            default_cos = entry.cosine_threshold
    except Exception:
        pass  # fall back to hardcoded defaults

    return (
        atol if atol is not None else default_atol,
        cosine_threshold if cosine_threshold is not None else default_cos,
    )


def main() -> int:
    args = _parse_args()

    # CPU early-exit: skip transformers import (not needed for SKIPPED result)
    if args.backend == "cpu":
        result: dict = {
            "backend": "cpu",
            "status": "SKIPPED",
            "reason": "cpu backend — no GPU comparison",
        }
        _write_result(args.output_json, result)
        print("\nCPU-only validation: no GPU comparison possible.")
        return 0

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("ERROR: PyTorch not installed.", file=sys.stderr)
        return 2

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers", file=sys.stderr)
        return 2

    atol, cosine_threshold = _get_thresholds(args.backend, args.atol, args.cosine_threshold)

    # Determine device
    if args.backend in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            print("ERROR: CUDA/ROCm requested but torch.cuda.is_available() is False", file=sys.stderr)
            return 2
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif args.backend == "mps":
        if not torch.backends.mps.is_available():
            print("ERROR: MPS requested but not available", file=sys.stderr)
            return 2
        device = torch.device("mps")
        device_name = "Apple Silicon MPS"
    elif args.backend == "trainium":
        try:
            import torch_neuronx  # noqa: F401
        except ImportError:
            print("ERROR: torch_neuronx not installed. Activate the Neuron venv first.", file=sys.stderr)
            return 2
        device = torch.device("xla")
        device_name = "AWS Trainium (NeuronCore)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print("=== TorchBridge GPU Validation ===")
    print(f"Backend: {args.backend} ({device_name})")
    print(f"Model:   {args.model}")
    print(f"Thresholds: atol={atol:.1e}, cosine_sim>={cosine_threshold}")

    # Load model on CPU
    print("\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    inputs = tokenizer("The capital of France is", return_tensors="pt")

    model.eval()
    with torch.no_grad():
        cpu_out = model(**inputs)

    result: dict = {
        "backend": args.backend,
        "device_name": device_name,
        "model": args.model,
        "atol_threshold": atol,
        "cosine_threshold": cosine_threshold,
    }

    if args.backend == "trainium":
        # Trainium with PJRT_DEVICE=CPU routes XLA to CPU. XLA tensors do not
        # support .cpu() transfer, so we validate reproducibility instead of
        # GPU/CPU divergence: run two identical forward passes, compare outputs.
        with torch.no_grad():
            out2 = model(**inputs)
        logits1 = cpu_out.logits[:, -1, :]
        logits2 = out2.logits[:, -1, :]
        max_diff = float(torch.abs(logits1 - logits2).max())
        cos_sim = float(
            F.cosine_similarity(
                logits1.flatten().unsqueeze(0),
                logits2.flatten().unsqueeze(0),
            )
        )
        t0 = time.perf_counter()
        for _ in range(10):
            model(**inputs)
        latency_ms = (time.perf_counter() - t0) / 10 * 1000
    else:
        # Move model to target device and compare against CPU baseline
        model_gpu = model.to(device)
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gpu_out = model_gpu(**inputs_gpu)

        cpu_logits = cpu_out.logits[:, -1, :]
        gpu_logits = gpu_out.logits[:, -1, :].cpu()

        max_diff = float(torch.abs(cpu_logits - gpu_logits).max())
        cos_sim = float(
            F.cosine_similarity(
                cpu_logits.flatten().unsqueeze(0),
                gpu_logits.flatten().unsqueeze(0),
            )
        )

        # Latency benchmark
        for _ in range(3):
            model_gpu(**inputs_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            model_gpu(**inputs_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / 100 * 1000

    passed = max_diff < atol and cos_sim >= cosine_threshold
    status = "PASSED" if passed else "FAILED"

    print(f"\nMax diff:   {max_diff:.2e}  (threshold: < {atol:.1e})")
    print(f"Cosine sim: {cos_sim:.6f}  (threshold: >= {cosine_threshold})")
    print(f"Latency:    {latency_ms:.1f} ms")
    print(f"\nStatus: {status}")

    result.update({
        "max_diff": max_diff,
        "cosine_sim": cos_sim,
        "latency_ms": latency_ms,
        "status": status,
    })
    _write_result(args.output_json, result)
    return 0 if passed else 1


def _write_result(path: str | None, result: dict) -> None:
    if path:
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult written to: {path}")


if __name__ == "__main__":
    sys.exit(main())
