#!/usr/bin/env python3
"""
Cross-Backend Validation for BERT SQuAD

Validates that BERT produces consistent outputs across different backends
(CUDA, ROCm, XPU, MPS, CPU) to ensure numerical parity.

Usage:
    python validate_cross_backend.py
    python validate_cross_backend.py --model checkpoints/bert_squad_best.pt
    python validate_cross_backend.py --tolerance 1e-4
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# TorchBridge imports
try:
    from torchbridge.backends import detect_best_backend, BackendType
    from torchbridge.core.hardware_detector import detect_hardware
    TORCHBRIDGE_AVAILABLE = True
except ImportError:
    TORCHBRIDGE_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of cross-backend validation."""
    backend: str
    device: str
    inference_time_ms: float
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    memory_mb: float | None = None


def get_available_backends() -> list[tuple[str, torch.device]]:
    """Detect all available compute backends."""
    backends = []

    # CPU is always available
    backends.append(("cpu", torch.device("cpu")))

    # CUDA/ROCm (both use torch.cuda API)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if "AMD" in device_name or "Radeon" in device_name:
            backends.append(("rocm", torch.device("cuda")))
        else:
            backends.append(("cuda", torch.device("cuda")))

    # Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.append(("mps", torch.device("mps")))

    # Intel XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        backends.append(("xpu", torch.device("xpu")))

    return backends


def synchronize_device(device: torch.device):
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def get_memory_allocated(device: torch.device) -> float | None:
    """Get memory allocated in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    elif device.type == "xpu" and hasattr(torch.xpu, "memory_allocated"):
        return torch.xpu.memory_allocated(device) / 1024 / 1024
    return None


def run_inference(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 3,
    iterations: int = 10,
) -> ValidationResult:
    """Run inference on a specific device."""
    backend_name = device.type
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        if "AMD" in device_name or "Radeon" in device_name:
            backend_name = "rocm"

    # Move model and inputs to device
    model = model.to(device)
    model.eval()
    inputs_device = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs_device)
    synchronize_device(device)

    # Timed inference
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            synchronize_device(device)
            start = time.perf_counter()
            outputs = model(**inputs_device)
            synchronize_device(device)
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    memory = get_memory_allocated(device)

    return ValidationResult(
        backend=backend_name,
        device=str(device),
        inference_time_ms=avg_time,
        start_logits=outputs.start_logits.cpu(),
        end_logits=outputs.end_logits.cpu(),
        memory_mb=memory,
    )


def compare_outputs(
    reference: ValidationResult,
    target: ValidationResult,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare outputs between two backends."""
    # Compute differences
    start_diff = torch.abs(reference.start_logits - target.start_logits)
    end_diff = torch.abs(reference.end_logits - target.end_logits)

    # Statistics
    start_max_diff = start_diff.max().item()
    start_mean_diff = start_diff.mean().item()
    end_max_diff = end_diff.max().item()
    end_mean_diff = end_diff.mean().item()

    # Cosine similarity
    start_cos = F.cosine_similarity(
        reference.start_logits.flatten().unsqueeze(0),
        target.start_logits.flatten().unsqueeze(0),
    ).item()
    end_cos = F.cosine_similarity(
        reference.end_logits.flatten().unsqueeze(0),
        target.end_logits.flatten().unsqueeze(0),
    ).item()

    # Check if within tolerance
    passed = start_max_diff < tolerance and end_max_diff < tolerance

    return {
        "passed": passed,
        "start_logits": {
            "max_diff": start_max_diff,
            "mean_diff": start_mean_diff,
            "cosine_sim": start_cos,
        },
        "end_logits": {
            "max_diff": end_max_diff,
            "mean_diff": end_mean_diff,
            "cosine_sim": end_cos,
        },
        "within_tolerance": tolerance,
    }


def print_section(title: str, width: int = 60):
    """Print a section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Backend Validation")
    parser.add_argument("--model", default="bert-base-uncased",
                       help="Model name or checkpoint path")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                       help="Numerical tolerance for comparison")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--reference", default="cpu",
                       help="Reference backend for comparison")
    args = parser.parse_args()

    print_section("Cross-Backend Validation for BERT")

    # Detect available backends
    backends = get_available_backends()
    print(f"\nAvailable backends: {[b[0] for b in backends]}")

    if len(backends) < 2:
        print("\nWarning: Only one backend available. Cross-backend validation requires multiple backends.")
        print("Proceeding with single-backend validation...")

    # Load tokenizer and model
    print(f"\nLoading model: {args.model}")

    if Path(args.model).exists():
        # Load from checkpoint
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        checkpoint = torch.load(args.model, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from: {args.model}")
    else:
        # Load from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    # Save reference state dict to ensure all backends use identical weights
    reference_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Create test input
    question = "What is the capital of France?"
    context = "France is a country in Western Europe. Paris is the capital and largest city of France."

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    print(f"\nTest input:")
    print(f"  Question: {question}")
    print(f"  Context: {context[:50]}...")
    print(f"  Input shape: {inputs['input_ids'].shape}")

    # Run on each backend
    print_section("Running Inference on Each Backend")

    results: dict[str, ValidationResult] = {}

    for backend_name, device in backends:
        print(f"\n  Testing {backend_name.upper()} ({device})...")
        try:
            # Clone model with identical weights for each backend
            model_copy = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
            model_copy.load_state_dict(reference_state_dict)

            result = run_inference(model_copy, inputs, device)
            results[backend_name] = result

            mem_str = f", {result.memory_mb:.1f} MB" if result.memory_mb else ""
            print(f"    Latency: {result.inference_time_ms:.2f} ms{mem_str}")

            # Clean up
            del model_copy
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Failed: {e}")

    # Compare backends
    print_section("Cross-Backend Comparison")

    reference_backend = args.reference
    if reference_backend not in results:
        reference_backend = list(results.keys())[0]

    reference = results[reference_backend]
    print(f"\nReference backend: {reference_backend.upper()}")

    comparisons = {}
    all_passed = True

    for backend_name, result in results.items():
        if backend_name == reference_backend:
            continue

        comparison = compare_outputs(reference, result, args.tolerance)
        comparisons[backend_name] = comparison

        status = "PASS" if comparison["passed"] else "FAIL"
        all_passed = all_passed and comparison["passed"]

        print(f"\n  {reference_backend.upper()} vs {backend_name.upper()}: [{status}]")
        print(f"    Start logits - max diff: {comparison['start_logits']['max_diff']:.2e}, "
              f"cosine sim: {comparison['start_logits']['cosine_sim']:.6f}")
        print(f"    End logits   - max diff: {comparison['end_logits']['max_diff']:.2e}, "
              f"cosine sim: {comparison['end_logits']['cosine_sim']:.6f}")
        print(f"    Speedup vs CPU: {results['cpu'].inference_time_ms / result.inference_time_ms:.2f}x")

    # Performance summary
    print_section("Performance Summary")

    print(f"\n  {'Backend':<10} {'Latency (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
    print(f"  {'-' * 50}")

    cpu_time = results.get("cpu", results[list(results.keys())[0]]).inference_time_ms
    for backend_name, result in results.items():
        mem_str = f"{result.memory_mb:.1f}" if result.memory_mb else "N/A"
        speedup = cpu_time / result.inference_time_ms
        print(f"  {backend_name.upper():<10} {result.inference_time_ms:<15.2f} {mem_str:<15} {speedup:<10.2f}x")

    # Summary
    print_section("Validation Summary")

    if all_passed:
        print(f"\n  All backends produce consistent outputs within tolerance ({args.tolerance})")
    else:
        print(f"\n  WARNING: Some backends show numerical differences beyond tolerance ({args.tolerance})")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "tolerance": args.tolerance,
            "reference_backend": reference_backend,
            "backends_tested": list(results.keys()),
            "all_passed": all_passed,
            "performance": {
                name: {
                    "latency_ms": r.inference_time_ms,
                    "memory_mb": r.memory_mb,
                }
                for name, r in results.items()
            },
            "comparisons": comparisons,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved: {args.output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
