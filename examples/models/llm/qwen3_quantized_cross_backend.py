"""
Qwen3-0.6B Auto-Quantized Cross-Backend Inference

Demonstrates TorchBridge's backend-aware quantization by auto-selecting
the optimal format for the detected hardware, then comparing quality
against the full-precision baseline.

Usage:
    python examples/models/llm/qwen3_quantized_cross_backend.py
"""

import time

import torch
import torch.nn.functional as F

from torchbridge.precision.quantization import QuantizationEngine


def main():
    print("=== Qwen3-0.6B Auto-Quantized Cross-Backend Inference ===\n")

    # Detect hardware
    engine = QuantizationEngine()
    print(f"Backend: {engine.backend_name}")
    print(f"Optimal format: {engine.get_optimal_format().value}")
    print(f"Supported formats: {[f.value for f in engine.get_supported_formats()]}")

    # Load model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\nError: transformers required. Install with: pip install transformers")
        return

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model.eval()

    # Baseline inference
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        baseline_out = model(**inputs)
    baseline_logits = baseline_out.logits[:, -1, :]

    # Auto-quantize
    print("\nQuantizing with auto-selected format...")
    t0 = time.perf_counter()
    result = engine.quantize(model, format="auto")
    quant_time = (time.perf_counter() - t0) * 1000

    if not result.success:
        print(f"Quantization failed: {result.errors}")
        return

    print(f"Format applied: {result.format_applied.value}")
    print(f"Memory: {result.memory_before_mb:.1f} MB -> {result.memory_after_mb:.1f} MB")
    print(f"Reduction: {result.memory_reduction_pct:.1f}%")
    print(f"Quantization time: {quant_time:.0f} ms")

    if result.used_fallback:
        print(f"Fallback used: {[f.value for f in result.fallback_chain]}")

    # Quantized inference
    quantized_model = result.model
    quantized_model.eval()

    with torch.no_grad():
        quant_out = quantized_model(**inputs)
    quant_logits = quant_out.logits[:, -1, :]

    # Quality comparison
    max_diff = torch.abs(baseline_logits - quant_logits).max().item()
    cos_sim = F.cosine_similarity(
        baseline_logits.flatten().unsqueeze(0),
        quant_logits.flatten().unsqueeze(0),
    ).item()

    print("\n--- Quality Comparison ---")
    print(f"Max logit diff: {max_diff:.2e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Status: {'PASSED' if cos_sim > 0.99 else 'WARNING'}")

    # Latency comparison
    with torch.no_grad():
        for _ in range(3):
            quantized_model(**inputs)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            quantized_model(**inputs)
    quant_latency = (time.perf_counter() - t0) / 20 * 1000

    with torch.no_grad():
        for _ in range(3):
            model(**inputs)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            model(**inputs)
    baseline_latency = (time.perf_counter() - t0) / 20 * 1000

    print(f"\nBaseline latency: {baseline_latency:.1f} ms")
    print(f"Quantized latency: {quant_latency:.1f} ms")
    print(f"Speedup: {baseline_latency / quant_latency:.2f}x")


if __name__ == "__main__":
    main()
