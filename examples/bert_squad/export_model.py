#!/usr/bin/env python3
"""
Export BERT SQuAD Model to ONNX

Export trained model to ONNX format for portable deployment.

Usage:
    python export_model.py --model checkpoints/bert_squad_best.pt --output bert_squad.onnx
    python export_model.py --model bert-base-uncased --output bert_qa.onnx --validate
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> bool:
    """
    Export model to ONNX format.

    Returns:
        True if export successful.
    """
    print("=" * 60)
    print("  ONNX Export")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {model_path}")
    base_model = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if Path(model_path).exists():
        model = AutoModelForQuestionAnswering.from_pretrained(base_model)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {model_path}")
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    model.eval()

    # Create dummy input
    dummy_text = "What is the capital of France?"
    dummy_context = "Paris is the capital of France."

    inputs = tokenizer(
        dummy_text,
        dummy_context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Export
    print(f"\nExporting to: {output_path}")
    print(f"ONNX opset version: {opset_version}")

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["start_logits", "end_logits"]

    if dynamic_axes:
        dynamic_axes_config = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "start_logits": {0: "batch_size", 1: "sequence_length"},
            "end_logits": {0: "batch_size", 1: "sequence_length"},
        }
    else:
        dynamic_axes_config = None

    try:
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_config,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print("Export successful!")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def validate_onnx(
    onnx_path: str,
    pytorch_model_path: str,
    tolerance: float = 1e-4,
) -> bool:
    """
    Validate ONNX model against PyTorch model.

    Returns:
        True if outputs match within tolerance.
    """
    print("\n" + "=" * 60)
    print("  ONNX Validation")
    print("=" * 60)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Error: onnx and onnxruntime required for validation")
        print("Install: pip install onnx onnxruntime")
        return False

    # Check ONNX model
    print(f"\nLoading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Create ONNX runtime session
    print("\nCreating ONNX Runtime session...")
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Using providers: {session.get_providers()}")

    # Load PyTorch model
    print(f"\nLoading PyTorch model: {pytorch_model_path}")
    base_model = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if Path(pytorch_model_path).exists():
        pytorch_model = AutoModelForQuestionAnswering.from_pretrained(base_model)
        checkpoint = torch.load(pytorch_model_path, map_location="cpu")
        pytorch_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        pytorch_model = AutoModelForQuestionAnswering.from_pretrained(pytorch_model_path)
    pytorch_model.eval()

    # Test input
    question = "What is the capital of France?"
    context = "Paris is the capital and largest city of France."

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pytorch_outputs = pytorch_model(**inputs)

    pytorch_start = pytorch_outputs.start_logits.numpy()
    pytorch_end = pytorch_outputs.end_logits.numpy()

    # ONNX inference
    print("Running ONNX inference...")
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
        "token_type_ids": inputs["token_type_ids"].numpy(),
    }
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_start = onnx_outputs[0]
    onnx_end = onnx_outputs[1]

    # Compare outputs
    print("\nComparing outputs...")

    start_diff = np.abs(pytorch_start - onnx_start).max()
    end_diff = np.abs(pytorch_end - onnx_end).max()

    print(f"  Start logits max diff: {start_diff:.2e}")
    print(f"  End logits max diff: {end_diff:.2e}")

    passed = start_diff < tolerance and end_diff < tolerance

    if passed:
        print(f"\n  PASSED: Outputs match within tolerance ({tolerance})")
    else:
        print(f"\n  FAILED: Outputs differ beyond tolerance ({tolerance})")

    # Benchmark
    print("\n" + "-" * 40)
    print("  Performance Comparison")
    print("-" * 40)

    # PyTorch benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            _ = pytorch_model(**inputs)
        times.append((time.perf_counter() - start) * 1000)
    pytorch_time = sum(times) / len(times)

    # ONNX benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = session.run(None, onnx_inputs)
        times.append((time.perf_counter() - start) * 1000)
    onnx_time = sum(times) / len(times)

    print(f"\n  PyTorch:      {pytorch_time:.2f} ms")
    print(f"  ONNX Runtime: {onnx_time:.2f} ms")
    print(f"  Speedup:      {pytorch_time / onnx_time:.2f}x")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Export BERT to ONNX")
    parser.add_argument("--model", default="bert-base-uncased",
                       help="Model name or checkpoint path")
    parser.add_argument("--output", "-o", required=True,
                       help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14,
                       help="ONNX opset version")
    parser.add_argument("--validate", action="store_true",
                       help="Validate ONNX against PyTorch")
    parser.add_argument("--no-dynamic", action="store_true",
                       help="Disable dynamic axes")

    args = parser.parse_args()

    # Export
    success = export_to_onnx(
        args.model,
        args.output,
        opset_version=args.opset,
        dynamic_axes=not args.no_dynamic,
    )

    if not success:
        return 1

    # Get file size
    output_path = Path(args.output)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\nONNX model size: {size_mb:.1f} MB")

    # Validate
    if args.validate:
        if not validate_onnx(args.output, args.model):
            return 1

    print("\n" + "=" * 60)
    print("  Export Complete!")
    print("=" * 60)
    print(f"\n  Output: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
