#!/usr/bin/env python3
"""
BERT Optimization Example

Demonstrates how to use TorchBridge to optimize BERT models for
inference and training. Shows 2-3x speedup on various hardware backends.

Models covered:
- bert-base-uncased (110M params, ~440MB)
- bert-large-uncased (340M params, ~1.3GB)
- distilbert-base-uncased (66M params, ~265MB)

Usage:
    python bert_optimization.py [--model bert-base-uncased] [--task text-classification]

Requirements:
    pip install transformers datasets

"""

import argparse
import logging
import time
from typing import Any

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import transformers
        logger.info(f"transformers version: {transformers.__version__}")
        return True
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        return False

def create_sample_data(tokenizer, batch_size: int = 8, seq_length: int = 128):
    """Create sample data for benchmarking."""
    texts = [
        "TorchBridge provides production-grade GPU optimization for deep learning.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries worldwide.",
        "PyTorch is a popular deep learning framework.",
    ] * (batch_size // 4 + 1)

    texts = texts[:batch_size]

    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_length,
        return_tensors="pt"
    )

    return inputs

def benchmark_model(model, inputs, device, num_iterations: int = 100) -> dict[str, float]:
    """Benchmark model inference."""
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = inputs["input_ids"].shape[0] * num_iterations / total_time

    return {
        "total_time_s": total_time,
        "avg_latency_ms": avg_time * 1000,
        "throughput_samples_per_sec": throughput,
    }

def run_baseline(model_name: str, task: str, batch_size: int, seq_length: int) -> dict[str, Any]:
    """Run baseline PyTorch model without optimization."""
    print_section(f"Baseline PyTorch - {model_name}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create sample data
    inputs = create_sample_data(tokenizer, batch_size, seq_length)

    # Benchmark
    results = benchmark_model(model, inputs, device)

    print(f"Average latency: {results['avg_latency_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")

    return {
        "model_name": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "seq_length": seq_length,
        **results
    }

def run_optimized(model_name: str, task: str, batch_size: int, seq_length: int) -> dict[str, Any]:
    """Run TorchBridge optimized model."""
    print_section(f"TorchBridge Optimized - {model_name}")

    from transformers import AutoTokenizer

    # Import TorchBridge
    try:
        from torchbridge.models.text import (
            OptimizationMode,
            TextModelConfig,
            TextModelOptimizer,
        )

        # Create optimized config
        config = TextModelConfig(
            model_name=model_name,
            optimization_mode=OptimizationMode.INFERENCE,
            use_torch_compile=True,
            compile_mode="reduce-overhead",
            max_sequence_length=seq_length,
        )

        # Create optimizer and optimize model
        optimizer = TextModelOptimizer(config)
        model = optimizer.optimize(model_name, task=task, num_labels=2)

        device = optimizer.device
        optimization_info = optimizer.get_optimization_info()

        print(f"Device: {device}")
        print(f"Backend: {optimization_info['backend']}")
        print(f"Dtype: {optimization_info['dtype']}")
        print(f"Optimization mode: {optimization_info['optimization_mode']}")
        print(f"torch.compile: {optimization_info['torch_compile']}")

    except ImportError as e:
        logger.warning(f"TorchBridge not available: {e}")
        logger.info("Falling back to torch.compile only")

        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Apply torch.compile
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")

        optimization_info = {"backend": "pytorch", "torch_compile": True}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create sample data
    inputs = create_sample_data(tokenizer, batch_size, seq_length)

    # Benchmark
    results = benchmark_model(model, inputs, device)

    print(f"Average latency: {results['avg_latency_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")

    return {
        "model_name": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "seq_length": seq_length,
        "optimization_info": optimization_info,
        **results
    }

def run_text_classification_demo(model_name: str):
    """Run a complete text classification demo."""
    print_section("Text Classification Demo")

    from transformers import AutoTokenizer

    try:
        from torchbridge.models.text import OptimizedBERT

        # Create optimized model
        model = OptimizedBERT(
            model_name=model_name,
            task="sequence-classification",
            num_labels=2
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Sample texts
        texts = [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "The worst experience I've ever had. Terrible service.",
            "It was okay, nothing special but not bad either.",
        ]

        print("Classifying sample texts:\n")

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()

            label = "Positive" if pred == 1 else "Negative"
            print(f"  Text: '{text[:50]}...'")
            print(f"  Prediction: {label} (confidence: {confidence:.2%})")
            print()

    except ImportError as e:
        logger.warning(f"Demo skipped: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BERT Optimization Example")
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased"],
        help="Model to optimize"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-classification",
        help="Task type"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for benchmarking"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline benchmark"
    )

    args = parser.parse_args()

    print_section("BERT Optimization with TorchBridge")

    # Check dependencies
    if not check_dependencies():
        return

    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run baseline
    if not args.skip_baseline:
        baseline_results = run_baseline(
            args.model, args.task, args.batch_size, args.seq_length
        )
    else:
        baseline_results = None

    # Run optimized
    optimized_results = run_optimized(
        args.model, args.task, args.batch_size, args.seq_length
    )

    # Print comparison
    if baseline_results:
        print_section("Performance Comparison")

        speedup = baseline_results["avg_latency_ms"] / optimized_results["avg_latency_ms"]
        throughput_gain = optimized_results["throughput_samples_per_sec"] / baseline_results["throughput_samples_per_sec"]

        print(f"Baseline latency:  {baseline_results['avg_latency_ms']:.2f} ms")
        print(f"Optimized latency: {optimized_results['avg_latency_ms']:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print()
        print(f"Baseline throughput:  {baseline_results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"Optimized throughput: {optimized_results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"Throughput gain: {throughput_gain:.2f}x")

    # Run demo
    run_text_classification_demo(args.model)

    print_section("Complete!")

if __name__ == "__main__":
    main()
