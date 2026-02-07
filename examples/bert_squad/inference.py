#!/usr/bin/env python3
"""
BERT SQuAD Inference with Benchmarking

Run inference on trained BERT model with optional benchmarking.

Usage:
    python inference.py --question "What is the capital?" --context "Paris is the capital of France."
    python inference.py --model checkpoints/bert_squad_best.pt --benchmark
    python inference.py --interactive
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# TorchBridge imports
try:
    from torchbridge.backends import detect_best_backend
    from torchbridge.core.unified_manager import get_manager
    TORCHBRIDGE_AVAILABLE = True
except ImportError:
    TORCHBRIDGE_AVAILABLE = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def synchronize(device: torch.device):
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


class BERTSquadInference:
    """BERT Question Answering inference engine."""

    def __init__(self, model_path: str | None = None, device: torch.device | None = None):
        self.device = device or get_device()
        self.model_path = model_path or "bert-base-uncased"

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}")

        # Load tokenizer
        base_model = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load model
        if Path(self.model_path).exists():
            # Load from checkpoint
            self.model = AutoModelForQuestionAnswering.from_pretrained(base_model)
            checkpoint = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint: {self.model_path}")
        else:
            # Load from HuggingFace
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)

        # Apply TorchBridge optimizations
        if TORCHBRIDGE_AVAILABLE:
            try:
                manager = get_manager()
                self.model = manager.prepare_model(self.model, optimization_level="O2")
                print("Applied TorchBridge O2 optimizations")
            except Exception as e:
                print(f"TorchBridge optimization skipped: {e}")

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, question: str, context: str, top_k: int = 3) -> list[dict]:
        """
        Answer a question given context.

        Returns:
            List of top-k answer candidates with scores.
        """
        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            max_length=384,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get answer spans
        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()

        # Find top-k start/end positions
        start_indices = torch.argsort(start_logits, descending=True)[:top_k * 2]
        end_indices = torch.argsort(end_logits, descending=True)[:top_k * 2]

        # Generate candidate answers
        candidates = []
        for start_idx in start_indices:
            for end_idx in end_indices:
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx > 50:  # Max answer length
                    continue

                score = start_logits[start_idx] + end_logits[end_idx]

                # Get answer text
                start_char = offset_mapping[start_idx][0].item()
                end_char = offset_mapping[end_idx][1].item()

                if start_char == 0 and end_char == 0:
                    answer_text = "[No Answer]"
                else:
                    answer_text = context[start_char:end_char]

                candidates.append({
                    "answer": answer_text,
                    "score": score.item(),
                    "start": start_char,
                    "end": end_char,
                })

        # Sort by score and return top-k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def benchmark(self, question: str, context: str, warmup: int = 10, iterations: int = 100) -> dict:
        """
        Benchmark inference latency.

        Returns:
            Dictionary with timing statistics.
        """
        inputs = self.tokenizer(
            question,
            context,
            max_length=384,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(**inputs)
        synchronize(self.device)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                synchronize(self.device)
                start = time.perf_counter()
                _ = self.model(**inputs)
                synchronize(self.device)
                times.append((time.perf_counter() - start) * 1000)

        times_tensor = torch.tensor(times)

        return {
            "mean_ms": times_tensor.mean().item(),
            "std_ms": times_tensor.std().item(),
            "min_ms": times_tensor.min().item(),
            "max_ms": times_tensor.max().item(),
            "p50_ms": times_tensor.median().item(),
            "p95_ms": times_tensor.quantile(0.95).item(),
            "p99_ms": times_tensor.quantile(0.99).item(),
            "throughput_qps": 1000 / times_tensor.mean().item(),
            "iterations": iterations,
            "device": str(self.device),
        }


def interactive_mode(engine: BERTSquadInference):
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("  Interactive Question Answering")
    print("=" * 60)
    print("\nEnter context, then ask questions. Type 'quit' to exit.\n")

    while True:
        print("-" * 40)
        context = input("Context: ").strip()
        if context.lower() == "quit":
            break
        if not context:
            continue

        while True:
            question = input("Question (or 'new' for new context): ").strip()
            if question.lower() in ["quit", "exit"]:
                return
            if question.lower() == "new":
                break
            if not question:
                continue

            # Get answer
            start = time.perf_counter()
            answers = engine.predict(question, context)
            latency = (time.perf_counter() - start) * 1000

            print(f"\nAnswers ({latency:.1f}ms):")
            for i, ans in enumerate(answers, 1):
                print(f"  {i}. {ans['answer']} (score: {ans['score']:.2f})")
            print()


def main():
    parser = argparse.ArgumentParser(description="BERT SQuAD Inference")
    parser.add_argument("--model", default="bert-base-uncased",
                       help="Model name or checkpoint path")
    parser.add_argument("--question", "-q", help="Question to answer")
    parser.add_argument("--context", "-c", help="Context for the question")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--output", "-o", help="Save results to JSON")

    args = parser.parse_args()

    # Initialize engine
    engine = BERTSquadInference(model_path=args.model)

    # Interactive mode
    if args.interactive:
        interactive_mode(engine)
        return 0

    # Default context and question for demo/benchmark
    question = args.question or "What is the capital of France?"
    context = args.context or (
        "France is a country in Western Europe. Paris is the capital and largest city of France. "
        "The city is known for the Eiffel Tower, the Louvre museum, and its rich cultural heritage."
    )

    # Run inference
    print("\n" + "=" * 60)
    print("  BERT Question Answering")
    print("=" * 60)
    print(f"\nContext: {context[:80]}...")
    print(f"Question: {question}")

    answers = engine.predict(question, context)
    print("\nAnswers:")
    for i, ans in enumerate(answers, 1):
        print(f"  {i}. {ans['answer']} (score: {ans['score']:.2f})")

    # Benchmark
    if args.benchmark:
        print("\n" + "=" * 60)
        print("  Benchmark Results")
        print("=" * 60)

        results = engine.benchmark(question, context, iterations=args.iterations)

        print(f"\nDevice: {results['device']}")
        print(f"Iterations: {results['iterations']}")
        print(f"\nLatency:")
        print(f"  Mean:   {results['mean_ms']:.2f} ms")
        print(f"  Std:    {results['std_ms']:.2f} ms")
        print(f"  P50:    {results['p50_ms']:.2f} ms")
        print(f"  P95:    {results['p95_ms']:.2f} ms")
        print(f"  P99:    {results['p99_ms']:.2f} ms")
        print(f"\nThroughput: {results['throughput_qps']:.1f} queries/sec")

        if args.output:
            output_data = {
                "model": args.model,
                "question": question,
                "context": context,
                "answers": answers,
                "benchmark": results,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
