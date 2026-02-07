#!/usr/bin/env python3
"""
Small Model Benchmark Suite

Comprehensive benchmarks for small text models (BERT, GPT-2, DistilBERT)
comparing baseline PyTorch vs TorchBridge optimized performance.

Metrics measured:
- Inference latency (ms)
- Throughput (samples/sec or tokens/sec)
- Memory usage (MB)
- VRAM utilization (%)

Usage:
    python small_model_benchmark.py [--models all] [--batch-sizes 1,4,8,16]

Requirements:
    pip install transformers

"""

import argparse
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_name: str
    batch_size: int
    seq_length: int
    optimization: str  # "baseline" or "optimized"
    device: str
    dtype: str

    # Performance metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput: float  # samples/sec or tokens/sec
    throughput_unit: str

    # Memory metrics
    memory_allocated_mb: float
    memory_reserved_mb: float
    peak_memory_mb: float

    # Metadata
    timestamp: str
    pytorch_version: str
    warmup_iterations: int
    benchmark_iterations: int

@dataclass
class ComparisonResult:
    """Comparison between baseline and optimized."""
    model_name: str
    batch_size: int
    seq_length: int

    baseline_latency_ms: float
    optimized_latency_ms: float
    latency_speedup: float

    baseline_throughput: float
    optimized_throughput: float
    throughput_speedup: float

    memory_reduction_pct: float

class SmallModelBenchmark:
    """Benchmark suite for small text models."""

    # Models to benchmark
    ENCODER_MODELS = [
        "bert-base-uncased",
        "distilbert-base-uncased",
    ]

    DECODER_MODELS = [
        "gpt2",
    ]

    def __init__(
        self,
        batch_sizes: list[int] = None,
        seq_lengths: list[int] = None,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        output_dir: str = "benchmark_results"
    ):
        self.batch_sizes = batch_sizes or [1, 4, 8, 16]
        self.seq_lengths = seq_lengths or [64, 128, 256, 512]
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.output_dir = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results: list[BenchmarkResult] = []
        self.comparisons: list[ComparisonResult] = []

        os.makedirs(output_dir, exist_ok=True)

    def _get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics."""
        if self.device.type == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}

    def _reset_memory_stats(self) -> None:
        """Reset memory tracking."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()

    def _create_encoder_inputs(
        self,
        batch_size: int,
        seq_length: int
    ) -> dict[str, torch.Tensor]:
        """Create inputs for encoder models (BERT, DistilBERT)."""
        return {
            "input_ids": torch.randint(0, 30000, (batch_size, seq_length), device=self.device),
            "attention_mask": torch.ones(batch_size, seq_length, device=self.device, dtype=torch.long),
        }

    def _create_decoder_inputs(
        self,
        batch_size: int,
        seq_length: int
    ) -> dict[str, torch.Tensor]:
        """Create inputs for decoder models (GPT-2)."""
        return {
            "input_ids": torch.randint(0, 50000, (batch_size, seq_length), device=self.device),
        }

    def _benchmark_model(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        is_generation: bool = False
    ) -> dict[str, Any]:
        """Run benchmark on a model."""
        latencies = []

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                if is_generation:
                    _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                else:
                    _ = model(**inputs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self._reset_memory_stats()

        # Benchmark
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                start = time.perf_counter()
                if is_generation:
                    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                else:
                    outputs = model(**inputs)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                latencies.append((time.perf_counter() - start) * 1000)

        memory_stats = self._get_memory_stats()

        # Calculate statistics
        latencies_sorted = sorted(latencies)
        avg_latency = sum(latencies) / len(latencies)
        p50 = latencies_sorted[int(len(latencies) * 0.50)]
        p95 = latencies_sorted[int(len(latencies) * 0.95)]
        p99 = latencies_sorted[int(len(latencies) * 0.99)]

        batch_size = inputs["input_ids"].shape[0]
        throughput = batch_size * 1000 / avg_latency  # samples/sec

        return {
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "throughput": throughput,
            **memory_stats
        }

    def benchmark_baseline_encoder(
        self,
        model_name: str,
        batch_size: int,
        seq_length: int
    ) -> BenchmarkResult:
        """Benchmark baseline encoder model."""
        logger.info(f"Baseline: {model_name} batch={batch_size} seq={seq_length}")

        try:
            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
            model = model.to(self.device)
            model.eval()

            inputs = self._create_encoder_inputs(batch_size, seq_length)
            metrics = self._benchmark_model(model, inputs)

            result = BenchmarkResult(
                model_name=model_name,
                batch_size=batch_size,
                seq_length=seq_length,
                optimization="baseline",
                device=str(self.device),
                dtype="float32",
                avg_latency_ms=metrics["avg_latency_ms"],
                p50_latency_ms=metrics["p50_latency_ms"],
                p95_latency_ms=metrics["p95_latency_ms"],
                p99_latency_ms=metrics["p99_latency_ms"],
                throughput=metrics["throughput"],
                throughput_unit="samples/sec",
                memory_allocated_mb=metrics["allocated_mb"],
                memory_reserved_mb=metrics["reserved_mb"],
                peak_memory_mb=metrics["peak_mb"],
                timestamp=datetime.now().isoformat(),
                pytorch_version=torch.__version__,
                warmup_iterations=self.warmup_iterations,
                benchmark_iterations=self.benchmark_iterations,
            )

            del model
            self._reset_memory_stats()

            return result

        except Exception as e:
            logger.error(f"Baseline benchmark failed: {e}")
            raise

    def benchmark_optimized_encoder(
        self,
        model_name: str,
        batch_size: int,
        seq_length: int
    ) -> BenchmarkResult:
        """Benchmark optimized encoder model."""
        logger.info(f"Optimized: {model_name} batch={batch_size} seq={seq_length}")

        try:
            from torchbridge.models.text import (
                OptimizationMode,
                TextModelConfig,
                TextModelOptimizer,
            )

            config = TextModelConfig(
                model_name=model_name,
                optimization_mode=OptimizationMode.INFERENCE,
                use_torch_compile=True,
                compile_mode="reduce-overhead",
                max_sequence_length=seq_length,
            )

            optimizer = TextModelOptimizer(config)
            model = optimizer.optimize(model_name, task="sequence-classification", num_labels=2)

            inputs = self._create_encoder_inputs(batch_size, seq_length)
            metrics = self._benchmark_model(model, inputs)

            result = BenchmarkResult(
                model_name=model_name,
                batch_size=batch_size,
                seq_length=seq_length,
                optimization="optimized",
                device=str(optimizer.device),
                dtype=str(optimizer.dtype),
                avg_latency_ms=metrics["avg_latency_ms"],
                p50_latency_ms=metrics["p50_latency_ms"],
                p95_latency_ms=metrics["p95_latency_ms"],
                p99_latency_ms=metrics["p99_latency_ms"],
                throughput=metrics["throughput"],
                throughput_unit="samples/sec",
                memory_allocated_mb=metrics["allocated_mb"],
                memory_reserved_mb=metrics["reserved_mb"],
                peak_memory_mb=metrics["peak_mb"],
                timestamp=datetime.now().isoformat(),
                pytorch_version=torch.__version__,
                warmup_iterations=self.warmup_iterations,
                benchmark_iterations=self.benchmark_iterations,
            )

            del model
            self._reset_memory_stats()

            return result

        except ImportError:
            logger.warning("TorchBridge not available, using torch.compile fallback")
            return self._benchmark_torch_compile_encoder(model_name, batch_size, seq_length)

    def _benchmark_torch_compile_encoder(
        self,
        model_name: str,
        batch_size: int,
        seq_length: int
    ) -> BenchmarkResult:
        """Fallback benchmark using torch.compile."""
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model = model.to(self.device)
        model.eval()

        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")

        inputs = self._create_encoder_inputs(batch_size, seq_length)
        metrics = self._benchmark_model(model, inputs)

        result = BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            seq_length=seq_length,
            optimization="torch_compile",
            device=str(self.device),
            dtype="float32",
            avg_latency_ms=metrics["avg_latency_ms"],
            p50_latency_ms=metrics["p50_latency_ms"],
            p95_latency_ms=metrics["p95_latency_ms"],
            p99_latency_ms=metrics["p99_latency_ms"],
            throughput=metrics["throughput"],
            throughput_unit="samples/sec",
            memory_allocated_mb=metrics["allocated_mb"],
            memory_reserved_mb=metrics["reserved_mb"],
            peak_memory_mb=metrics["peak_mb"],
            timestamp=datetime.now().isoformat(),
            pytorch_version=torch.__version__,
            warmup_iterations=self.warmup_iterations,
            benchmark_iterations=self.benchmark_iterations,
        )

        del model
        self._reset_memory_stats()

        return result

    def benchmark_baseline_decoder(
        self,
        model_name: str,
        batch_size: int,
        seq_length: int
    ) -> BenchmarkResult:
        """Benchmark baseline decoder model (GPT-2)."""
        logger.info(f"Baseline decoder: {model_name} batch={batch_size} seq={seq_length}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = model.to(self.device)
            model.eval()

            inputs = self._create_decoder_inputs(batch_size, seq_length)
            inputs["pad_token_id"] = tokenizer.eos_token_id

            metrics = self._benchmark_model(model, inputs, is_generation=True)

            result = BenchmarkResult(
                model_name=model_name,
                batch_size=batch_size,
                seq_length=seq_length,
                optimization="baseline",
                device=str(self.device),
                dtype="float32",
                avg_latency_ms=metrics["avg_latency_ms"],
                p50_latency_ms=metrics["p50_latency_ms"],
                p95_latency_ms=metrics["p95_latency_ms"],
                p99_latency_ms=metrics["p99_latency_ms"],
                throughput=metrics["throughput"],
                throughput_unit="samples/sec",
                memory_allocated_mb=metrics["allocated_mb"],
                memory_reserved_mb=metrics["reserved_mb"],
                peak_memory_mb=metrics["peak_mb"],
                timestamp=datetime.now().isoformat(),
                pytorch_version=torch.__version__,
                warmup_iterations=self.warmup_iterations,
                benchmark_iterations=self.benchmark_iterations,
            )

            del model
            self._reset_memory_stats()

            return result

        except Exception as e:
            logger.error(f"Decoder baseline benchmark failed: {e}")
            raise

    def run_encoder_benchmarks(self, models: list[str] = None) -> list[BenchmarkResult]:
        """Run all encoder model benchmarks."""
        models = models or self.ENCODER_MODELS
        results = []

        for model_name in models:
            for batch_size in self.batch_sizes:
                for seq_length in self.seq_lengths:
                    try:
                        # Baseline
                        baseline = self.benchmark_baseline_encoder(
                            model_name, batch_size, seq_length
                        )
                        results.append(baseline)

                        # Optimized
                        optimized = self.benchmark_optimized_encoder(
                            model_name, batch_size, seq_length
                        )
                        results.append(optimized)

                        # Create comparison
                        speedup = baseline.avg_latency_ms / optimized.avg_latency_ms
                        throughput_speedup = optimized.throughput / baseline.throughput
                        memory_reduction = 0
                        if baseline.peak_memory_mb > 0:
                            memory_reduction = (
                                (baseline.peak_memory_mb - optimized.peak_memory_mb)
                                / baseline.peak_memory_mb * 100
                            )

                        comparison = ComparisonResult(
                            model_name=model_name,
                            batch_size=batch_size,
                            seq_length=seq_length,
                            baseline_latency_ms=baseline.avg_latency_ms,
                            optimized_latency_ms=optimized.avg_latency_ms,
                            latency_speedup=speedup,
                            baseline_throughput=baseline.throughput,
                            optimized_throughput=optimized.throughput,
                            throughput_speedup=throughput_speedup,
                            memory_reduction_pct=memory_reduction,
                        )
                        self.comparisons.append(comparison)

                    except Exception as e:
                        logger.error(f"Benchmark failed for {model_name}: {e}")

        self.results.extend(results)
        return results

    def run_decoder_benchmarks(self, models: list[str] = None) -> list[BenchmarkResult]:
        """Run all decoder model benchmarks."""
        models = models or self.DECODER_MODELS
        results = []

        for model_name in models:
            for batch_size in self.batch_sizes[:2]:  # Limit batch sizes for generation
                for seq_length in self.seq_lengths[:2]:  # Limit seq lengths
                    try:
                        baseline = self.benchmark_baseline_decoder(
                            model_name, batch_size, seq_length
                        )
                        results.append(baseline)
                    except Exception as e:
                        logger.error(f"Decoder benchmark failed for {model_name}: {e}")

        self.results.extend(results)
        return results

    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"small_model_benchmark_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "device": str(self.device),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "batch_sizes": self.batch_sizes,
                "seq_lengths": self.seq_lengths,
            },
            "results": [asdict(r) for r in self.results],
            "comparisons": [asdict(c) for c in self.comparisons],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("  Small Model Benchmark Summary")
        print("="*80)

        print(f"\nDevice: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print("\nLatency Comparisons:")
        print("-" * 80)
        print(f"{'Model':<25} {'Batch':<6} {'Seq':<5} {'Baseline':<12} {'Optimized':<12} {'Speedup':<8}")
        print("-" * 80)

        for comp in self.comparisons:
            print(
                f"{comp.model_name:<25} "
                f"{comp.batch_size:<6} "
                f"{comp.seq_length:<5} "
                f"{comp.baseline_latency_ms:>10.2f}ms "
                f"{comp.optimized_latency_ms:>10.2f}ms "
                f"{comp.latency_speedup:>6.2f}x"
            )

        print("-" * 80)

        # Calculate average speedup
        if self.comparisons:
            avg_speedup = sum(c.latency_speedup for c in self.comparisons) / len(self.comparisons)
            print(f"\nAverage speedup: {avg_speedup:.2f}x")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Small Model Benchmark Suite")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to benchmark (comma-separated or 'all')"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8",
        help="Batch sizes (comma-separated)"
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="64,128",
        help="Sequence lengths (comma-separated)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/models",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Parse arguments
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]

    # Create benchmark
    benchmark = SmallModelBenchmark(
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        benchmark_iterations=args.iterations,
        output_dir=args.output_dir,
    )

    # Run benchmarks
    print("Running encoder model benchmarks...")
    benchmark.run_encoder_benchmarks()

    print("\nRunning decoder model benchmarks...")
    benchmark.run_decoder_benchmarks()

    # Save and print results
    benchmark.save_results()
    benchmark.print_summary()

if __name__ == "__main__":
    main()
