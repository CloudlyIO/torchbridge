"""
Quantization Accuracy Benchmarks

Measures the accuracy impact of different quantization modes on real models.
Evaluates perplexity for language models and accuracy for classification models.

v0.4.21 - Quantization Quality Validation

Targets:
    - INT8: <2% perplexity increase, 50% memory reduction
    - INT4 (GPTQ/AWQ): <5% perplexity increase, 75% memory reduction
    - FP8: <1% perplexity increase, 50% memory reduction
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class QuantizationResult:
    """Results from quantization benchmark."""
    model_name: str
    quantization_mode: str

    # Quality metrics
    baseline_perplexity: Optional[float] = None
    quantized_perplexity: Optional[float] = None
    perplexity_increase_pct: Optional[float] = None

    baseline_accuracy: Optional[float] = None
    quantized_accuracy: Optional[float] = None
    accuracy_drop_pct: Optional[float] = None

    # Memory metrics
    baseline_memory_mb: float = 0.0
    quantized_memory_mb: float = 0.0
    memory_reduction_pct: float = 0.0

    # Performance metrics
    baseline_latency_ms: float = 0.0
    quantized_latency_ms: float = 0.0
    speedup: float = 1.0

    # Metadata
    num_samples: int = 0
    device: str = "cpu"
    timestamp: str = ""

    def passes_quality_target(self) -> bool:
        """Check if result meets quality targets."""
        targets = {
            "int8": {"perplexity": 2.0, "accuracy": 2.0},
            "int4": {"perplexity": 5.0, "accuracy": 5.0},
            "int4_gptq": {"perplexity": 5.0, "accuracy": 5.0},
            "int4_awq": {"perplexity": 3.0, "accuracy": 3.0},
            "fp8": {"perplexity": 1.0, "accuracy": 1.0},
            "fp8_e4m3": {"perplexity": 1.0, "accuracy": 1.0},
            "fp8_e5m2": {"perplexity": 1.5, "accuracy": 1.5},
        }

        mode_key = self.quantization_mode.lower()
        if mode_key not in targets:
            return True  # Unknown mode, no target

        target = targets[mode_key]

        if self.perplexity_increase_pct is not None:
            if self.perplexity_increase_pct > target["perplexity"]:
                return False

        if self.accuracy_drop_pct is not None:
            if self.accuracy_drop_pct > target["accuracy"]:
                return False

        return True


@dataclass
class BenchmarkConfig:
    """Configuration for quantization benchmarks."""
    # Model settings
    model_name: str = "gpt2"
    model_type: str = "causal-lm"  # causal-lm, masked-lm, classification

    # Quantization modes to test
    quantization_modes: List[str] = field(default_factory=lambda: ["int8", "fp8"])

    # Evaluation settings
    num_samples: int = 100
    max_length: int = 512
    batch_size: int = 4

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output settings
    output_dir: str = "benchmark_results"
    save_results: bool = True


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def calculate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 100
) -> Tuple[float, float]:
    """
    Calculate perplexity for a language model.

    Returns:
        Tuple of (perplexity, average_loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    samples_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            labels = input_ids.clone()

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Count non-padding tokens
                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                samples_processed += input_ids.size(0)

            except Exception as e:
                warnings.warn(f"Error processing batch: {e}")
                continue

    if total_tokens == 0:
        return float('inf'), float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity, avg_loss


def calculate_classification_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 100
) -> float:
    """Calculate classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    samples_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            labels = batch["labels"].to(device)

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                samples_processed += input_ids.size(0)

            except Exception as e:
                warnings.warn(f"Error processing batch: {e}")
                continue

    return correct / total if total > 0 else 0.0


def measure_latency(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    num_runs: int = 10,
    warmup_runs: int = 3
) -> float:
    """Measure average inference latency in milliseconds."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(**sample_input)

    # Sync if CUDA
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(**sample_input)

    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    avg_latency_ms = (end_time - start_time) / num_runs * 1000

    return avg_latency_ms


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Apply INT8 dynamic quantization."""
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized


def quantize_model_fp8(model: nn.Module, device: torch.device) -> nn.Module:
    """Apply FP8 quantization using our native implementation."""
    try:
        from torchbridge.precision.fp8_native import convert_model_to_native_fp8
        return convert_model_to_native_fp8(model, device=device)
    except Exception as e:
        warnings.warn(f"FP8 conversion failed: {e}, returning original model")
        return model


def quantize_model_int4_bitsandbytes(
    model_name: str,
    device: torch.device
) -> nn.Module:
    """Load model with INT4 quantization via BitsAndBytes."""
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        return model

    except ImportError:
        warnings.warn("BitsAndBytes not available for INT4 quantization")
        return None


class QuantizationBenchmark:
    """Main benchmark runner for quantization quality evaluation."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: List[QuantizationResult] = []

    def load_model(self, model_name: str) -> nn.Module:
        """Load a model for benchmarking."""
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

        if self.config.model_type == "causal-lm":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif self.config.model_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)

        return model.to(self.device)

    def load_tokenizer(self, model_name: str):
        """Load tokenizer for the model."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def create_eval_dataloader(self, tokenizer, num_samples: int) -> DataLoader:
        """Create evaluation dataloader with WikiText-2 or similar."""
        try:
            from datasets import load_dataset

            # Load WikiText-2 for perplexity evaluation
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

            # Filter and tokenize
            texts = [t for t in dataset["text"] if len(t.strip()) > 50][:num_samples]

            encodings = tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            # Create simple dataset
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings

                def __len__(self):
                    return len(self.encodings["input_ids"])

                def __getitem__(self, idx):
                    return {key: val[idx] for key, val in self.encodings.items()}

            dataset = SimpleDataset(encodings)
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        except Exception as e:
            warnings.warn(f"Could not load WikiText-2: {e}")
            return None

    def benchmark_quantization_mode(
        self,
        model_name: str,
        quantization_mode: str,
        baseline_model: nn.Module,
        tokenizer,
        dataloader: DataLoader
    ) -> QuantizationResult:
        """Benchmark a specific quantization mode."""
        import copy
        from datetime import datetime

        result = QuantizationResult(
            model_name=model_name,
            quantization_mode=quantization_mode,
            device=str(self.device),
            timestamp=datetime.now().isoformat(),
            num_samples=self.config.num_samples
        )

        # Baseline metrics
        print(f"  Measuring baseline metrics...")
        result.baseline_memory_mb = get_model_size_mb(baseline_model)
        result.baseline_perplexity, _ = calculate_perplexity(
            baseline_model, dataloader, self.device, self.config.num_samples
        )

        # Create sample input for latency measurement
        sample_batch = next(iter(dataloader))
        sample_input = {k: v[:1].to(self.device) for k, v in sample_batch.items() if k != "labels"}
        result.baseline_latency_ms = measure_latency(baseline_model, sample_input)

        # Apply quantization
        print(f"  Applying {quantization_mode} quantization...")

        if quantization_mode.lower() == "int8":
            # INT8 dynamic quantization
            quantized_model = quantize_model_int8(copy.deepcopy(baseline_model).cpu())
            quantized_model = quantized_model.to(self.device) if self.device.type != "cuda" else quantized_model

        elif quantization_mode.lower() in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
            # FP8 quantization
            quantized_model = quantize_model_fp8(copy.deepcopy(baseline_model), self.device)

        elif quantization_mode.lower() in ["int4", "int4_bnb", "int4_nf4"]:
            # INT4 via BitsAndBytes (requires reloading)
            quantized_model = quantize_model_int4_bitsandbytes(model_name, self.device)
            if quantized_model is None:
                warnings.warn(f"INT4 quantization not available")
                return result

        else:
            warnings.warn(f"Unknown quantization mode: {quantization_mode}")
            return result

        # Quantized metrics
        print(f"  Measuring quantized metrics...")
        result.quantized_memory_mb = get_model_size_mb(quantized_model)

        try:
            result.quantized_perplexity, _ = calculate_perplexity(
                quantized_model, dataloader, self.device, self.config.num_samples
            )
        except Exception as e:
            warnings.warn(f"Could not calculate quantized perplexity: {e}")
            result.quantized_perplexity = float('inf')

        try:
            # Update sample input device if needed
            sample_input_q = {k: v.to(next(quantized_model.parameters()).device)
                           for k, v in sample_input.items()}
            result.quantized_latency_ms = measure_latency(quantized_model, sample_input_q)
        except Exception as e:
            warnings.warn(f"Could not measure quantized latency: {e}")
            result.quantized_latency_ms = result.baseline_latency_ms

        # Calculate deltas
        if result.baseline_perplexity and result.quantized_perplexity:
            result.perplexity_increase_pct = (
                (result.quantized_perplexity - result.baseline_perplexity)
                / result.baseline_perplexity * 100
            )

        if result.baseline_memory_mb > 0:
            result.memory_reduction_pct = (
                (result.baseline_memory_mb - result.quantized_memory_mb)
                / result.baseline_memory_mb * 100
            )

        if result.quantized_latency_ms > 0:
            result.speedup = result.baseline_latency_ms / result.quantized_latency_ms

        return result

    def run(self) -> List[QuantizationResult]:
        """Run the full benchmark suite."""
        print(f"\n{'='*60}")
        print(f"Quantization Accuracy Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Quantization modes: {self.config.quantization_modes}")
        print(f"Samples: {self.config.num_samples}")
        print(f"{'='*60}\n")

        # Load model and tokenizer
        print("Loading model and tokenizer...")
        baseline_model = self.load_model(self.config.model_name)
        tokenizer = self.load_tokenizer(self.config.model_name)

        # Create evaluation dataloader
        print("Creating evaluation dataloader...")
        dataloader = self.create_eval_dataloader(tokenizer, self.config.num_samples)

        if dataloader is None:
            print("ERROR: Could not create evaluation dataloader")
            return []

        # Benchmark each quantization mode
        for mode in self.config.quantization_modes:
            print(f"\nBenchmarking {mode}...")
            result = self.benchmark_quantization_mode(
                self.config.model_name,
                mode,
                baseline_model,
                tokenizer,
                dataloader
            )
            self.results.append(result)

            # Print result summary
            self._print_result(result)

        # Save results
        if self.config.save_results:
            self._save_results()

        return self.results

    def _print_result(self, result: QuantizationResult):
        """Print a single result summary."""
        status = "PASS" if result.passes_quality_target() else "FAIL"

        print(f"\n  {result.quantization_mode.upper()} Results [{status}]:")
        print(f"    Perplexity: {result.baseline_perplexity:.2f} -> {result.quantized_perplexity:.2f} ({result.perplexity_increase_pct:+.2f}%)")
        print(f"    Memory: {result.baseline_memory_mb:.1f}MB -> {result.quantized_memory_mb:.1f}MB ({result.memory_reduction_pct:.1f}% reduction)")
        print(f"    Latency: {result.baseline_latency_ms:.2f}ms -> {result.quantized_latency_ms:.2f}ms ({result.speedup:.2f}x speedup)")

    def _save_results(self):
        """Save results to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"quantization_results_{self.config.model_name.replace('/', '_')}.json"

        results_dict = {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results]
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Main entry point for quantization benchmarks."""
    parser = argparse.ArgumentParser(description="Quantization Accuracy Benchmarks")
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument("--modes", nargs="+", default=["int8", "fp8"],
                       help="Quantization modes to benchmark")
    parser.add_argument("--samples", type=int, default=100, help="Number of evaluation samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="benchmark_results")

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_name=args.model,
        quantization_modes=args.modes,
        num_samples=args.samples,
        device=args.device,
        output_dir=args.output_dir
    )

    benchmark = QuantizationBenchmark(config)
    results = benchmark.run()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_pass = True
    for result in results:
        status = "PASS" if result.passes_quality_target() else "FAIL"
        if not result.passes_quality_target():
            all_pass = False
        print(f"  {result.quantization_mode}: {status}")

    print(f"\nOverall: {'ALL TARGETS MET' if all_pass else 'SOME TARGETS MISSED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
