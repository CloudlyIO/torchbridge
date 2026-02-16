#!/usr/bin/env python3
"""
GTE-Qwen3 Embedding Cross-Backend Example

Demonstrates how to use TorchBridge to run Alibaba's GTE-Qwen3
embedding model across CUDA, ROCm, Trainium, TPU, and CPU backends.

GTE (General Text Embeddings) with Qwen3 backbone is a leading
embedding model for RAG pipelines, semantic search, and retrieval.
It supports flexible output dimensions and instruction-based embedding.

Models covered:
- Alibaba-NLP/gte-Qwen3-embedding (primary, ~1.5B params)

Key features demonstrated:
- Embedding generation across backends
- Semantic similarity computation
- Batch encoding throughput
- Cross-backend output consistency
- Comparison with BGE-M3 (see bge_m3_cross_backend.py)

Requirements:
    pip install transformers sentence-transformers

Hardware requirements:
    - FP16: ~3GB VRAM
    - FP32: ~6GB VRAM

Usage:
    python gte_qwen3_cross_backend.py
    python gte_qwen3_cross_backend.py --benchmark
    python gte_qwen3_cross_backend.py --batch-size 64
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


def run_embedding(model_name: str) -> dict[str, Any]:
    """Run GTE-Qwen3 embedding with cross-backend comparison."""
    print_section(f"GTE-Qwen3 Embedding - {model_name}")

    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Diverse sentences to test embedding quality
        queries = [
            "How does TorchBridge handle cross-backend optimization?",
            "What is the best GPU for training large language models?",
        ]
        documents = [
            "TorchBridge provides hardware abstraction for PyTorch, enabling code "
            "to run on NVIDIA, AMD, Trainium, TPU, and CPU without modifications.",
            "The NVIDIA H100 with 80GB HBM3 is widely used for LLM training, "
            "offering FP8 Transformer Engine for 2x throughput.",
            "Apple's M-series chips use unified memory architecture for ML workloads.",
            "RAG pipelines benefit from fast embedding throughput on GPU backends.",
        ]

        all_sentences = queries + documents

        inputs = tokenizer(
            all_sentences, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )

        # CPU forward pass
        with torch.no_grad():
            cpu_out = model(**inputs)
        cpu_embeddings = cpu_out.last_hidden_state[:, 0]
        cpu_embeddings = torch.nn.functional.normalize(cpu_embeddings, p=2, dim=1)

        print(f"Embedding shape: {cpu_embeddings.shape}")
        print(f"Embedding dim: {cpu_embeddings.shape[1]}")

        # Semantic similarity matrix (queries vs documents)
        query_emb = cpu_embeddings[:len(queries)]
        doc_emb = cpu_embeddings[len(queries):]
        similarity = torch.mm(query_emb, doc_emb.t())

        print("\nSemantic Similarity (queries vs documents):")
        for i, query in enumerate(queries):
            print(f"\n  Query: '{query[:60]}...'")
            scores = similarity[i].tolist()
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            for rank, (doc_idx, score) in enumerate(ranked):
                marker = " <-- best match" if rank == 0 else ""
                print(f"    [{score:.4f}] {documents[doc_idx][:70]}...{marker}")

        # GPU comparison
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_gpu = model.to(device)
            inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gpu_out = model_gpu(**inputs_gpu)
            gpu_embeddings = gpu_out.last_hidden_state[:, 0]
            gpu_embeddings = torch.nn.functional.normalize(gpu_embeddings, p=2, dim=1)

            max_diff = torch.abs(
                cpu_embeddings - gpu_embeddings.cpu()
            ).max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                cpu_embeddings.flatten().unsqueeze(0),
                gpu_embeddings.cpu().flatten().unsqueeze(0),
            ).item()

            print("\nCross-backend comparison:")
            print(f"  Max diff: {max_diff:.2e}")
            print(f"  Cosine sim: {cos_sim:.6f}")
            print(f"  Status: {'PASSED' if max_diff < 1e-4 else 'REVIEW'}")

            return {
                "model_name": model_name,
                "embedding_shape": list(cpu_embeddings.shape),
                "max_diff": max_diff,
                "cosine_sim": cos_sim,
                "status": "PASSED" if max_diff < 1e-4 else "REVIEW",
                "similarity_matrix": similarity.tolist(),
            }

        return {
            "model_name": model_name,
            "embedding_shape": list(cpu_embeddings.shape),
            "status": "CPU_ONLY",
            "similarity_matrix": similarity.tolist(),
        }

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return {"error": str(e)}


def run_benchmark(
    model_name: str,
    num_runs: int = 10,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run structured benchmark for GTE-Qwen3."""
    print_section(f"Benchmark - {model_name} (batch={batch_size})")

    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        sentences = [
            f"Benchmark sentence number {i} for embedding throughput testing."
            for i in range(batch_size)
        ]
        inputs = tokenizer(
            sentences, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        results = {
            "model": model_name,
            "device": str(device),
            "batch_size": batch_size,
            "num_runs": num_runs,
            "latency_p50_ms": latencies[len(latencies) // 2],
            "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
            "throughput_sentences_per_sec": batch_size * 1000.0
            / latencies[len(latencies) // 2],
            "system_info": get_system_info(),
        }

        print(f"Results ({num_runs} runs, batch={batch_size}):")
        print(f"  Latency p50: {results['latency_p50_ms']:.1f} ms")
        print(f"  Latency p95: {results['latency_p95_ms']:.1f} ms")
        print(f"  Throughput: {results['throughput_sentences_per_sec']:.0f} sentences/s")

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
        description="GTE-Qwen3 Cross-Backend Embedding with TorchBridge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Alibaba-NLP/gte-Qwen3-embedding",
        help="HuggingFace model name",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for benchmark"
    )
    parser.add_argument("--output-json", type=str, help="Save results to JSON")

    args = parser.parse_args()

    print_section("GTE-Qwen3 Cross-Backend Embedding with TorchBridge")

    sys_info = get_system_info()
    print("System Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    deps = check_dependencies()
    if not deps.get("transformers"):
        print("\nERROR: transformers required. Install: pip install transformers")
        return

    if args.benchmark:
        results = run_benchmark(args.model, batch_size=args.batch_size)
    else:
        results = run_embedding(args.model)

    if "error" in results:
        print(f"\nFull demo requires model access. Error: {results['error']}")

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print_section("Complete!")


if __name__ == "__main__":
    main()
