"""
Kernel Benchmark Cache

Caches per-kernel latency measurements to avoid repeated benchmarking.
Cache is invalidated when the hardware fingerprint changes (new GPU,
PyTorch version, or TorchBridge version).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .kernel_types import AttentionKernelType

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".torchbridge")
_DEFAULT_CACHE_FILE = "kernel_benchmarks.json"


@dataclass
class BenchmarkEntry:
    """Single benchmark measurement."""

    kernel_type: str
    latency_ms: float
    throughput_tflops: float
    seq_length: int
    num_heads: int
    head_dim: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkFingerprint:
    """Hardware fingerprint used for cache invalidation."""

    backend: str
    architecture: str
    device_name: str
    pytorch_version: str
    torchbridge_version: str

    @classmethod
    def current(cls) -> BenchmarkFingerprint:
        """Build fingerprint from current environment."""
        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)

        try:
            from importlib.metadata import PackageNotFoundError, version

            tb_version = version("torchbridge-ml")
        except (PackageNotFoundError, ImportError):
            tb_version = "dev"

        return cls(
            backend="cuda" if torch.cuda.is_available() else "cpu",
            architecture="unknown",
            device_name=device_name,
            pytorch_version=torch.__version__,
            torchbridge_version=tb_version,
        )


class KernelBenchmarkCache:
    """Persistent benchmark cache with fingerprint invalidation."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_path = os.path.join(self._cache_dir, _DEFAULT_CACHE_FILE)
        self._entries: dict[str, BenchmarkEntry] = {}
        self._fingerprint: BenchmarkFingerprint | None = None
        self._load()

    # ── public API ───────────────────────────────────────────────────

    def get_cached_latency(
        self,
        kernel_type: AttentionKernelType,
        seq_length: int,
        num_heads: int,
        head_dim: int,
    ) -> float | None:
        """Return cached latency in ms, or None if not cached."""
        key = self._make_key(kernel_type, seq_length, num_heads, head_dim)
        entry = self._entries.get(key)
        return entry.latency_ms if entry else None

    def run_benchmark(
        self,
        kernel_type: AttentionKernelType,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 3,
        iterations: int = 10,
    ) -> BenchmarkEntry:
        """Benchmark a kernel using PyTorch SDPA as a proxy and cache the result."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = 1
        q = torch.randn(batch, num_heads, seq_length, head_dim, device=device)
        k = torch.randn(batch, num_heads, seq_length, head_dim, device=device)
        v = torch.randn(batch, num_heads, seq_length, head_dim, device=device)

        # warm-up
        for _ in range(warmup):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iterations):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / iterations * 1000  # ms

        # Rough TFLOPS estimate: 2 * B * H * S^2 * D
        flops = 2 * batch * num_heads * seq_length * seq_length * head_dim
        tflops = flops / (elapsed / 1000) / 1e12 if elapsed > 0 else 0.0

        entry = BenchmarkEntry(
            kernel_type=kernel_type.value,
            latency_ms=elapsed,
            throughput_tflops=tflops,
            seq_length=seq_length,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        key = self._make_key(kernel_type, seq_length, num_heads, head_dim)
        self._entries[key] = entry
        self._save()
        return entry

    def warm_cache(
        self,
        kernel_types: list[AttentionKernelType],
        seq_length: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
    ) -> dict[str, float]:
        """Benchmark all given kernels and return {kernel_name: latency_ms}."""
        results: dict[str, float] = {}
        for kt in kernel_types:
            entry = self.run_benchmark(kt, seq_length, num_heads, head_dim)
            results[kt.value] = entry.latency_ms
        return results

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            stored_fp = data.get("fingerprint", {})
            current_fp = asdict(BenchmarkFingerprint.current())
            if stored_fp != current_fp:
                logger.debug("Benchmark cache fingerprint mismatch — clearing cache")
                self._entries = {}
                return
            self._fingerprint = BenchmarkFingerprint(**stored_fp)
            for key, entry_dict in data.get("entries", {}).items():
                self._entries[key] = BenchmarkEntry(**entry_dict)
        except Exception:
            logger.debug("Failed to load benchmark cache", exc_info=True)
            self._entries = {}

    def _save(self) -> None:
        os.makedirs(self._cache_dir, exist_ok=True)
        fp = BenchmarkFingerprint.current()
        data = {
            "fingerprint": asdict(fp),
            "entries": {k: asdict(v) for k, v in self._entries.items()},
        }
        try:
            with open(self._cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.debug("Failed to save benchmark cache", exc_info=True)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_key(
        kernel_type: AttentionKernelType,
        seq_length: int,
        num_heads: int,
        head_dim: int,
    ) -> str:
        return f"{kernel_type.value}_{seq_length}_{num_heads}_{head_dim}"
