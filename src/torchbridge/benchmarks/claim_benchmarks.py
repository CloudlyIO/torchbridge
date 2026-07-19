# SPDX-License-Identifier: Apache-2.0
"""
Claim-Level Benchmark Harness

Measures a TorchBridge performance claim by timing a baseline function
(vanilla PyTorch) against an optimized function (TorchBridge-applied),
computing speedup percentage, and determining pass/fail against a threshold.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ClaimResult:
    """Result of benchmarking a single performance claim."""

    claim_name: str
    baseline_ms: float
    optimized_ms: float
    speedup_pct: float
    passed: bool
    threshold_pct: float
    runs: int
    std_baseline_ms: float
    std_optimized_ms: float
    device: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_name": self.claim_name,
            "baseline_ms": round(self.baseline_ms, 4),
            "optimized_ms": round(self.optimized_ms, 4),
            "speedup_pct": round(self.speedup_pct, 2),
            "passed": self.passed,
            "threshold_pct": self.threshold_pct,
            "runs": self.runs,
            "std_baseline_ms": round(self.std_baseline_ms, 4),
            "std_optimized_ms": round(self.std_optimized_ms, 4),
            "device": self.device,
            "notes": self.notes,
        }


def _time_fn(fn: Callable[[], Any], warmup: int, runs: int) -> tuple[float, float]:
    """Time a callable, returning (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    std = variance**0.5
    return mean, std


class ClaimBenchmark:
    """Measures a single TorchBridge performance claim."""

    def __init__(
        self,
        name: str,
        baseline_fn: Callable[[], Any],
        optimized_fn: Callable[[], Any],
        *,
        warmup: int = 5,
        runs: int = 20,
        threshold_pct: float = 3.0,
        notes: list[str] | None = None,
        requires_backend: str | None = None,
        skip_reason: str | None = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._baseline_fn = baseline_fn
        self._optimized_fn = optimized_fn
        self._warmup = warmup
        self._runs = runs
        self._threshold_pct = threshold_pct
        self._notes = notes or []
        self.requires_backend = requires_backend
        self.skip_reason = skip_reason

    def run(self, device: str = "cpu") -> ClaimResult:
        """Time baseline vs optimized, compute speedup.

        If ``skip_reason`` is set, returns a zero-runs skipped result immediately
        without executing either function (e.g., required library not installed).
        """
        if self.skip_reason:
            return ClaimResult(
                claim_name=self.name,
                baseline_ms=0.0,
                optimized_ms=0.0,
                speedup_pct=0.0,
                passed=False,
                threshold_pct=self._threshold_pct,
                runs=0,
                std_baseline_ms=0.0,
                std_optimized_ms=0.0,
                device=device,
                notes=[f"SKIPPED: {self.skip_reason}"] + list(self._notes),
            )

        base_mean, base_std = _time_fn(self._baseline_fn, self._warmup, self._runs)
        opt_mean, opt_std = _time_fn(self._optimized_fn, self._warmup, self._runs)

        speedup = ((base_mean - opt_mean) / base_mean * 100.0) if base_mean > 0 else 0.0
        passed = speedup >= self._threshold_pct

        return ClaimResult(
            claim_name=self.name,
            baseline_ms=base_mean,
            optimized_ms=opt_mean,
            speedup_pct=speedup,
            passed=passed,
            threshold_pct=self._threshold_pct,
            runs=self._runs,
            std_baseline_ms=base_std,
            std_optimized_ms=opt_std,
            device=device,
            notes=list(self._notes),
        )


class BenchmarkSuite:
    """Runs all claim benchmarks and produces a report."""

    def __init__(self) -> None:
        self._benchmarks: list[ClaimBenchmark] = []

    def run_all(self, device: str = "cpu") -> BenchmarkReport:
        """Run all benchmarks, skipping those that require unavailable backends."""
        results: list[ClaimResult] = []
        for bench in self._benchmarks:
            if bench.requires_backend and bench.requires_backend != device:
                results.append(
                    ClaimResult(
                        claim_name=bench.name,
                        baseline_ms=0.0,
                        optimized_ms=0.0,
                        speedup_pct=0.0,
                        passed=False,
                        threshold_pct=bench._threshold_pct,
                        runs=0,
                        std_baseline_ms=0.0,
                        std_optimized_ms=0.0,
                        device=device,
                        notes=[
                            f"SKIPPED: requires {bench.requires_backend}, "
                            f"running on {device}"
                        ],
                    )
                )
                continue
            results.append(bench.run(device=device))
        return BenchmarkReport(results=results, device=device)


@dataclass
class BenchmarkReport:
    """Aggregated report from running a benchmark suite."""

    results: list[ClaimResult]
    device: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        """One-line summary of pass/fail counts."""
        ran = [r for r in self.results if r.runs > 0]
        skipped = [r for r in self.results if r.runs == 0]
        passed = sum(1 for r in ran if r.passed)
        failed = len(ran) - passed
        parts = [f"{passed} passed", f"{failed} failed"]
        if skipped:
            parts.append(f"{len(skipped)} skipped")
        return ", ".join(parts)

    def claims_to_delete(self) -> list[str]:
        """Return names of claims that failed the threshold (candidates for deletion)."""
        return [
            r.claim_name
            for r in self.results
            if r.runs > 0 and not r.passed
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "timestamp": self.timestamp,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
            "claims_to_delete": self.claims_to_delete(),
        }

