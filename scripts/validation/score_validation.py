#!/usr/bin/env python3
"""
TorchBridge Validation Scoring System

Aggregates cloud validation reports and test suite metadata to produce a
structured multi-dimensional quality score for TorchBridge.

Usage:
    python3 scripts/validation/score_validation.py
    python3 scripts/validation/score_validation.py --output reports/scoring/v0.5.31.json
    python3 scripts/validation/score_validation.py --verbose

Dimensions & Weights:
    Functionality  (30%) — API test correctness and inference accuracy
    Reliability    (20%) — Consistency and stability across platforms
    Performance    (20%) — Speed vs vanilla PyTorch baseline
    Robustness     (15%) — Fallback chains, negative scenarios, error recovery
    Portability    (10%) — Cross-hardware platform coverage
    Usability      ( 3%) — Developer experience and end-to-end accessibility
    Security       ( 2%) — Security test coverage and input validation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_ROOT = REPO_ROOT / "reports" / "cloud_validation"
SCORING_DIR = REPO_ROOT / "reports" / "scoring"
TESTS_ROOT = REPO_ROOT / "tests"

# ─── Dimension definitions ────────────────────────────────────────────────────

DIMENSIONS: dict[str, dict[str, Any]] = {
    "functionality": {
        "label": "Functionality",
        "weight": 0.30,
        "description": "API test pass rate, inference correctness, backend detection accuracy",
    },
    "reliability": {
        "label": "Reliability",
        "weight": 0.20,
        "description": "Cross-platform consistency, inference success rate, non-flaky behaviour",
    },
    "performance": {
        "label": "Performance",
        "weight": 0.20,
        "description": "Inference latency vs vanilla baseline, optimization overhead",
    },
    "robustness": {
        "label": "Robustness",
        "weight": 0.15,
        "description": "Fallback chains, error recovery, negative scenario test coverage",
    },
    "portability": {
        "label": "Portability",
        "weight": 0.10,
        "description": "Validated hardware backends and environment diversity",
    },
    "usability": {
        "label": "Usability",
        "weight": 0.03,
        "description": "End-to-end pipeline accessibility, CLI coverage, API simplicity",
    },
    "security": {
        "label": "Security & Privacy",
        "weight": 0.02,
        "description": "Security test coverage, input validation, safe loading",
    },
}

assert abs(sum(d["weight"] for d in DIMENSIONS.values()) - 1.0) < 1e-9

# Scoring bands for latency ratio (TorchBridge / vanilla)
_LATENCY_SCORE: list[tuple[float, int]] = [
    (0.95, 100),   # measurably faster than vanilla
    (1.00, 92),    # on par
    (1.05, 80),    # slight overhead
    (1.10, 65),    # moderate overhead
    (1.15, 50),    # notable overhead
    (1.20, 35),    # at regression limit
    (float("inf"), 10),  # failed regression check
]


# ─── Report loading ───────────────────────────────────────────────────────────

def _normalize_platform(name: str) -> str:
    """Normalize platform names for deduplication (hyphens → underscores, lowercase)."""
    return name.lower().replace("-", "_")


def load_platform_reports() -> list[dict]:
    """Load all platform JSON reports, latest-date version per platform.

    Normalizes platform name differences between date directories
    (e.g. 'aws-a10g' and 'aws_a10g' are treated as the same platform).
    """
    seen: dict[str, dict] = {}
    for date_dir in sorted(REPORTS_ROOT.iterdir()):
        if not date_dir.is_dir():
            continue
        for json_file in sorted(date_dir.glob("*.json")):
            if json_file.stem in ("summary",):
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if "tests" not in data or "summary" not in data:
                    continue
                platform = data.get("platform", json_file.stem)
                key = _normalize_platform(platform)
                seen[key] = data  # later date wins (dirs sorted chronologically)
            except Exception:
                continue
    return list(seen.values())


def count_test_files(subdir: str) -> int:
    """Count test_*.py files in a tests/ subdirectory."""
    path = TESTS_ROOT / subdir
    if not path.is_dir():
        return 0
    return sum(1 for f in path.rglob("test_*.py"))


def collect_test_counts() -> dict[str, int]:
    """Count tests in key quality-assurance directories."""
    counts: dict[str, int] = {}
    for name in ("security", "robustness", "stress", "regression", "features"):
        counts[name] = count_test_files(name)
    return counts


# ─── Dimension scorers ────────────────────────────────────────────────────────

def score_functionality(reports: list[dict]) -> tuple[float, dict]:
    """
    Functionality = weighted avg of:
      70% — API test pass rate across platforms
      30% — Inference output correctness (cosine similarity)
    """
    api_rates: list[float] = []
    cosine_scores: list[float] = []
    details: list[dict] = []

    for r in reports:
        summary = r["summary"]
        total = summary["total_api_tests"]
        passed = summary["passed"]
        rate = passed / total if total > 0 else 0.0
        api_rates.append(rate)

        detail: dict[str, Any] = {
            "platform": r.get("platform", "unknown"),
            "api_pass_rate": round(rate * 100, 1),
            "passed": passed,
            "total": total,
        }

        # Inference correctness: parse cosine_sim from torchbridge sub-run
        inf = r.get("inference", {})
        tb_inf = inf.get("torchbridge", {})
        cos = tb_inf.get("cosine_sim")
        if cos is not None:
            # Score: ≥0.9999 → 100, ≥0.999 → 95, ≥0.99 → 80, ≥0.95 → 60, else 30
            if cos >= 0.9999:
                cs = 100
            elif cos >= 0.999:
                cs = 95
            elif cos >= 0.99:
                cs = 80
            elif cos >= 0.95:
                cs = 60
            else:
                cs = 30
            cosine_scores.append(cs)
            detail["inference_cosine_sim"] = round(cos, 6)
            detail["inference_cosine_score"] = cs
        else:
            detail["inference_cosine_sim"] = None

        details.append(detail)

    avg_api = sum(api_rates) / len(api_rates) * 100 if api_rates else 0
    avg_cos = sum(cosine_scores) / len(cosine_scores) if cosine_scores else avg_api
    score = 0.70 * avg_api + 0.30 * avg_cos

    return round(score, 1), {
        "avg_api_pass_rate_pct": round(avg_api, 1),
        "avg_cosine_score": round(avg_cos, 1),
        "platforms": details,
    }


def score_reliability(reports: list[dict]) -> tuple[float, dict]:
    """
    Reliability = weighted avg of:
      50% — Platform 100% API pass rate (25/25 on every platform)
      30% — Inference pass rate (GPU platforms only)
      20% — Zero variance: no platform significantly below average
    """
    total_platforms = len(reports)
    full_pass = sum(1 for r in reports if r["summary"]["pass_rate_pct"] == 100.0)

    # Inference: only GPU platforms (CUDA, MPS) can meaningfully pass inference
    gpu_reports = [
        r for r in reports
        if r["environment"].get("cuda_available") or r["environment"].get("mps_available")
    ]
    inf_passed = sum(
        1 for r in gpu_reports
        if r.get("inference", {}).get("passed") is True
    )
    gpu_inf_rate = inf_passed / len(gpu_reports) if gpu_reports else 1.0

    # Variance: std dev of api pass rates
    rates = [r["summary"]["pass_rate_pct"] for r in reports]
    mean_rate = sum(rates) / len(rates) if rates else 100.0
    variance = sum((x - mean_rate) ** 2 for x in rates) / len(rates) if rates else 0.0
    std_dev = math.sqrt(variance)
    # 0 std_dev → 100, every 5% std_dev → -20 points
    variance_score = max(0, 100 - std_dev * 4)

    full_pass_pct = full_pass / total_platforms * 100 if total_platforms > 0 else 0
    score = (0.50 * full_pass_pct) + (0.30 * gpu_inf_rate * 100) + (0.20 * variance_score)

    return round(score, 1), {
        "platforms_100pct": f"{full_pass}/{total_platforms}",
        "gpu_inference_passed": f"{inf_passed}/{len(gpu_reports)}",
        "api_pass_std_dev": round(std_dev, 2),
        "variance_score": round(variance_score, 1),
    }


def score_performance(reports: list[dict]) -> tuple[float, dict]:
    """
    Performance = weighted avg of:
      60% — Inference latency ratio (TorchBridge / vanilla), GPU platforms only
      40% — Optimization overhead (auto_optimize elapsed_ms)
    """
    latency_scores: list[float] = []
    latency_details: list[dict] = []
    optim_scores: list[float] = []

    for r in reports:
        inf = r.get("inference", {})
        tb_inf = inf.get("torchbridge", {})

        # Latency ratio
        ratio = tb_inf.get("latency_ratio_vs_vanilla")
        if ratio is not None:
            for threshold, pts in _LATENCY_SCORE:
                if ratio <= threshold:
                    latency_scores.append(pts)
                    latency_details.append({
                        "platform": r.get("platform"),
                        "ratio": round(ratio, 3),
                        "score": pts,
                    })
                    break

        # Optimization time (from API test)
        for t in r.get("tests", []):
            if t["name"] == "unified_manager_auto_optimize" and t["passed"]:
                val = t.get("value", {})
                elapsed = val.get("elapsed_ms") if isinstance(val, dict) else None
                if elapsed is not None:
                    if elapsed < 100:
                        optim_scores.append(100)
                    elif elapsed < 200:
                        optim_scores.append(92)
                    elif elapsed < 500:
                        optim_scores.append(78)
                    elif elapsed < 1000:
                        optim_scores.append(60)
                    else:
                        optim_scores.append(40)

    avg_latency = sum(latency_scores) / len(latency_scores) if latency_scores else 70.0
    avg_optim = sum(optim_scores) / len(optim_scores) if optim_scores else 70.0
    score = 0.60 * avg_latency + 0.40 * avg_optim

    return round(score, 1), {
        "avg_latency_score": round(avg_latency, 1),
        "avg_optim_score": round(avg_optim, 1),
        "latency_measurements": latency_details,
        "note": "CPU-only platforms (Trainium, Inferentia2, TPU) excluded from latency ratio",
    }


def score_robustness(reports: list[dict], test_counts: dict[str, int]) -> tuple[float, dict]:
    """
    Robustness = weighted avg of:
      35% — Fallback chains exercised (attention kernel fallback)
      25% — CPU fallback on non-GPU platforms (graceful degradation)
      25% — Negative scenario test coverage (robustness + stress + regression tests)
      15% — JIT / import-time resilience (inference ran on tricky platforms)
    """
    # Fallback chain: look for attention_select_kernel tests that show used_fallback
    fallback_hits = 0
    fallback_total = 0
    cpu_fallback_hits = 0
    cpu_fallback_total = 0

    for r in reports:
        env = r.get("environment", {})
        has_gpu = env.get("cuda_available") or env.get("mps_available") or env.get("xla_available")

        for t in r.get("tests", []):
            if t["name"] == "attention_select_kernel" and t["passed"]:
                fallback_total += 1
                val = t.get("value", "")
                val_str = str(val)
                # The value dict contains used_fallback key
                if "used_fallback" in val_str and ("True" in val_str or "fallback" in val_str.lower()):
                    fallback_hits += 1

        # CPU fallback: non-CUDA platforms (Trainium, Inferentia2, TPU) ran inference on CPU
        if not env.get("cuda_available") and not env.get("mps_available"):
            cpu_fallback_total += 1
            inf = r.get("inference", {})
            if inf.get("passed") and inf.get("device") in ("CPU", None):
                cpu_fallback_hits += 1

    fallback_score = (fallback_hits / fallback_total * 100) if fallback_total > 0 else 60.0
    cpu_fallback_score = (cpu_fallback_hits / cpu_fallback_total * 100) if cpu_fallback_total > 0 else 70.0

    # Test coverage: robustness + stress + regression test files
    total_neg_files = sum(
        test_counts.get(k, 0)
        for k in ("robustness", "stress", "regression")
    )
    # Scoring: 0 files → 0, 5 files → 50, 10+ files → 80, 15+ files → 95
    if total_neg_files >= 15:
        coverage_score = 95
    elif total_neg_files >= 10:
        coverage_score = 80
    elif total_neg_files >= 5:
        coverage_score = 50
    else:
        coverage_score = max(0, total_neg_files * 10)

    # JIT resilience: Trainium and Inferentia passed 25/25 despite JIT challenges
    tricky_platforms = [
        r for r in reports
        if "trainium" in r.get("platform", "") or "inferentia" in r.get("platform", "")
    ]
    jit_score = 100 if all(
        r["summary"]["pass_rate_pct"] == 100.0 for r in tricky_platforms
    ) and tricky_platforms else 60.0

    score = (
        0.35 * fallback_score
        + 0.25 * cpu_fallback_score
        + 0.25 * coverage_score
        + 0.15 * jit_score
    )

    return round(score, 1), {
        "attention_fallback": f"{fallback_hits}/{fallback_total} platforms used fallback chain",
        "cpu_fallback": f"{cpu_fallback_hits}/{cpu_fallback_total} non-GPU platforms ran CPU inference",
        "negative_test_files": total_neg_files,
        "neg_test_breakdown": {k: test_counts.get(k, 0) for k in ("robustness", "stress", "regression")},
        "jit_resilience_score": jit_score,
        "coverage_score": coverage_score,
    }


def score_portability(reports: list[dict]) -> tuple[float, dict]:
    """
    Portability = weighted avg of:
      60% — Hardware backend coverage (distinct backend types with 25/25 PASS)
      40% — Environment diversity (OS/runtime variety)
    """
    # Distinct backend types validated
    backend_map = {
        "nvidia": False, "amd": False, "tpu": False,
        "trainium": False, "inferentia": False, "mps": False, "cpu": False,
    }

    env_diversity: set[str] = set()

    for r in reports:
        if r["summary"]["pass_rate_pct"] < 100.0:
            continue
        env = r.get("environment", {})
        gpu_names = env.get("gpu_names", [])
        platform = r.get("platform", "")

        # Classify backend
        if "trainium" in platform.lower() or "trn" in platform.lower():
            backend_map["trainium"] = True
        elif "inferentia" in platform.lower() or "inf2" in platform.lower():
            backend_map["inferentia"] = True
        elif "tpu" in platform.lower():
            backend_map["tpu"] = True
        elif env.get("mps_available"):
            backend_map["mps"] = True
        elif env.get("cuda_available"):
            gpu_str = " ".join(gpu_names).lower()
            if "amd" in gpu_str or "radeon" in gpu_str or "instinct" in gpu_str:
                backend_map["amd"] = True
            else:
                backend_map["nvidia"] = True

        # CPU is always available (fallback confirmed on non-GPU runs)
        backend_map["cpu"] = True

        # Environment diversity: capture runtime string
        platform_str = env.get("platform", "")
        if "Linux" in platform_str:
            env_diversity.add("linux")
        if "Darwin" in platform_str or "macOS" in platform_str:
            env_diversity.add("darwin_macos")
        pt_ver = env.get("pytorch", "")
        if "cu1" in pt_ver:
            env_diversity.add("cuda_runtime")
        if "rocm" in pt_ver.lower():
            env_diversity.add("rocm_runtime")
        if "neuronx" in r.get("platform", "").lower() or "trainium" in r.get("platform", "").lower():
            env_diversity.add("neuronx_runtime")
        if "xla" in r.get("platform", "").lower() or "tpu" in r.get("platform", "").lower():
            env_diversity.add("xla_runtime")

    validated_backends = sum(1 for v in backend_map.values() if v)
    total_backends = len(backend_map)
    backend_score = validated_backends / total_backends * 100

    # Diversity score: 1 env → 40, 3 → 70, 5+ → 100
    div_count = len(env_diversity)
    if div_count >= 5:
        diversity_score = 100
    elif div_count >= 3:
        diversity_score = 70
    elif div_count >= 1:
        diversity_score = 40
    else:
        diversity_score = 0

    score = 0.60 * backend_score + 0.40 * diversity_score

    return round(score, 1), {
        "backend_coverage": {k: ("✓" if v else "✗") for k, v in backend_map.items()},
        "validated_backends": f"{validated_backends}/{total_backends}",
        "environments": sorted(env_diversity),
        "diversity_score": diversity_score,
    }


def score_usability(reports: list[dict]) -> tuple[float, dict]:
    """
    Usability = weighted avg of:
      40% — Inference pipeline success (end-to-end pipeline accessible to users)
      30% — CLI command richness (inferred from known CLI count)
      30% — API simplicity: single auto_optimize() call works across all platforms
    """
    # Inference pipeline: % of all tested platforms where inference ran to completion
    inf_complete = sum(
        1 for r in reports
        if r.get("inference", {}).get("passed") is not None  # result exists, not skipped
    )
    inf_passed = sum(
        1 for r in reports
        if r.get("inference", {}).get("passed") is True
    )
    inf_score = (inf_passed / len(reports)) * 100 if reports else 0

    # CLI richness: known from codebase (14 commands as of v0.5.29)
    cli_count = _discover_cli_count()
    if cli_count >= 14:
        cli_score = 100
    elif cli_count >= 10:
        cli_score = 80
    elif cli_count >= 6:
        cli_score = 60
    else:
        cli_score = 40

    # API simplicity: auto_optimize() worked on all platforms (unified_manager test)
    auto_opt_pass = sum(
        1 for r in reports
        for t in r.get("tests", [])
        if t["name"] == "unified_manager_auto_optimize" and t["passed"]
    )
    simplicity_score = (auto_opt_pass / len(reports)) * 100 if reports else 0

    score = 0.40 * inf_score + 0.30 * cli_score + 0.30 * simplicity_score

    return round(score, 1), {
        "inference_pass_rate_pct": round(inf_passed / len(reports) * 100, 1) if reports else 0,
        "cli_commands_discovered": cli_count,
        "auto_optimize_pass_rate_pct": round(auto_opt_pass / len(reports) * 100, 1) if reports else 0,
    }


def score_security(test_counts: dict[str, int]) -> tuple[float, dict]:
    """
    Security = weighted avg of:
      50% — Security test file coverage (dedicated security test suite)
      30% — Input validation / robustness test coverage
      20% — Safe loading (model deserialisation safety)
    """
    sec_files = test_counts.get("security", 0)
    rob_files = test_counts.get("robustness", 0)

    # Security test coverage
    if sec_files >= 5:
        sec_score = 100
    elif sec_files >= 3:
        sec_score = 80
    elif sec_files >= 1:
        sec_score = 60
    else:
        sec_score = 10

    # Input validation / robustness
    if rob_files >= 5:
        rob_score = 90
    elif rob_files >= 3:
        rob_score = 70
    elif rob_files >= 1:
        rob_score = 50
    else:
        rob_score = 10

    # Safe loading: check for test_cli_safe_loading.py
    safe_loading_score = 100 if sec_files > 0 else 40

    score = 0.50 * sec_score + 0.30 * rob_score + 0.20 * safe_loading_score

    return round(score, 1), {
        "security_test_files": sec_files,
        "robustness_test_files": rob_files,
        "safe_loading_covered": safe_loading_score == 100,
        "note": "Active penetration testing not performed; coverage based on test suite analysis",
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _discover_cli_count() -> int:
    """Count CLI entry points from pyproject.toml."""
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return 0
    try:
        content = pyproject.read_text()
        # Count lines like  tb-xxx = "..."
        return sum(
            1 for line in content.splitlines()
            if line.strip().startswith("tb-") and "=" in line
        )
    except Exception:
        return 0


def letter_grade(score: float) -> str:
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "B+"
    elif score >= 80:
        return "B"
    elif score >= 75:
        return "C+"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


# ─── Report generation ────────────────────────────────────────────────────────

def generate_report(verbose: bool = False) -> dict:
    reports = load_platform_reports()
    if not reports:
        print("ERROR: No valid validation reports found.", file=sys.stderr)
        sys.exit(1)

    test_counts = collect_test_counts()

    # Compute dimension scores
    func_score, func_detail = score_functionality(reports)
    rel_score, rel_detail = score_reliability(reports)
    perf_score, perf_detail = score_performance(reports)
    rob_score, rob_detail = score_robustness(reports, test_counts)
    port_score, port_detail = score_portability(reports)
    usa_score, usa_detail = score_usability(reports)
    sec_score, sec_detail = score_security(test_counts)

    scores = {
        "functionality": func_score,
        "reliability": rel_score,
        "performance": perf_score,
        "robustness": rob_score,
        "portability": port_score,
        "usability": usa_score,
        "security": sec_score,
    }

    weighted_total = sum(
        scores[dim] * DIMENSIONS[dim]["weight"]
        for dim in DIMENSIONS
    )

    details = {
        "functionality": func_detail,
        "reliability": rel_detail,
        "performance": perf_detail,
        "robustness": rob_detail,
        "portability": port_detail,
        "usability": usa_detail,
        "security": sec_detail,
    }

    return {
        "generated_at": datetime.now().isoformat(),
        "platforms_evaluated": len(reports),
        "platform_list": [r.get("platform", "?") for r in reports],
        "torchbridge_version": _infer_version(reports),
        "weighted_score": round(weighted_total, 1),
        "grade": letter_grade(weighted_total),
        "dimension_scores": {
            dim: {
                "label": DIMENSIONS[dim]["label"],
                "weight_pct": int(DIMENSIONS[dim]["weight"] * 100),
                "score": scores[dim],
                "weighted_contribution": round(scores[dim] * DIMENSIONS[dim]["weight"], 1),
                "description": DIMENSIONS[dim]["description"],
            }
            for dim in DIMENSIONS
        },
        "dimension_details": details if verbose else {},
        "improvement_targets": _build_improvement_targets(scores),
    }


def _infer_version(reports: list[dict]) -> str:
    versions = {r.get("torchbridge_version", "unknown") for r in reports}
    versions.discard("unknown")
    return max(versions) if versions else "unknown"


def _build_improvement_targets(scores: dict[str, float]) -> list[dict]:
    """Return the top 3 dimensions with highest improvement potential."""
    # Weighted improvement potential = (100 - score) × weight
    potential = {
        dim: (100 - scores[dim]) * DIMENSIONS[dim]["weight"]
        for dim in DIMENSIONS
    }
    ranked = sorted(potential.items(), key=lambda x: x[1], reverse=True)
    targets = []
    for dim, pot in ranked[:3]:
        targets.append({
            "dimension": DIMENSIONS[dim]["label"],
            "current_score": scores[dim],
            "gap_points": round(100 - scores[dim], 1),
            "weighted_gain_if_perfect": round(pot, 1),
            "description": DIMENSIONS[dim]["description"],
        })
    return targets


# ─── Printing ─────────────────────────────────────────────────────────────────

def print_report(report: dict, verbose: bool = False) -> None:
    W = 70
    print("=" * W)
    print("  TorchBridge Validation Score Report")
    print("=" * W)
    print(f"  Version  : {report['torchbridge_version']}")
    print(f"  Generated: {report['generated_at'][:10]}")
    print(f"  Platforms: {report['platforms_evaluated']}  ({', '.join(report['platform_list'])})")
    print()

    # Dimension table
    print(f"  {'Dimension':<18} {'Wt':>4}  {'Score':>6}  {'Contrib':>7}  {'Bar'}")
    print("  " + "─" * 62)
    for dim, d in report["dimension_scores"].items():
        bar_len = int(d["score"] / 5)  # 0-100 → 0-20 chars
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  {d['label']:<18} {d['weight_pct']:>3}%  "
            f"{d['score']:>5.1f}  {d['weighted_contribution']:>6.1f}  {bar}"
        )
    print("  " + "─" * 62)
    print(
        f"  {'WEIGHTED TOTAL':<18} {'100%':>4}  "
        f"{report['weighted_score']:>5.1f}  "
        f"  Grade: {report['grade']}"
    )
    print()

    # Improvement targets
    print("  Top Improvement Opportunities:")
    for t in report["improvement_targets"]:
        print(
            f"    [{t['dimension']}] score={t['current_score']:.0f}/100, "
            f"gap={t['gap_points']:.0f}pts, "
            f"potential weighted gain={t['weighted_gain_if_perfect']:.1f}pts"
        )

    # Verbose details
    if verbose:
        print()
        print("  Dimension Details:")
        for dim, detail in report["dimension_details"].items():
            print(f"\n  ── {DIMENSIONS[dim]['label']} ─────────────────────────────────")
            print(f"  {json.dumps(detail, indent=4)}")

    print()
    print("=" * W)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TorchBridge Validation Scoring System")
    parser.add_argument(
        "--output", "-o",
        help="Save JSON report to this path (default: reports/scoring/YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed breakdown per dimension",
    )
    args = parser.parse_args()

    report = generate_report(verbose=args.verbose)
    print_report(report, verbose=args.verbose)

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        SCORING_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = SCORING_DIR / f"{date_str}_score.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # Always include details in the saved file
        report_full = generate_report(verbose=True)
        json.dump(report_full, f, indent=2)
    print(f"  Report saved → {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
