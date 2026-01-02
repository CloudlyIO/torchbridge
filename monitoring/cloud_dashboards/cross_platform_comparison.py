"""
Cross-Platform Comparison Tool for KernelPyTorch.

This module provides utilities for comparing benchmark results
across different cloud platforms (AWS vs GCP) and hardware types.

Features:
- Compare NVIDIA performance: AWS P5/P4d vs GCP A3/A2
- Compare TPU performance across TPU generations
- Generate comparison reports and visualizations
- Detect cross-platform inconsistencies

Version: 0.3.7
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlatformMetrics:
    """Metrics for a single platform."""
    platform_name: str  # e.g., "AWS P4d", "GCP A2"
    cloud_provider: str  # "aws" or "gcp"
    instance_type: str
    hardware_type: str  # "nvidia", "amd", "tpu"
    gpu_model: str

    # Aggregated metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_throughput: float = 0.0
    avg_memory_mb: float = 0.0
    avg_gpu_utilization: float = 0.0

    # Test results
    total_tests: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    pass_rate: float = 0.0

    # Cost metrics
    avg_cost_per_hour: float = 0.0
    cost_per_1k_inferences: float = 0.0

    # Sample count
    sample_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform_name": self.platform_name,
            "cloud_provider": self.cloud_provider,
            "instance_type": self.instance_type,
            "hardware_type": self.hardware_type,
            "gpu_model": self.gpu_model,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_throughput": self.avg_throughput,
            "avg_memory_mb": self.avg_memory_mb,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "total_tests": self.total_tests,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "pass_rate": self.pass_rate,
            "avg_cost_per_hour": self.avg_cost_per_hour,
            "cost_per_1k_inferences": self.cost_per_1k_inferences,
            "sample_count": self.sample_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ComparisonMetric:
    """A single comparison metric between two platforms."""
    metric_name: str
    platform_a_value: float
    platform_b_value: float
    ratio: float  # platform_b / platform_a
    difference_pct: float  # (platform_b - platform_a) / platform_a * 100
    winner: str  # "platform_a", "platform_b", or "tie"
    significance: str  # "significant", "marginal", "none"


@dataclass
class ComparisonReport:
    """Full comparison report between two platforms."""
    platform_a: PlatformMetrics
    platform_b: PlatformMetrics
    metrics: List[ComparisonMetric]
    overall_winner: str
    summary: str
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform_a": self.platform_a.to_dict(),
            "platform_b": self.platform_b.to_dict(),
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "platform_a_value": m.platform_a_value,
                    "platform_b_value": m.platform_b_value,
                    "ratio": m.ratio,
                    "difference_pct": m.difference_pct,
                    "winner": m.winner,
                    "significance": m.significance,
                }
                for m in self.metrics
            ],
            "overall_winner": self.overall_winner,
            "summary": self.summary,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            f"# Cross-Platform Comparison Report",
            f"",
            f"**Generated**: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Platforms Compared",
            f"",
            f"| Property | {self.platform_a.platform_name} | {self.platform_b.platform_name} |",
            f"|----------|{'-' * len(self.platform_a.platform_name)}--|{'-' * len(self.platform_b.platform_name)}--|",
            f"| Cloud | {self.platform_a.cloud_provider.upper()} | {self.platform_b.cloud_provider.upper()} |",
            f"| Instance | {self.platform_a.instance_type} | {self.platform_b.instance_type} |",
            f"| GPU | {self.platform_a.gpu_model} | {self.platform_b.gpu_model} |",
            f"| Samples | {self.platform_a.sample_count} | {self.platform_b.sample_count} |",
            f"",
            f"## Metrics Comparison",
            f"",
            f"| Metric | {self.platform_a.platform_name} | {self.platform_b.platform_name} | Ratio | Winner |",
            f"|--------|{'-' * 10}|{'-' * 10}|-------|--------|",
        ]

        for m in self.metrics:
            winner_emoji = "🏆" if m.significance == "significant" else ""
            lines.append(
                f"| {m.metric_name} | {m.platform_a_value:.2f} | {m.platform_b_value:.2f} | "
                f"{m.ratio:.2f}x | {m.winner} {winner_emoji} |"
            )

        lines.extend([
            f"",
            f"## Summary",
            f"",
            f"**Overall Winner**: {self.overall_winner}",
            f"",
            self.summary,
        ])

        return "\n".join(lines)


# ============================================================================
# Cross-Platform Comparison
# ============================================================================

class CrossPlatformComparison:
    """
    Compare benchmark results across cloud platforms.

    Example:
        >>> comparison = CrossPlatformComparison()
        >>> comparison.add_platform(aws_metrics)
        >>> comparison.add_platform(gcp_metrics)
        >>> report = comparison.generate_report("AWS P4d", "GCP A2")
        >>> print(report.to_markdown())
    """

    # Significance thresholds
    SIGNIFICANT_THRESHOLD = 0.10  # 10% difference is significant
    MARGINAL_THRESHOLD = 0.05    # 5% difference is marginal

    def __init__(self):
        """Initialize comparison tool."""
        self.platforms: Dict[str, PlatformMetrics] = {}

    def add_platform(self, metrics: PlatformMetrics) -> None:
        """Add platform metrics for comparison."""
        self.platforms[metrics.platform_name] = metrics
        logger.info(f"Added platform: {metrics.platform_name}")

    def add_from_benchmark_results(
        self,
        results: Dict[str, Any],
        platform_name: str,
    ) -> None:
        """
        Add platform from raw benchmark results.

        Args:
            results: Dictionary of benchmark results
            platform_name: Name for this platform
        """
        metrics = PlatformMetrics(
            platform_name=platform_name,
            cloud_provider=results.get("cloud_provider", "unknown"),
            instance_type=results.get("instance_type", "unknown"),
            hardware_type=results.get("hardware_type", "unknown"),
            gpu_model=results.get("gpu_model", "unknown"),
            avg_latency_ms=results.get("avg_latency_ms", 0.0),
            p50_latency_ms=results.get("p50_latency_ms", 0.0),
            p95_latency_ms=results.get("p95_latency_ms", 0.0),
            p99_latency_ms=results.get("p99_latency_ms", 0.0),
            avg_throughput=results.get("avg_throughput", 0.0),
            avg_memory_mb=results.get("avg_memory_mb", 0.0),
            avg_gpu_utilization=results.get("avg_gpu_utilization", 0.0),
            total_tests=results.get("total_tests", 0),
            tests_passed=results.get("tests_passed", 0),
            tests_failed=results.get("tests_failed", 0),
            tests_skipped=results.get("tests_skipped", 0),
            pass_rate=results.get("pass_rate", 0.0),
            avg_cost_per_hour=results.get("avg_cost_per_hour", 0.0),
            cost_per_1k_inferences=results.get("cost_per_1k_inferences", 0.0),
            sample_count=results.get("sample_count", 1),
        )
        self.add_platform(metrics)

    def compare_metric(
        self,
        metric_name: str,
        value_a: float,
        value_b: float,
        lower_is_better: bool = True,
    ) -> ComparisonMetric:
        """
        Compare a single metric between platforms.

        Args:
            metric_name: Name of the metric
            value_a: Value for platform A
            value_b: Value for platform B
            lower_is_better: If True, lower values win (latency)

        Returns:
            ComparisonMetric with analysis
        """
        if value_a == 0:
            ratio = float('inf') if value_b > 0 else 1.0
            difference_pct = 100.0 if value_b > 0 else 0.0
        else:
            ratio = value_b / value_a
            difference_pct = (value_b - value_a) / value_a * 100

        # Determine winner
        if abs(difference_pct) < 1.0:  # Within 1% is a tie
            winner = "tie"
        elif lower_is_better:
            winner = "platform_a" if value_a < value_b else "platform_b"
        else:
            winner = "platform_a" if value_a > value_b else "platform_b"

        # Determine significance
        abs_diff = abs(difference_pct) / 100
        if abs_diff >= self.SIGNIFICANT_THRESHOLD:
            significance = "significant"
        elif abs_diff >= self.MARGINAL_THRESHOLD:
            significance = "marginal"
        else:
            significance = "none"

        return ComparisonMetric(
            metric_name=metric_name,
            platform_a_value=value_a,
            platform_b_value=value_b,
            ratio=ratio,
            difference_pct=difference_pct,
            winner=winner,
            significance=significance,
        )

    def generate_report(
        self,
        platform_a_name: str,
        platform_b_name: str,
    ) -> ComparisonReport:
        """
        Generate comparison report between two platforms.

        Args:
            platform_a_name: Name of first platform
            platform_b_name: Name of second platform

        Returns:
            ComparisonReport with full analysis
        """
        if platform_a_name not in self.platforms:
            raise ValueError(f"Platform not found: {platform_a_name}")
        if platform_b_name not in self.platforms:
            raise ValueError(f"Platform not found: {platform_b_name}")

        platform_a = self.platforms[platform_a_name]
        platform_b = self.platforms[platform_b_name]

        # Compare all metrics
        metrics = [
            self.compare_metric("Latency (ms)", platform_a.avg_latency_ms, platform_b.avg_latency_ms, lower_is_better=True),
            self.compare_metric("P95 Latency (ms)", platform_a.p95_latency_ms, platform_b.p95_latency_ms, lower_is_better=True),
            self.compare_metric("Throughput", platform_a.avg_throughput, platform_b.avg_throughput, lower_is_better=False),
            self.compare_metric("Memory (MB)", platform_a.avg_memory_mb, platform_b.avg_memory_mb, lower_is_better=True),
            self.compare_metric("GPU Utilization (%)", platform_a.avg_gpu_utilization, platform_b.avg_gpu_utilization, lower_is_better=False),
            self.compare_metric("Pass Rate (%)", platform_a.pass_rate, platform_b.pass_rate, lower_is_better=False),
            self.compare_metric("Cost/Hour ($)", platform_a.avg_cost_per_hour, platform_b.avg_cost_per_hour, lower_is_better=True),
            self.compare_metric("Cost/1K Inferences ($)", platform_a.cost_per_1k_inferences, platform_b.cost_per_1k_inferences, lower_is_better=True),
        ]

        # Determine overall winner
        significant_wins = {"platform_a": 0, "platform_b": 0}
        for m in metrics:
            if m.significance == "significant" and m.winner != "tie":
                significant_wins[m.winner] += 1

        if significant_wins["platform_a"] > significant_wins["platform_b"]:
            overall_winner = platform_a_name
        elif significant_wins["platform_b"] > significant_wins["platform_a"]:
            overall_winner = platform_b_name
        else:
            overall_winner = "Tie"

        # Generate summary
        summary_parts = []
        for m in metrics:
            if m.significance == "significant":
                winner_name = platform_a_name if m.winner == "platform_a" else platform_b_name
                summary_parts.append(
                    f"- **{m.metric_name}**: {winner_name} is {abs(m.difference_pct):.1f}% better"
                )

        summary = "\n".join(summary_parts) if summary_parts else "No significant differences found."

        return ComparisonReport(
            platform_a=platform_a,
            platform_b=platform_b,
            metrics=metrics,
            overall_winner=overall_winner,
            summary=summary,
        )

    def get_platform_names(self) -> List[str]:
        """Get list of available platform names."""
        return list(self.platforms.keys())


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_comparison_report(
    platform_a_results: Dict[str, Any],
    platform_b_results: Dict[str, Any],
    platform_a_name: str = "Platform A",
    platform_b_name: str = "Platform B",
) -> ComparisonReport:
    """
    Generate comparison report from raw results.

    Args:
        platform_a_results: Results dictionary for platform A
        platform_b_results: Results dictionary for platform B
        platform_a_name: Name for platform A
        platform_b_name: Name for platform B

    Returns:
        ComparisonReport
    """
    comparison = CrossPlatformComparison()
    comparison.add_from_benchmark_results(platform_a_results, platform_a_name)
    comparison.add_from_benchmark_results(platform_b_results, platform_b_name)
    return comparison.generate_report(platform_a_name, platform_b_name)


def create_comparison_chart(
    report: ComparisonReport,
    output_path: Optional[str] = None,
) -> str:
    """
    Create a text-based comparison chart.

    Args:
        report: ComparisonReport to visualize
        output_path: Optional path to save chart

    Returns:
        Text chart as string
    """
    lines = [
        "=" * 70,
        f"Cross-Platform Comparison: {report.platform_a.platform_name} vs {report.platform_b.platform_name}",
        "=" * 70,
        "",
    ]

    max_name_len = max(len(m.metric_name) for m in report.metrics)

    for m in report.metrics:
        # Create bar chart
        bar_width = 20
        if m.ratio > 0 and m.ratio != float('inf'):
            if m.ratio < 1:
                a_bar = int(bar_width * (1 / m.ratio) / 2)
                b_bar = int(bar_width / 2)
            else:
                a_bar = int(bar_width / 2)
                b_bar = int(bar_width * m.ratio / 2)
        else:
            a_bar = bar_width // 2
            b_bar = bar_width // 2

        a_bar = min(a_bar, bar_width)
        b_bar = min(b_bar, bar_width)

        bar = "█" * a_bar + "░" * (bar_width - a_bar) + " | " + "░" * (bar_width - b_bar) + "█" * b_bar

        winner_indicator = ""
        if m.winner == "platform_a" and m.significance == "significant":
            winner_indicator = " ◀"
        elif m.winner == "platform_b" and m.significance == "significant":
            winner_indicator = " ▶"

        lines.append(f"{m.metric_name:<{max_name_len}} {bar}{winner_indicator}")

    lines.extend([
        "",
        "-" * 70,
        f"Overall Winner: {report.overall_winner}",
        "-" * 70,
    ])

    chart = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(chart)
        logger.info(f"Chart saved to {output_path}")

    return chart
