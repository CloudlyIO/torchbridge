#!/usr/bin/env python3
"""
Generate markdown benchmark reports from pytest-benchmark JSON output.

Usage:
    python scripts/generate_benchmark_report.py --input benchmark.json --output report.md
    python scripts/generate_benchmark_report.py --input benchmark.json --baseline baseline.json --output report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_benchmark_data(path: Path) -> dict[str, Any]:
    """Load benchmark JSON data."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} \u00b5s"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def format_change(current: float, baseline: float) -> str:
    """Format percentage change with emoji indicator."""
    if baseline == 0:
        return "N/A"

    change = ((current - baseline) / baseline) * 100

    if change < -5:
        return f"\u2705 {change:+.1f}%"  # Green check - faster
    elif change > 15:
        return f"\u274c {change:+.1f}%"  # Red X - much slower
    elif change > 5:
        return f"\u26a0\ufe0f {change:+.1f}%"  # Warning - slower
    else:
        return f"\u2796 {change:+.1f}%"  # Neutral - similar


def generate_report(
    current_data: dict[str, Any],
    baseline_data: dict[str, Any] | None = None,
    title: str = "Benchmark Results"
) -> str:
    """Generate markdown report from benchmark data."""
    lines = []

    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Machine info
    machine = current_data.get("machine_info", {})
    if machine:
        lines.append("")
        lines.append("## Environment")
        lines.append("")
        lines.append(f"- **Python**: {machine.get('python_version', 'N/A')}")
        lines.append(f"- **Platform**: {machine.get('platform', 'N/A')}")
        lines.append(f"- **CPU**: {machine.get('cpu', {}).get('brand', 'N/A')}")
        lines.append(f"- **CPU Count**: {machine.get('cpu', {}).get('count', 'N/A')}")

    # Commit info
    commit = current_data.get("commit_info", {})
    if commit:
        lines.append("")
        lines.append("## Git Info")
        lines.append("")
        lines.append(f"- **Branch**: {commit.get('branch', 'N/A')}")
        lines.append(f"- **Commit**: {commit.get('id', 'N/A')[:8]}")

    # Benchmarks table
    benchmarks = current_data.get("benchmarks", [])
    if not benchmarks:
        lines.append("")
        lines.append("*No benchmark data available.*")
        return "\n".join(lines)

    # Build baseline lookup
    baseline_lookup = {}
    if baseline_data:
        for b in baseline_data.get("benchmarks", []):
            baseline_lookup[b["name"]] = b

    lines.append("")
    lines.append("## Results")
    lines.append("")

    # Group benchmarks by group
    groups: dict[str, list] = {}
    for b in benchmarks:
        group = b.get("group", "default")
        if group not in groups:
            groups[group] = []
        groups[group].append(b)

    for group_name, group_benchmarks in sorted(groups.items()):
        if len(groups) > 1:
            lines.append(f"### {group_name}")
            lines.append("")

        # Table header
        if baseline_data:
            lines.append("| Benchmark | Mean | Std Dev | Min | Max | Baseline | Change |")
            lines.append("|-----------|------|---------|-----|-----|----------|--------|")
        else:
            lines.append("| Benchmark | Mean | Std Dev | Min | Max | Rounds |")
            lines.append("|-----------|------|---------|-----|-----|--------|")

        for b in sorted(group_benchmarks, key=lambda x: x["name"]):
            name = b["name"]
            stats = b.get("stats", {})

            mean = stats.get("mean", 0)
            stddev = stats.get("stddev", 0)
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)
            rounds = stats.get("rounds", 0)

            if baseline_data and name in baseline_lookup:
                baseline_mean = baseline_lookup[name]["stats"].get("mean", 0)
                change = format_change(mean, baseline_mean)
                lines.append(
                    f"| `{name}` | {format_time(mean)} | {format_time(stddev)} | "
                    f"{format_time(min_val)} | {format_time(max_val)} | "
                    f"{format_time(baseline_mean)} | {change} |"
                )
            else:
                lines.append(
                    f"| `{name}` | {format_time(mean)} | {format_time(stddev)} | "
                    f"{format_time(min_val)} | {format_time(max_val)} | {rounds} |"
                )

        lines.append("")

    # Summary statistics
    if benchmarks:
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Benchmarks**: {len(benchmarks)}")

        if baseline_data:
            faster = 0
            slower = 0
            similar = 0

            for b in benchmarks:
                name = b["name"]
                if name in baseline_lookup:
                    current_mean = b["stats"]["mean"]
                    baseline_mean = baseline_lookup[name]["stats"]["mean"]
                    if baseline_mean > 0:
                        change = (current_mean - baseline_mean) / baseline_mean
                        if change < -0.05:
                            faster += 1
                        elif change > 0.05:
                            slower += 1
                        else:
                            similar += 1

            lines.append(f"- **Faster**: {faster}")
            lines.append(f"- **Slower**: {slower}")
            lines.append(f"- **Similar**: {similar}")

    # Regressions alert
    if baseline_data:
        regressions = []
        for b in benchmarks:
            name = b["name"]
            if name in baseline_lookup:
                current_mean = b["stats"]["mean"]
                baseline_mean = baseline_lookup[name]["stats"]["mean"]
                if baseline_mean > 0:
                    change = (current_mean - baseline_mean) / baseline_mean
                    if change > 0.15:  # 15% regression threshold
                        regressions.append({
                            "name": name,
                            "current": current_mean,
                            "baseline": baseline_mean,
                            "change": change * 100
                        })

        if regressions:
            lines.append("")
            lines.append("## \u26a0\ufe0f Regressions Detected")
            lines.append("")
            lines.append("The following benchmarks show significant regressions (>15%):")
            lines.append("")
            for r in sorted(regressions, key=lambda x: -x["change"]):
                lines.append(
                    f"- **{r['name']}**: {r['change']:.1f}% slower "
                    f"({format_time(r['baseline'])} \u2192 {format_time(r['current'])})"
                )

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by TorchBridge benchmark suite*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown benchmark reports"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input benchmark JSON file"
    )
    parser.add_argument(
        "--baseline", "-b",
        type=Path,
        default=None,
        help="Baseline benchmark JSON file for comparison"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output markdown file (stdout if not specified)"
    )
    parser.add_argument(
        "--title", "-t",
        default="Benchmark Results",
        help="Report title"
    )

    args = parser.parse_args()

    # Load data
    current_data = load_benchmark_data(args.input)
    if not current_data:
        print(f"Error: Could not load benchmark data from {args.input}", file=sys.stderr)
        sys.exit(1)

    baseline_data = None
    if args.baseline:
        baseline_data = load_benchmark_data(args.baseline)
        if not baseline_data:
            print(f"Warning: Could not load baseline data from {args.baseline}", file=sys.stderr)

    # Generate report
    report = generate_report(current_data, baseline_data, args.title)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
