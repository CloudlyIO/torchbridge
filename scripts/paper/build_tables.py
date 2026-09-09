#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn trace result files into the paper's tables.

Reads the JSON produced by ``tb-validate --compare A B --trace`` and emits a
Markdown table. Two things drive the design.

**Every field is optional.** Results written before provenance was recorded have
no ``atol``, ``atol_source``, ``input`` or ``env_*``. The four committed A10G
runs are exactly that shape and are the only real hardware measurements we have,
so dropping them was never an option. Missing values render as
``not recorded`` rather than being hidden — a reader has to be able to see which
rows cannot be re-checked.

**The peak step is derived.** The paper reports amplification as
"peak 5.75x @ step 12", and the step is not stored. It comes from
``step_results``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

NOT_RECORDED = "not recorded"

_REQUIRED_TRACE_KEYS = ("backend_a", "backend_b", "dtype", "max_amplification")


def peak_amplification_step(result: dict[str, Any]) -> int | None:
    """Step at which cumulative amplification peaked, or None if unknowable."""
    steps = result.get("step_results") or []
    if not steps:
        return None
    peak = max(steps, key=lambda s: s.get("cumulative_amplification", 0.0))
    return peak.get("step")


def _describe_input(value: Any) -> str:
    """One-line summary of the recorded input, or ``not recorded``."""
    if not isinstance(value, dict) or not value:
        return NOT_RECORDED
    shape = "x".join(str(n) for n in value.get("shape", [])) or "?"
    if value.get("synthetic"):
        return f"{shape} synthetic ({value.get('unique_values', '?')} unique)"
    return f"{shape} ({value.get('unique_values', '?')} unique)"


def load_result(path: str | Path) -> dict[str, Any] | None:
    """Read one result file into a flat row, or None if it is not a trace."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not all(k in data for k in _REQUIRED_TRACE_KEYS):
        return None  # a validation report or something else, not a trace

    env_a = data.get("env_a") or {}
    env_b = data.get("env_b") or {}
    return {
        "file": path.name,
        "model": data.get("model") or NOT_RECORDED,
        "model_family": data.get("model_family") or NOT_RECORDED,
        "backend_a": data["backend_a"],
        "backend_b": data["backend_b"],
        "dtype": data["dtype"],
        "steps": data.get("steps"),
        "autoregressive": bool(data.get("autoregressive")),
        "first_divergence_step": data.get("first_divergence_step"),
        "max_amplification": data.get("max_amplification"),
        "peak_step": peak_amplification_step(data),
        "final_passed": data.get("final_passed"),
        "atol": data.get("atol"),
        "atol_source": data.get("atol_source") or NOT_RECORDED,
        "tolerance_rule": data.get("tolerance_rule") or NOT_RECORDED,
        "input": _describe_input(data.get("input")),
        "device_a": env_a.get("device") or env_a.get("device_type") or NOT_RECORDED,
        "device_b": env_b.get("device") or env_b.get("device_type") or NOT_RECORDED,
        "torch": env_a.get("torch") or NOT_RECORDED,
    }


def collect_rows(directory: str | Path) -> list[dict[str, Any]]:
    """Load every trace result in ``directory``, sorted for a stable table."""
    rows = [
        row
        for path in sorted(Path(directory).glob("*.json"))
        if (row := load_result(path)) is not None
    ]
    rows.sort(key=lambda r: (r["backend_a"], r["backend_b"], r["dtype"]))
    return rows


def _amplification_cell(row: dict[str, Any]) -> str:
    amp = row["max_amplification"]
    if amp is None:
        return NOT_RECORDED
    step = row["peak_step"]
    return f"{amp:.2f}x" + (f" @ step {step}" if step is not None else "")


def _tolerance_cell(row: dict[str, Any]) -> str:
    if row["atol"] is None:
        return NOT_RECORDED
    return f"{row['atol']:.1e} ({row['atol_source']})"


def _verdict_cell(row: dict[str, Any]) -> str:
    """PASS, FAIL, or ``not recorded`` when the file never stored a verdict.

    Truthiness would turn a missing value into FAIL, which reports a run that
    may well have passed as a failure. Every field here is optional by design,
    so absence has to read as absence.
    """
    passed = row["final_passed"]
    if passed is None:
        return NOT_RECORDED
    return "PASS" if passed else "FAIL"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the paper table. Rows lacking provenance say so, in the table."""
    if not rows:
        return "_No trace results found._\n"

    header = (
        "| Model | Comparison | dtype | Steps | First divergence | "
        "Peak amplification | Tolerance applied | Input | Result |"
    )
    lines = [header, "|" + "---|" * 9]
    for r in rows:
        first = r["first_divergence_step"]
        lines.append(
            "| {model} | {a} vs {b} | {dtype} | {steps} | {first} | {amp} | "
            "{tol} | {inp} | {verdict} |".format(
                model=r["model"].split("/")[-1],
                a=r["backend_a"],
                b=r["backend_b"],
                dtype=r["dtype"],
                steps=r["steps"] if r["steps"] is not None else "?",
                first="none" if first is None else f"step {first}",
                amp=_amplification_cell(r),
                tol=_tolerance_cell(r),
                inp=r["input"],
                verdict=_verdict_cell(r),
            )
        )

    unrecorded = [r for r in rows if r["atol"] is None]
    if unrecorded:
        lines += [
            "",
            f"**{len(unrecorded)} of {len(rows)} rows predate tolerance recording.** "
            "Their verdict cannot be re-checked from the file: the limit that "
            "produced it was not saved. Re-run them once the fixes are released.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default="results/paper1",
        help="directory of trace result JSON files (default: results/paper1)",
    )
    parser.add_argument("-o", "--output", help="write the table here instead of stdout")
    args = parser.parse_args(argv)

    rows = collect_rows(args.directory)
    table = render_markdown(rows)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(table)
        print(f"Wrote {len(rows)} row(s) to {args.output}")
    else:
        sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
