"""
Unit tests for scripts/paper/build_tables.py.

Turns trace result files into the paper's tables. Two constraints shape it.

Files written before the provenance work lack ``model_family``, ``atol``,
``atol_source``, ``input`` and the ``env_*`` blocks, so every field has to be
optional — the four committed A10G results are exactly that shape, and silently
dropping them would lose the only real measurements we have.

And the step at which amplification peaks is not a stored field. The paper
reports it ("peak DAF 5.75x @ step 12"), so it is derived from ``step_results``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.paper.build_tables import (
    collect_rows,
    load_result,
    peak_amplification_step,
    render_markdown,
)


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def _trace(**over):
    d = {
        "backend_a": "cuda",
        "backend_b": "cpu",
        "steps": 3,
        "dtype": "float32",
        "autoregressive": True,
        "first_divergence_step": 2,
        "max_amplification": 5.75,
        "final_passed": False,
        "model": "Qwen/Qwen3-0.6B",
        "step_results": [
            {
                "step": 1,
                "max_diff": 1e-5,
                "cosine_sim": 1.0,
                "within_tolerance": True,
                "cumulative_amplification": 1.0,
            },
            {
                "step": 2,
                "max_diff": 5e-5,
                "cosine_sim": 1.0,
                "within_tolerance": False,
                "cumulative_amplification": 5.75,
            },
            {
                "step": 3,
                "max_diff": 3e-5,
                "cosine_sim": 1.0,
                "within_tolerance": True,
                "cumulative_amplification": 3.0,
            },
        ],
    }
    d.update(over)
    return d


class TestPeakStep:
    def test_peak_step_is_derived_from_step_results(self):
        assert peak_amplification_step(_trace()) == 2

    def test_no_step_results_gives_none(self):
        assert peak_amplification_step({"step_results": []}) is None

    def test_missing_key_gives_none(self):
        assert peak_amplification_step({}) is None


class TestOldFilesStillLoad:
    """The four committed A10G results predate the provenance fields."""

    def test_an_old_shaped_file_loads(self, tmp_path):
        p = _write(tmp_path, "old.json", _trace())
        row = load_result(p)
        assert row["dtype"] == "float32"
        assert row["max_amplification"] == 5.75

    def test_missing_provenance_is_marked_not_recorded(self, tmp_path):
        p = _write(tmp_path, "old.json", _trace())
        row = load_result(p)
        assert row["atol"] is None
        assert row["atol_source"] == "not recorded"
        assert row["input"] == "not recorded"

    def test_new_provenance_is_used_when_present(self, tmp_path):
        p = _write(
            tmp_path,
            "new.json",
            _trace(
                atol=4e-4,
                atol_source="derived",
                tolerance_rule="atol_only",
                model_family="decoder-medium",
                input={"shape": [1, 64], "synthetic": True, "unique_values": 1},
                env_a={"device": "NVIDIA A10G", "torch": "2.11.0"},
            ),
        )
        row = load_result(p)
        assert row["atol"] == 4e-4
        assert row["atol_source"] == "derived"
        assert row["device_a"] == "NVIDIA A10G"
        assert "synthetic" in row["input"]


class TestSkippingNonTraceFiles:
    def test_a_non_trace_json_is_skipped(self, tmp_path):
        _write(tmp_path, "other.json", {"summary": "not a trace"})
        assert collect_rows(tmp_path) == []

    def test_unreadable_file_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not json")
        _write(tmp_path, "good.json", _trace())
        assert len(collect_rows(tmp_path)) == 1


class TestOrdering:
    def test_rows_are_sorted_for_a_stable_table(self, tmp_path):
        _write(tmp_path, "b.json", _trace(dtype="float16"))
        _write(tmp_path, "a.json", _trace(dtype="bfloat16"))
        rows = collect_rows(tmp_path)
        assert [r["dtype"] for r in rows] == ["bfloat16", "float16"]


class TestRendering:
    def test_markdown_has_a_header_and_one_row_per_result(self, tmp_path):
        _write(tmp_path, "a.json", _trace())
        out = render_markdown(collect_rows(tmp_path))
        table_lines = [ln for ln in out.split("\n") if ln.startswith("|")]
        assert len(table_lines) == 3, table_lines  # header, separator, one row
        assert "5.75" in out

    def test_peak_step_appears(self, tmp_path):
        _write(tmp_path, "a.json", _trace())
        assert "@ step 2" in render_markdown(collect_rows(tmp_path))

    def test_unrecorded_provenance_is_visible_in_the_table(self, tmp_path):
        """A reader must see which rows cannot be re-checked."""
        _write(tmp_path, "a.json", _trace())
        assert "not recorded" in render_markdown(collect_rows(tmp_path))

    def test_empty_input_renders_without_crashing(self):
        assert isinstance(render_markdown([]), str)


RESULTS_DIR = Path("results/paper1")


# Keyed on the A10G files themselves, not on the directory. The directory can
# exist and hold unrelated runs — a local control sweep, say — and then a bare
# is_dir() check turns "these results are elsewhere" into a failure.
_A10G_RESULTS = sorted(RESULTS_DIR.glob("*a10g*.json")) if RESULTS_DIR.is_dir() else []


@pytest.mark.skipif(
    not _A10G_RESULTS,
    reason="the A10G traces live on the paper-1 results branch, not on main",
)
class TestRealCommittedFiles:
    """The script must work on the real result files, where those are present.

    The committed traces live on the paper-1 results branch rather than on main,
    so this class skips when the directory is absent instead of failing. It still
    runs — and still guards the numbers — on the branch that holds them.
    """

    def test_the_committed_results_produce_rows(self):
        rows = collect_rows("results/paper1")
        assert len(rows) >= 4, f"only found {len(rows)}"
        assert any(r["dtype"] == "bfloat16" for r in rows)

    def test_the_known_a10g_numbers_appear(self):
        rows = collect_rows("results/paper1")
        amps = {r["dtype"]: r["max_amplification"] for r in rows if "a10g" in r["file"]}
        assert amps.get("bfloat16") == pytest.approx(10.75, abs=0.2)
        assert amps.get("float16") == pytest.approx(5.75, abs=0.2)


class TestUnknownVerdictIsNotReportedAsFailure:
    """A missing verdict must not render as FAIL.

    The script exists to tolerate result files written before the provenance
    fields, so every field can be absent. Rendering a missing ``final_passed``
    through truthiness turns "we do not know" into "it failed", which misreports
    a run that may well have passed — the opposite of what a table tolerant of
    old shapes should do.
    """

    def test_missing_verdict_renders_as_not_recorded(self, tmp_path):
        p = _write(tmp_path, "old.json", _trace(final_passed=None))
        out = render_markdown([load_result(p)])
        assert "FAIL" not in out
        assert "not recorded" in out

    def test_a_real_failure_still_renders_as_fail(self, tmp_path):
        p = _write(tmp_path, "f.json", _trace(final_passed=False))
        assert "FAIL" in render_markdown([load_result(p)])

    def test_a_pass_still_renders_as_pass(self, tmp_path):
        p = _write(tmp_path, "p.json", _trace(final_passed=True))
        assert "PASS" in render_markdown([load_result(p)])

    def test_a_file_with_no_verdict_key_at_all(self, tmp_path):
        payload = _trace()
        del payload["final_passed"]
        p = _write(tmp_path, "none.json", payload)
        out = render_markdown([load_result(p)])
        assert "FAIL" not in out


class TestResultConstructorContractAcrossTheStack:
    """Every provenance field added by this stack sits after the released ones.

    ``TraceValidationResult`` is public and was constructible positionally with
    ``step_results`` sixth. Three commits here add fields; if any of them lands
    ahead of ``step_results``, a caller written against the last release binds
    its step list to the wrong name and gets a result that looks valid.
    """

    def test_released_positional_prefix_is_unchanged(self):
        import dataclasses

        from torchbridge.testing.trace_validator import TraceValidationResult

        released = [
            "backend_a",
            "backend_b",
            "steps",
            "dtype",
            "autoregressive",
            "step_results",
            "first_divergence_step",
            "max_amplification",
            "final_passed",
        ]
        names = [f.name for f in dataclasses.fields(TraceValidationResult)]
        assert names[: len(released)] == released

    def test_every_new_field_is_optional(self):
        """A positional caller supplies none of them, so each needs a default."""
        import dataclasses

        from torchbridge.testing.trace_validator import TraceValidationResult

        for f in dataclasses.fields(TraceValidationResult)[9:]:
            has_default = (
                f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING
            )
            assert has_default, f"{f.name} has no default"
