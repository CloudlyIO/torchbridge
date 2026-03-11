"""
Unit tests for torchbridge.testing.trace_validator.

Tests cover MultiStepTracer, TraceStepResult, TraceValidationResult, and helpers.
All tests run on CPU — no GPU required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchbridge.testing.trace_validator import (
    MultiStepTracer,
    TraceStepResult,
    TraceValidationResult,
    _extract_tensor,
    _greedy_token,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

class _IdentityModel(nn.Module):
    """Returns input unchanged — zero divergence between any two instances."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NoisyModel(nn.Module):
    """Adds a small fixed offset — creates controlled non-zero divergence."""

    def __init__(self, offset: float = 1e-4) -> None:
        super().__init__()
        self._offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._offset


def _make_tracer(model: nn.Module, *, dtype: str = "float32") -> MultiStepTracer:
    cpu = torch.device("cpu")
    return MultiStepTracer(
        model=model,
        device_a=cpu,
        device_b=cpu,
        backend_a="cpu",
        backend_b="cpu",
        dtype=dtype,
    )


# ── TraceStepResult / TraceValidationResult dataclass tests ──────────────────

class TestTraceResultDataclasses:
    def test_step_result_fields_exist(self):
        sr = TraceStepResult(
            step=1,
            max_diff=0.0,
            cosine_sim=1.0,
            within_tolerance=True,
            cumulative_amplification=1.0,
        )
        assert sr.step == 1
        assert sr.max_diff == 0.0
        assert sr.cosine_sim == 1.0
        assert sr.within_tolerance is True
        assert sr.cumulative_amplification == 1.0

    def test_validation_result_fields_exist(self):
        vr = TraceValidationResult(
            backend_a="cpu",
            backend_b="cpu",
            steps=5,
            dtype="float32",
            autoregressive=False,
        )
        assert vr.backend_a == "cpu"
        assert vr.backend_b == "cpu"
        assert vr.steps == 5
        assert vr.dtype == "float32"
        assert vr.autoregressive is False
        assert vr.step_results == []
        assert vr.first_divergence_step is None
        assert vr.max_amplification == 1.0
        assert vr.final_passed is True

    def test_to_dict_contains_all_required_keys(self):
        vr = TraceValidationResult(
            backend_a="cuda", backend_b="rocm", steps=3, dtype="float16",
            autoregressive=True,
        )
        d = vr.to_dict()
        assert "backend_a" in d
        assert "backend_b" in d
        assert "steps" in d
        assert "dtype" in d
        assert "autoregressive" in d
        assert "first_divergence_step" in d
        assert "max_amplification" in d
        assert "final_passed" in d
        assert "step_results" in d
        assert isinstance(d["step_results"], list)

    def test_to_dict_step_results_have_all_keys(self):
        sr = TraceStepResult(
            step=1, max_diff=1e-6, cosine_sim=0.9999,
            within_tolerance=True, cumulative_amplification=1.0,
        )
        vr = TraceValidationResult(
            backend_a="cpu", backend_b="cpu", steps=1, dtype="float32",
            autoregressive=False,
        )
        vr.step_results.append(sr)
        d = vr.to_dict()
        assert len(d["step_results"]) == 1
        row = d["step_results"][0]
        for key in ("step", "max_diff", "cosine_sim", "within_tolerance", "cumulative_amplification"):
            assert key in row


# ── Validation: steps boundary ────────────────────────────────────────────────

class TestStepsValidation:
    def test_steps_zero_raises(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 4)
        with pytest.raises(ValueError, match="steps must be >= 1"):
            tracer.run(x, steps=0)

    def test_steps_negative_raises(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 4)
        with pytest.raises(ValueError, match="steps must be >= 1"):
            tracer.run(x, steps=-5)

    def test_steps_one_is_valid(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 4)
        result = tracer.run(x, steps=1)
        assert len(result.step_results) == 1

    def test_empty_input_ids_raises(self):
        tracer = _make_tracer(_IdentityModel())
        empty = torch.zeros(0, 4)
        with pytest.raises(ValueError, match="non-empty"):
            tracer.run(empty, steps=1)


# ── CPU vs CPU — identical outputs ───────────────────────────────────────────

class TestCpuCpuIdentical:
    def test_zero_divergence_all_steps(self):
        """Identical model on both devices → max_diff = 0 every step."""
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=5)
        for sr in result.step_results:
            assert sr.max_diff == 0.0

    def test_all_steps_pass_identical_backends(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=5)
        assert result.final_passed is True

    def test_amplification_all_ones_when_zero_diff(self):
        """When step1_max_diff == 0, amplification stays 1.0 for all steps."""
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=5)
        for sr in result.step_results:
            assert sr.cumulative_amplification == pytest.approx(1.0)

    def test_step_count_matches_requested(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        for n in (1, 3, 10):
            result = tracer.run(x, steps=n)
            assert len(result.step_results) == n

    def test_steps_are_one_indexed(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 4)
        result = tracer.run(x, steps=3)
        assert [sr.step for sr in result.step_results] == [1, 2, 3]


# ── Amplification logic ───────────────────────────────────────────────────────

class TestAmplification:
    def test_amplification_starts_at_one(self):
        """Step 1 cumulative_amplification must always be 1.0."""
        tracer = _make_tracer(_NoisyModel(offset=1e-4))
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=3)
        assert result.step_results[0].cumulative_amplification == pytest.approx(1.0)

    def test_amplification_equals_one_for_constant_offset(self):
        """Constant offset every step → max_diff[k] == max_diff[1] → amplif == 1.0."""
        # NoisyModel adds same offset each step, so amplification stays 1.0
        tracer = _make_tracer(_NoisyModel(offset=1e-4))
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=5)
        for sr in result.step_results:
            assert sr.cumulative_amplification == pytest.approx(1.0, rel=1e-5)

    def test_max_amplification_tracked(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=5)
        expected = max(sr.cumulative_amplification for sr in result.step_results)
        assert result.max_amplification == pytest.approx(expected)

    def test_max_amplification_is_max_across_all_steps(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=8)
        all_amplifs = [sr.cumulative_amplification for sr in result.step_results]
        assert result.max_amplification == pytest.approx(max(all_amplifs))


# ── Pass/fail and first_divergence_step ──────────────────────────────────────

class TestPassFail:
    def test_final_passed_true_if_all_steps_pass(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=5)
        assert all(sr.within_tolerance for sr in result.step_results)
        assert result.final_passed is True

    def test_final_passed_false_if_any_step_fails(self):
        """Force failures by setting atol=-1.0: max_diff >= 0 > -1.0, so always fails."""
        from torchbridge.testing.tolerance_db import ToleranceDB, TolerancePair

        class _AlwaysFailDB(ToleranceDB):
            def get(self, backend, dtype):
                # atol=-1.0 → within_tol = max_diff <= -1.0, always False
                return TolerancePair(atol=-1.0, rtol=0.0)

        cpu = torch.device("cpu")
        tracer = MultiStepTracer(
            model=_IdentityModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a="cpu",
            backend_b="cpu",
            dtype="float32",
            tolerance_db=_AlwaysFailDB(),
        )
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=3)
        assert result.final_passed is False
        assert all(not sr.within_tolerance for sr in result.step_results)
        assert result.first_divergence_step == 1

    def test_final_passed_false_when_no_steps_ran(self):
        """Vacuous truth guard: 0 step_results must yield final_passed=False, not True.

        Regression guard for: all([], ...) = True in Python.
        If inference fails on step 1 and the loop breaks with empty step_results,
        the caller must see final_passed=False, not a silent pass.
        """
        # Use a model that raises on forward to force break on step 1
        class _CrashModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("simulated inference failure")

        tracer = _make_tracer(_CrashModel())
        x = torch.randn(1, 4)
        result = tracer.run(x, steps=3)
        assert len(result.step_results) == 0
        assert result.final_passed is False

    def test_first_divergence_step_none_when_all_pass(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 4)
        result = tracer.run(x, steps=5)
        assert result.first_divergence_step is None

    def test_tolerance_db_used_for_pass_fail(self):
        """ToleranceDB is queried and its atol is used for within_tolerance."""
        from torchbridge.testing.tolerance_db import ToleranceDB, TolerancePair

        class _ZeroAtolDB(ToleranceDB):
            def get(self, backend, dtype):
                # atol=0 means even 0.0 diff is within tolerance (0 <= 0 is True)
                return TolerancePair(atol=1e10, rtol=0.0)

        cpu = torch.device("cpu")
        tracer = MultiStepTracer(
            model=_IdentityModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a="cpu",
            backend_b="cpu",
            dtype="float32",
            tolerance_db=_ZeroAtolDB(),
        )
        x = torch.randn(1, 4)
        result = tracer.run(x, steps=3)
        # With huge atol, all steps pass regardless
        assert result.final_passed is True
        for sr in result.step_results:
            assert sr.within_tolerance is True


# ── Non-autoregressive: same input each step ──────────────────────────────────

class TestNonAutoregressive:
    def test_non_ar_step_count(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=4, autoregressive=False)
        assert len(result.step_results) == 4

    def test_non_ar_result_is_not_autoregressive(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=4, autoregressive=False)
        assert result.autoregressive is False

    def test_non_ar_same_input_does_not_grow_sequence(self):
        """Non-autoregressive mode should not extend input between steps."""
        calls = []

        class _TrackingModel(nn.Module):
            def forward(self, x):
                calls.append(x.shape)
                return x

        tracer = _make_tracer(_TrackingModel())
        x = torch.randn(1, 4)
        tracer.run(x, steps=3, autoregressive=False)
        # Shape should be constant across all calls
        assert all(s == calls[0] for s in calls)


# ── Autoregressive mode ───────────────────────────────────────────────────────

class TestAutoregressive:
    def test_ar_mode_flag_in_result(self):
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=3, autoregressive=True)
        assert result.autoregressive is True

    def test_ar_non_lm_does_not_crash(self):
        """autoregressive=True on a non-LM model should run without error
        (no greedy token appended since is_lm=False)."""
        tracer = _make_tracer(_IdentityModel())
        x = torch.randn(1, 8)
        result = tracer.run(x, steps=3, autoregressive=True)
        assert len(result.step_results) == 3

    def test_ar_lm_appends_token_and_grows_sequence(self):
        """autoregressive=True with is_lm=True should grow input by 1 token/step."""
        vocab_size = 16
        seq_len = 4

        class _FakeLM(nn.Module):
            """Returns logits of shape (batch, seq, vocab)."""
            def forward(self, input_ids):
                batch, seq = input_ids.shape
                return type("Out", (), {"logits": torch.zeros(batch, seq, vocab_size)})()

        cpu = torch.device("cpu")
        tracer = MultiStepTracer(
            model=_FakeLM(),
            device_a=cpu,
            device_b=cpu,
            backend_a="cpu",
            backend_b="cpu",
            dtype="float32",
            is_lm=True,
        )
        # batch=2 — verifies unsqueeze(-1) not unsqueeze(0)
        x = torch.zeros(2, seq_len, dtype=torch.long)
        result = tracer.run(x, steps=3, autoregressive=True)
        assert len(result.step_results) == 3


# ── Model copy isolation ──────────────────────────────────────────────────────

class TestNaNGuard:
    def test_zero_output_cosine_sim_is_not_nan(self):
        """Regression guard: cosine_sim must be JSON-serialisable (not NaN).

        F.cosine_similarity(zero_vec, zero_vec) = NaN (0/0).
        json.dumps({'x': float('nan')}) raises ValueError.
        Zero-output models (e.g. identity with zero input) must not crash CI JSON.
        """
        import json

        tracer = _make_tracer(_IdentityModel())
        x = torch.zeros(1, 8)  # all-zero input → all-zero output → cos_sim = NaN
        result = tracer.run(x, steps=1)
        # Must not raise
        as_dict = result.to_dict()
        serialised = json.dumps(as_dict)  # must not raise ValueError
        assert '"cosine_sim"' in serialised
        # cos_sim should be 0.0 (the NaN replacement)
        assert result.step_results[0].cosine_sim == 0.0


class TestModelCopyIsolation:
    def test_original_model_not_mutated_after_run(self):
        """MultiStepTracer must not move the caller's model to a different device.
        (Regression guard for deepcopy fix — nn.Module.to() mutates in-place.)"""
        model = nn.Linear(4, 4)  # has parameters we can inspect
        assert next(model.parameters()).device == torch.device("cpu")
        tracer = _make_tracer(model)
        tracer.run(torch.randn(1, 4), steps=3)
        # Caller's model must still be on CPU
        assert next(model.parameters()).device == torch.device("cpu")


# ── _extract_tensor helper ────────────────────────────────────────────────────

class TestExtractTensor:
    def test_tensor_passthrough(self):
        t = torch.randn(2, 4)
        assert _extract_tensor(t, is_lm=False) is t

    def test_tuple_first_element(self):
        a = torch.randn(2, 4)
        b = torch.randn(2, 4)
        result = _extract_tensor((a, b), is_lm=False)
        assert result is a

    def test_list_first_element(self):
        a = torch.randn(2, 4)
        result = _extract_tensor([a], is_lm=False)
        assert result is a

    def test_lm_mode_uses_last_position(self):
        """is_lm=True should extract logits[:, -1, :] from output.logits."""
        class _FakeOutput:
            logits = torch.randn(1, 5, 32)  # (batch, seq, vocab)

        result = _extract_tensor(_FakeOutput(), is_lm=True)
        assert result.shape == (1, 32)

    def test_lm_mode_1d_logits(self):
        """is_lm=True with 1-D logits tensor should return it unchanged."""
        class _FakeOutput:
            logits = torch.randn(32)

        result = _extract_tensor(_FakeOutput(), is_lm=True)
        assert result.shape == (32,)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _extract_tensor("not_a_tensor", is_lm=False)


# ── _greedy_token helper ──────────────────────────────────────────────────────

class TestGreedyToken:
    def test_returns_none_for_2d_logits_does_not_crash(self):
        """Regression guard: _greedy_token must handle 2D logits without IndexError.

        logits[:, -1, :] on a (batch, vocab) tensor raises IndexError.
        2D logits occur for single-token LM outputs (no sequence dimension).
        """
        vocab_size = 8
        logits = torch.zeros(2, vocab_size)  # (batch=2, vocab) — no seq dim
        logits[0, 3] = 10.0  # batch 0 → token 3
        logits[1, 7] = 10.0  # batch 1 → token 7

        class _FakeOutput:
            pass

        obj = _FakeOutput()
        obj.logits = logits
        token = _greedy_token(obj)
        assert token is not None
        assert token.shape == (2,)
        assert token[0].item() == 3
        assert token[1].item() == 7

    def test_returns_none_for_no_logits(self):
        class _NoLogits:
            pass
        assert _greedy_token(_NoLogits()) is None

    def test_returns_none_for_non_tensor_logits(self):
        class _BadLogits:
            logits = "not_a_tensor"
        assert _greedy_token(_BadLogits()) is None

    def test_returns_argmax_of_last_position(self):
        """Should return argmax over vocab at last position."""
        vocab_size = 16
        logits = torch.zeros(1, 3, vocab_size)
        logits[0, -1, 5] = 10.0  # token 5 wins at last position
        class _FakeOutput:
            pass
        obj = _FakeOutput()
        obj.logits = logits
        token = _greedy_token(obj)
        assert token is not None
        assert token.shape == (1,)
        assert token.item() == 5

    def test_batch_size_preserved(self):
        vocab_size = 8
        batch_size = 3
        logits = torch.randn(batch_size, 2, vocab_size)
        class _FakeOutput:
            pass
        obj = _FakeOutput()
        obj.logits = logits
        token = _greedy_token(obj)
        assert token is not None
        assert token.shape == (batch_size,)
