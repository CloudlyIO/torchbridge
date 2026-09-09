"""
Unit tests for torchbridge.testing.trace_validator.

Tests cover MultiStepTracer, TraceStepResult, TraceValidationResult, and helpers.
All tests run on CPU — no GPU required.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from torchbridge.testing.trace_validator import (
    RECORD_FORMAT_VERSION,
    MultiStepTracer,
    SplitTraceRecord,
    TraceStepResult,
    TraceValidationResult,
    _extract_tensor,
    _greedy_token,
    compare_records,
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
            backend_a="cuda",
            backend_b="rocm",
            steps=3,
            dtype="float16",
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
            step=1,
            max_diff=1e-6,
            cosine_sim=0.9999,
            within_tolerance=True,
            cumulative_amplification=1.0,
        )
        vr = TraceValidationResult(
            backend_a="cpu",
            backend_b="cpu",
            steps=1,
            dtype="float32",
            autoregressive=False,
        )
        vr.step_results.append(sr)
        d = vr.to_dict()
        assert len(d["step_results"]) == 1
        row = d["step_results"][0]
        for key in (
            "step",
            "max_diff",
            "cosine_sim",
            "within_tolerance",
            "cumulative_amplification",
        ):
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
            def get(self, backend, dtype, model_family=None):
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
            def get(self, backend, dtype, model_family=None):
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
                return type(
                    "Out", (), {"logits": torch.zeros(batch, seq, vocab_size)}
                )()

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


# ── Split trace: record / replay / compare ───────────────────────────────────


class _FakeLMOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _TinyLM(nn.Module):
    """Deterministic pseudo-LM over token ids, for autoregressive tests."""

    def __init__(self, vocab: int = 37, dim: int = 16, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor) -> _FakeLMOutput:
        return _FakeLMOutput(self.proj(self.emb(input_ids)))


def _perturbed(model: _TinyLM, offset: float = 3e-5) -> _TinyLM:
    """A copy with slightly shifted weights, standing in for a second backend."""
    clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.proj.weight += offset
    return clone


def _split_tracer(model: nn.Module) -> MultiStepTracer:
    cpu = torch.device("cpu")
    return MultiStepTracer(
        model=model,
        device_a=cpu,
        device_b=cpu,
        backend_a="cuda",
        backend_b="rocm",
        dtype="float32",
        is_lm=True,
    )


class TestSplitTraceEquivalence:
    """A split trace must reproduce a single-process run exactly, not approximately.

    This is the property that makes cross-vendor comparison possible at all:
    ``run`` derives the next token only from backend A, so backend B never
    influences the input trajectory and can be replayed on another machine.
    """

    def test_record_replay_matches_single_process_run(self):
        model = _TinyLM()
        tracer = _split_tracer(model)
        prompt = torch.randint(0, 37, (1, 6))

        single = tracer.run(prompt.clone(), steps=12, autoregressive=True)

        record = tracer.record(prompt.clone(), steps=12, autoregressive=True)
        split = compare_records(record, tracer.replay(record))

        assert split.steps == single.steps
        assert split.max_amplification == single.max_amplification
        assert split.first_divergence_step == single.first_divergence_step
        assert split.final_passed == single.final_passed
        for s_single, s_split in zip(single.step_results, split.step_results):
            assert s_split.max_diff == s_single.max_diff
            assert s_split.cumulative_amplification == s_single.cumulative_amplification
            assert s_split.within_tolerance == s_single.within_tolerance

    def test_matches_ground_truth_under_real_divergence(self):
        """With two genuinely different models, every metric must still match."""
        model_a = _TinyLM()
        model_b = _perturbed(model_a)
        prompt = torch.randint(0, 37, (1, 6))
        steps = 15

        # Hand-rolled two-model loop: the definition the split must reproduce.
        ma = copy.deepcopy(model_a).eval()
        mb = copy.deepcopy(model_b).eval()
        current = prompt.clone()
        expected_diffs = []
        expected_inputs = []
        for _ in range(steps):
            with torch.no_grad():
                raw_a = ma(input_ids=current)
                raw_b = mb(input_ids=current)
            expected_inputs.append(current.clone())
            expected_diffs.append(
                float((raw_a.logits[:, -1, :] - raw_b.logits[:, -1, :]).abs().max())
            )
            current = torch.cat([current, _greedy_token(raw_a).unsqueeze(-1)], dim=-1)

        record = _split_tracer(model_a).record(
            prompt.clone(), steps=steps, autoregressive=True
        )
        replayed = _split_tracer(model_b).replay(record)
        result = compare_records(record, replayed)

        assert result.step_results[0].max_diff > 0.0, "divergence must be non-zero"
        for expected, actual in zip(expected_inputs, record.token_inputs):
            assert torch.equal(expected, actual)
        for expected, step in zip(expected_diffs, result.step_results):
            assert step.max_diff == expected

    def test_backend_names_come_from_the_two_halves(self):
        model = _TinyLM()
        tracer = _split_tracer(model)
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=3)
        result = compare_records(record, tracer.replay(record))
        assert result.backend_a == "cuda"
        assert result.backend_b == "rocm"


class TestSplitTraceRoundTrip:
    def test_save_load_preserves_record(self, tmp_path):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(
            torch.randint(0, 37, (1, 5)), steps=6, autoregressive=True
        )

        path = str(tmp_path / "record.pt")
        record.save(path)
        loaded = SplitTraceRecord.load(path)

        assert loaded.backend == record.backend
        assert loaded.dtype == record.dtype
        assert loaded.autoregressive == record.autoregressive
        assert loaded.is_lm == record.is_lm
        assert loaded.env == record.env
        assert loaded.steps == record.steps
        for before, after in zip(record.token_inputs, loaded.token_inputs):
            assert torch.equal(before, after)
        for before, after in zip(record.outputs, loaded.outputs):
            assert torch.equal(before, after)

    def test_load_rejects_a_future_format_version(self, tmp_path):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=2)
        record.format_version = RECORD_FORMAT_VERSION + 1

        path = str(tmp_path / "future.pt")
        record.save(path)
        with pytest.raises(ValueError, match="format"):
            SplitTraceRecord.load(path)

    def test_steps_property_counts_completed_outputs(self):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=7)
        assert record.steps == 7 == len(record.outputs)


class TestSplitTraceGuards:
    def test_rejects_halves_fed_different_inputs(self):
        """A replay that did not follow the record is not a valid comparison."""
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(
            torch.randint(0, 37, (1, 5)), steps=6, autoregressive=True
        )
        replayed = tracer.replay(record)
        replayed.token_inputs[2] = replayed.token_inputs[2] + 1

        with pytest.raises(ValueError, match="different inputs"):
            compare_records(record, replayed)

    def test_rejects_the_same_record_used_as_both_halves(self):
        """Passing one record twice is always meaningless, so it must raise.

        Distinct from two records that merely share a backend name: that pair is
        the split path's control run and only warns. What cannot be allowed is a
        follower that was never replayed, because then nothing was compared.
        """
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=3)
        with pytest.raises(ValueError, match="replay"):
            compare_records(record, record)

    def test_rejects_a_follower_that_is_not_a_replay(self):
        tracer = _split_tracer(_TinyLM())
        a = tracer.record(torch.randint(0, 37, (1, 4)), steps=3)
        b = tracer.record(torch.randint(0, 37, (1, 4)), steps=3)
        b.backend = "rocm"
        with pytest.raises(ValueError, match="replay"):
            compare_records(a, b)

    def test_rejects_empty_records(self):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=3)
        empty = SplitTraceRecord(
            backend="rocm", dtype="float32", autoregressive=False, is_lm=True
        )
        with pytest.raises(ValueError, match="at least one completed step"):
            compare_records(record, empty)

    def test_replay_rejects_an_empty_record(self):
        tracer = _split_tracer(_TinyLM())
        empty = SplitTraceRecord(
            backend="cuda", dtype="float32", autoregressive=False, is_lm=True
        )
        with pytest.raises(ValueError, match="no steps to replay"):
            tracer.replay(empty)

    def test_record_rejects_bad_steps_and_empty_input(self):
        tracer = _split_tracer(_TinyLM())
        with pytest.raises(ValueError, match="steps must be >= 1"):
            tracer.record(torch.randint(0, 37, (1, 4)), steps=0)
        with pytest.raises(ValueError, match="non-empty"):
            tracer.record(torch.empty(0, dtype=torch.long), steps=3)

    def test_env_mismatch_warns_by_default_but_still_compares(self, caplog):
        """A version mismatch must be visible, because it invalidates the result."""
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=4)
        replayed = tracer.replay(record)
        replayed.env = {**replayed.env, "torch": "0.0.0-fake"}

        with caplog.at_level("WARNING"):
            result = compare_records(record, replayed)
        assert result.steps == 4
        assert "different environments" in caplog.text

    def test_strict_env_raises_on_version_mismatch(self):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=4)
        replayed = tracer.replay(record)
        replayed.env = {**replayed.env, "torch": "0.0.0-fake"}

        with pytest.raises(ValueError, match="different environments"):
            compare_records(record, replayed, strict_env=True)

    def test_unequal_step_counts_compare_the_common_prefix(self):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(
            torch.randint(0, 37, (1, 5)), steps=8, autoregressive=True
        )
        replayed = tracer.replay(record)
        del replayed.token_inputs[5:]
        del replayed.outputs[5:]

        result = compare_records(record, replayed)
        assert result.steps == 5
        assert len(result.step_results) == 5

    def test_rejects_reversed_arguments(self):
        """Order matters: atol comes from record_a's backend, so a swap changes the verdict."""
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=3)
        replayed = tracer.replay(record)

        with pytest.raises(ValueError, match="reversed"):
            compare_records(replayed, record)


class TestSplitTraceModelFingerprint:
    """Both halves must have loaded the same weights.

    ``run`` deep-copies one model so this is structural, but a split trace calls
    ``from_pretrained`` twice on two machines. A checkpoint that drifts between
    them would masquerade as backend divergence.
    """

    def test_fingerprint_is_recorded(self):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=2)
        assert record.env["model_fingerprint"]
        assert record.env["model_fingerprint"] != "unavailable"

    def test_identical_weights_produce_identical_fingerprints(self):
        model = _TinyLM()
        tracer = _split_tracer(model)
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=2)
        replayed = tracer.replay(record)
        assert record.env["model_fingerprint"] == replayed.env["model_fingerprint"]

    def test_drifted_weights_warn_and_strict_env_raises(self, caplog):
        model_a = _TinyLM()
        model_b = _perturbed(model_a)
        prompt = torch.randint(0, 37, (1, 5))

        record = _split_tracer(model_a).record(prompt, steps=4, autoregressive=True)
        replayed = _split_tracer(model_b).replay(record)

        assert record.env["model_fingerprint"] != replayed.env["model_fingerprint"], (
            "a perturbed checkpoint must change the fingerprint"
        )

        # Warn-level, not fatal: a legitimate cross-backend run may load weights
        # that differ in the low bits, so the comparison must still be possible.
        with caplog.at_level("WARNING"):
            result = compare_records(record, replayed)
        assert result.steps == 4
        assert "model_fingerprint" in caplog.text

        with pytest.raises(ValueError, match="model_fingerprint"):
            compare_records(record, replayed, strict_env=True)

    def test_fingerprint_detects_a_structural_difference(self):
        """A different shape must change the fingerprint, not just different values."""
        from torchbridge.testing.trace_validator import _model_fingerprint

        assert _model_fingerprint(_TinyLM(vocab=37, dim=16)) != _model_fingerprint(
            _TinyLM(vocab=37, dim=24)
        )

    def test_fingerprint_is_stable_across_calls(self):
        from torchbridge.testing.trace_validator import _model_fingerprint

        model = _TinyLM()
        assert _model_fingerprint(model) == _model_fingerprint(model)


class TestReportedStepCount:
    def test_run_reports_steps_actually_measured(self):
        """An early break must not leave `steps` at the requested count."""

        class _FailsAfterFour(nn.Module):
            """Raises on its 5th forward call.

            The tracer deep-copies the model per device, so each copy keeps its
            own counter — 4 steps complete, then step 5 breaks the loop.
            """

            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                if self.calls > 4:
                    raise RuntimeError("simulated backend failure")
                return x

        tracer = _make_tracer(_FailsAfterFour())
        result = tracer.run(torch.randn(1, 4), steps=10)
        assert result.steps == len(result.step_results) == 4
        assert result.steps != 10, "must not report the requested count"


class TestModelFamilyReachesToleranceDB:
    """The 3D tolerance table is useless if the family never reaches the lookup.

    ToleranceDB stores atol under (model_family, backend, dtype) because a
    deeper model accumulates more error. When the family is dropped the trace
    silently falls back to the coarser (backend, dtype) row, which is too
    strict for a large model and turns an acceptable run into a failure.
    """

    class _RecordingDB:
        """Captures the arguments the tracer actually passes."""

        def __init__(self, atol=1.0):
            self.calls = []
            self._atol = atol

        def get(self, backend, dtype, model_family=None):
            self.calls.append((backend, dtype, model_family))
            from torchbridge.testing.tolerance_db import TolerancePair

            return TolerancePair(atol=self._atol, rtol=0.0)

    def _tracer(self, db, model_family):
        cpu = torch.device("cpu")
        return MultiStepTracer(
            model=_IdentityModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a="cuda",
            backend_b="cpu",
            dtype="float32",
            tolerance_db=db,
            model_family=model_family,
        )

    def test_family_is_forwarded_to_the_lookup(self):
        db = self._RecordingDB()
        self._tracer(db, "decoder-large").run(torch.randn(1, 8), steps=2)
        assert db.calls == [("cuda", "float32", "decoder-large")]

    def test_no_family_still_looks_up_without_one(self):
        db = self._RecordingDB()
        self._tracer(db, None).run(torch.randn(1, 8), steps=2)
        assert db.calls == [("cuda", "float32", None)]

    def test_family_changes_the_verdict(self):
        """A difference between the two atols must flip pass/fail."""
        from torchbridge.testing.tolerance_db import TolerancePair

        class _FamilyAwareDB:
            def get(self, backend, dtype, model_family=None):
                atol = 4.0e-4 if model_family == "decoder-large" else 1.0e-4
                return TolerancePair(atol=atol, rtol=0.0)

        class _DriftingModel(nn.Module):
            """Second copy returns the first copy's output shifted by 2.0e-4."""

            calls = 0

            def forward(self, x):
                out = x.clone()
                if _DriftingModel.calls % 2 == 1:
                    out = out + 2.0e-4
                _DriftingModel.calls += 1
                return out

        def verdict(family):
            _DriftingModel.calls = 0
            cpu = torch.device("cpu")
            tracer = MultiStepTracer(
                model=_DriftingModel(),
                device_a=cpu,
                device_b=cpu,
                backend_a="cuda",
                backend_b="cpu",
                dtype="float32",
                tolerance_db=_FamilyAwareDB(),
                model_family=family,
            )
            return tracer.run(torch.randn(1, 8), steps=5)

        strict = verdict(None)
        loose = verdict("decoder-large")
        assert strict.final_passed is False
        assert strict.first_divergence_step == 1
        assert loose.final_passed is True
        assert loose.first_divergence_step is None

    def test_family_is_always_passed_even_when_absent(self):
        """The third argument goes on every call, not only when a family is set.

        Branching on "was a family given" looks safer but is not: a database with
        a two-argument ``get`` still breaks the moment a family *is* supplied, so
        the branch buys nothing and leaves two code paths where one will do.
        ToleranceDB already declares ``model_family`` optional, so the contract
        for any substitute database is simply to accept it.
        """
        db = self._RecordingDB()
        self._tracer(db, None).run(torch.randn(1, 8), steps=1)
        backend, dtype, family = db.calls[0]
        assert (backend, dtype, family) == ("cuda", "float32", None)

    def test_database_without_a_family_parameter_fails_loudly(self):
        """A substitute DB that cannot take the family must raise, not fall back.

        Silently retrying without the family would reinstate the coarse-table
        bug this class exists to prevent.
        """
        from torchbridge.testing.tolerance_db import TolerancePair

        class _TwoArgDB:
            def get(self, backend, dtype):  # deliberately missing model_family
                return TolerancePair(atol=1.0, rtol=0.0)

        cpu = torch.device("cpu")
        tracer = MultiStepTracer(
            model=_IdentityModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a="cuda",
            backend_b="cpu",
            tolerance_db=_TwoArgDB(),
        )
        with pytest.raises(TypeError):
            tracer.run(torch.randn(1, 8), steps=2)


class TestCliForwardsModelFamilyToTrace:
    """--model-family is accepted in trace mode; it must not be dropped there."""

    def test_run_trace_passes_model_family_to_the_tracer(self):
        import argparse
        from unittest.mock import patch

        from torchbridge.cli.validate import ValidateCommand

        args = argparse.Namespace(
            compare=["cpu", "cpu"],
            trace=True,
            steps=2,
            autoregressive=False,
            model=None,
            input_shape="1,4",
            dtype="float32",
            output=None,
            trace_output=None,
            ci=False,
            verbose=False,
            model_family="decoder-large",
        )
        seen = {}
        real = MultiStepTracer

        def _spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        with patch("torchbridge.testing.trace_validator.MultiStepTracer", _spy):
            ValidateCommand._run_trace(args)

        assert seen.get("model_family") == "decoder-large"


class TestResultRecordsTheToleranceUsed:
    """A result file must say which tolerance decided its verdict.

    first_divergence_step and final_passed both depend on the atol that was
    applied, and the family selects it. Without the family in the JSON, a saved
    run cannot be re-checked or defended later.
    """

    def _run(self, family):
        cpu = torch.device("cpu")
        return MultiStepTracer(
            model=_IdentityModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a="cuda",
            backend_b="cpu",
            dtype="float32",
            model_family=family,
        ).run(torch.randn(1, 8), steps=2)

    def test_family_appears_in_the_dict(self):
        assert self._run("decoder-large").to_dict()["model_family"] == "decoder-large"

    def test_absent_family_is_recorded_as_null(self):
        assert self._run(None).to_dict()["model_family"] is None

    def test_atol_actually_applied_is_recorded(self):
        d = self._run("decoder-large").to_dict()
        assert d["atol"] == pytest.approx(4.0e-4)


class TestToleranceLookupUsesKeywordArgument:
    """The family must be passed by keyword, not position.

    A substitute database may legitimately declare it keyword-only —
    ``def get(self, backend, dtype, *, model_family=None)`` — and a positional
    call raises ``TypeError`` against that signature. The CLI already calls
    ``ToleranceDB.get`` with the keyword, so using it here costs nothing and
    removes an avoidable incompatibility. It does not reintroduce the silent
    fallback: the argument is still always sent.
    """

    def test_keyword_only_database_is_supported(self):
        from torchbridge.testing.tolerance_db import TolerancePair
        from torchbridge.testing.trace_validator import _lookup_tolerance

        class _KeywordOnlyDB:
            def get(self, backend, dtype, *, model_family=None):
                assert model_family == "decoder-large"
                return TolerancePair(atol=1.0e-4, rtol=0.0)

        entry = _lookup_tolerance(_KeywordOnlyDB(), "cuda", "float32", "decoder-large")
        assert entry.atol == pytest.approx(1.0e-4)

    def test_family_is_still_always_sent(self):
        """Using the keyword must not turn into "only send it sometimes"."""
        from torchbridge.testing.tolerance_db import TolerancePair
        from torchbridge.testing.trace_validator import _lookup_tolerance

        seen = []

        class _RecordingDB:
            def get(self, backend, dtype, model_family=None):
                seen.append(model_family)
                return TolerancePair(atol=1.0, rtol=0.0)

        _lookup_tolerance(_RecordingDB(), "cuda", "float32", None)
        _lookup_tolerance(_RecordingDB(), "cuda", "float32", "decoder-large")
        assert seen == [None, "decoder-large"]


class TestOfflineCompareRecordsProvenance:
    """The split path backs the cross-vendor claim, so it needs the same audit
    trail as the in-process path: which family, and which atol it selected."""

    def _records(self):
        from torchbridge.testing.trace_validator import SplitTraceRecord

        common = {"dtype": "float32", "autoregressive": False, "is_lm": False}
        a = SplitTraceRecord(backend="cuda", role="record", **common)
        b = SplitTraceRecord(backend="rocm", role="replay", **common)
        for rec in (a, b):
            rec.token_inputs = [torch.zeros(1, 4)]
            rec.env = {"torch": "2.0", "transformers": "4.0", "model_fingerprint": "x"}
        a.outputs = [torch.zeros(1, 4)]
        b.outputs = [torch.zeros(1, 4)]
        return a, b

    def test_family_is_recorded(self):
        from torchbridge.testing.trace_validator import compare_records

        a, b = self._records()
        d = compare_records(a, b, model_family="decoder-large").to_dict()
        assert d["model_family"] == "decoder-large"

    def test_atol_is_recorded(self):
        from torchbridge.testing.trace_validator import compare_records

        a, b = self._records()
        d = compare_records(a, b, model_family="decoder-large").to_dict()
        assert d["atol"] is not None and d["atol"] > 0


class TestSameBackendComparisonIsAllowedWithAWarning:
    """Comparing two records from the same backend is the split-path control run.

    It proves record-and-replay adds no error of its own: the same backend on
    both sides must agree exactly. Refusing it outright removes that check, so it
    warns instead — the warning still catches the real mistake of pairing two
    records from one machine by accident.
    """

    def _records(self, backend_a, backend_b):
        from torchbridge.testing.trace_validator import SplitTraceRecord

        common = {"dtype": "float32", "autoregressive": False, "is_lm": False}
        a = SplitTraceRecord(backend=backend_a, role="record", **common)
        b = SplitTraceRecord(backend=backend_b, role="replay", **common)
        for rec in (a, b):
            rec.token_inputs = [torch.zeros(1, 4)]
            rec.outputs = [torch.zeros(1, 4)]
            rec.env = {"torch": "2.0", "transformers": "4.0", "model_fingerprint": "x"}
        return a, b

    def test_same_backend_comparison_succeeds(self):
        from torchbridge.testing.trace_validator import compare_records

        result = compare_records(*self._records("cpu", "cpu"))
        assert result.final_passed is True
        assert result.first_divergence_step is None

    def test_same_backend_comparison_warns(self, caplog):
        import logging

        from torchbridge.testing.trace_validator import compare_records

        with caplog.at_level(logging.WARNING):
            compare_records(*self._records("cpu", "cpu"))
        assert any("same backend" in r.message.lower() for r in caplog.records)

    def test_different_backends_do_not_warn(self, caplog):
        import logging

        from torchbridge.testing.trace_validator import compare_records

        with caplog.at_level(logging.WARNING):
            compare_records(*self._records("cuda", "rocm"))
        assert not any("same backend" in r.message.lower() for r in caplog.records)


class TestSplitTraceRoleAndCompleteness:
    """Two guards that decide whether a comparison means anything.

    record_a supplies the backend name, the dtype and the tolerance for the
    whole result, so which half lands there is not a detail. And a comparison
    that stopped early measured something other than the trace requested.
    """

    @staticmethod
    def _halves(steps=3):
        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=steps)
        return record, tracer.replay(record)

    def test_two_replay_halves_are_refused(self):
        """Previously only the reversed pair was rejected, so a follower could
        become the primary side and supply the tolerance."""
        _, replayed = self._halves()
        with pytest.raises(ValueError, match="recorded half and one replayed"):
            compare_records(replayed, replayed)

    def test_two_recorded_halves_are_refused(self):
        record, _ = self._halves()
        with pytest.raises(ValueError, match="recorded half and one replayed"):
            compare_records(record, record)

    def test_the_reversed_pair_still_says_so(self):
        record, replayed = self._halves()
        with pytest.raises(ValueError, match="reversed"):
            compare_records(replayed, record)

    def test_a_correct_pair_still_compares(self):
        record, replayed = self._halves()
        assert compare_records(record, replayed).final_passed is True

    def test_a_truncated_replay_cannot_pass(self):
        """The surviving steps all agree, so every per-step number says PASS.
        The verdict must still be False: those are not the steps asked for."""
        record, replayed = self._halves(steps=5)
        del replayed.outputs[3:]
        del replayed.token_inputs[3:]

        result = compare_records(record, replayed)
        assert all(s.within_tolerance for s in result.step_results)
        assert result.steps == 3
        assert result.final_passed is False

    def test_a_shape_change_mid_trace_cannot_pass(self):
        record, replayed = self._halves(steps=4)
        replayed.outputs[2] = torch.zeros(1, 1)

        result = compare_records(record, replayed)
        assert result.final_passed is False

    def test_a_complete_comparison_still_passes(self):
        record, replayed = self._halves(steps=4)
        result = compare_records(record, replayed)
        assert result.steps == 4
        assert result.final_passed is True


class TestFingerprintStatesItsBlindSpot:
    """Endpoint sampling cannot see an interior edit. The warning has to say so,
    because a reader who takes fingerprint silence as proof of matching weights
    will attribute a checkpoint difference to the backend."""

    def test_an_interior_edit_is_genuinely_invisible(self):
        """Pins the limitation itself, so a later change to the sampling that
        closed this gap would fail here and prompt updating the warning."""
        from torchbridge.testing.trace_validator import _model_fingerprint

        model = torch.nn.Linear(256, 256, bias=False)
        before = _model_fingerprint(model)
        with torch.no_grad():
            model.weight[128, 128] += 1.0
        assert _model_fingerprint(model) == before

    def test_a_fingerprint_mismatch_warning_names_the_blind_spot(self, caplog):
        import logging

        from torchbridge.testing.trace_validator import compare_records

        tracer = _split_tracer(_TinyLM())
        record = tracer.record(torch.randint(0, 37, (1, 4)), steps=2)
        replayed = tracer.replay(record)
        replayed.env = dict(replayed.env, model_fingerprint="different")

        with caplog.at_level(logging.WARNING):
            compare_records(record, replayed)

        assert any("does not prove they match" in r.message for r in caplog.records), (
            "the warning must say that silence is not proof"
        )


class TestXlaIsNotSilentlyRunOnCpu:
    """An XLA device must never be quietly swapped for the CPU.

    TPU and Trainium both present as device type ``xla``. Substituting native
    CPU is numerically equivalent only when ``PJRT_DEVICE=CPU``, which is a
    CPU-backed XLA runtime. Doing it unconditionally means a real accelerator
    run reports CPU-vs-CPU zeros under the accelerator's name — a fabricated
    result that passes, with nothing in the output to reveal it.
    """

    def _tracer(self):
        return MultiStepTracer(
            model=_IdentityModel(),
            device_a=torch.device("xla"),
            device_b=torch.device("cpu"),
            backend_a="tpu",
            backend_b="cpu",
            dtype="float32",
        )

    def test_real_xla_device_refuses_instead_of_running_on_cpu(self, monkeypatch):
        monkeypatch.delenv("PJRT_DEVICE", raising=False)
        with pytest.raises(RuntimeError, match="record"):
            self._tracer().run(torch.randn(1, 8), steps=3)

    def test_tpu_pjrt_device_also_refuses(self, monkeypatch):
        monkeypatch.setenv("PJRT_DEVICE", "TPU")
        with pytest.raises(RuntimeError, match="record"):
            self._tracer().run(torch.randn(1, 8), steps=3)

    def test_cpu_backed_xla_still_falls_back(self, monkeypatch):
        """The one case where substitution is genuinely equivalent is preserved."""
        monkeypatch.setenv("PJRT_DEVICE", "CPU")
        result = self._tracer().run(torch.randn(1, 8), steps=3)
        assert result.final_passed is True
        assert len(result.step_results) == 3

    def test_lowercase_pjrt_value_is_accepted(self, monkeypatch):
        monkeypatch.setenv("PJRT_DEVICE", "cpu")
        assert self._tracer().run(torch.randn(1, 8), steps=2).final_passed is True

    def test_no_xla_device_is_unaffected(self, monkeypatch):
        """A plain cpu-vs-cpu run must not be touched by any of this."""
        monkeypatch.delenv("PJRT_DEVICE", raising=False)
        tracer = _make_tracer(_IdentityModel())
        assert tracer.run(torch.randn(1, 8), steps=3).final_passed is True

    def test_the_error_says_which_device_and_what_to_do(self, monkeypatch):
        monkeypatch.delenv("PJRT_DEVICE", raising=False)
        with pytest.raises(RuntimeError) as exc:
            self._tracer().run(torch.randn(1, 8), steps=2)
        message = str(exc.value)
        assert "PJRT_DEVICE" in message
        assert "--record" in message and "--replay" in message

    def test_record_does_not_refuse_on_a_real_xla_device(self, monkeypatch):
        """record() is the supported route, so it must stay open.

        It never substitutes, and each half stays on its own machine, so the
        XLA-to-CPU transfer problem that motivated the fallback never arises.
        """
        monkeypatch.delenv("PJRT_DEVICE", raising=False)
        tracer = MultiStepTracer(
            model=_IdentityModel(),
            device_a=torch.device("cpu"),  # stands in for the accelerator
            device_b=torch.device("cpu"),
            backend_a="tpu",
            backend_b="cpu",
            dtype="float32",
        )
        rec = tracer.record(torch.randn(1, 8), steps=3)
        assert rec.steps == 3 and rec.backend == "tpu"


class TestUnmeasuredToleranceIsSurfaced:
    """A verdict reached with an unmeasured fallback tolerance must say so.

    ``ToleranceDB`` answers an unknown backend with a safe default and marks the
    entry ``source="fallback"``. That is sensible for exploration and wrong for a
    paper: the run passes or fails against a number nobody measured, and today
    nothing in the result reveals which. The signal already exists on the entry;
    it simply was not being read.
    """

    def _run(self, backend_a):
        cpu = torch.device("cpu")
        return MultiStepTracer(
            model=_IdentityModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a=backend_a,
            backend_b="cpu",
            dtype="float32",
        ).run(torch.randn(1, 8), steps=2)

    def test_measured_backend_is_marked_measured(self):
        assert self._run("cuda").to_dict()["atol_source"] == "measured"

    def test_unmeasured_backend_is_marked_fallback(self):
        """A backend the table has never seen, so the default is all there is."""
        assert self._run("some-future-chip").to_dict()["atol_source"] == "fallback"

    def test_fallback_is_warned_about(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            self._run("some-future-chip")
        joined = " ".join(r.message.lower() for r in caplog.records)
        assert "fallback" in joined or "not measured" in joined

    def test_tpu_is_no_longer_a_fallback_after_the_alias_fix(self):
        """tpu resolves to the measured xla row rather than a safe default."""
        assert self._run("tpu").to_dict()["atol_source"] == "measured"


class TestBackendAliasesShareOneToleranceKey:
    """An alias must not change the tolerance for identical hardware.

    ``neuron`` and ``trainium`` name the same chip, and ``tpu`` is how the CLI
    exposes the XLA path. Today the table has ``trainium`` and ``xla`` but not
    ``neuron`` or ``tpu``, so the two names for one device get different limits —
    ``trainium`` a measured 1.0e-4, ``neuron`` an unmeasured 1.0e-3.
    """

    def _atol(self, backend):
        cpu = torch.device("cpu")
        result = MultiStepTracer(
            model=_IdentityModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a=backend,
            backend_b="cpu",
            dtype="float32",
        ).run(torch.randn(1, 8), steps=2)
        return result.to_dict()["atol"], result.to_dict()["atol_source"]

    def test_neuron_matches_trainium(self):
        assert self._atol("neuron") == self._atol("trainium")

    def test_tpu_matches_xla(self):
        assert self._atol("tpu") == self._atol("xla")

    def test_an_unrelated_backend_is_untouched(self):
        atol, source = self._atol("cuda")
        assert atol == pytest.approx(1.0e-4)
        assert source == "measured"


class TestNormalRunRecordsProvenance:
    """A normal trace result must say which hardware produced it.

    The split path already captures this per half. The in-process path recorded
    only the backend *name the caller typed* — which is precisely the field that
    lies when a name and the real device disagree, as in the cuda/rocm and XLA
    cases. Rented machines are destroyed after each run, so whatever the file
    omits is unrecoverable.
    """

    def _dict(self):
        cpu = torch.device("cpu")
        return (
            MultiStepTracer(
                model=_IdentityModel(),
                device_a=cpu,
                device_b=cpu,
                backend_a="cuda",
                backend_b="cpu",
                dtype="float32",
            )
            .run(torch.randn(1, 8), steps=2)
            .to_dict()
        )

    def test_both_sides_are_recorded_separately(self):
        d = self._dict()
        assert "env_a" in d and "env_b" in d

    def test_library_versions_are_recorded(self):
        env = self._dict()["env_a"]
        assert env["torch"] == torch.__version__
        assert "platform" in env

    def test_real_device_is_recorded_not_the_typed_name(self):
        """The point of the field: what actually ran, not what was asked for."""
        d = self._dict()
        assert d["backend_a"] == "cuda"  # what we typed
        assert d["env_a"]["device_type"] == "cpu"  # what actually ran

    def test_weight_fingerprint_is_recorded(self):
        assert "model_fingerprint" in self._dict()["env_a"]

    def test_both_halves_share_the_same_fingerprint(self):
        """One model is deep-copied, so a difference here means a real problem."""
        d = self._dict()
        assert d["env_a"]["model_fingerprint"] == d["env_b"]["model_fingerprint"]

    def test_result_is_json_serialisable(self):
        import json

        json.dumps(self._dict())


class TestInputIsDescribedInTheResult:
    """A result must describe the input it was measured on.

    Our figures were all measured on ``torch.ones(1, 64)`` — one token repeated
    sixty-four times. Whether that is a fair test is a methodology question for
    the supervisor, and changing it would invalidate every completed run. What
    is not a judgement call is that the file should say what the input was: right
    now the question cannot be answered from the saved results at all.

    The tensor itself is not stored. A description is enough to answer the
    question and to spot two runs measured on different inputs, and it avoids
    writing a real prompt's contents into a shared artifact.
    """

    def _dict(self, x):
        cpu = torch.device("cpu")
        return (
            MultiStepTracer(
                model=_IdentityModel(),
                device_a=cpu,
                device_b=cpu,
                backend_a="cpu",
                backend_b="cpu",
            )
            .run(x, steps=2)
            .to_dict()
        )

    def test_shape_and_dtype_are_recorded(self):
        d = self._dict(torch.ones(1, 64, dtype=torch.long))["input"]
        assert d["shape"] == [1, 64]
        assert d["dtype"] == "torch.int64"

    def test_a_single_repeated_value_is_flagged_synthetic(self):
        d = self._dict(torch.ones(1, 64, dtype=torch.long))["input"]
        assert d["synthetic"] is True
        assert d["unique_values"] == 1

    def test_varied_input_is_not_flagged_synthetic(self):
        d = self._dict(torch.randn(1, 32))["input"]
        assert d["synthetic"] is False
        assert d["unique_values"] > 1

    def test_the_same_input_gives_the_same_fingerprint(self):
        a = self._dict(torch.ones(1, 8, dtype=torch.long))["input"]["fingerprint"]
        b = self._dict(torch.ones(1, 8, dtype=torch.long))["input"]["fingerprint"]
        assert a == b

    def test_a_different_input_gives_a_different_fingerprint(self):
        a = self._dict(torch.ones(1, 8, dtype=torch.long))["input"]["fingerprint"]
        b = self._dict(torch.zeros(1, 8, dtype=torch.long))["input"]["fingerprint"]
        assert a != b

    def test_the_tensor_itself_is_not_stored(self):
        import json

        d = self._dict(torch.ones(1, 64, dtype=torch.long))
        json.dumps(d)  # must stay serialisable
        assert "values" not in d["input"] and "data" not in d["input"]


class TestToleranceRuleIsExplicit:
    """The result must state which tolerance rule was applied.

    The database supplies two limits per entry, ``atol`` and ``rtol``, and the
    trace applies only ``atol`` (``max_diff <= tol.atol``). Meanwhile the
    single-step path prints ``rtol`` and writes ``tolerance_rtol`` into its
    output, which reads as though both were used.

    Starting to apply ``rtol`` is not a code decision: ``atol + rtol*|b|`` is
    looser, so it would change verdicts and alter how every completed run reads.
    That is the supervisor's call. What can be fixed without touching a single
    number is the ambiguity — say which rule ran, and record the limit that was
    available but unused.
    """

    def _dict(self):
        cpu = torch.device("cpu")
        return (
            MultiStepTracer(
                model=_IdentityModel(),
                device_a=cpu,
                device_b=cpu,
                backend_a="cuda",
                backend_b="cpu",
                dtype="float32",
            )
            .run(torch.randn(1, 8), steps=2)
            .to_dict()
        )

    def test_the_applied_rule_is_named(self):
        assert self._dict()["tolerance_rule"] == "atol_only"

    def test_the_unused_limit_is_still_recorded(self):
        d = self._dict()
        assert d["rtol"] == pytest.approx(1.0e-5)
        assert d["atol"] == pytest.approx(1.0e-4)

    def test_verdicts_are_unchanged_by_this(self):
        """Recording the rule must not alter any pass or fail.

        A real difference of 2.0e-4 against cuda/float32's atol of 1.0e-4 must
        still fail. Applying rtol as well would have loosened the threshold and
        turned this into a pass, which is exactly the change not being made.
        """

        class _DriftingModel(nn.Module):
            """Second copy returns the first copy's output shifted by 2.0e-4."""

            calls = 0

            def forward(self, x):
                out = x.clone()
                if _DriftingModel.calls % 2 == 1:
                    out = out + 2.0e-4
                _DriftingModel.calls += 1
                return out

        _DriftingModel.calls = 0
        cpu = torch.device("cpu")
        result = MultiStepTracer(
            model=_DriftingModel(),
            device_a=cpu,
            device_b=cpu,
            backend_a="cuda",
            backend_b="cpu",
            dtype="float32",
        ).run(torch.ones(1, 8), steps=3)
        assert result.final_passed is False
        assert result.first_divergence_step == 1


class TestGpuAliasFollowsTheLocalVendor:
    """``gpu`` means the local accelerator, so its tolerance must follow the build.

    ``gpu`` is deliberately neutral in the resolver — it resolves to whichever
    accelerator this machine has. Hard-mapping it to ``cuda`` therefore applies
    NVIDIA's limit to an AMD run: 1.0e-4 instead of 1.0e-3, ten times stricter,
    and from the wrong vendor entirely.

    That is the exact failure this alias table exists to remove — a tolerance
    that changes with the name typed rather than the hardware used.
    """

    def test_gpu_maps_to_rocm_on_a_rocm_build(self):
        from unittest.mock import patch

        from torchbridge.testing.trace_validator import _tolerance_key

        with patch.object(torch.version, "hip", "6.0.32830", create=True):
            assert _tolerance_key("gpu") == "rocm"

    def test_gpu_maps_to_cuda_on_a_cuda_build(self):
        from unittest.mock import patch

        from torchbridge.testing.trace_validator import _tolerance_key

        with patch.object(torch.version, "hip", None, create=True):
            assert _tolerance_key("gpu") == "cuda"

    def test_the_two_builds_give_different_limits(self):
        """If they agreed, the mapping would not matter and neither would this."""
        from unittest.mock import patch

        from torchbridge.testing.tolerance_db import ToleranceDB
        from torchbridge.testing.trace_validator import _tolerance_key

        db = ToleranceDB()
        with patch.object(torch.version, "hip", "6.0.32830", create=True):
            rocm_atol = db.get(_tolerance_key("gpu"), "float32").atol
        with patch.object(torch.version, "hip", None, create=True):
            cuda_atol = db.get(_tolerance_key("gpu"), "float32").atol
        assert rocm_atol != cuda_atol
        assert rocm_atol == pytest.approx(1.0e-3)
        assert cuda_atol == pytest.approx(1.0e-4)

    def test_the_other_aliases_are_unaffected(self):
        from torchbridge.testing.trace_validator import _tolerance_key

        assert _tolerance_key("neuron") == "trainium"
        assert _tolerance_key("tpu") == "xla"
        assert _tolerance_key("cuda") == "cuda"
