# SPDX-License-Identifier: Apache-2.0
"""
Multi-Step Trace Validator

Runs N sequential forward passes on two hardware backends and tracks per-step
numerical divergence. Detects the compounding divergence pattern specific to
agentic AI: a single-step divergence of 2e-5 can amplify 500× or more over
50 reasoning steps, causing backends to branch semantically.

Two modes:
- Standard (autoregressive=False): same input repeated N times. Measures
  run-to-run cross-backend consistency. Divergence does NOT propagate.
- Autoregressive (autoregressive=True): greedy token from backend_a appended
  to the input sequence at each step. Simulates actual LLM generation where
  step k's output becomes step k+1's input. Divergence propagates.

Usage::

    from torchbridge.testing.trace_validator import MultiStepTracer
    import torch
    import torch.nn as nn

    model = nn.Linear(64, 64)
    tracer = MultiStepTracer(
        model,
        device_a=torch.device("cpu"),
        device_b=torch.device("cpu"),
        backend_a="cpu",
        backend_b="cpu",
    )
    result = tracer.run(
        input_ids=torch.randn(1, 8, 64),
        steps=10,
    )
    print(result.final_passed, result.max_amplification)
"""

from __future__ import annotations

import copy
import hashlib
import logging
import platform
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbridge.testing.tolerance_db import ToleranceDB

logger = logging.getLogger(__name__)

RECORD_FORMAT_VERSION = 1
"""Schema version for :class:`SplitTraceRecord` artifacts on disk."""


@dataclass
class TraceStepResult:
    """Divergence statistics for a single step in a multi-step trace."""

    step: int
    """1-indexed step number."""

    max_diff: float
    """Max absolute difference between backend_a and backend_b outputs at this step."""

    cosine_sim: float
    """Cosine similarity between flattened outputs at this step."""

    within_tolerance: bool
    """True if max_diff <= tolerance threshold from ToleranceDB."""

    cumulative_amplification: float
    """max_diff[step] / max_diff[1]. Always 1.0 at step 1. 1.0 if step 1 max_diff == 0."""


@dataclass
class TraceValidationResult:
    """Result from a complete multi-step trace validation run."""

    backend_a: str
    backend_b: str
    steps: int
    dtype: str
    autoregressive: bool

    step_results: list[TraceStepResult] = field(default_factory=list)

    first_divergence_step: int | None = None
    """First step (1-indexed) where within_tolerance is False. None if all steps pass."""

    max_amplification: float = 1.0
    """Peak cumulative_amplification across all steps."""

    final_passed: bool = True
    """True only if all step_results have within_tolerance=True."""

    # Appended rather than grouped with the other tolerance fields on purpose.
    # This dataclass is public and was constructible positionally, with
    # step_results sixth; inserting ahead of it would silently bind a caller's
    # step list to model_family. New fields go at the end.
    model_family: str | None = None
    """Family used for the tolerance lookup, or None if the coarse table was used."""

    atol: float | None = None
    """The absolute tolerance actually applied — it decides every verdict below."""

    atol_source: str | None = None
    """Where that tolerance came from: measured, derived, or fallback."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "backend_a": self.backend_a,
            "backend_b": self.backend_b,
            "steps": self.steps,
            "dtype": self.dtype,
            "autoregressive": self.autoregressive,
            # Without these two a saved run cannot be re-checked: the family
            # selects the atol, and the atol decides pass/fail.
            "model_family": self.model_family,
            "atol": self.atol,
            "atol_source": self.atol_source,
            "first_divergence_step": self.first_divergence_step,
            "max_amplification": self.max_amplification,
            "final_passed": self.final_passed,
            "step_results": [
                {
                    "step": s.step,
                    "max_diff": s.max_diff,
                    "cosine_sim": s.cosine_sim,
                    "within_tolerance": s.within_tolerance,
                    "cumulative_amplification": s.cumulative_amplification,
                }
                for s in self.step_results
            ],
        }


@dataclass
class SplitTraceRecord:
    """One backend's half of a split (record/replay) trace.

    A split trace exists because a cross-vendor pair such as ``cuda vs rocm``
    cannot be run in a single process — NVIDIA and AMD hardware never share a
    machine. Splitting is numerically exact rather than an approximation: in
    :meth:`MultiStepTracer.run` the autoregressive token is taken only from
    ``backend_a``, and both backends are then fed that same tensor. ``backend_b``
    never influences the input trajectory, so replaying ``backend_a``'s recorded
    token sequence on another machine reproduces the identical comparison.

    Attributes:
        backend: Backend name string for ToleranceDB lookup (e.g. ``"cuda"``).
        dtype: Dtype string used for the run.
        autoregressive: Whether the recording grew the input sequence per step.
        is_lm: Whether the model was treated as a HuggingFace-style LM.
        token_inputs: Exact input tensor fed at each step, CPU-side. In replay
            these are consumed verbatim — no tokens are re-derived.
        outputs: Per-step extracted output tensor, float32 on CPU.
        env: Library versions, weight fingerprint and platform captured at run
            time. Compared across halves to catch a divergence caused by
            differing ``torch``/``transformers`` builds or a drifted checkpoint
            rather than by the backend itself.
        role: ``"record"`` for the leading half, ``"replay"`` for the follower.
        format_version: :data:`RECORD_FORMAT_VERSION` at write time.
    """

    backend: str
    dtype: str
    autoregressive: bool
    is_lm: bool
    token_inputs: list[torch.Tensor] = field(default_factory=list)
    outputs: list[torch.Tensor] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    role: str = "record"
    format_version: int = RECORD_FORMAT_VERSION
    model_family: str | None = None

    @property
    def steps(self) -> int:
        """Number of steps that actually completed."""
        return len(self.outputs)

    def save(self, path: str) -> None:
        """Write this record to ``path`` via :func:`torch.save`.

        A binary artifact rather than JSON: the per-step outputs are
        last-position logits, so a 50-step trace over a 150k vocabulary is
        millions of floats and would be unwieldy as text.
        """
        torch.save(
            {
                "backend": self.backend,
                "dtype": self.dtype,
                "autoregressive": self.autoregressive,
                "is_lm": self.is_lm,
                "token_inputs": self.token_inputs,
                "outputs": self.outputs,
                "env": self.env,
                "role": self.role,
                "format_version": self.format_version,
                "model_family": self.model_family,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> SplitTraceRecord:
        """Read a record previously written by :meth:`save`.

        Raises:
            ValueError: If the artifact's ``format_version`` is newer than this
                build understands.
        """
        # weights_only=True: a record is carried here from another machine, so
        # it is untrusted input — unpickling arbitrary objects from it would let
        # a tampered file execute code. The payload is only tensors, strings,
        # bools and ints, all of which this mode supports, so nothing is lost.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        version = int(payload.get("format_version", 0))
        if version > RECORD_FORMAT_VERSION:
            raise ValueError(
                f"{path} was written with record format v{version}, but this "
                f"build understands at most v{RECORD_FORMAT_VERSION} — upgrade "
                f"TorchBridge to read it"
            )
        return cls(
            backend=payload["backend"],
            dtype=payload["dtype"],
            autoregressive=payload["autoregressive"],
            is_lm=payload["is_lm"],
            token_inputs=payload["token_inputs"],
            outputs=payload["outputs"],
            env=payload.get("env", {}),
            role=payload.get("role", "record"),
            model_family=payload.get("model_family"),
            format_version=version,
        )


class MultiStepTracer:
    """Runs N forward passes across two backends, tracking per-step divergence.

    Args:
        model: PyTorch model to trace. Must accept the input tensor as its
            sole positional argument (or via ``input_ids`` keyword for LLMs).
        device_a: Primary device (backend A).
        device_b: Comparison device (backend B).
        backend_a: Backend name string for ToleranceDB lookup (e.g. ``"cuda"``).
        backend_b: Backend name string for ToleranceDB lookup (e.g. ``"rocm"``).
        dtype: Dtype string for ToleranceDB lookup (e.g. ``"float32"``).
        is_lm: If True, model returns an object with a ``.logits`` attribute
            (HuggingFace-style). If False, model output is used directly.
        tolerance_db: Custom tolerance database. Defaults to the built-in
            empirically calibrated database.
        model_family: Optional model-family string (e.g. ``"decoder-large"``)
            used for the 3D ToleranceDB lookup. When omitted the lookup falls
            back to the coarser ``(backend, dtype)`` table.
    """

    def __init__(
        self,
        model: nn.Module,
        device_a: torch.device,
        device_b: torch.device,
        backend_a: str,
        backend_b: str,
        dtype: str = "float32",
        is_lm: bool = False,
        tolerance_db: ToleranceDB | None = None,
        model_family: str | None = None,
    ) -> None:
        self._model = model
        self._device_a = device_a
        self._device_b = device_b
        self._backend_a = backend_a.lower()
        self._backend_b = backend_b.lower()
        self._dtype = dtype.lower()
        self._is_lm = is_lm
        self._tol_db = tolerance_db if tolerance_db is not None else ToleranceDB()
        self._model_family = model_family

    def run(
        self,
        input_ids: torch.Tensor,
        steps: int = 10,
        autoregressive: bool = False,
    ) -> TraceValidationResult:
        """Run the multi-step trace and return divergence results.

        Args:
            input_ids: Initial input tensor. For autoregressive mode this is
                the prompt; each step appends the greedy-decoded token from
                backend_a. For standard mode the same tensor is used every step.
            steps: Number of forward passes to run. Must be >= 1.
            autoregressive: If True, append greedy token from backend_a at each
                step (LLM generation simulation). If False, repeat same input.

        Returns:
            :class:`TraceValidationResult` with per-step divergence data.

        Raises:
            ValueError: If ``steps < 1``.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if input_ids.numel() == 0:
            raise ValueError(
                "input_ids must be non-empty (got a tensor with 0 elements)"
            )

        tol = _lookup_tolerance(
            self._tol_db, self._backend_a, self._dtype, self._model_family
        )

        result = TraceValidationResult(
            backend_a=self._backend_a,
            backend_b=self._backend_b,
            steps=steps,
            dtype=self._dtype,
            autoregressive=autoregressive,
            model_family=self._model_family,
            atol=float(tol.atol),
            atol_source=getattr(tol, "source", None),
        )

        # Prepare independent model copies on each device.
        # deepcopy is required: nn.Module.to() mutates in-place, so without it
        # model_a and model_b would be the same object on the last-assigned device.
        #
        # XLA (Neuron/PJRT_DEVICE=CPU): XLA operations route to CPU but XLA tensor
        # → CPU transfer breaks for complex LLM ops (RoPE, GQA, SiLU), producing NaN
        # or a RuntimeError on .cpu(). Since XLA/CPU is numerically identical to native
        # CPU in this mode, run both copies on CPU to get valid divergence data.
        _eff_a = torch.device("cpu") if self._device_a.type == "xla" else self._device_a
        _eff_b = torch.device("cpu") if self._device_b.type == "xla" else self._device_b
        if _eff_a != self._device_a or _eff_b != self._device_b:
            logger.info(
                "XLA device detected with PJRT_DEVICE=CPU — running both model copies "
                "on CPU (XLA CPU-backed execution is numerically identical to native CPU)"
            )
        model_a = copy.deepcopy(self._model).to(_eff_a)
        model_b = copy.deepcopy(self._model).to(_eff_b)
        model_a.eval()
        model_b.eval()

        # Initialise running input (may grow in autoregressive mode)
        # Pinned to CPU because the running trajectory is grown by
        # concatenating next_token.cpu() each step. A caller who passes
        # input_ids already on the accelerator — the natural thing to do —
        # would otherwise hit a device mismatch on the first append.
        current_input = input_ids.detach().clone().cpu()

        step1_max_diff: float | None = None

        for step_idx in range(steps):
            step_num = step_idx + 1  # 1-indexed

            x_a = current_input.to(_eff_a)
            x_b = current_input.to(_eff_b)

            try:
                with torch.no_grad():
                    raw_a = model_a(x_a) if not self._is_lm else model_a(input_ids=x_a)
                    raw_b = model_b(x_b) if not self._is_lm else model_b(input_ids=x_b)
            except Exception as exc:
                logger.warning("Step %d inference failed: %s", step_num, exc)
                break

            # Extract tensor outputs
            out_a = _extract_tensor(raw_a, self._is_lm).float().cpu()
            out_b = _extract_tensor(raw_b, self._is_lm).float().cpu()

            # Compute divergence metrics
            if out_a.shape != out_b.shape:
                logger.warning(
                    "Step %d: output shapes differ (%s vs %s) — skipping",
                    step_num,
                    out_a.shape,
                    out_b.shape,
                )
                break

            diff = torch.abs(out_a - out_b)
            max_diff = float(diff.max())
            _cos = float(
                F.cosine_similarity(
                    out_a.flatten().unsqueeze(0),
                    out_b.flatten().unsqueeze(0),
                )
            )
            # Guard against NaN (occurs when both outputs are all-zero: 0/0).
            # Replace with 0.0 so the result remains JSON-serialisable.
            cos_sim = _cos if _cos == _cos else 0.0

            # Amplification factor relative to step 1
            if step1_max_diff is None:
                step1_max_diff = max_diff
            if step1_max_diff == 0.0:
                amplification = 1.0
            else:
                amplification = max_diff / step1_max_diff

            within_tol = max_diff <= tol.atol

            step_result = TraceStepResult(
                step=step_num,
                max_diff=max_diff,
                cosine_sim=cos_sim,
                within_tolerance=within_tol,
                cumulative_amplification=amplification,
            )
            result.step_results.append(step_result)

            # Track first failure
            if not within_tol and result.first_divergence_step is None:
                result.first_divergence_step = step_num

            # Track peak amplification
            if amplification > result.max_amplification:
                result.max_amplification = amplification

            # Autoregressive: append greedy token from backend_a output.
            # next_token shape: (batch,) → unsqueeze(-1) → (batch, 1)
            # current_input shape: (batch, seq) → cat → (batch, seq+1)
            if autoregressive and self._is_lm:
                next_token = _greedy_token(raw_a)
                if next_token is not None:
                    current_input = torch.cat(
                        [current_input, next_token.unsqueeze(-1).cpu()], dim=-1
                    )

        # A step that raised or produced mismatched shapes breaks the loop early,
        # so report the steps actually measured rather than the number requested.
        result.steps = len(result.step_results)

        # Empty step_results (all inference steps failed) must be a failure,
        # not vacuous True from all() on an empty sequence.
        result.final_passed = bool(result.step_results) and all(
            s.within_tolerance for s in result.step_results
        )
        return result

    def record(
        self,
        input_ids: torch.Tensor,
        steps: int = 10,
        autoregressive: bool = False,
    ) -> SplitTraceRecord:
        """Run the leading half of a split trace on ``device_a`` alone.

        Use this when the two backends of a comparison cannot coexist in one
        process (``cuda vs rocm``, ``cuda vs mps``, ``cuda vs xla``). Ship the
        returned record to the second machine, call :meth:`replay` there, then
        :func:`compare_records` to obtain the same
        :class:`TraceValidationResult` a single-process :meth:`run` would give.

        Unlike :meth:`run`, no XLA-to-CPU substitution happens here: a silent
        fallback would report CPU-vs-CPU numbers under an accelerator's name.

        Args:
            input_ids: Initial input tensor (the prompt in autoregressive mode).
            steps: Number of forward passes. Must be >= 1.
            autoregressive: If True, append this backend's greedy token each
                step, exactly as :meth:`run` does.

        Returns:
            :class:`SplitTraceRecord` with ``role="record"``.

        Raises:
            ValueError: If ``steps < 1`` or ``input_ids`` is empty.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if input_ids.numel() == 0:
            raise ValueError(
                "input_ids must be non-empty (got a tensor with 0 elements)"
            )

        record = SplitTraceRecord(
            backend=self._backend_a,
            dtype=self._dtype,
            autoregressive=autoregressive,
            is_lm=self._is_lm,
            env=_capture_env(self._device_a, self._model),
            role="record",
            # Carried in the artifact, not left to the compare host's command
            # line. compare_records() runs offline, possibly on a third machine,
            # and without this it falls back to the coarse tolerance row — the
            # same silent wrong-limit bug this workflow exists to avoid.
            model_family=self._model_family,
        )

        model = copy.deepcopy(self._model).to(self._device_a)
        model.eval()

        # Pinned to CPU because the running trajectory is grown by
        # concatenating next_token.cpu() each step. A caller who passes
        # input_ids already on the accelerator — the natural thing to do —
        # would otherwise hit a device mismatch on the first append.
        current_input = input_ids.detach().clone().cpu()

        for step_idx in range(steps):
            step_num = step_idx + 1
            x = current_input.to(self._device_a)

            try:
                with torch.no_grad():
                    raw = model(x) if not self._is_lm else model(input_ids=x)
                _sync_device(self._device_a)
            except Exception as exc:
                logger.warning("Record step %d inference failed: %s", step_num, exc)
                break

            out = _extract_tensor(raw, self._is_lm).float().cpu()
            record.token_inputs.append(current_input.clone().cpu())
            record.outputs.append(out)

            if autoregressive and self._is_lm:
                next_token = _greedy_token(raw)
                if next_token is None:
                    break
                current_input = torch.cat(
                    [current_input, next_token.unsqueeze(-1).cpu()], dim=-1
                )

        return record

    def replay(self, record: SplitTraceRecord) -> SplitTraceRecord:
        """Run the following half of a split trace on ``device_b`` alone.

        Consumes ``record.token_inputs`` verbatim. No tokens are re-derived —
        that is what makes the split exact: the follower sees precisely the
        inputs the leader saw.

        Args:
            record: Artifact produced by :meth:`record` on the other machine.

        Returns:
            :class:`SplitTraceRecord` with ``role="replay"``, aligned step-for-
            step with the input record.

        Raises:
            ValueError: If ``record`` contains no steps.
        """
        if record.role != "record":
            # A library-level invariant, separate from the CLI's backend check:
            # replaying a follower produces a second follower, and
            # compare_records() then has no recorded half at all.
            raise ValueError(
                f"can only replay a record with role 'record', got {record.role!r}"
            )
        if not record.token_inputs:
            raise ValueError("record contains no steps to replay")
        if record.dtype != self._dtype:
            logger.warning(
                "Replaying a %s record under dtype %s — compare_records() looks "
                "the tolerance up from the recorded half's dtype, not this one",
                record.dtype,
                self._dtype,
            )

        replayed = SplitTraceRecord(
            backend=self._backend_b,
            dtype=self._dtype,
            autoregressive=record.autoregressive,
            is_lm=record.is_lm,
            env=_capture_env(self._device_b, self._model),
            role="replay",
            model_family=self._model_family,
        )

        model = copy.deepcopy(self._model).to(self._device_b)
        model.eval()

        for step_idx, token_input in enumerate(record.token_inputs):
            step_num = step_idx + 1
            x = token_input.to(self._device_b)

            try:
                with torch.no_grad():
                    raw = model(x) if not record.is_lm else model(input_ids=x)
                _sync_device(self._device_b)
            except Exception as exc:
                logger.warning("Replay step %d inference failed: %s", step_num, exc)
                break

            replayed.token_inputs.append(token_input.clone().cpu())
            replayed.outputs.append(_extract_tensor(raw, record.is_lm).float().cpu())

        return replayed


def compare_records(
    record_a: SplitTraceRecord,
    record_b: SplitTraceRecord,
    tolerance_db: ToleranceDB | None = None,
    model_family: str | None = None,
    strict_env: bool = False,
) -> TraceValidationResult:
    """Diff two halves of a split trace offline.

    Produces the same :class:`TraceValidationResult` that a single-process
    :meth:`MultiStepTracer.run` would produce for the equivalent device pair.

    Args:
        record_a: The recorded (leading) half.
        record_b: The replayed (following) half.
        tolerance_db: Custom tolerance database. Defaults to the built-in one.
        model_family: Optional family string for the 3D ToleranceDB lookup.
        strict_env: If True, raise when the two halves were produced under
            different ``torch``/``transformers`` versions. Left False by default
            so an exploratory comparison still runs, but a mismatch is always
            logged as a warning: differing library builds mean differing RoPE
            and attention kernels, so the measured divergence would reflect the
            libraries rather than the backends.

    Returns:
        :class:`TraceValidationResult` over the steps common to both halves.

    Raises:
        ValueError: If either half is empty, if their dtypes differ, if the
            second half was not produced by :meth:`MultiStepTracer.replay`, if
            their token inputs disagree (so they are not halves of the same
            trace), or if ``strict_env`` is set and the environments differ.
            Two halves sharing a backend name only warn — that pairing is the
            split path's own control run.
    """
    if not record_a.outputs or not record_b.outputs:
        raise ValueError(
            "both records must contain at least one completed step "
            f"(got {len(record_a.outputs)} and {len(record_b.outputs)})"
        )
    if record_a.backend == record_b.backend:
        # A warning, not an error: two records from one backend is the split
        # path's control run, and it must stay possible — the same backend on
        # both sides has to agree exactly, which is how record/replay proves it
        # adds no error of its own. The warning still surfaces the real mistake
        # of pairing two halves from the same machine by accident.
        logger.warning(
            "Both records report the same backend %r. This is only meaningful as "
            "a control run; if it was not intended, one half is from the wrong "
            "machine.",
            record_a.backend,
        )
    # Order is not cosmetic: the tolerance threshold is looked up from
    # record_a's backend, so swapping the arguments changes atol and therefore
    # first_divergence_step.
    # Both roles are checked directly. Rejecting only the reversed pair left
    # two replay halves accepted, which silently promotes a follower to the
    # primary side — and the primary side supplies the backend name, the dtype
    # and the tolerance for the whole comparison.
    if record_a.role != "record" or record_b.role != "replay":
        raise ValueError(
            f"a comparison needs one recorded half and one replayed half, but "
            f"got roles {record_a.role!r} and {record_b.role!r}"
            + (
                " — the arguments are reversed; pass the recorded half first"
                if record_a.role == "replay" and record_b.role == "record"
                else " — the second half must come from replay() on the other "
                "machine, otherwise the two halves were never fed the same inputs"
            )
        )

    if record_a.dtype != record_b.dtype:
        # The tolerance comes from record_a.dtype, so a mismatched pair would be
        # judged by one half's precision while half the data came from another.
        # Silent, and wrong in the same way the vendor check exists to prevent.
        raise ValueError(
            f"the two halves were recorded under different dtypes "
            f"({record_a.dtype!r} and {record_b.dtype!r}); the tolerance is "
            f"looked up from the recorded half, so the comparison would apply "
            f"the wrong limit to half the data"
        )

    mismatches = _env_mismatches(record_a.env, record_b.env)
    if mismatches:
        message = (
            "split-trace halves were produced under different environments "
            f"({'; '.join(mismatches)}) — measured divergence may reflect the "
            f"libraries rather than the backends"
        )
        if any(m.startswith("model_fingerprint") for m in mismatches):
            message += (
                "; note the fingerprint samples each parameter's ends, so it "
                "proves the weights differ but its silence does not prove they "
                "match"
            )
        if strict_env:
            raise ValueError(message)
        logger.warning(message)

    n_steps = min(len(record_a.outputs), len(record_b.outputs))
    if len(record_a.outputs) != len(record_b.outputs):
        logger.warning(
            "step counts differ (%d vs %d) — comparing the first %d",
            len(record_a.outputs),
            len(record_b.outputs),
            n_steps,
        )

    db = tolerance_db if tolerance_db is not None else ToleranceDB()
    # An explicit argument wins — the operator may know something the artifact
    # does not. Otherwise the family travels with the record, so an offline
    # comparison applies the same limit the recording run would have.
    family = model_family if model_family is not None else record_a.model_family
    if (
        model_family is None
        and record_b.model_family is not None
        and record_b.model_family != record_a.model_family
    ):
        logger.warning(
            "The two halves record different model families (%r and %r); "
            "using %r from the recorded half",
            record_a.model_family,
            record_b.model_family,
            record_a.model_family,
        )
    tol = _lookup_tolerance(db, record_a.backend, record_a.dtype, family)

    result = TraceValidationResult(
        backend_a=record_a.backend,
        backend_b=record_b.backend,
        steps=n_steps,
        dtype=record_a.dtype,
        autoregressive=record_a.autoregressive,
        model_family=family,
        atol=float(tol.atol),
        atol_source=getattr(tol, "source", None),
    )

    step1_max_diff: float | None = None

    for step_idx in range(n_steps):
        step_num = step_idx + 1
        in_a = record_a.token_inputs[step_idx]
        in_b = record_b.token_inputs[step_idx]

        # The halves must have been fed identical inputs, otherwise they are not
        # two views of one trace and any diff between them is meaningless.
        if in_a.shape != in_b.shape or not torch.equal(in_a, in_b):
            raise ValueError(
                f"step {step_num}: the two halves were fed different inputs — "
                f"record_b was not replayed from record_a"
            )

        out_a = record_a.outputs[step_idx]
        out_b = record_b.outputs[step_idx]
        if out_a.shape != out_b.shape:
            logger.warning(
                "Step %d: output shapes differ (%s vs %s) — stopping",
                step_num,
                out_a.shape,
                out_b.shape,
            )
            break

        diff = torch.abs(out_a - out_b)
        max_diff = float(diff.max())
        _cos = float(
            F.cosine_similarity(
                out_a.flatten().unsqueeze(0),
                out_b.flatten().unsqueeze(0),
            )
        )
        cos_sim = _cos if _cos == _cos else 0.0

        if step1_max_diff is None:
            step1_max_diff = max_diff
        amplification = 1.0 if step1_max_diff == 0.0 else max_diff / step1_max_diff

        within_tol = max_diff <= tol.atol
        result.step_results.append(
            TraceStepResult(
                step=step_num,
                max_diff=max_diff,
                cosine_sim=cos_sim,
                within_tolerance=within_tol,
                cumulative_amplification=amplification,
            )
        )

        if not within_tol and result.first_divergence_step is None:
            result.first_divergence_step = step_num
        if amplification > result.max_amplification:
            result.max_amplification = amplification

    result.steps = len(result.step_results)
    # Every pair the halves offered has to have been compared. Without this a
    # replay that died after three good steps, or a shape change at step four,
    # reports final_passed on the prefix and the CLI exits 0 — having measured
    # something other than the trace that was asked for. The per-step numbers
    # are still returned, because a partial trace is worth reading; it just
    # cannot be called a pass.
    complete = (
        len(result.step_results) == len(record_a.outputs) == len(record_b.outputs)
    )
    if not complete:
        logger.warning(
            "compared %d step(s) of %d recorded and %d replayed — the result "
            "cannot pass, because the requested trace was not measured in full",
            len(result.step_results),
            len(record_a.outputs),
            len(record_b.outputs),
        )
    result.final_passed = (
        complete
        and bool(result.step_results)
        and all(s.within_tolerance for s in result.step_results)
    )
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _lookup_tolerance(
    db: Any, backend: str, dtype: str, model_family: str | None
) -> Any:
    """Look up a tolerance entry, always passing the family.

    Branching on whether a family was supplied would leave two code paths and
    still break a two-argument get() the moment a family *is* given, so it
    buys nothing. Catching the resulting TypeError and retrying without the
    family would be worse: a genuine bug inside a database would be swallowed
    and the coarse (backend, dtype) row applied silently — exactly the failure
    this argument exists to prevent. ToleranceDB already declares
    model_family optional, so accepting it is the contract for any substitute
    database.
    """
    # By keyword, not position: a substitute database may declare the family
    # keyword-only, and a positional call raises TypeError against that
    # signature. It is still always sent, so no silent fallback is reintroduced.
    return db.get(backend, dtype, model_family=model_family)


def _extract_tensor(output: Any, is_lm: bool) -> torch.Tensor:
    """Extract a flat tensor from model output."""
    if is_lm:
        # HuggingFace: use last-position logits
        logits = getattr(output, "logits", output)
        if isinstance(logits, torch.Tensor):
            if logits.ndim >= 2:
                return logits[:, -1, :]  # (batch, vocab)
            return logits
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Cannot extract tensor from model output of type {type(output)}")


_ENV_COMPARED_KEYS = ("torch", "transformers", "model_fingerprint")
"""Env keys whose mismatch invalidates a split comparison.

The libraries that supply the model's forward implementation, plus the weights
they loaded. ``platform`` and ``device`` are recorded for provenance but are
expected to differ — that they differ is the entire point of a cross-backend
comparison.
"""

_FINGERPRINT_SAMPLE = 8
"""Elements taken from each end of a parameter when fingerprinting."""


def _model_fingerprint(model: nn.Module) -> str:
    """Hash a model's parameters, to prove both halves loaded the same weights.

    :meth:`MultiStepTracer.run` deep-copies one model, so identical weights are
    structural. A split trace has no such guarantee: each machine calls
    ``from_pretrained`` independently, so a checkpoint that drifts between the
    two runs — an updated HuggingFace revision, a partial download — would show
    up as backend divergence when it is really checkpoint divergence.

    Samples each parameter's ends rather than hashing every byte: an 8B model in
    bfloat16 is ~16 GB, and hashing it twice per run would dominate the runtime.
    Key names, shapes and dtypes are hashed in full, so a structural difference
    is always caught.

    **This is a one-way signal.** A difference proves the weights differ; a
    match does not prove they agree, because an edit confined to the interior
    of a large parameter changes no sampled value. That is why the comparison
    only ever warns on a mismatch and never decides a verdict from one. Closing
    the gap would mean hashing the full checkpoint on both machines, which
    costs more than the trace it guards — so the limitation is stated rather
    than removed.
    """
    hasher = hashlib.sha256()
    try:
        state = model.state_dict()
    except Exception as exc:  # pragma: no cover - exotic modules
        logger.debug("Could not fingerprint model: %s", exc)
        return "unavailable"

    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        hasher.update(key.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(str(tensor.dtype).encode())
        # Slice first, convert second. Converting the whole parameter to float32
        # on the host before taking 16 values costs a full copy of the model —
        # for an 8B bfloat16 checkpoint the embedding alone is ~2.5 GB as
        # float32, and this runs twice per trace. Only the sample is moved.
        flat = tensor.detach().reshape(-1)
        if flat.numel() <= 2 * _FINGERPRINT_SAMPLE:
            sample = flat
        else:
            sample = torch.cat(
                [flat[:_FINGERPRINT_SAMPLE], flat[-_FINGERPRINT_SAMPLE:]]
            )
        hasher.update(sample.float().cpu().numpy().tobytes())

    return hasher.hexdigest()[:16]


def _capture_env(
    device: torch.device, model: nn.Module | None = None
) -> dict[str, str]:
    """Record the library, weight and hardware provenance of one split-trace half."""
    # str() on every value, not cosmetic: torch.__version__ is a TorchVersion
    # object, and a record holding one cannot be reloaded under
    # weights_only=True — which is how these files must be read, since they
    # arrive from another machine.
    env = {
        "torch": str(torch.__version__),
        "platform": str(platform.platform()),
        "device_type": str(device.type),
    }

    if model is not None:
        env["model_fingerprint"] = _model_fingerprint(model)

    try:  # transformers is an optional dependency
        import transformers

        env["transformers"] = str(transformers.__version__)
    except Exception:  # pragma: no cover - depends on install extras
        pass

    try:
        if device.type == "cuda" and torch.cuda.is_available():
            env["device"] = torch.cuda.get_device_name(device)
            # torch.version.hip is set on ROCm builds, where devices still
            # report type "cuda" — needed to tell an MI300X from an H100.
            hip = getattr(torch.version, "hip", None)
            env["vendor"] = "rocm" if hip else "cuda"
            if hip:
                env["hip"] = str(hip)
        else:
            env["device"] = str(device)
    except Exception:  # pragma: no cover - driver-dependent
        env["device"] = str(device)

    return env


def _env_mismatches(env_a: dict[str, str], env_b: dict[str, str]) -> list[str]:
    """Return human-readable descriptions of forward-affecting env differences."""
    mismatches = []
    for key in _ENV_COMPARED_KEYS:
        val_a = env_a.get(key)
        val_b = env_b.get(key)
        if val_a is not None and val_b is not None and val_a != val_b:
            mismatches.append(f"{key} {val_a} vs {val_b}")
    return mismatches


def _sync_device(device: torch.device) -> None:
    """Force pending work to complete before the output tensor is read.

    Lazy backends need an explicit barrier: without it the tensor read back is
    not guaranteed to reflect the forward pass just issued. XLA in particular
    requires ``mark_step`` to materialise its graph.
    """
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "xla":
            import torch_xla.core.xla_model as xm

            xm.mark_step()
    except Exception as exc:  # pragma: no cover - backend-dependent
        logger.debug("Device sync on %s was a no-op: %s", device, exc)


def _greedy_token(lm_output: Any) -> torch.Tensor | None:
    """Return greedy (argmax) next token from LM output logits."""
    logits = getattr(lm_output, "logits", None)
    if logits is None or not isinstance(logits, torch.Tensor):
        return None
    if logits.ndim == 3:
        # (batch, seq, vocab) — take last position, argmax over vocab
        return logits[:, -1, :].argmax(dim=-1)  # (batch,)
    if logits.ndim == 2:
        # (batch, vocab) — already at single token position
        return logits.argmax(dim=-1)  # (batch,)
    return None
