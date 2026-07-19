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
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbridge.testing.tolerance_db import ToleranceDB

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "backend_a": self.backend_a,
            "backend_b": self.backend_b,
            "steps": self.steps,
            "dtype": self.dtype,
            "autoregressive": self.autoregressive,
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
    ) -> None:
        self._model = model
        self._device_a = device_a
        self._device_b = device_b
        self._backend_a = backend_a.lower()
        self._backend_b = backend_b.lower()
        self._dtype = dtype.lower()
        self._is_lm = is_lm
        self._tol_db = tolerance_db if tolerance_db is not None else ToleranceDB()

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

        tol = self._tol_db.get(self._backend_a, self._dtype)

        result = TraceValidationResult(
            backend_a=self._backend_a,
            backend_b=self._backend_b,
            steps=steps,
            dtype=self._dtype,
            autoregressive=autoregressive,
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
        current_input = input_ids.clone()

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

        # Empty step_results (all inference steps failed) must be a failure,
        # not vacuous True from all() on an empty sequence.
        result.final_passed = bool(result.step_results) and all(
            s.within_tolerance for s in result.step_results
        )
        return result


# ── Helpers ──────────────────────────────────────────────────────────────────


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
