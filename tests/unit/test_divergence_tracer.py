"""
Unit tests for torchbridge.testing.divergence.DivergenceTracer.

Covers:
  - Empty tensor guard in compare_with() and layers_with_divergence()
  - max_layers parameter limits hook registrations
  - Basic compare_with() and layers_with_divergence() functionality
  - Context manager protocol (__enter__/__exit__)
All tests run on CPU — no GPU required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchbridge.testing.divergence import (
    DivergenceTracer,  # noqa: F401 (LayerDivergence imported for completeness)
)

# ── Helpers ──────────────────────────────────────────────────────────────────

class _SimpleModel(nn.Module):
    """3-layer sequential model for testing."""
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class _EmptyOutputModule(nn.Module):
    """Module that emits an empty tensor (shape [0, dim])."""
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(0, self.dim)


class _EmptyOutputModel(nn.Module):
    """Wraps _EmptyOutputModule in a simple container."""
    def __init__(self) -> None:
        super().__init__()
        self.empty_layer = _EmptyOutputModule(8)
        self.normal_layer = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.empty_layer(x)
        return self.normal_layer(x)


# ── Basic context manager ─────────────────────────────────────────────────────

class TestContextManager:
    def test_enter_registers_hooks(self):
        model = _SimpleModel()
        tracer = DivergenceTracer(model)
        assert len(tracer._handles) == 0
        with tracer:
            assert len(tracer._handles) > 0

    def test_exit_removes_hooks(self):
        model = _SimpleModel()
        tracer = DivergenceTracer(model)
        with tracer:
            pass
        assert len(tracer._handles) == 0

    def test_layer_names_populated_after_forward(self):
        model = _SimpleModel()
        x = torch.randn(2, 8)
        tracer = DivergenceTracer(model)
        with tracer:
            model(x)
        assert len(tracer.layer_names) > 0


# ── Empty tensor guard ────────────────────────────────────────────────────────

class TestEmptyTensorGuard:
    def test_compare_with_skips_empty_tensor_no_crash(self):
        """compare_with() must not crash or produce NaN when a layer emits []."""
        model = _EmptyOutputModel()
        x = torch.randn(2, 8)

        ref_tracer = DivergenceTracer(model)
        with ref_tracer:
            model(x)

        test_tracer = DivergenceTracer(model)
        with test_tracer:
            model(x)

        # Should not raise; the empty_layer capture is silently skipped
        results = test_tracer.compare_with(ref_tracer)
        for r in results:
            assert r.max_diff == r.max_diff  # not NaN
            assert r.cosine_sim == r.cosine_sim  # not NaN

    def test_layers_with_divergence_skips_empty_no_crash(self):
        """layers_with_divergence() must not crash on empty-tensor layers."""
        model = _EmptyOutputModel()
        x = torch.randn(2, 8)

        tracer = DivergenceTracer(model)
        with tracer:
            model(x)
            model(x)  # second pass for consecutive comparison

        # Should not raise
        results = tracer.layers_with_divergence()
        for r in results:
            assert r.max_diff == r.max_diff


# ── max_layers parameter ──────────────────────────────────────────────────────

class TestMaxLayers:
    def test_max_layers_limits_captures(self):
        """max_layers=1 should register only 1 layer hook."""
        model = _SimpleModel()
        tracer = DivergenceTracer(model, max_layers=1)
        x = torch.randn(2, 8)
        with tracer:
            model(x)
        assert len(tracer.layer_names) == 1

    def test_max_layers_two_captures_two(self):
        model = _SimpleModel()
        tracer = DivergenceTracer(model, max_layers=2)
        x = torch.randn(2, 8)
        with tracer:
            model(x)
        assert len(tracer.layer_names) == 2

    def test_max_layers_none_captures_all(self):
        model = _SimpleModel()
        tracer_unlimited = DivergenceTracer(model, max_layers=None)
        x = torch.randn(2, 8)
        with tracer_unlimited:
            model(x)
        # _SimpleModel has 3 named submodules (linear1, relu, linear2)
        assert len(tracer_unlimited.layer_names) == 3

    def test_max_layers_larger_than_actual_captures_all(self):
        model = _SimpleModel()
        tracer = DivergenceTracer(model, max_layers=100)
        x = torch.randn(2, 8)
        with tracer:
            model(x)
        # Should capture all 3 layers, not raise
        assert len(tracer.layer_names) == 3

    def test_max_layers_zero_captures_none(self):
        model = _SimpleModel()
        tracer = DivergenceTracer(model, max_layers=0)
        x = torch.randn(2, 8)
        with tracer:
            model(x)
        assert len(tracer.layer_names) == 0


# ── Basic compare_with correctness ───────────────────────────────────────────

class TestCompareWith:
    def test_identical_outputs_zero_diff(self):
        model = _SimpleModel()
        x = torch.randn(2, 8)

        ref = DivergenceTracer(model)
        with ref:
            model(x)

        test = DivergenceTracer(model)
        with test:
            model(x)

        results = test.compare_with(ref)
        for r in results:
            assert r.max_diff == 0.0

    def test_results_sorted_by_max_diff_descending(self):
        model = _SimpleModel()
        x = torch.randn(2, 8)

        ref = DivergenceTracer(model)
        with ref:
            model(x)

        test = DivergenceTracer(model)
        with test:
            model(x)

        results = test.compare_with(ref)
        diffs = [r.max_diff for r in results]
        assert diffs == sorted(diffs, reverse=True)
