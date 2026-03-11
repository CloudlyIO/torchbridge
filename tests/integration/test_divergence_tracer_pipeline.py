"""
Integration tests for DivergenceTracer in realistic pipeline scenarios.

Tests verify:
- max_layers parameter caps hook registrations in a real multi-layer model
- Empty-tensor-emitting modules are silently skipped without NaN in results
- DivergenceTracer integrates correctly with tb-validate --compare --per-layer
All tests run on CPU — no GPU required.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from torchbridge.testing.divergence import DivergenceTracer

# ── Helpers ───────────────────────────────────────────────────────────────────

class _DeepModel(nn.Module):
    """10-layer sequential linear model for testing max_layers cap."""
    def __init__(self, layers: int = 10, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _EmptyTensorLayer(nn.Module):
    """Module that emits a shape-[0, dim] tensor."""
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(0, self.dim)


class _MixedModel(nn.Module):
    """Has both normal layers and a layer that emits empty tensors."""
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.empty_layer = _EmptyTensorLayer(16)
        self.linear2 = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        _ = self.empty_layer(x)  # produces empty tensor (side effect only)
        return self.linear2(out)


# ── max_layers in pipeline ─────────────────────────────────────────────────────

class TestMaxLayersInPipeline:
    def test_max_layers_caps_captures_on_deep_model(self):
        """max_layers=3 on a 10-layer model must register exactly 3 hooks."""
        model = _DeepModel(layers=10)
        tracer = DivergenceTracer(model, max_layers=3)
        x = torch.randn(2, 16)
        with tracer:
            model(x)
        # layers ModuleList has 10 children; max_layers=3 should cap it
        assert len(tracer.layer_names) == 3

    def test_max_layers_results_subset_of_full(self):
        """max_layers results are a prefix of the full (no cap) results."""
        model = _DeepModel(layers=6)
        x = torch.randn(2, 16)

        # Full run
        ref_full = DivergenceTracer(model)
        with ref_full:
            model(x)
        test_full = DivergenceTracer(model)
        with test_full:
            model(x)
        full_names = {r.layer_name for r in test_full.compare_with(ref_full)}

        # Capped run (max_layers=3)
        ref_cap = DivergenceTracer(model, max_layers=3)
        with ref_cap:
            model(x)
        test_cap = DivergenceTracer(model, max_layers=3)
        with test_cap:
            model(x)
        cap_names = {r.layer_name for r in test_cap.compare_with(ref_cap)}

        # Every layer in the capped result should appear in the full result
        assert cap_names.issubset(full_names)
        assert len(cap_names) <= len(full_names)

    def test_full_run_without_max_layers_captures_all(self):
        """Without max_layers, all 10 layers are captured."""
        model = _DeepModel(layers=10)
        tracer = DivergenceTracer(model)
        x = torch.randn(2, 16)
        with tracer:
            model(x)
        # _DeepModel has layers ModuleList (1) + 10 Linear children (10 named submodules total)
        # named_modules skips root; ModuleList + each Linear = 11 entries
        assert len(tracer.layer_names) >= 10


# ── Empty tensor in pipeline ───────────────────────────────────────────────────

class TestEmptyTensorInPipeline:
    def test_pipeline_with_empty_tensor_layer_no_crash(self):
        """Full compare_with() pipeline must not crash with mixed empty/normal layers."""
        model = _MixedModel()
        x = torch.randn(2, 16)

        ref = DivergenceTracer(model)
        with ref:
            model(x)

        test = DivergenceTracer(model)
        with test:
            model(x)

        # Should not raise, should not produce NaN
        results = test.compare_with(ref)
        for r in results:
            assert r.max_diff == r.max_diff  # not NaN
            assert r.cosine_sim == r.cosine_sim  # not NaN

    def test_empty_tensor_layer_absent_from_results(self):
        """The empty_layer must not appear in compare_with() results."""
        model = _MixedModel()
        x = torch.randn(2, 16)

        ref = DivergenceTracer(model)
        with ref:
            model(x)

        test = DivergenceTracer(model)
        with test:
            model(x)

        results = test.compare_with(ref)
        layer_names = {r.layer_name for r in results}
        assert "empty_layer" not in layer_names

    def test_normal_layers_still_compared_despite_empty_layer(self):
        """Normal layers (linear1, linear2) must appear in results."""
        model = _MixedModel()
        x = torch.randn(2, 16)

        ref = DivergenceTracer(model)
        with ref:
            model(x)

        test = DivergenceTracer(model)
        with test:
            model(x)

        results = test.compare_with(ref)
        layer_names = {r.layer_name for r in results}
        # Both normal linear layers must be compared
        assert "linear1" in layer_names
        assert "linear2" in layer_names


# ── Integration with tb-validate --per-layer ──────────────────────────────────

class TestDivergenceTracerWithValidatePipeline:
    """End-to-end: per_layer=True feeds DivergenceTracer output into _run_compare result."""

    def _make_args(self, **kwargs) -> argparse.Namespace:
        defaults = {
            "compare": ["cpu", "cpu"],
            "model": None,
            "input_shape": "1,32",
            "per_layer": True,
            "dtype": "float32",
            "output": None,
            "ci": True,
            "verbose": False,
            "level": "standard",
            "quantized": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_per_layer_key_present_in_ci_output(self, capsys):
        import json

        from torchbridge.cli.validate import ValidateCommand
        args = self._make_args()
        ValidateCommand._run_compare(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "per_layer" in data

    def test_per_layer_entries_have_required_keys(self, capsys):
        import json

        from torchbridge.cli.validate import ValidateCommand
        args = self._make_args()
        ValidateCommand._run_compare(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        for entry in data.get("per_layer", []):
            assert "layer" in entry
            assert "max_diff" in entry
            assert "cosine_sim" in entry
            assert "exceeds_threshold" in entry

    def test_per_layer_max_diff_non_negative(self, capsys):
        """max_diff values must never be negative (abs difference property)."""
        import json

        from torchbridge.cli.validate import ValidateCommand
        args = self._make_args()
        ValidateCommand._run_compare(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        for entry in data.get("per_layer", []):
            assert entry["max_diff"] >= 0.0
