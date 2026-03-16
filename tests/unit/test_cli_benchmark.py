"""
Unit tests for tb-benchmark CLI fixes (v0.5.75).

Covers:
- _load_model() raises ValueError on unknown model name (instead of silent fallback)
- _apply_optimization() calls CompileCompatibility.get_compile_mode when level='compile'
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn


class TestLoadModelUnknownRaises:
    """_load_model must raise ValueError on unknown model name."""

    def test_unknown_model_raises_value_error(self):
        """A model name that is not a file and not a predefined name must raise ValueError."""
        from torchbridge.cli.benchmark import BenchmarkCommand

        device = torch.device("cpu")
        with pytest.raises(ValueError, match="nonexistent_xyz_model"):
            BenchmarkCommand._load_model("nonexistent_xyz_model", device)

    def test_error_message_lists_predefined_names(self):
        """The ValueError message must mention the valid predefined names."""
        from torchbridge.cli.benchmark import BenchmarkCommand

        device = torch.device("cpu")
        with pytest.raises(ValueError, match="linear_stress_test|resnet50"):
            BenchmarkCommand._load_model("definitely_not_a_model", device)

    def test_known_predefined_model_still_works(self):
        """linear_stress_test must still load without error."""
        from torchbridge.cli.benchmark import BenchmarkCommand

        device = torch.device("cpu")
        model = BenchmarkCommand._load_model("linear_stress_test", device)
        assert isinstance(model, nn.Module)


class TestApplyOptimizationUsesCompat:
    """_apply_optimization must consult CompileCompatibility when level='compile'."""

    def test_compile_level_calls_compat_get_compile_mode(self):
        """When level='compile', CompileCompatibility.get_compile_mode must be called."""
        from torchbridge.cli.benchmark import BenchmarkCommand

        model = nn.Linear(8, 4).eval()
        device = torch.device("cpu")

        with patch(
            "torchbridge.cli.benchmark.CompileCompatibility.get_compile_mode",
            return_value="default",
        ) as mock_mode:
            with patch("torch.compile", return_value=model):
                BenchmarkCommand._apply_optimization(model, "compile", (1, 8), device)

        mock_mode.assert_called_once()

    def test_compile_passes_mode_from_compat(self):
        """The compile mode returned by CompileCompatibility must be passed to torch.compile."""
        from torchbridge.cli.benchmark import BenchmarkCommand

        model = nn.Linear(8, 4).eval()
        device = torch.device("cpu")

        with patch(
            "torchbridge.cli.benchmark.CompileCompatibility.get_compile_mode",
            return_value="reduce-overhead",
        ):
            with patch("torch.compile", return_value=model) as mock_compile:
                BenchmarkCommand._apply_optimization(model, "compile", (1, 8), device)

        call_kwargs = mock_compile.call_args
        # mode kwarg must match what compat returned
        assert call_kwargs.kwargs.get("mode") == "reduce-overhead" or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == "reduce-overhead"
        )

    def test_basic_level_does_not_call_compat(self):
        """When level='basic', CompileCompatibility.get_compile_mode must NOT be called."""
        from torchbridge.cli.benchmark import BenchmarkCommand

        model = nn.Linear(8, 4).eval()
        device = torch.device("cpu")

        with patch(
            "torchbridge.cli.benchmark.CompileCompatibility.get_compile_mode"
        ) as mock_mode:
            BenchmarkCommand._apply_optimization(model, "basic", (1, 8), device)

        mock_mode.assert_not_called()
