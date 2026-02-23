"""
Security tests for model serialization and safe loading.

Verifies that all CLI commands and source modules default to safe model loading
(weights_only=True) and only allow unsafe loading when explicitly requested via
--trust-source. Extends test_cli_safe_loading.py to cover tb-quantize.
"""

import argparse
import inspect
import re

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def unsafe_model_path(tmp_path):
    """Full nn.Module saved with pickle — requires weights_only=False to load."""
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    path = tmp_path / "full_model.pt"
    torch.save(model, path)
    return path


@pytest.fixture
def state_dict_path(tmp_path):
    """State dict saved — loadable with weights_only=True."""
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    path = tmp_path / "state_dict.pt"
    torch.save(model.state_dict(), path)
    return path


class TestQuantizeSafeLoading:
    """tb-quantize must now respect --trust-source (bug fix in v0.5.32)."""

    def test_quantize_rejects_unsafe_model_by_default(self, unsafe_model_path):
        """QuantizeCommand._load_model returns None (safe failure) on full model without trust_source."""
        from torchbridge.cli.quantize import QuantizeCommand

        # weights_only=True (default) rejects full-model pickles — returns None, does not raise
        result = QuantizeCommand._load_model(
            str(unsafe_model_path), verbose=False, trust_source=False
        )
        assert result is None, (
            "Should return None when loading an unsafe model without trust_source"
        )

    def test_quantize_allows_unsafe_model_with_trust_source(self, unsafe_model_path):
        """QuantizeCommand._load_model succeeds with trust_source=True."""
        from torchbridge.cli.quantize import QuantizeCommand

        model = QuantizeCommand._load_model(
            str(unsafe_model_path), verbose=False, trust_source=True
        )
        assert isinstance(model, nn.Module)

    def test_quantize_trust_source_in_help(self):
        """--trust-source flag must appear in tb-quantize help output."""
        from torchbridge.cli.quantize import QuantizeCommand

        sub_parser = argparse.ArgumentParser()
        QuantizeCommand.register(
            type("SP", (), {"add_parser": lambda *a, **k: sub_parser})()
        )
        assert "--trust-source" in sub_parser.format_help()

    def test_quantize_execute_rejects_unsafe_model_no_trust(self, unsafe_model_path, tmp_path):
        """Full CLI execute returns error code when loading unsafe model without --trust-source."""
        from torchbridge.cli.quantize import QuantizeCommand

        args = argparse.Namespace(
            model=str(unsafe_model_path),
            strategy="auto",
            format="auto",
            backend="auto",
            output=None,
            validate=False,
            calibration_samples=512,
            trust_source=False,
            verbose=False,
            ci=True,
        )
        result = QuantizeCommand.execute(args)
        assert result == 1, "Quantize must return error code when loading unsafe model"


class TestNoUngatedWeightsOnlyFalseAllCLI:
    """No CLI command file may have ungated weights_only=False (post v0.5.32 fix)."""

    def test_no_ungated_weights_only_false_in_quantize(self):
        """QuantizeCommand must not have literal weights_only=False after the fix."""
        from torchbridge.cli.quantize import QuantizeCommand

        source = inspect.getsource(QuantizeCommand)
        matches = re.findall(
            r"torch\.load\([^)]*weights_only\s*=\s*False", source
        )
        assert len(matches) == 0, (
            f"QuantizeCommand has ungated weights_only=False: {matches}"
        )

    def test_no_ungated_weights_only_false_in_export(self):
        """ExportCommand must not have ungated weights_only=False."""
        from torchbridge.cli.export import ExportCommand

        source = inspect.getsource(ExportCommand)
        matches = re.findall(
            r"torch\.load\([^)]*weights_only\s*=\s*False", source
        )
        assert len(matches) == 0, (
            f"ExportCommand has ungated weights_only=False: {matches}"
        )

    def test_no_ungated_weights_only_false_in_optimize(self):
        """OptimizeCommand must not have ungated weights_only=False."""
        from torchbridge.cli.optimize import OptimizeCommand

        source = inspect.getsource(OptimizeCommand)
        matches = re.findall(
            r"torch\.load\([^)]*weights_only\s*=\s*False", source
        )
        assert len(matches) == 0, (
            f"OptimizeCommand has ungated weights_only=False: {matches}"
        )

    def test_no_ungated_weights_only_false_in_profile(self):
        """ProfileCommand must not have ungated weights_only=False."""
        from torchbridge.cli.profile import ProfileCommand

        source = inspect.getsource(ProfileCommand)
        matches = re.findall(
            r"torch\.load\([^)]*weights_only\s*=\s*False", source
        )
        assert len(matches) == 0, (
            f"ProfileCommand has ungated weights_only=False: {matches}"
        )


class TestSourceFileSafeLoading:
    """Core source modules must always use weights_only=True for torch.load."""

    def test_checkpoint_manager_uses_weights_only_true(self):
        """checkpoint/manager.py uses weights_only=True for torch.load calls."""
        from torchbridge.checkpoint import manager as checkpoint_manager

        source = inspect.getsource(checkpoint_manager)
        # Find all torch.load calls
        load_calls = re.findall(r"torch\.load\([^)]+\)", source)
        unsafe = [c for c in load_calls if "weights_only=False" in c]
        assert len(unsafe) == 0, (
            f"checkpoint/manager.py has unsafe torch.load calls: {unsafe}"
        )

    def test_torchserve_handler_uses_weights_only_true(self):
        """deployment/serving/torchserve_handler.py uses weights_only=True."""
        from torchbridge.deployment.serving import torchserve_handler

        source = inspect.getsource(torchserve_handler)
        load_calls = re.findall(r"torch\.load\([^)]+\)", source)
        unsafe = [c for c in load_calls if "weights_only=False" in c]
        assert len(unsafe) == 0, (
            f"torchserve_handler.py has unsafe torch.load calls: {unsafe}"
        )

    def test_trainium_backend_uses_weights_only_true(self):
        """backends/trainium/trainium_backend.py uses weights_only=True."""
        from torchbridge.backends.trainium import trainium_backend

        source = inspect.getsource(trainium_backend)
        load_calls = re.findall(r"torch\.load\([^)]+\)", source)
        unsafe = [c for c in load_calls if "weights_only=False" in c]
        assert len(unsafe) == 0, (
            f"trainium_backend.py has unsafe torch.load calls: {unsafe}"
        )

    def test_tpu_backend_uses_weights_only_true(self):
        """backends/tpu/tpu_backend.py uses weights_only=True."""
        from torchbridge.backends.tpu import tpu_backend

        source = inspect.getsource(tpu_backend)
        load_calls = re.findall(r"torch\.load\([^)]+\)", source)
        unsafe = [c for c in load_calls if "weights_only=False" in c]
        assert len(unsafe) == 0, (
            f"tpu_backend.py has unsafe torch.load calls: {unsafe}"
        )
