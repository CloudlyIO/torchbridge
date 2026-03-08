"""
Security tests for CLI model loading.

Verifies that CLI commands default to weights_only=True (safe loading)
and only allow weights_only=False when --trust-source is explicitly passed.
"""

import argparse

import pytest
import torch

from torchbridge.cli.quantize import QuantizeCommand


@pytest.fixture
def unsafe_model_path(tmp_path):
    """Create a model file saved as a full nn.Module (requires weights_only=False to load)."""
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )
    path = tmp_path / "full_model.pt"
    torch.save(model, path)
    return path


@pytest.fixture
def safe_model_path(tmp_path):
    """Create a model file saved as a state dict (loadable with weights_only=True)."""
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )
    path = tmp_path / "state_dict_model.pt"
    torch.save(model.state_dict(), path)
    return path


class TestQuantizeSafeLoading:
    """Test that quantize command defaults to safe loading."""

    def test_quantize_rejects_unsafe_model_by_default(self, unsafe_model_path):
        """Quantize should fail on full model file without --trust-source."""
        result = QuantizeCommand._load_model(
            str(unsafe_model_path), verbose=False, trust_source=False
        )
        # _load_model catches the exception and returns None when safe loading fails
        assert result is None

    def test_quantize_allows_unsafe_model_with_trust_source(self, unsafe_model_path):
        """Quantize should succeed on full model file with --trust-source."""
        model = QuantizeCommand._load_model(
            str(unsafe_model_path), verbose=False, trust_source=True
        )
        assert isinstance(model, torch.nn.Module)

    def test_quantize_trust_source_in_help(self):
        """Verify --trust-source appears in quantize help."""
        sub_parser = argparse.ArgumentParser()
        QuantizeCommand.register(
            type('SP', (), {'add_parser': lambda *a, **k: sub_parser})()
        )
        assert '--trust-source' in sub_parser.format_help()


class TestNoUngatedWeightsOnlyFalse:
    """Verify no ungated weights_only=False remains in source."""

    def test_no_ungated_weights_only_false_in_cli(self):
        """Ensure torch.load never uses literal weights_only=False in code."""
        import inspect
        import re

        for cls in [QuantizeCommand]:
            source = inspect.getsource(cls)
            torch_load_calls = re.findall(
                r'torch\.load\([^)]*weights_only\s*=\s*False', source
            )
            assert len(torch_load_calls) == 0, (
                f"{cls.__name__} has torch.load with literal weights_only=False: "
                f"{torch_load_calls}"
            )
