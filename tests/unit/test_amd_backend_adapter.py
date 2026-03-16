"""
Tests for AMDBackend optimize_for_inference / optimize_for_training

Verifies device-placement-only prepare_model, eval/train mode switches,
param freezing, and arch-aware compile mode selection.  No AMDAdapter
delegation — that was removed in v0.5.74 (Contraction IX).
"""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from torchbridge.backends.amd.amd_adapter import AMDAdapter
from torchbridge.backends.amd.amd_backend import AMDBackend
from torchbridge.core.config import AMDArchitecture, AMDConfig


def _make_cpu_backend() -> AMDBackend:
    """Create an AMDBackend in CPU fallback mode (no ROCm hardware required)."""
    backend = AMDBackend(AMDConfig())
    backend._cpu_fallback = True
    backend._current_amd_device = None
    return backend


def _make_amd_backend_mock() -> AMDBackend:
    """Create an AMDBackend with mocked AMD device (no actual GPU needed)."""
    backend = AMDBackend(AMDConfig(architecture=AMDArchitecture.CDNA3))
    backend._cpu_fallback = False
    backend._current_amd_device = MagicMock()
    backend._current_amd_device.architecture = AMDArchitecture.CDNA3
    return backend


def _simple_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4))


class TestAMDAdapterImport:
    """Basic import and instantiation tests."""

    def test_amd_adapter_importable(self):
        """AMDAdapter should be importable from the AMD package."""
        assert AMDAdapter is not None

    def test_amd_adapter_instantiates_with_config(self):
        """AMDAdapter should accept an AMDConfig."""
        config = AMDConfig()
        adapter = AMDAdapter(config)
        assert adapter.config is config


class TestAMDOptimizeForInference:
    """optimize_for_inference: eval mode, param freeze, returns nn.Module."""

    def test_returns_nn_module(self):
        """optimize_for_inference must return an nn.Module."""
        backend = _make_cpu_backend()
        model = _simple_model()
        result = backend.optimize_for_inference(model)
        assert isinstance(result, nn.Module)

    def test_model_in_eval_mode(self):
        """optimize_for_inference must set the model to eval mode."""
        backend = _make_cpu_backend()
        model = _simple_model()
        result = backend.optimize_for_inference(model)
        assert not result.training

    def test_params_frozen(self):
        """optimize_for_inference must freeze all parameters."""
        backend = _make_cpu_backend()
        model = _simple_model()
        result = backend.optimize_for_inference(model)
        for param in result.parameters():
            assert not param.requires_grad

    def test_amd_device_path_returns_model(self):
        """optimize_for_inference on AMD path also returns eval-mode model."""
        backend = _make_amd_backend_mock()
        model = _simple_model()
        with patch.object(backend, "prepare_model", return_value=model):
            result = backend.optimize_for_inference(model)
        assert isinstance(result, nn.Module)
        assert not result.training


class TestAMDOptimizeForTraining:
    """optimize_for_training: train mode, optional optimizer tuple."""

    def test_returns_nn_module(self):
        """optimize_for_training must return an nn.Module."""
        backend = _make_cpu_backend()
        model = _simple_model()
        result = backend.optimize_for_training(model)
        assert isinstance(result, nn.Module)

    def test_model_in_train_mode(self):
        """optimize_for_training must set the model to train mode."""
        backend = _make_cpu_backend()
        model = _simple_model()
        result = backend.optimize_for_training(model)
        assert result.training

    def test_returns_tuple_with_optimizer(self):
        """When optimizer is provided, a (model, optimizer) tuple is returned."""
        backend = _make_cpu_backend()
        model = _simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        result = backend.optimize_for_training(model, optimizer=optimizer)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], nn.Module)

    def test_amd_device_path_returns_model(self):
        """optimize_for_training on AMD path also returns train-mode model."""
        backend = _make_amd_backend_mock()
        model = _simple_model()
        with patch.object(backend, "prepare_model", return_value=model):
            result = backend.optimize_for_training(model)
        assert isinstance(result, nn.Module)
        assert result.training


class TestAMDPrepareModel:
    """prepare_model: device placement only."""

    def test_cpu_fallback_returns_model(self):
        """prepare_model in CPU fallback mode returns a model."""
        backend = _make_cpu_backend()
        model = _simple_model()
        result = backend.prepare_model(model)
        assert isinstance(result, nn.Module)

    def test_cpu_fallback_places_on_cpu(self):
        """prepare_model in CPU fallback mode places model on CPU."""
        backend = _make_cpu_backend()
        model = _simple_model()
        result = backend.prepare_model(model)
        for param in result.parameters():
            assert param.device.type == "cpu"
