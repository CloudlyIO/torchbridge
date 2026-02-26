"""
Tests for AMDAdapter wiring in AMDBackend

Verifies that optimize_for_inference() and optimize_for_training() call
AMDAdapter when an AMD GPU is present, and skip it in CPU fallback mode.
"""

from unittest.mock import MagicMock, patch

import torch.nn as nn

from torchbridge.backends.amd.amd_adapter import AMDAdapter
from torchbridge.backends.amd.amd_backend import AMDBackend
from torchbridge.core.config import AMDArchitecture, AMDConfig


def _make_cpu_backend() -> AMDBackend:
    """Create an AMDBackend in CPU fallback mode (no ROCm hardware required)."""
    backend = AMDBackend(AMDConfig())
    # Ensure CPU fallback regardless of test environment
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


class TestAdapterCalledWhenAMDPresent:
    """Verify AMDAdapter is invoked when an AMD GPU device is detected."""

    def test_optimize_for_inference_calls_adapter(self):
        """optimize_for_inference must call AMDAdapter.optimize with 'balanced'."""
        backend = _make_amd_backend_mock()
        model = _simple_model()

        with (
            patch.object(backend, "prepare_model", return_value=model),
            patch(
                "torchbridge.backends.amd.amd_backend.AMDAdapter"
            ) as MockAdapter,
        ):
            mock_instance = MagicMock()
            mock_instance.optimize.return_value = model
            MockAdapter.return_value = mock_instance

            backend.optimize_for_inference(model)

        MockAdapter.assert_called_once_with(backend._amd_config)
        mock_instance.optimize.assert_called_once_with(model, level="balanced")

    def test_optimize_for_training_calls_adapter(self):
        """optimize_for_training must call AMDAdapter.optimize with 'conservative'."""
        backend = _make_amd_backend_mock()
        model = _simple_model()

        with (
            patch.object(backend, "prepare_model", return_value=model),
            patch(
                "torchbridge.backends.amd.amd_backend.AMDAdapter"
            ) as MockAdapter,
        ):
            mock_instance = MagicMock()
            mock_instance.optimize.return_value = model
            MockAdapter.return_value = mock_instance

            backend.optimize_for_training(model)

        MockAdapter.assert_called_once_with(backend._amd_config)
        mock_instance.optimize.assert_called_once_with(model, level="conservative")

    def test_optimize_for_inference_returns_model(self):
        """optimize_for_inference must return a model even when adapter is called."""
        backend = _make_amd_backend_mock()
        model = _simple_model()

        with (
            patch.object(backend, "prepare_model", return_value=model),
            patch("torchbridge.backends.amd.amd_backend.AMDAdapter") as MockAdapter,
        ):
            mock_instance = MagicMock()
            mock_instance.optimize.return_value = model
            MockAdapter.return_value = mock_instance

            result = backend.optimize_for_inference(model)

        assert isinstance(result, nn.Module)

    def test_optimize_for_training_returns_model(self):
        """optimize_for_training must return a model even when adapter is called."""
        backend = _make_amd_backend_mock()
        model = _simple_model()

        with (
            patch.object(backend, "prepare_model", return_value=model),
            patch("torchbridge.backends.amd.amd_backend.AMDAdapter") as MockAdapter,
        ):
            mock_instance = MagicMock()
            mock_instance.optimize.return_value = model
            MockAdapter.return_value = mock_instance

            result = backend.optimize_for_training(model)

        assert isinstance(result, nn.Module)


class TestAdapterSkippedInCPUFallback:
    """Verify AMDAdapter is NOT invoked when CPU fallback is active."""

    def test_optimize_for_inference_skips_adapter_on_cpu_fallback(self):
        """AMDAdapter must not be called when _cpu_fallback is True."""
        backend = _make_cpu_backend()
        model = _simple_model()

        with patch(
            "torchbridge.backends.amd.amd_backend.AMDAdapter"
        ) as MockAdapter:
            backend.optimize_for_inference(model)

        MockAdapter.assert_not_called()

    def test_optimize_for_training_skips_adapter_on_cpu_fallback(self):
        """AMDAdapter must not be called when _cpu_fallback is True."""
        backend = _make_cpu_backend()
        model = _simple_model()

        with patch(
            "torchbridge.backends.amd.amd_backend.AMDAdapter"
        ) as MockAdapter:
            backend.optimize_for_training(model)

        MockAdapter.assert_not_called()

    def test_optimize_for_inference_skips_adapter_when_no_device(self):
        """AMDAdapter must not be called when _current_amd_device is None."""
        backend = AMDBackend(AMDConfig())
        backend._cpu_fallback = False
        backend._current_amd_device = None
        model = _simple_model()

        with (
            patch.object(backend, "prepare_model", return_value=model),
            patch(
                "torchbridge.backends.amd.amd_backend.AMDAdapter"
            ) as MockAdapter,
        ):
            backend.optimize_for_inference(model)

        MockAdapter.assert_not_called()


class TestAdapterFailureGraceful:
    """Verify that AMDAdapter failure is caught and does not propagate."""

    def test_inference_continues_if_adapter_raises(self):
        """If AMDAdapter.optimize raises, optimize_for_inference must still return model."""
        backend = _make_amd_backend_mock()
        model = _simple_model()

        with (
            patch.object(backend, "prepare_model", return_value=model),
            patch(
                "torchbridge.backends.amd.amd_backend.AMDAdapter"
            ) as MockAdapter,
        ):
            mock_instance = MagicMock()
            mock_instance.optimize.side_effect = RuntimeError("ROCm kernel error")
            MockAdapter.return_value = mock_instance

            result = backend.optimize_for_inference(model)

        assert isinstance(result, nn.Module)

    def test_training_continues_if_adapter_raises(self):
        """If AMDAdapter.optimize raises, optimize_for_training must still return model."""
        backend = _make_amd_backend_mock()
        model = _simple_model()

        with (
            patch.object(backend, "prepare_model", return_value=model),
            patch(
                "torchbridge.backends.amd.amd_backend.AMDAdapter"
            ) as MockAdapter,
        ):
            mock_instance = MagicMock()
            mock_instance.optimize.side_effect = RuntimeError("ROCm kernel error")
            MockAdapter.return_value = mock_instance

            result = backend.optimize_for_training(model)

        assert isinstance(result, nn.Module)
