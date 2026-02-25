"""
Tests for NVIDIABackend Tensor Core alignment — facade fix

Verifies that _optimize_for_tensor_cores() actually replaces misaligned
Linear layers with _TensorCoreAlignedLinear wrappers that produce
numerically identical outputs.
"""

import torch
import torch.nn as nn

from torchbridge.backends.nvidia.nvidia_backend import (
    NVIDIABackend,
    _ceil_to_multiple,
    _TensorCoreAlignedLinear,
)


class TestCeilToMultiple:
    """Tests for _ceil_to_multiple helper."""

    def test_already_aligned(self):
        assert _ceil_to_multiple(16, 16) == 16

    def test_needs_rounding(self):
        assert _ceil_to_multiple(15, 16) == 16

    def test_zero(self):
        assert _ceil_to_multiple(0, 16) == 0


class TestTensorCoreAlignedLinear:
    """Tests for _TensorCoreAlignedLinear wrapper."""

    def test_produces_same_output_as_original(self):
        """Aligned wrapper produces numerically identical output."""
        original = nn.Linear(15, 13, bias=True)
        aligned = _TensorCoreAlignedLinear(original, optimal_multiple=8)

        x = torch.randn(4, 15)
        with torch.no_grad():
            expected = original(x)
            actual = aligned(x)

        assert actual.shape == expected.shape, "Output shape must match original"
        assert torch.allclose(actual, expected, atol=1e-5), (
            f"Max diff: {(actual - expected).abs().max().item()}"
        )

    def test_output_shape_preserved(self):
        """Aligned wrapper preserves original output dimensions."""
        original = nn.Linear(13, 7, bias=False)
        aligned = _TensorCoreAlignedLinear(original, optimal_multiple=8)

        x = torch.randn(3, 13)
        out = aligned(x)
        assert out.shape == (3, 7)

    def test_padded_dimensions_are_multiples(self):
        """Padded dimensions are multiples of optimal_multiple."""
        original = nn.Linear(13, 7, bias=True)
        aligned = _TensorCoreAlignedLinear(original, optimal_multiple=8)
        assert aligned.padded_in % 8 == 0
        assert aligned.padded_out % 8 == 0

    def test_original_dimensions_preserved(self):
        """orig_in and orig_out match original Linear dimensions."""
        original = nn.Linear(13, 7, bias=True)
        aligned = _TensorCoreAlignedLinear(original, optimal_multiple=8)
        assert aligned.orig_in == 13
        assert aligned.orig_out == 7

    def test_no_bias_handled(self):
        """Aligned wrapper works with bias=False."""
        original = nn.Linear(9, 5, bias=False)
        aligned = _TensorCoreAlignedLinear(original, optimal_multiple=8)
        x = torch.randn(2, 9)
        with torch.no_grad():
            expected = original(x)
            actual = aligned(x)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_batch_dimensions_preserved(self):
        """Aligned wrapper works with batched 3D input."""
        original = nn.Linear(11, 9, bias=True)
        aligned = _TensorCoreAlignedLinear(original, optimal_multiple=8)
        x = torch.randn(2, 5, 11)
        with torch.no_grad():
            expected = original(x)
            actual = aligned(x)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-5)


class TestNVIDIABackendTensorCoreAlignment:
    """Tests for NVIDIABackend._optimize_for_tensor_cores() actual replacement."""

    def _make_backend(self, compute_capability=(8, 0)):
        """Create a NVIDIABackend with mocked compute capability."""
        from unittest.mock import patch

        with patch.object(NVIDIABackend, "__init__", lambda self, config=None: None):
            backend = NVIDIABackend.__new__(NVIDIABackend)
            backend._compute_capability = compute_capability
        return backend

    def test_misaligned_layers_replaced_in_eval_mode(self):
        """Misaligned Linear layers are replaced in eval mode."""
        backend = self._make_backend(compute_capability=(8, 0))

        model = nn.Sequential(
            nn.Linear(13, 7),  # both misaligned
            nn.ReLU(),
        )
        model.eval()
        result = backend._optimize_for_tensor_cores(model)

        # The Linear should have been replaced
        assert isinstance(result[0], _TensorCoreAlignedLinear)

    def test_aligned_layers_not_replaced(self):
        """Already-aligned Linear layers are not replaced."""
        backend = self._make_backend(compute_capability=(8, 0))

        model = nn.Sequential(
            nn.Linear(16, 32),  # both aligned to 16
        )
        model.eval()
        result = backend._optimize_for_tensor_cores(model)

        # Should be unchanged
        assert isinstance(result[0], nn.Linear)
        assert not isinstance(result[0], _TensorCoreAlignedLinear)

    def test_training_mode_skips_replacement(self):
        """Training mode skips Tensor Core replacement (inference-only)."""
        backend = self._make_backend(compute_capability=(8, 0))

        model = nn.Sequential(nn.Linear(13, 7))
        model.train()
        result = backend._optimize_for_tensor_cores(model)

        # No replacement in training mode
        assert isinstance(result[0], nn.Linear)
        assert not isinstance(result[0], _TensorCoreAlignedLinear)

    def test_output_numerically_identical_after_optimization(self):
        """Model output is numerically identical before and after optimization."""
        backend = self._make_backend(compute_capability=(8, 0))

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(13, 7),
            nn.ReLU(),
            nn.Linear(7, 3),
        )
        model.eval()

        x = torch.randn(4, 13)
        with torch.no_grad():
            before = model(x)

        model = backend._optimize_for_tensor_cores(model)
        with torch.no_grad():
            after = model(x)

        assert before.shape == after.shape
        assert torch.allclose(before, after, atol=1e-5), (
            f"Max diff after optimization: {(before - after).abs().max().item()}"
        )

    def test_optimal_divisor_is_16_for_ampere(self):
        """Ampere+ uses divisor 16 (not 8)."""
        backend = self._make_backend(compute_capability=(8, 0))

        model = nn.Sequential(nn.Linear(8, 8))  # aligned to 8 but not 16
        model.eval()
        result = backend._optimize_for_tensor_cores(model)

        # 8 % 16 != 0, so should be replaced with padded_in=16
        assert isinstance(result[0], _TensorCoreAlignedLinear)
        assert result[0].padded_in == 16
        assert result[0].padded_out == 16
