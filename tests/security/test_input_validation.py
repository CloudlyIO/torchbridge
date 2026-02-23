"""
Security tests for input validation.

Verifies that TorchBridge APIs handle malformed, out-of-range, or invalid
inputs gracefully — raising clean exceptions or degrading safely rather than
crashing ungracefully or leaking internals.
"""

import argparse
import time

import pytest
import torch.nn as nn


class TestQuantizationFormatValidation:
    """QuantizationFormat.from_string() must reject unknown format strings cleanly."""

    def test_invalid_format_string_raises_value_error(self):
        """Unknown quantization format string raises ValueError with descriptive message."""
        from torchbridge.precision.quantization.formats import QuantizationFormat

        with pytest.raises(ValueError, match="Unknown quantization format"):
            QuantizationFormat.from_string("totally_invalid_format_xyz")

    def test_valid_format_strings_parse_correctly(self):
        """All documented format strings parse without error."""
        from torchbridge.precision.quantization.formats import QuantizationFormat

        for fmt in QuantizationFormat:
            parsed = QuantizationFormat.from_string(fmt.value)
            assert parsed == fmt

    def test_quantize_with_invalid_format_does_not_crash(self):
        """QuantizationEngine.quantize() with invalid format string never propagates uncaught."""
        from torchbridge.precision.quantization import QuantizationEngine

        model = nn.Linear(32, 16)
        engine = QuantizationEngine()
        # Must not raise an unhandled exception — either returns error result or raises ValueError cleanly
        try:
            result = engine.quantize(model, format="not_a_real_format_xyz")
            # If it didn't raise, the result must indicate failure
            assert not result.success or result.errors, (
                "Engine should report failure for unknown format, not silently succeed"
            )
        except ValueError as e:
            # A clean ValueError is acceptable
            assert "format" in str(e).lower() or "unknown" in str(e).lower(), (
                f"ValueError message should mention the format issue: {e}"
            )


class TestAttentionDispatcherInputs:
    """AttentionDispatcher.select_kernel() must degrade gracefully on odd inputs."""

    def test_non_power_of_two_head_dim_does_not_crash(self):
        """head_dim=7 is non-standard but must not raise an unhandled exception."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        dispatcher = AttentionDispatcher()
        # Must not raise — should fall back to pytorch_sdpa
        result = dispatcher.select_kernel(seq_length=128, num_heads=8, head_dim=7)
        assert result is not None
        assert result.kernel_type is not None

    def test_very_large_seq_length_does_not_hang(self):
        """select_kernel() with very large seq_length returns quickly (no blocking)."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        dispatcher = AttentionDispatcher()
        t0 = time.perf_counter()
        result = dispatcher.select_kernel(seq_length=1_000_000, num_heads=8, head_dim=64)
        elapsed = time.perf_counter() - t0
        assert result is not None
        assert elapsed < 5.0, f"select_kernel should return quickly, took {elapsed:.1f}s"

    def test_zero_num_heads_does_not_crash(self):
        """select_kernel() with num_heads=0 must not crash ungracefully."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        dispatcher = AttentionDispatcher()
        try:
            result = dispatcher.select_kernel(seq_length=128, num_heads=0, head_dim=64)
            assert result is not None
        except (ValueError, ZeroDivisionError):
            pass  # Clean exception is also acceptable


class TestBackendFactoryInputs:
    """BackendFactory.create() must be robust to unknown or invalid backend strings."""

    def test_unknown_backend_string_returns_cpu_backend(self):
        """Unknown backend string falls back to CPU backend, never raises."""
        from torchbridge.backends.backend_factory import BackendFactory

        backend = BackendFactory.create("totally_unknown_backend_xyz")
        assert backend is not None
        # Must return a working backend (CPU fallback)
        device_info = backend.get_device_info()
        assert device_info is not None

    def test_empty_string_backend_does_not_crash(self):
        """Empty string backend name is handled gracefully."""
        from torchbridge.backends.backend_factory import BackendFactory

        try:
            backend = BackendFactory.create("")
            assert backend is not None
        except (ValueError, KeyError, AttributeError):
            pass  # Clean exception acceptable


class TestCLIPathInputs:
    """CLI commands must handle path-traversal or non-existent model paths safely."""

    def test_quantize_path_traversal_returns_error_not_crash(self, tmp_path):
        """Path traversal in --model arg returns error code, not unhandled exception."""
        from torchbridge.cli.quantize import QuantizeCommand

        args = argparse.Namespace(
            model="../../etc/passwd",
            strategy="auto",
            format="auto",
            backend="auto",
            output=None,
            validate=False,
            calibration_samples=512,
            trust_source=False,
            verbose=False,
            ci=False,
        )
        result = QuantizeCommand.execute(args)
        assert result == 1, "CLI should return error code 1 for non-existent model path"

    def test_quantize_nonexistent_model_path_returns_error(self, tmp_path):
        """Non-existent model path returns error code cleanly."""
        from torchbridge.cli.quantize import QuantizeCommand

        args = argparse.Namespace(
            model=str(tmp_path / "does_not_exist.pt"),
            strategy="auto",
            format="auto",
            backend="auto",
            output=None,
            validate=False,
            calibration_samples=512,
            trust_source=False,
            verbose=False,
            ci=False,
        )
        result = QuantizeCommand.execute(args)
        assert result == 1
