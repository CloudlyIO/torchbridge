"""
Tests for silent feature downgrade logging.

Verifies that feature downgrades are visible at WARNING/INFO level, not silently
swallowed at DEBUG. These tests FAIL until log levels are elevated in engine.py
and dispatcher.py.
"""

import logging
from unittest.mock import patch


class TestDowngradeLogging:
    """Feature downgrades must be visible in logs, not buried at DEBUG."""

    def test_torchao_int8_failure_logs_warning(self, caplog):
        """When torchao INT8 raises, a WARNING must appear in the log."""
        from torchbridge.core.config import HardwareBackend
        from torchbridge.precision.engine import QuantizationEngine
        from torchbridge.precision.torchao_integration import (
            TorchAOBackend,
        )

        engine = QuantizationEngine.__new__(QuantizationEngine)
        engine._backend = None
        engine._hw_backend = HardwareBackend.CUDA
        engine._architecture = None

        with (
            patch.object(TorchAOBackend, "is_available_on_backend", return_value=True),
            patch.object(
                TorchAOBackend,
                "quantize_int8_dynamic",
                side_effect=RuntimeError("simulated torchao failure"),
            ),
            patch(
                "torch.quantization.quantize_dynamic",
                side_effect=RuntimeError("quantize_dynamic unavailable"),
            ),
            caplog.at_level(logging.WARNING, logger="torchbridge.precision.engine"),
        ):
            import torch.nn as nn

            model = nn.Linear(4, 4)
            try:
                engine._apply_int8_dynamic(model)
            except Exception:
                pass

        assert any(
            "WARNING" in r.levelname or r.levelno >= logging.WARNING
            for r in caplog.records
        ), "Expected a WARNING log when torchao INT8 fails; got only: " + str(
            [r.levelname + ": " + r.message for r in caplog.records]
        )

    def test_torchao_unsupported_backend_logs_info(self, caplog):
        """When torchao is present but backend is not cuda/rocm, INFO must appear."""
        from torchbridge.core.config import HardwareBackend
        from torchbridge.precision.engine import QuantizationEngine
        from torchbridge.precision.torchao_integration import (
            TorchAOBackend,
        )

        engine = QuantizationEngine.__new__(QuantizationEngine)
        engine._backend = None
        engine._hw_backend = HardwareBackend.CPU
        engine._architecture = None

        with (
            patch.object(TorchAOBackend, "is_available_on_backend", return_value=False),
            patch("torchbridge.precision.engine.TORCHAO_AVAILABLE", True),
            patch(
                "torch.quantization.quantize_dynamic",
                side_effect=RuntimeError("quantize_dynamic unavailable"),
            ),
            caplog.at_level(logging.INFO, logger="torchbridge.precision.engine"),
        ):
            import torch.nn as nn

            model = nn.Linear(4, 4)
            try:
                engine._apply_int8_dynamic(model)
            except Exception:
                pass

        assert any(r.levelno >= logging.INFO for r in caplog.records), (
            "Expected at least INFO log when torchao is present but backend unsupported; "
            "got: " + str([r.levelname + ": " + r.message for r in caplog.records])
        )

    def test_kernel_downgrade_logs_warning(self, caplog):
        """When a kernel is not available at runtime, a WARNING must appear in logs."""
        from unittest.mock import patch

        from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher
        from torchbridge.attention.dispatch.kernel_types import AttentionKernelType

        dispatcher = AttentionDispatcher(use_benchmark_cache=False)

        # Force supported list to [FLEX_ATTENTION, PYTORCH_SDPA] so fallback fires
        def patched_supported(backend, architecture=None):
            return [
                AttentionKernelType.FLEX_ATTENTION,
                AttentionKernelType.PYTORCH_SDPA,
            ]

        def patched_check(kernel_type: AttentionKernelType) -> bool:
            return kernel_type == AttentionKernelType.PYTORCH_SDPA

        with (
            patch.object(
                AttentionDispatchMatrix, "get_supported_kernels", patched_supported
            ),
            patch.object(dispatcher, "_check_kernel_availability", patched_check),
            caplog.at_level(
                logging.WARNING, logger="torchbridge.attention.dispatch.dispatcher"
            ),
        ):
            result = dispatcher.select_kernel(seq_length=64, num_heads=2, head_dim=16)

        warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_logs, (
            "Expected WARNING log when kernel is unavailable and fallback is used; "
            f"result.warnings={result.warnings}, "
            "but nothing was emitted at WARNING level. "
            "Add logger.warning(...) alongside result_warnings.append(...) in dispatcher.py"
        )
