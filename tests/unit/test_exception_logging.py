"""
Exception logging regression tests.

Verifies that previously-silent bare `except ... pass` paths now emit
logger.debug() calls. These tests FAIL until debug logging is added to:
- core/config.py  (AMD GPU arch detection)
- distributed/fsdp.py  (NCCL version check)
- testing/plugin.py  (pytest_configure backend config)
- testing/otel_exporter.py  (span attribute coercion ×2)
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch


class TestExceptionLogging:
    """Silent exception paths must emit logger.debug after fix."""

    # ------------------------------------------------------------------
    # AMD arch detection (core/config.py)
    # ------------------------------------------------------------------

    def test_amd_arch_detection_failure_logs_debug(self, caplog):
        """AMDConfig._detect_architecture() must log when torch.hip fails."""
        from torchbridge.core.config import AMDArchitecture, AMDConfig

        # Bypass __post_init__ auto-detection by specifying architecture
        cfg = AMDConfig(architecture=AMDArchitecture.CDNA2)

        # Build a mock torch where .hip is present but .get_device_properties raises
        mock_hip = MagicMock()
        mock_hip.is_available.return_value = True
        mock_hip.get_device_properties.side_effect = RuntimeError("mock AMD GPU error")
        mock_torch = MagicMock()
        mock_torch.hip = mock_hip

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with caplog.at_level(logging.DEBUG, logger="torchbridge.core.config"):
                result = cfg._detect_architecture()

        assert result == AMDArchitecture.CDNA2, "must return default on error"
        debug_messages = [
            r.message for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(
            "amd" in m.lower() or "detection" in m.lower() or "arch" in m.lower()
            for m in debug_messages
        ), f"Expected AMD detection debug log, got: {debug_messages}"

    # ------------------------------------------------------------------
    # NCCL version check (distributed/fsdp.py)
    # ------------------------------------------------------------------

    def test_nccl_check_failure_logs_debug(self, caplog):
        """FSDPManager._supports_float8_all_gather() must log when NCCL check raises."""
        from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture
        from torchbridge.distributed.fsdp import FSDPManager

        # HOPPER is in _FLOAT8_ALLGATHER_ARCHS; CUDA backend passes the first guard
        manager = FSDPManager(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )

        with patch(
            "torch.distributed.get_nccl_version",
            side_effect=RuntimeError("mock NCCL error"),
            create=True,
        ):
            with caplog.at_level(logging.DEBUG, logger="torchbridge.distributed.fsdp"):
                result = manager._supports_float8_all_gather()

        assert result is False, "must return False when NCCL check raises"
        debug_messages = [
            r.message for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(
            "nccl" in m.lower() or "float8" in m.lower() or "failed" in m.lower()
            for m in debug_messages
        ), f"Expected NCCL debug log, got: {debug_messages}"

    # ------------------------------------------------------------------
    # pytest plugin configure (testing/plugin.py)
    # ------------------------------------------------------------------

    def test_plugin_configure_failure_logs_debug(self, caplog):
        """pytest_configure() must log when config attribute access fails."""
        from torchbridge.testing import plugin as plugin_module

        mock_config = MagicMock()
        mock_config.getoption.side_effect = RuntimeError("mock pytest config error")

        with caplog.at_level(logging.DEBUG, logger="torchbridge.testing.plugin"):
            # Must not raise — exception is caught and logged
            plugin_module.pytest_configure(mock_config)

        debug_messages = [
            r.message for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(
            "config" in m.lower() or "backend" in m.lower() or "failed" in m.lower()
            for m in debug_messages
        ), f"Expected plugin config debug log, got: {debug_messages}"

    # ------------------------------------------------------------------
    # OTel exporter span attribute coercion (testing/otel_exporter.py)
    # ------------------------------------------------------------------

    def test_otel_exporter_bad_top_level_attribute_logs_debug(self, caplog):
        """export() must log debug when a span attribute coercion raises ValueError."""
        from torchbridge.testing.otel_exporter import ValidationSpanExporter

        # Build exporter without real OTel — bypass __init__ and inject mock tracer
        exporter = ValidationSpanExporter.__new__(ValidationSpanExporter)
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = lambda s, *a: (
            mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )
        exporter._tracer = mock_tracer

        # "max_diff" expects float; "not-a-number" causes float("not-a-number") → ValueError
        result = {"max_diff": "not-a-number"}

        with caplog.at_level(logging.DEBUG, logger="torchbridge.testing.otel_exporter"):
            exporter.export(result)  # must not raise

        debug_messages = [
            r.message for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(
            "skipped" in m.lower()
            or "coercion" in m.lower()
            or "attribute" in m.lower()
            for m in debug_messages
        ), f"Expected span attribute coercion debug log, got: {debug_messages}"

    def test_otel_exporter_bad_layer_attribute_logs_debug(self, caplog):
        """export() must log debug when a per-layer span attribute coercion fails."""
        from torchbridge.testing.otel_exporter import ValidationSpanExporter

        exporter = ValidationSpanExporter.__new__(ValidationSpanExporter)
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = lambda s, *a: (
            mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )
        exporter._tracer = mock_tracer

        # "max_diff" in per-layer row expects float; bad value triggers coercion error
        result = {
            "per_layer": [{"max_diff": "not-a-number"}],
        }

        with caplog.at_level(logging.DEBUG, logger="torchbridge.testing.otel_exporter"):
            exporter.export(result)  # must not raise

        debug_messages = [
            r.message for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(
            "skipped" in m.lower()
            or "coercion" in m.lower()
            or "attribute" in m.lower()
            for m in debug_messages
        ), (
            f"Expected per-layer span attribute coercion debug log, got: {debug_messages}"
        )
