"""
Unit tests for torchbridge.testing.otel_exporter.

Tests are written first (TDD). They define the contract:
- _SPAN_ATTRIBUTE_SCHEMA maps result keys → (otel_attr_name, type)
- _LAYER_SPAN_SCHEMA maps layer row keys → (otel_attr_name, type)
- ValidationSpanExporter raises RuntimeError when opentelemetry is unavailable
- ValidationSpanExporter selects ConsoleSpanExporter when no endpoint configured
- ValidationSpanExporter selects OTLPSpanExporter when endpoint is given or env var set
- export() sets all present result keys as span attributes
- export() creates child spans for per_layer rows
- shutdown() flushes the processor
"""

from unittest.mock import MagicMock, patch

import pytest

# ── Schema tests (no opentelemetry needed) ─────────────────────────────────

class TestSpanSchema:
    def test_schema_has_required_keys(self):
        from torchbridge.testing.otel_exporter import _SPAN_ATTRIBUTE_SCHEMA
        required = {
            "backend1", "backend2", "model", "dtype",
            "max_diff", "cosine_sim", "tolerance_atol", "tolerance_rtol",
            "passed", "duration_ms",
        }
        assert required.issubset(_SPAN_ATTRIBUTE_SCHEMA.keys())

    def test_schema_values_are_tuples_of_str_and_type(self):
        from torchbridge.testing.otel_exporter import _SPAN_ATTRIBUTE_SCHEMA
        for key, val in _SPAN_ATTRIBUTE_SCHEMA.items():
            assert isinstance(val, tuple) and len(val) == 2, key
            attr_name, attr_type = val
            assert isinstance(attr_name, str) and attr_name.startswith("torchbridge."), key
            assert attr_type in (str, float, bool, int), key

    def test_layer_schema_has_required_keys(self):
        from torchbridge.testing.otel_exporter import _LAYER_SPAN_SCHEMA
        required = {"layer", "max_diff", "cosine_sim", "exceeds_threshold"}
        assert required.issubset(_LAYER_SPAN_SCHEMA.keys())

    def test_layer_schema_values_are_valid_tuples(self):
        from torchbridge.testing.otel_exporter import _LAYER_SPAN_SCHEMA
        for key, val in _LAYER_SPAN_SCHEMA.items():
            assert isinstance(val, tuple) and len(val) == 2, key
            attr_name, attr_type = val
            assert isinstance(attr_name, str) and attr_name.startswith("torchbridge."), key

    def test_otel_available_is_bool(self):
        from torchbridge.testing.otel_exporter import OTEL_AVAILABLE
        assert isinstance(OTEL_AVAILABLE, bool)


# ── Unavailable guard ──────────────────────────────────────────────────────

class TestValidationSpanExporterUnavailable:
    def test_raises_runtime_error_when_otel_unavailable(self):
        with patch("torchbridge.testing.otel_exporter._OTEL_AVAILABLE", False):
            from torchbridge.testing.otel_exporter import ValidationSpanExporter
            with pytest.raises(RuntimeError, match="opentelemetry"):
                ValidationSpanExporter()


# ── Main exporter tests (require opentelemetry.sdk) ───────────────────────

otel_sdk = pytest.importorskip("opentelemetry.sdk", reason="opentelemetry-sdk not installed")


class TestValidationSpanExporterInit:
    def test_uses_console_exporter_when_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        exp = ValidationSpanExporter(endpoint=None)
        assert isinstance(exp._raw_exporter, ConsoleSpanExporter)
        exp.shutdown()

    def test_uses_otlp_exporter_when_endpoint_given(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        exp = ValidationSpanExporter(endpoint="http://localhost:4318")
        assert isinstance(exp._raw_exporter, OTLPSpanExporter)
        exp.shutdown()

    def test_uses_otlp_exporter_from_env_var(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-endpoint:4318")
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        exp = ValidationSpanExporter(endpoint=None)
        assert isinstance(exp._raw_exporter, OTLPSpanExporter)
        exp.shutdown()

    def test_explicit_endpoint_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env:4318")
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        exp = ValidationSpanExporter(endpoint="http://explicit:4318")
        assert isinstance(exp._raw_exporter, OTLPSpanExporter)
        assert "explicit" in exp._raw_exporter._endpoint
        exp.shutdown()


class TestValidationSpanExporterExport:
    _RESULT = {
        "backend1": "cuda",
        "backend2": "rocm",
        "model": "smoke",
        "dtype": "float32",
        "max_diff": 1.5e-6,
        "cosine_sim": 0.999999,
        "tolerance_atol": 1e-4,
        "tolerance_rtol": 1e-5,
        "passed": True,
        "duration_ms": 42.1,
        "per_layer": [],
    }

    def _make_exporter(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        return ValidationSpanExporter(endpoint=None)

    def test_export_sets_schema_attributes(self, monkeypatch):
        from torchbridge.testing.otel_exporter import _SPAN_ATTRIBUTE_SCHEMA
        exp = self._make_exporter(monkeypatch)
        recorded = {}

        mock_span = MagicMock()
        mock_span.__enter__ = lambda s: s
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.set_attribute.side_effect = lambda k, v: recorded.__setitem__(k, v)

        with patch.object(exp._tracer, "start_as_current_span", return_value=mock_span):
            exp.export(self._RESULT)

        for key, (attr_name, _) in _SPAN_ATTRIBUTE_SCHEMA.items():
            assert attr_name in recorded, f"Missing attribute {attr_name} for key {key}"

        exp.shutdown()

    def test_export_skips_missing_keys(self, monkeypatch):
        exp = self._make_exporter(monkeypatch)
        partial_result = {"backend1": "cuda", "backend2": "cpu", "per_layer": []}

        mock_span = MagicMock()
        mock_span.__enter__ = lambda s: s
        mock_span.__exit__ = MagicMock(return_value=False)

        with patch.object(exp._tracer, "start_as_current_span", return_value=mock_span):
            # Must not raise even with missing keys
            exp.export(partial_result)

        exp.shutdown()

    def test_export_creates_child_spans_for_layers(self, monkeypatch):
        exp = self._make_exporter(monkeypatch)
        result_with_layers = dict(self._RESULT)
        result_with_layers["per_layer"] = [
            {"layer": "fc1", "max_diff": 1e-7, "cosine_sim": 1.0, "exceeds_threshold": False},
            {"layer": "fc2", "max_diff": 2e-7, "cosine_sim": 0.9999, "exceeds_threshold": False},
        ]

        child_spans_started = []
        mock_span = MagicMock()
        mock_span.__enter__ = lambda s: s
        mock_span.__exit__ = MagicMock(return_value=False)

        def fake_start(name, **kwargs):
            child_spans_started.append(name)
            return mock_span

        with patch.object(exp._tracer, "start_as_current_span", side_effect=fake_start):
            exp.export(result_with_layers)

        # One parent span + two layer child spans
        layer_spans = [n for n in child_spans_started if "layer" in n]
        assert len(layer_spans) == 2

        exp.shutdown()

    def test_export_no_child_spans_when_per_layer_empty(self, monkeypatch):
        exp = self._make_exporter(monkeypatch)
        spans_started = []
        mock_span = MagicMock()
        mock_span.__enter__ = lambda s: s
        mock_span.__exit__ = MagicMock(return_value=False)

        def fake_start(name, **kwargs):
            spans_started.append(name)
            return mock_span

        with patch.object(exp._tracer, "start_as_current_span", side_effect=fake_start):
            exp.export(self._RESULT)  # per_layer is []

        layer_spans = [n for n in spans_started if "layer" in n]
        assert len(layer_spans) == 0

        exp.shutdown()

    def test_shutdown_does_not_raise(self, monkeypatch):
        exp = self._make_exporter(monkeypatch)
        exp.shutdown()  # Must not raise


# ── v0.5.69: URL scheme validation ─────────────────────────────────────────

class TestEndpointUrlValidation:
    def test_invalid_scheme_logs_warning(self, monkeypatch, caplog):
        import logging
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        with caplog.at_level(logging.WARNING, logger="torchbridge.testing.otel_exporter"):
            exp = ValidationSpanExporter(endpoint="ftp://invalid-url.example.com")
            exp.shutdown()
        assert any("ftp://" in msg for msg in caplog.messages)

    def test_valid_https_scheme_no_warning(self, monkeypatch, caplog):
        import logging
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        with caplog.at_level(logging.WARNING, logger="torchbridge.testing.otel_exporter"):
            exp = ValidationSpanExporter(endpoint="https://cloud.langfuse.com/api/public/otel")
            exp.shutdown()
        url_warnings = [m for m in caplog.messages if "does not look like" in m]
        assert len(url_warnings) == 0

    def test_valid_http_scheme_no_warning(self, monkeypatch, caplog):
        import logging
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from torchbridge.testing.otel_exporter import ValidationSpanExporter
        with caplog.at_level(logging.WARNING, logger="torchbridge.testing.otel_exporter"):
            exp = ValidationSpanExporter(endpoint="http://localhost:4318")
            exp.shutdown()
        url_warnings = [m for m in caplog.messages if "does not look like" in m]
        assert len(url_warnings) == 0
