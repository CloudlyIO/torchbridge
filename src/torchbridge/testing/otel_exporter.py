"""
OpenTelemetry span exporter for TorchBridge validation results.

Emits a structured span after ``tb-validate --compare`` completes so that
Langfuse, W&B Weave, or any OTEL-compatible backend can ingest hardware
numerical divergence data alongside semantic LLM traces.

Usage::

    tb-validate --compare cuda rocm --model ./model.pt --otel
    tb-validate --compare cuda rocm --otel \\
        --otel-endpoint https://cloud.langfuse.com/api/public/otel

Endpoint resolution order:
1. ``endpoint`` constructor argument
2. ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable
3. Console (stdout) — safe fallback for development

All three opentelemetry packages are optional; import errors are caught and
surfaced as ``RuntimeError`` at construction time so the caller can log a
warning and continue without crashing validation.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

#: Module-level availability flag exported for testing and caller inspection.
OTEL_AVAILABLE: bool = _OTEL_AVAILABLE

# ---------------------------------------------------------------------------
# Span attribute schemas (the matrices)
# ---------------------------------------------------------------------------

#: Maps validation result dict keys → (OTEL attribute name, Python type).
#: All attribute names are namespaced under ``torchbridge.*``.
_SPAN_ATTRIBUTE_SCHEMA: dict[str, tuple[str, type]] = {
    "backend1":       ("torchbridge.backend.primary",   str),
    "backend2":       ("torchbridge.backend.secondary", str),
    "model":          ("torchbridge.model",             str),
    "dtype":          ("torchbridge.dtype",             str),
    "max_diff":       ("torchbridge.max_diff",          float),
    "cosine_sim":     ("torchbridge.cosine_sim",        float),
    "tolerance_atol": ("torchbridge.tolerance.atol",    float),
    "tolerance_rtol": ("torchbridge.tolerance.rtol",    float),
    "passed":         ("torchbridge.passed",            bool),
    "duration_ms":    ("torchbridge.duration_ms",       float),
}

#: Maps per-layer divergence row keys → (OTEL attribute name, Python type).
_LAYER_SPAN_SCHEMA: dict[str, tuple[str, type]] = {
    "layer":             ("torchbridge.layer.name",     str),
    "max_diff":          ("torchbridge.layer.max_diff", float),
    "cosine_sim":        ("torchbridge.layer.cosine_sim", float),
    "exceeds_threshold": ("torchbridge.layer.exceeded", bool),
}


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class ValidationSpanExporter:
    """
    Emits a ``torchbridge.validate.compare`` OTEL span from a validation
    result dict, with optional child spans for per-layer divergence rows.

    Args:
        endpoint: OTLP HTTP endpoint URL. If ``None``, reads
            ``OTEL_EXPORTER_OTLP_ENDPOINT``. If that is also unset, falls
            back to ``ConsoleSpanExporter`` (stdout).

    Raises:
        RuntimeError: if ``opentelemetry-sdk`` and
            ``opentelemetry-exporter-otlp-proto-http`` are not installed.
    """

    def __init__(self, endpoint: str | None = None) -> None:
        if not _OTEL_AVAILABLE:
            raise RuntimeError(
                "opentelemetry packages are required for span export. "
                "Install with: pip install torchbridge-ml[tracing]"
            )

        resolved = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

        if resolved and not (
            resolved.startswith("http://") or resolved.startswith("https://")
        ):
            logger.warning(
                "OTel endpoint %r does not look like a valid HTTP(S) URL; "
                "export may fail silently at span flush time. Expected https://...",
                resolved,
            )

        if resolved:
            raw_exporter = OTLPSpanExporter(endpoint=resolved)
        else:
            raw_exporter = ConsoleSpanExporter()

        self._raw_exporter = raw_exporter
        self._processor = BatchSpanProcessor(raw_exporter)
        provider = TracerProvider()
        provider.add_span_processor(self._processor)
        self._tracer = provider.get_tracer("torchbridge")

    def export(self, result: dict) -> None:
        """
        Build and emit a span from a validation result dict.

        Keys present in ``_SPAN_ATTRIBUTE_SCHEMA`` are set as span attributes;
        missing keys are silently skipped. Each row in ``result["per_layer"]``
        becomes a child span with attributes from ``_LAYER_SPAN_SCHEMA``.

        Args:
            result: The result dict produced by ``ValidateCommand._run_compare``.
        """
        with self._tracer.start_as_current_span("torchbridge.validate.compare") as span:
            for key, (attr_name, attr_type) in _SPAN_ATTRIBUTE_SCHEMA.items():
                if key in result:
                    try:
                        span.set_attribute(attr_name, attr_type(result[key]))
                    except (TypeError, ValueError) as e:
                        logger.debug(
                            "Skipped span attribute %s — coercion failed: %s",
                            attr_name, e,
                        )

            for row in result.get("per_layer", []):
                with self._tracer.start_as_current_span(
                    "torchbridge.validate.layer"
                ) as child:
                    for key, (attr_name, attr_type) in _LAYER_SPAN_SCHEMA.items():
                        if key in row:
                            try:
                                child.set_attribute(attr_name, attr_type(row[key]))
                            except (TypeError, ValueError) as e:
                                logger.debug(
                                    "Skipped span attribute %s — coercion failed: %s",
                                    attr_name, e,
                                )

    def shutdown(self) -> None:
        """Flush pending spans and release resources."""
        self._processor.shutdown()
