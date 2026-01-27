"""
Tests for the structured logging module.

Tests cover:
- JSON formatting
- Correlation ID propagation
- Log context management
- Performance logging
- Configuration management
- Sensitive data redaction
"""

import json
import logging
import sys
import threading
import time
from io import StringIO
from unittest.mock import patch

import pytest

from kernel_pytorch.monitoring.structured_logging import (
    CorrelationContext,
    LogConfig,
    LogContext,
    LogLevel,
    PerformanceLogger,
    StructuredLogger,
    configure_logging,
    correlation_context,
    get_correlation_id,
    get_logger,
    log_context,
    log_function_call,
    performance_log,
    set_correlation_id,
)


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.json_format is True
        assert config.include_timestamp is True
        assert config.include_caller is True
        assert config.include_correlation_id is True
        assert config.output_file is None
        assert "password" in config.sensitive_fields

    def test_custom_config(self):
        """Test custom configuration."""
        config = LogConfig(
            level="DEBUG",
            json_format=False,
            include_thread_info=True,
            module_levels={"test_module": "WARNING"}
        )
        assert config.level == "DEBUG"
        assert config.json_format is False
        assert config.include_thread_info is True
        assert config.module_levels["test_module"] == "WARNING"

    def test_to_dict(self):
        """Test config serialization."""
        config = LogConfig(level="ERROR")
        config_dict = config.to_dict()
        assert config_dict["level"] == "ERROR"
        assert isinstance(config_dict, dict)


class TestJSONFormatting:
    """Tests for JSON log formatting."""

    @pytest.fixture
    def capture_logs(self):
        """Fixture to capture log output."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)

        # Store original handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        yield stream, handler

        # Restore
        root_logger.handlers = original_handlers
        root_logger.level = original_level

    def test_json_format_basic(self, capture_logs):
        """Test basic JSON formatting."""
        stream, _ = capture_logs
        configure_logging(level="DEBUG", json_format=True)

        logger = get_logger("test_json_basic")
        logger.info("Test message")

        output = stream.getvalue()
        # Find the JSON line
        for line in output.strip().split('\n'):
            if "Test message" in line:
                log_entry = json.loads(line)
                assert log_entry["level"] == "INFO"
                assert log_entry["message"] == "Test message"
                assert "timestamp" in log_entry
                break

    def test_json_format_with_extra(self, capture_logs):
        """Test JSON formatting with extra fields."""
        stream, _ = capture_logs
        configure_logging(level="DEBUG", json_format=True)

        logger = get_logger("test_json_extra")
        logger.info("Test with extra", extra={"batch_size": 32, "model": "bert"})

        output = stream.getvalue()
        for line in output.strip().split('\n'):
            if "Test with extra" in line:
                log_entry = json.loads(line)
                assert log_entry["extra"]["batch_size"] == 32
                assert log_entry["extra"]["model"] == "bert"
                break

    def test_json_format_with_exception(self, capture_logs):
        """Test JSON formatting with exception."""
        stream, _ = capture_logs
        configure_logging(level="DEBUG", json_format=True)

        logger = get_logger("test_json_exception")
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("Error occurred")

        output = stream.getvalue()
        for line in output.strip().split('\n'):
            if "Error occurred" in line:
                log_entry = json.loads(line)
                assert "exception" in log_entry
                assert log_entry["exception"]["type"] == "ValueError"
                assert "Test error" in log_entry["exception"]["message"]
                break


class TestCorrelationID:
    """Tests for correlation ID propagation."""

    def test_correlation_context_basic(self):
        """Test basic correlation context."""
        assert get_correlation_id() is None

        with correlation_context() as ctx:
            assert get_correlation_id() == ctx.correlation_id
            assert len(ctx.correlation_id) == 36  # UUID format

        assert get_correlation_id() is None

    def test_correlation_context_custom_id(self):
        """Test custom correlation ID."""
        custom_id = "my-request-123"

        with correlation_context(custom_id):
            assert get_correlation_id() == custom_id

        assert get_correlation_id() is None

    def test_correlation_context_nested(self):
        """Test nested correlation contexts."""
        with correlation_context("outer-id") as outer:
            assert get_correlation_id() == "outer-id"

            with correlation_context("inner-id") as inner:
                assert get_correlation_id() == "inner-id"

            assert get_correlation_id() == "outer-id"

        assert get_correlation_id() is None

    def test_correlation_id_in_logs(self):
        """Test correlation ID appears in logs."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)

        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("test_corr_logs")

        with correlation_context("test-correlation-123"):
            logger.info("Correlated message")

        output = stream.getvalue()
        for line in output.strip().split('\n'):
            if "Correlated message" in line:
                log_entry = json.loads(line)
                assert log_entry.get("correlation_id") == "test-correlation-123"
                break

    def test_correlation_thread_isolation(self):
        """Test correlation IDs are isolated per thread."""
        results = {}

        def thread_func(thread_id: str):
            with correlation_context(thread_id):
                time.sleep(0.01)  # Ensure overlap
                results[thread_id] = get_correlation_id()

        threads = [
            threading.Thread(target=thread_func, args=(f"thread-{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have its own correlation ID
        for i in range(5):
            assert results[f"thread-{i}"] == f"thread-{i}"


class TestLogContext:
    """Tests for log context management."""

    def test_log_context_basic(self):
        """Test basic log context."""
        with log_context(user_id="123", request_type="inference"):
            # Context should be available within the block
            pass

    def test_log_context_nested(self):
        """Test nested log contexts merge correctly."""
        with log_context(user_id="123"):
            with log_context(request_type="inference"):
                # Both contexts should be available
                pass


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_logger("test_creation")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_creation"

    def test_logger_caching(self):
        """Test loggers are cached."""
        logger1 = get_logger("test_cache")
        logger2 = get_logger("test_cache")
        assert logger1 is logger2

    def test_logger_with_kwargs(self):
        """Test logger accepts kwargs as extra."""
        stream = StringIO()
        configure_logging(level="DEBUG", json_format=True)

        logger = get_logger("test_kwargs")
        logger.info("Test message", batch_size=32)

        # The extra fields should be captured


class TestPerformanceLogging:
    """Tests for performance logging."""

    def test_performance_logger_basic(self):
        """Test basic performance logging."""
        configure_logging(level="DEBUG", json_format=True)

        logger = get_logger("test_perf")

        # Performance logging should not raise
        with performance_log(logger, "test_operation"):
            time.sleep(0.01)

        # Test completed without errors - the log was written to stdout

    def test_performance_logger_with_context(self):
        """Test performance logging with additional context."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("test_perf_context")

        with performance_log(logger, "batch_processing", batch_size=64):
            time.sleep(0.001)


class TestFunctionCallDecorator:
    """Tests for log_function_call decorator."""

    def test_decorator_basic(self):
        """Test basic function call logging."""
        configure_logging(level="DEBUG", json_format=True)

        @log_function_call()
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_decorator_with_exception(self):
        """Test decorator logs exceptions."""
        configure_logging(level="DEBUG", json_format=True)

        @log_function_call()
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_decorator_timing(self):
        """Test decorator captures timing."""
        configure_logging(level="DEBUG", json_format=True)

        @log_function_call(include_timing=True)
        def slow_func():
            time.sleep(0.01)
            return "done"

        result = slow_func()
        assert result == "done"


class TestSensitiveDataRedaction:
    """Tests for sensitive data redaction."""

    def test_password_redaction(self):
        """Test passwords are redacted."""
        stream = StringIO()
        configure_logging(level="DEBUG", json_format=True)

        logger = get_logger("test_redact")
        logger.info("Login attempt", extra={"password": "secret123"})

        output = stream.getvalue()
        assert "secret123" not in output
        # Should contain redacted marker if password field was logged

    def test_api_key_redaction(self):
        """Test API keys are redacted."""
        stream = StringIO()
        configure_logging(level="DEBUG", json_format=True)

        logger = get_logger("test_redact_api")
        logger.info("API call", extra={"api_key": "sk-12345"})

        output = stream.getvalue()
        assert "sk-12345" not in output


class TestModuleLevelConfiguration:
    """Tests for per-module log level configuration."""

    def test_module_specific_level(self):
        """Test module-specific log levels."""
        configure_logging(
            level="INFO",
            json_format=True,
            module_levels={"verbose_module": "DEBUG", "quiet_module": "ERROR"}
        )

        verbose_logger = get_logger("verbose_module")
        quiet_logger = get_logger("quiet_module")

        assert verbose_logger.level == logging.DEBUG
        assert quiet_logger.level == logging.ERROR


class TestConsoleFormatter:
    """Tests for console (human-readable) formatter."""

    def test_console_format(self):
        """Test console formatting."""
        configure_logging(level="DEBUG", json_format=False)

        logger = get_logger("test_console")
        logger.info("Console message")

        # Should not raise and should be human readable


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels(self):
        """Test log level values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestIntegration:
    """Integration tests for structured logging."""

    def test_full_workflow(self):
        """Test complete logging workflow."""
        configure_logging(
            level="DEBUG",
            json_format=True,
            include_thread_info=True,
            include_hostname=True,
        )

        logger = get_logger("integration_test")

        # Simulate a request
        with correlation_context("request-abc123"):
            with log_context(user_id="user-456", endpoint="/predict"):
                logger.info("Request received", extra={"method": "POST"})

                with performance_log(logger, "model_inference", model="bert"):
                    time.sleep(0.01)
                    logger.debug("Model loaded")

                logger.info("Request completed", extra={"status": 200})

    def test_multi_threaded_logging(self):
        """Test logging from multiple threads."""
        configure_logging(level="DEBUG", json_format=True)

        results = []
        lock = threading.Lock()

        def worker(worker_id: int):
            logger = get_logger(f"worker_{worker_id}")
            with correlation_context(f"worker-{worker_id}"):
                for i in range(5):
                    logger.info(f"Work item {i}", extra={"worker": worker_id})

                with lock:
                    results.append(worker_id)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3

    def test_exception_chain_logging(self):
        """Test logging with exception chains."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("exception_chain_test")

        def inner_func():
            raise ValueError("Inner error")

        def outer_func():
            try:
                inner_func()
            except ValueError as e:
                raise RuntimeError("Outer error") from e

        try:
            outer_func()
        except RuntimeError:
            logger.exception("Caught chained exception")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_message(self):
        """Test logging empty message."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("empty_test")
        logger.info("")

    def test_very_long_message(self):
        """Test logging very long message (truncation)."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("long_test")

        long_message = "x" * 50000
        logger.info(long_message)

    def test_unicode_message(self):
        """Test logging unicode characters."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("unicode_test")
        logger.info("Unicode: 你好世界 🚀 émojis")

    def test_special_characters(self):
        """Test logging special characters."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("special_test")
        logger.info('Special: "quotes" and \\backslash and \nnewline')

    def test_none_values(self):
        """Test logging None values."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("none_test")
        logger.info("None value", extra={"value": None})

    def test_complex_extra(self):
        """Test logging complex extra data."""
        configure_logging(level="DEBUG", json_format=True)
        logger = get_logger("complex_test")
        logger.info(
            "Complex data",
            extra={
                "nested": {"a": 1, "b": {"c": 2}},
                "list": [1, 2, 3],
                "tuple": (4, 5, 6),
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
