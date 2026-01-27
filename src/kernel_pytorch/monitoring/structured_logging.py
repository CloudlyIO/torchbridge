"""
KernelPyTorch Structured Logging Module

Provides production-grade structured logging with:
- JSON formatting for log aggregation (ELK, Splunk, etc.)
- Correlation IDs for distributed tracing
- Contextual logging with automatic metadata
- Configurable log levels per module
- Performance metrics in logs
- Thread-safe context propagation

Usage:
    from kernel_pytorch.monitoring.structured_logging import (
        get_logger, configure_logging, correlation_context
    )

    # Configure logging
    configure_logging(level="INFO", json_format=True)

    # Get a logger
    logger = get_logger(__name__)

    # Log with correlation ID
    with correlation_context() as ctx:
        logger.info("Processing request", extra={"batch_size": 32})
        # All logs within this context share the same correlation_id
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

# Context variable for correlation ID (thread-safe)
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for additional context
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

# Global configuration lock
_config_lock = threading.Lock()

# Configured loggers registry
_configured_loggers: dict[str, logging.Logger] = {}


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """Logging configuration."""
    level: str = "INFO"
    json_format: bool = True
    include_timestamp: bool = True
    include_caller: bool = True
    include_correlation_id: bool = True
    include_thread_info: bool = False
    include_process_info: bool = False
    include_hostname: bool = True
    output_file: str | None = None
    max_message_length: int = 10000
    sensitive_fields: list[str] = field(default_factory=lambda: [
        "password", "token", "secret", "api_key", "credential"
    ])
    module_levels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "level": self.level,
            "json_format": self.json_format,
            "include_timestamp": self.include_timestamp,
            "include_caller": self.include_caller,
            "include_correlation_id": self.include_correlation_id,
            "include_thread_info": self.include_thread_info,
            "include_process_info": self.include_process_info,
            "include_hostname": self.include_hostname,
            "output_file": self.output_file,
            "max_message_length": self.max_message_length,
            "module_levels": self.module_levels,
        }


# Global configuration
_global_config = LogConfig()


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config
        self._hostname = self._get_hostname()

    def _get_hostname(self) -> str:
        """Get hostname for log entries."""
        try:
            import socket
            return socket.gethostname()
        except Exception:
            return "unknown"

    def _sanitize_value(self, value: Any, field_name: str = "") -> Any:
        """Sanitize sensitive values."""
        field_lower = field_name.lower()
        for sensitive in self.config.sensitive_fields:
            if sensitive in field_lower:
                return "***REDACTED***"

        if isinstance(value, dict):
            return {k: self._sanitize_value(v, k) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(v) for v in value]
        elif isinstance(value, str) and len(value) > self.config.max_message_length:
            return value[:self.config.max_message_length] + "...[TRUNCATED]"
        return value

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": self._sanitize_value(record.getMessage()),
        }

        # Add timestamp
        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()
            log_entry["timestamp_unix"] = record.created

        # Add correlation ID
        if self.config.include_correlation_id:
            correlation_id = _correlation_id.get()
            if correlation_id:
                log_entry["correlation_id"] = correlation_id

        # Add caller information
        if self.config.include_caller:
            log_entry["caller"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
                "module": record.module,
            }

        # Add thread info
        if self.config.include_thread_info:
            log_entry["thread"] = {
                "id": record.thread,
                "name": record.threadName,
            }

        # Add process info
        if self.config.include_process_info:
            log_entry["process"] = {
                "id": record.process,
                "name": record.processName,
            }

        # Add hostname
        if self.config.include_hostname:
            log_entry["hostname"] = self._hostname

        # Add context from context variable
        context = _log_context.get()
        if context:
            log_entry["context"] = self._sanitize_value(context)

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            }:
                extra_fields[key] = self._sanitize_value(value, key)

        if extra_fields:
            log_entry["extra"] = extra_fields

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config
        self._use_colors = sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        parts = []

        # Timestamp
        if self.config.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            parts.append(f"[{timestamp}]")

        # Level with color
        level = record.levelname
        if self._use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            parts.append(f"{color}{level:8}{reset}")
        else:
            parts.append(f"{level:8}")

        # Correlation ID
        if self.config.include_correlation_id:
            correlation_id = _correlation_id.get()
            if correlation_id:
                short_id = correlation_id[:8]
                parts.append(f"[{short_id}]")

        # Logger name
        parts.append(f"[{record.name}]")

        # Message
        parts.append(record.getMessage())

        # Extra fields
        extra_parts = []
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            }:
                extra_parts.append(f"{key}={value}")

        if extra_parts:
            parts.append(f"| {' '.join(extra_parts)}")

        result = " ".join(parts)

        # Add exception info
        if record.exc_info:
            result += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return result


class CorrelationContext:
    """Context manager for correlation ID propagation."""

    def __init__(self, correlation_id: str | None = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "CorrelationContext":
        self._token = _correlation_id.set(self.correlation_id)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _correlation_id.reset(self._token)


class LogContext:
    """Context manager for adding contextual information to logs."""

    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self._token: contextvars.Token | None = None
        self._previous: dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        self._previous = _log_context.get().copy()
        new_context = {**self._previous, **self.context}
        self._token = _log_context.set(new_context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _log_context.reset(self._token)


class StructuredLogger(logging.Logger):
    """Extended logger with structured logging support."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)

    def _log_with_context(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        extra: dict | None = None,
        **kwargs: Any
    ) -> None:
        """Log with automatic context injection."""
        if extra is None:
            extra = {}

        # Merge kwargs into extra
        extra.update(kwargs)

        super()._log(level, msg, args, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, msg, args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with context."""
        self._log_with_context(logging.INFO, msg, args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, msg, args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with context."""
        self._log_with_context(logging.ERROR, msg, args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, msg, args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback."""
        kwargs["exc_info"] = True
        self._log_with_context(logging.ERROR, msg, args, **kwargs)


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    output_file: str | None = None,
    module_levels: dict[str, str] | None = None,
    **kwargs: Any
) -> LogConfig:
    """
    Configure global logging settings.

    Args:
        level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True) or human-readable (False)
        output_file: Optional file path for log output
        module_levels: Per-module log level overrides
        **kwargs: Additional LogConfig parameters

    Returns:
        The active LogConfig

    Example:
        configure_logging(
            level="DEBUG",
            json_format=True,
            module_levels={"kernel_pytorch.attention": "WARNING"}
        )
    """
    global _global_config

    with _config_lock:
        _global_config = LogConfig(
            level=level,
            json_format=json_format,
            output_file=output_file,
            module_levels=module_levels or {},
            **kwargs
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Allow all, filter at handler

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        if json_format:
            formatter = JSONFormatter(_global_config)
        else:
            formatter = ConsoleFormatter(_global_config)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setLevel(getattr(logging, level))
            file_handler.setFormatter(JSONFormatter(_global_config))  # Always JSON for files
            root_logger.addHandler(file_handler)

        # Apply module-specific levels
        for module_name, module_level in _global_config.module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(getattr(logging, module_level))

        # Re-configure existing loggers
        for logger in _configured_loggers.values():
            _apply_config_to_logger(logger)

        return _global_config


def _apply_config_to_logger(logger: logging.Logger) -> None:
    """Apply current configuration to a logger."""
    # Check for module-specific level
    module_level = _global_config.module_levels.get(logger.name)
    if module_level:
        logger.setLevel(getattr(logging, module_level))
    else:
        logger.setLevel(getattr(logging, _global_config.level))


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing", batch_size=32, model="bert")
    """
    with _config_lock:
        if name in _configured_loggers:
            return _configured_loggers[name]  # type: ignore

        # Create new structured logger
        logging.setLoggerClass(StructuredLogger)
        logger = logging.getLogger(name)
        logging.setLoggerClass(logging.Logger)

        _apply_config_to_logger(logger)
        _configured_loggers[name] = logger

        return logger  # type: ignore


def correlation_context(correlation_id: str | None = None) -> CorrelationContext:
    """
    Create a correlation context for request tracing.

    Args:
        correlation_id: Optional ID to use (generates UUID if not provided)

    Returns:
        CorrelationContext context manager

    Example:
        with correlation_context() as ctx:
            logger.info("Request started")
            process_request()
            logger.info("Request completed")
    """
    return CorrelationContext(correlation_id)


def log_context(**kwargs: Any) -> LogContext:
    """
    Add contextual information to all logs within scope.

    Args:
        **kwargs: Key-value pairs to add to log context

    Returns:
        LogContext context manager

    Example:
        with log_context(user_id="123", request_type="inference"):
            logger.info("Processing request")  # Includes user_id, request_type
    """
    return LogContext(**kwargs)


def get_correlation_id() -> str | None:
    """Get current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> contextvars.Token:
    """Set correlation ID manually."""
    return _correlation_id.set(correlation_id)


T = TypeVar("T", bound=Callable[..., Any])


def log_function_call(
    logger: logging.Logger | None = None,
    level: str = "DEBUG",
    include_args: bool = True,
    include_result: bool = False,
    include_timing: bool = True,
) -> Callable[[T], T]:
    """
    Decorator to log function calls.

    Args:
        logger: Logger to use (creates one if not provided)
        level: Log level for function call logs
        include_args: Include function arguments in log
        include_result: Include return value in log
        include_timing: Include execution time in log

    Example:
        @log_function_call(include_timing=True)
        def process_batch(batch):
            ...
    """
    def decorator(func: T) -> T:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_level = getattr(logging, level)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            # Build log extras
            extra: dict[str, Any] = {"function": func.__name__}

            if include_args:
                extra["func_args"] = str(args)[:200]
                extra["func_kwargs"] = str(kwargs)[:200]

            logger.log(log_level, f"Calling {func.__name__}", extra=extra)

            try:
                result = func(*args, **kwargs)

                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000

                result_extra: dict[str, Any] = {"function": func.__name__}
                if include_timing:
                    result_extra["duration_ms"] = round(duration_ms, 3)
                if include_result:
                    result_extra["result"] = str(result)[:200]

                logger.log(
                    log_level,
                    f"Completed {func.__name__}",
                    extra=result_extra
                )

                return result

            except Exception as e:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000

                logger.exception(
                    f"Error in {func.__name__}: {e}",
                    extra={
                        "function": func.__name__,
                        "duration_ms": round(duration_ms, 3),
                        "error_type": type(e).__name__,
                    }
                )
                raise

        return wrapper  # type: ignore

    return decorator


class PerformanceLogger:
    """Context manager for logging performance metrics."""

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: str = "INFO",
        **context: Any
    ):
        self.logger = logger
        self.operation = operation
        self.level = getattr(logging, level)
        self.context = context
        self.start_time: float = 0
        self.start_memory: int = 0

    def __enter__(self) -> "PerformanceLogger":
        self.start_time = time.perf_counter()

        # Try to get GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                self.start_memory = torch.cuda.memory_allocated()
        except ImportError:
            pass

        return self

    def __exit__(self, *args: Any) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        extra = {
            "operation": self.operation,
            "duration_ms": round(duration_ms, 3),
            **self.context,
        }

        # Add memory delta if available
        try:
            import torch
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                extra["memory_delta_mb"] = round(
                    (end_memory - self.start_memory) / (1024 * 1024), 2
                )
        except ImportError:
            pass

        self.logger.log(
            self.level,
            f"Completed {self.operation}",
            extra=extra
        )


def performance_log(
    logger: logging.Logger,
    operation: str,
    level: str = "INFO",
    **context: Any
) -> PerformanceLogger:
    """
    Create a performance logging context.

    Args:
        logger: Logger to use
        operation: Name of the operation being measured
        level: Log level
        **context: Additional context to include

    Returns:
        PerformanceLogger context manager

    Example:
        with performance_log(logger, "batch_inference", batch_size=32):
            model(batch)
    """
    return PerformanceLogger(logger, operation, level, **context)


# Export all public APIs
__all__ = [
    "LogLevel",
    "LogConfig",
    "StructuredLogger",
    "CorrelationContext",
    "LogContext",
    "PerformanceLogger",
    "configure_logging",
    "get_logger",
    "correlation_context",
    "log_context",
    "get_correlation_id",
    "set_correlation_id",
    "log_function_call",
    "performance_log",
]
