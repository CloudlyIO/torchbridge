"""
TorchBridge Pytest Plugin

Adds a ``--torchbridge-backends`` CLI flag that restricts which backends the
``@cross_backend`` decorator will run on.

Register the plugin in your ``conftest.py``::

    pytest_plugins = ["torchbridge.testing.plugin"]

Or in ``pyproject.toml``::

    [tool.pytest.ini_options]
    addopts = "-p torchbridge.testing.plugin"

Then run::

    pytest --torchbridge-backends=cuda,cpu tests/

Without the flag, all available backends are used.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Registry key used to store the selected backends in pytest config
_CONFIG_KEY = "torchbridge_backends"


def pytest_addoption(parser: object) -> None:
    """Add ``--torchbridge-backends`` CLI option."""
    try:
        parser.addoption(  # type: ignore[attr-defined]
            "--torchbridge-backends",
            action="store",
            default=None,
            help=(
                "Comma-separated list of backends to test with @cross_backend. "
                "Example: --torchbridge-backends=cuda,cpu,mps"
            ),
        )
    except Exception as e:
        logger.debug("Could not add --torchbridge-backends option: %s", e)


def pytest_configure(config: object) -> None:
    """Store selected backends in config stash for use by @cross_backend."""
    try:
        raw = config.getoption("--torchbridge-backends", default=None)  # type: ignore[attr-defined]
        if raw:
            backends = [b.strip().lower() for b in raw.split(",") if b.strip()]
            config._torchbridge_backends = backends  # type: ignore[attr-defined]
            logger.info("TorchBridge: restricting @cross_backend to: %s", backends)
        else:
            config._torchbridge_backends = None  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Failed to finalize torchbridge backend config", exc_info=True)


def get_enabled_backends(config: object | None = None) -> list[str] | None:
    """Return the list of enabled backends set via ``--torchbridge-backends``.

    Returns ``None`` if no restriction is active (all available backends used).
    """
    if config is None:
        try:
            import pytest

            config = pytest.config  # type: ignore[attr-defined]
        except Exception:
            return None
    return getattr(config, "_torchbridge_backends", None)
