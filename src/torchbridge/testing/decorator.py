# SPDX-License-Identifier: Apache-2.0
"""
@cross_backend Pytest Decorator

Repeats a test function on every available hardware backend, collecting
results and surfacing per-backend failures with clear messages.

Usage::

    from torchbridge.testing import cross_backend

    @cross_backend(min_backends=1)
    def test_linear_forward(backend):
        import torch
        import torch.nn as nn
        device = backend.device
        model = nn.Linear(32, 16).to(device)
        x = torch.randn(4, 32).to(device)
        out = model(x)
        assert out.shape == (4, 16)
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _get_available_backends() -> list[Any]:
    """Return a list of available TorchBridge backends (non-CPU last)."""
    from torchbridge.backends.backend_factory import BackendFactory

    available = []
    for bt in BackendFactory.get_available_backends():
        try:
            backend = BackendFactory.create(bt)
            available.append(backend)
        except Exception as e:
            logger.debug("Skipping backend %s: %s", bt, e)

    # Sort: put CPU last so failures on real hardware are surfaced first
    cpu_backends = [
        b
        for b in available
        if getattr(b, "backend_type", None) and b.backend_type.value == "cpu"
    ]
    non_cpu = [b for b in available if b not in cpu_backends]
    return non_cpu + cpu_backends


def cross_backend(
    min_backends: int = 1,
    skip_if_unavailable: bool = True,
) -> Callable:
    """Decorator: run a test on every available hardware backend.

    The decorated function receives a ``backend`` keyword argument.  If the
    function raises on any backend, all failures are collected and reported
    together in a single ``AssertionError``.

    Args:
        min_backends: Skip the test if fewer than this many backends are
            available (default 1 — always run on at least CPU).
        skip_if_unavailable: If ``True``, use ``pytest.skip`` when
            ``min_backends`` is not met.  If ``False``, raise ``AssertionError``.

    Example::

        @cross_backend(min_backends=1)
        def test_forward(backend):
            model = nn.Linear(8, 4).to(backend.device)
            out = model(torch.randn(2, 8).to(backend.device))
            assert out.shape == (2, 4)
    """

    def decorator(test_fn: Callable) -> Callable:
        @functools.wraps(test_fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            import pytest

            available = _get_available_backends()
            if len(available) < min_backends:
                msg = (
                    f"cross_backend requires {min_backends} backend(s); "
                    f"only {len(available)} available."
                )
                if skip_if_unavailable:
                    pytest.skip(msg)
                else:
                    raise AssertionError(msg)

            results: dict[str, Any] = {}
            failures: dict[str, Exception] = {}

            for backend in available:
                backend_name = getattr(backend, "backend_type", type(backend).__name__)
                try:
                    result = test_fn(*args, backend=backend, **kwargs)
                    results[str(backend_name)] = result
                    logger.debug("cross_backend PASS: %s", backend_name)
                except Exception as exc:  # noqa: BLE001
                    failures[str(backend_name)] = exc
                    logger.warning("cross_backend FAIL: %s — %s", backend_name, exc)

            if failures:
                lines = [f"Cross-backend failures ({len(failures)}/{len(available)}):"]
                for name, exc in failures.items():
                    lines.append(f"  [{name}] {type(exc).__name__}: {exc}")
                raise AssertionError("\n".join(lines))

            return results

        return wrapper

    return decorator
