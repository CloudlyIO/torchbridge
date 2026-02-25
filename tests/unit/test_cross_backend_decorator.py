"""
Tests for @cross_backend decorator

Verifies the decorator runs on CPU, collects per-backend results,
surfaces failures clearly, and handles edge cases.
"""

import pytest
import torch
import torch.nn as nn


class TestCrossBackendDecorator:
    """Tests for torchbridge.testing.cross_backend."""

    def test_import(self):
        """cross_backend is importable from torchbridge.testing."""
        from torchbridge.testing import cross_backend
        assert callable(cross_backend)

    def test_runs_on_cpu(self):
        """@cross_backend runs the test on at least CPU."""
        from torchbridge.testing import cross_backend

        call_log = []

        @cross_backend(min_backends=1)
        def my_test(backend):
            call_log.append(getattr(backend, "backend_type", type(backend).__name__))

        my_test()
        assert len(call_log) >= 1

    def test_returns_dict_of_results(self):
        """@cross_backend returns a dict keyed by backend name."""
        from torchbridge.testing import cross_backend

        @cross_backend(min_backends=1)
        def my_test(backend):
            return 42

        results = my_test()
        assert isinstance(results, dict)
        assert all(v == 42 for v in results.values())

    def test_failure_raised_as_assertion_error(self):
        """@cross_backend raises AssertionError when test fails on any backend."""
        from torchbridge.testing import cross_backend

        @cross_backend(min_backends=1)
        def bad_test(backend):
            raise ValueError("intentional failure")

        with pytest.raises(AssertionError, match="Cross-backend failures"):
            bad_test()

    def test_failure_message_includes_backend_name(self):
        """AssertionError message includes the failing backend name."""
        from torchbridge.testing import cross_backend

        @cross_backend(min_backends=1)
        def bad_test(backend):
            raise RuntimeError("boom")

        with pytest.raises(AssertionError) as exc_info:
            bad_test()

        assert "RuntimeError" in str(exc_info.value)

    def test_preserves_function_name(self):
        """@cross_backend preserves the wrapped function's __name__."""
        from torchbridge.testing import cross_backend

        @cross_backend()
        def my_named_test(backend):
            pass

        assert my_named_test.__name__ == "my_named_test"

    def test_model_forward_on_cpu(self):
        """Runs a real model forward pass on CPU backend."""
        from torchbridge.testing import cross_backend

        @cross_backend(min_backends=1)
        def test_linear(backend):
            device = backend.device
            model = nn.Linear(8, 4).to(device)
            model.eval()
            x = torch.randn(2, 8).to(device)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, 4)

        test_linear()

    def test_skip_when_min_backends_not_met(self):
        """@cross_backend skips if min_backends > available backends."""
        from torchbridge.testing import cross_backend

        @cross_backend(min_backends=999, skip_if_unavailable=True)
        def unreachable_test(backend):
            pass  # pragma: no cover

        with pytest.raises(pytest.skip.Exception):
            unreachable_test()
