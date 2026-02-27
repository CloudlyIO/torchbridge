"""
Tests for LLMInferenceServer auth and rate-limiting (v0.5.41).

Uses TestClient from starlette (bundled with fastapi) so no real server
process is needed.  The model/tokenizer are fully mocked — these tests
exercise the security layer only, not generation logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Skip the entire module if FastAPI is not installed.
pytest.importorskip("fastapi", reason="FastAPI required for LLM server tests")

from fastapi.testclient import TestClient  # noqa: E402

from torchbridge.deployment.serving.llm_server import (  # noqa: E402
    LLMInferenceServer,
    LLMServerConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_server(config: LLMServerConfig) -> LLMInferenceServer:
    """Return a server backed by a tiny MagicMock model/tokenizer."""
    mock_model = MagicMock()
    mock_model.device = MagicMock()
    mock_model.device.type = "cpu"
    mock_model.eval.return_value = mock_model

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    return LLMInferenceServer(mock_model, mock_tokenizer, config)


@pytest.fixture()
def server_no_auth():
    """Server with no API key (default — all requests allowed)."""
    return _make_server(LLMServerConfig(enable_dynamic_batching=False, enable_llm_metrics=False))


@pytest.fixture()
def server_with_auth():
    """Server with API key = 'secret-key'."""
    return _make_server(LLMServerConfig(
        api_key="secret-key",
        enable_dynamic_batching=False,
        enable_llm_metrics=False,
    ))


@pytest.fixture()
def server_rate_limited():
    """Server with rate limit of 2 rpm (easy to exceed in tests)."""
    return _make_server(LLMServerConfig(
        rate_limit_rpm=2,
        enable_dynamic_batching=False,
        enable_llm_metrics=False,
    ))


@pytest.fixture()
def server_custom_cors():
    """Server with specific CORS origins."""
    return _make_server(LLMServerConfig(
        cors_origins=["https://example.com"],
        enable_dynamic_batching=False,
        enable_llm_metrics=False,
    ))


# ---------------------------------------------------------------------------
# Tests: api_key=None (auth disabled)
# ---------------------------------------------------------------------------

class TestAuthDisabled:
    def test_health_accessible_without_auth(self, server_no_auth):
        client = TestClient(server_no_auth.app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_root_accessible_without_auth(self, server_no_auth):
        client = TestClient(server_no_auth.app, raise_server_exceptions=False)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_metrics_accessible_without_auth(self, server_no_auth):
        client = TestClient(server_no_auth.app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        # 200 or 500 (if metrics internals need config) — but NOT 401
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Tests: api_key set — valid / invalid / missing header
# ---------------------------------------------------------------------------

class TestAuthEnabled:
    def test_valid_key_returns_not_401(self, server_with_auth):
        """A valid key must NOT be rejected (may still 422/500 for empty body)."""
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/metrics", headers={"Authorization": "Bearer secret-key"})
        assert resp.status_code != 401

    def test_missing_auth_header_returns_401(self, server_with_auth):
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, server_with_auth):
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/metrics", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_empty_token_returns_401(self, server_with_auth):
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/metrics", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401

    def test_no_bearer_prefix_returns_401(self, server_with_auth):
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/metrics", headers={"Authorization": "secret-key"})
        assert resp.status_code == 401

    def test_health_always_exempt_from_auth(self, server_with_auth):
        """Health endpoints must never require auth (infra probes)."""
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_liveness_always_exempt_from_auth(self, server_with_auth):
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_root_always_exempt_from_auth(self, server_with_auth):
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_401_response_includes_www_authenticate_header(self, server_with_auth):
        client = TestClient(server_with_auth.app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert "www-authenticate" in resp.headers or "WWW-Authenticate" in resp.headers


# ---------------------------------------------------------------------------
# Tests: rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_requests_under_limit_pass(self, server_rate_limited):
        """First 2 requests (the limit) must succeed (not 429)."""
        client = TestClient(server_rate_limited.app, raise_server_exceptions=False)
        for _ in range(2):
            resp = client.get("/metrics")
            assert resp.status_code != 429

    def test_request_over_limit_returns_429(self, server_rate_limited):
        """Third request in the same window must be rejected (limit = 2 rpm)."""
        client = TestClient(server_rate_limited.app, raise_server_exceptions=False)
        for _ in range(2):
            client.get("/metrics")
        resp = client.get("/metrics")
        assert resp.status_code == 429

    def test_rate_limit_retry_after_header(self, server_rate_limited):
        """429 response should include Retry-After header."""
        client = TestClient(server_rate_limited.app, raise_server_exceptions=False)
        for _ in range(3):
            resp = client.get("/metrics")
        assert resp.status_code == 429
        assert "retry-after" in resp.headers or "Retry-After" in resp.headers


# ---------------------------------------------------------------------------
# Tests: CORS
# ---------------------------------------------------------------------------

class TestCORSConfig:
    def test_default_cors_allows_all_origins(self, server_no_auth):
        assert server_no_auth.config.cors_origins == ["*"]

    def test_custom_cors_stored_in_config(self, server_custom_cors):
        assert "https://example.com" in server_custom_cors.config.cors_origins

    def test_api_key_none_disables_auth_check(self, server_no_auth):
        assert server_no_auth.config.api_key is None

    def test_rate_limit_none_disables_limiter(self, server_no_auth):
        assert server_no_auth._rate_limiter is None
