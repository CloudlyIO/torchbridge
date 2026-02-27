"""
Tests for LLMInferenceServer security defaults (v0.5.42+).

Covers:
- Startup warning logged when binding to 0.0.0.0 without an API key
- No warning when auth is configured or host is restricted
- LLM_SERVER_API_KEY env-var picked up by LLMServerConfig
- Env-var key authenticates / rejects requests via TestClient
- CORS wildcard warning when auth is enabled but origins are open
- Rate limiter thread safety under concurrent load
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="FastAPI required for LLM server tests")

from fastapi.testclient import TestClient  # noqa: E402

from torchbridge.deployment.serving.llm_server import (  # noqa: E402
    LLMInferenceServer,
    LLMServerConfig,
    _RateLimiter,
)

_ENV_KEY = "LLM_SERVER_API_KEY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server(config: LLMServerConfig) -> LLMInferenceServer:
    mock_model = MagicMock()
    mock_model.device = MagicMock()
    mock_model.device.type = "cpu"
    mock_model.eval.return_value = mock_model
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    return LLMInferenceServer(mock_model, mock_tokenizer, config)


def _cfg(**kwargs) -> LLMServerConfig:
    defaults = {"enable_dynamic_batching": False, "enable_llm_metrics": False}
    defaults.update(kwargs)
    return LLMServerConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests: startup warning
# ---------------------------------------------------------------------------

class TestStartupSecurityWarning:
    def test_warning_logged_no_auth_on_all_interfaces(self, caplog, monkeypatch):
        """0.0.0.0 + no api_key must log a WARNING."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg(host="0.0.0.0")
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("0.0.0.0" in m and "no API key" in m for m in warns)

    def test_warning_mentions_env_var_solution(self, caplog, monkeypatch):
        """Warning should reference LLM_SERVER_API_KEY as the remedy."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg(host="0.0.0.0")
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(_ENV_KEY in m for m in warns)

    def test_no_warning_when_api_key_set(self, caplog):
        """api_key present → no security warning."""
        cfg = _cfg(host="0.0.0.0", api_key="my-secret")
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        assert not any("no API key" in r.message for r in caplog.records)

    def test_no_warning_when_localhost(self, caplog, monkeypatch):
        """127.0.0.1 without api_key should NOT trigger the warning."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg(host="127.0.0.1")
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        assert not any("no API key" in r.message for r in caplog.records)

    def test_env_var_key_suppresses_warning(self, caplog, monkeypatch):
        """LLM_SERVER_API_KEY set in env → auth is active → no warning."""
        monkeypatch.setenv(_ENV_KEY, "env-secret")
        cfg = _cfg(host="0.0.0.0")
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        assert not any("no API key" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: LLM_SERVER_API_KEY env-var
# ---------------------------------------------------------------------------

class TestEnvVarApiKey:
    def test_env_var_sets_api_key(self, monkeypatch):
        """LLM_SERVER_API_KEY env var populates config.api_key at instantiation."""
        monkeypatch.setenv(_ENV_KEY, "env-secret")
        cfg = _cfg()
        assert cfg.api_key == "env-secret"

    def test_absent_env_var_gives_none(self, monkeypatch):
        """Without the env var, api_key defaults to None."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg()
        assert cfg.api_key is None

    def test_explicit_kwarg_overrides_env_var(self, monkeypatch):
        """Explicit api_key= kwarg takes precedence over env var."""
        monkeypatch.setenv(_ENV_KEY, "env-secret")
        cfg = _cfg(api_key="explicit-key")
        assert cfg.api_key == "explicit-key"

    def test_env_var_key_authenticates_request(self, monkeypatch):
        """Server built via env-var key accepts requests using that key."""
        monkeypatch.setenv(_ENV_KEY, "env-key-123")
        server = _make_server(_cfg())
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.get("/metrics", headers={"Authorization": "Bearer env-key-123"})
        assert resp.status_code != 401

    def test_env_var_key_rejects_wrong_key(self, monkeypatch):
        """Server built via env-var key rejects a wrong key."""
        monkeypatch.setenv(_ENV_KEY, "env-key-123")
        server = _make_server(_cfg())
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.get("/metrics", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_env_var_key_rejects_missing_header(self, monkeypatch):
        """Server built via env-var key rejects requests with no auth header."""
        monkeypatch.setenv(_ENV_KEY, "env-key-123")
        server = _make_server(_cfg())
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: CORS wildcard warning
# ---------------------------------------------------------------------------

class TestCorsWildcardWarning:
    def test_cors_warning_when_auth_with_wildcard(self, caplog, monkeypatch):
        """api_key set + cors_origins=['*'] must log a CORS warning."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg(api_key="secret", cors_origins=["*"])
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("CORS" in m and "allow_origins" in m for m in warns)

    def test_no_cors_warning_without_auth(self, caplog, monkeypatch):
        """No api_key → CORS warning should NOT fire (open API anyway)."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg(cors_origins=["*"])
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("CORS" in m for m in warns)

    def test_no_cors_warning_with_explicit_origins(self, caplog, monkeypatch):
        """api_key set + explicit origins → no CORS warning."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg(api_key="secret", cors_origins=["https://myapp.com"])
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("CORS" in m for m in warns)

    def test_cors_warning_mentions_config_field(self, caplog, monkeypatch):
        """CORS warning should reference cors_origins= as the remedy."""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        cfg = _cfg(api_key="secret", cors_origins=["*"])
        with caplog.at_level(logging.WARNING, logger="torchbridge"):
            _make_server(cfg)
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("cors_origins" in m for m in warns)


# ---------------------------------------------------------------------------
# Tests: rate limiter concurrency
# ---------------------------------------------------------------------------

class TestRateLimiterConcurrency:
    def test_concurrent_requests_respect_limit(self):
        """20 threads × 5 requests each against rpm=10 — at most 10 accepted."""
        limiter = _RateLimiter(rpm=10)
        results: list[bool] = []
        lock = threading.Lock()

        def _send_requests():
            for _ in range(5):
                allowed = limiter.is_allowed("10.0.0.1")
                with lock:
                    results.append(allowed)

        threads = [threading.Thread(target=_send_requests) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        accepted = sum(1 for r in results if r)
        assert accepted <= 10, f"Expected <=10 accepted, got {accepted}"
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"

    def test_no_deadlock_under_load(self):
        """Many threads must complete without deadlock within 5 seconds."""
        limiter = _RateLimiter(rpm=5)
        barrier = threading.Barrier(50, timeout=5)

        def _burst():
            barrier.wait()
            for _ in range(10):
                limiter.is_allowed("10.0.0.2")

        threads = [threading.Thread(target=_burst) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        alive = [t for t in threads if t.is_alive()]
        assert len(alive) == 0, f"{len(alive)} threads still alive (deadlock?)"

    def test_per_ip_isolation_under_concurrency(self):
        """Different IPs must not interfere with each other."""
        limiter = _RateLimiter(rpm=3)
        ip_results: dict[str, list[bool]] = {"A": [], "B": []}
        lock = threading.Lock()

        def _send(ip: str, key: str):
            for _ in range(3):
                allowed = limiter.is_allowed(ip)
                with lock:
                    ip_results[key].append(allowed)

        threads = [
            threading.Thread(target=_send, args=("10.0.0.10", "A")),
            threading.Thread(target=_send, args=("10.0.0.20", "B")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Each IP gets its own 3-request bucket — all 3 should be accepted
        assert sum(ip_results["A"]) == 3, f"IP A: {ip_results['A']}"
        assert sum(ip_results["B"]) == 3, f"IP B: {ip_results['B']}"
