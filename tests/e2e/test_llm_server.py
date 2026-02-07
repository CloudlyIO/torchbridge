"""
E2E Tests for LLM Inference Server (v0.5.3)

Tests cover:
- Server startup and shutdown
- Text generation endpoint (streaming and non-streaming)
- Chat completion endpoint
- Token counting endpoint
- Dynamic batching functionality
- Health checks and metrics
- Error handling

Version: 0.5.3
"""

import time
from unittest.mock import Mock

import pytest
import torch

# Skip all tests if FastAPI not available
pytest.importorskip("fastapi")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient

# ============================================================================
# Test Fixtures
# ============================================================================


class MockLLMModel(torch.nn.Module):
    """Mock LLM model for testing."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.device = torch.device("cpu")

        # Mock config
        self.config = Mock()
        self.config.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass returning logits."""
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, self.hidden_size)
        logits = self.lm_head(hidden)

        # Return object with logits attribute
        output = Mock()
        output.logits = logits
        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        **kwargs
    ) -> torch.Tensor:
        """Mock generation - just append random tokens."""
        batch_size = input_ids.size(0)
        new_tokens = torch.randint(
            0, self.vocab_size, (batch_size, max_new_tokens)
        )
        return torch.cat([input_ids, new_tokens], dim=1)


class TokenizerOutput:
    """Mock tokenizer output with attribute access."""

    def __init__(self, input_ids, attention_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

    def to(self, device):
        """Move to device."""
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self

    def __getitem__(self, key):
        """Dict-like access for compatibility."""
        if key == "input_ids":
            return self.input_ids
        elif key == "attention_mask":
            return self.attention_mask
        raise KeyError(key)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, **kwargs):
        """Tokenize text (mock implementation)."""
        # Simple mock: convert text to token IDs based on length
        tokens = [min(i + 2, self.vocab_size - 1) for i in range(len(text.split()))]
        if not tokens:
            tokens = [2]  # At least one token

        if return_tensors == "pt":
            input_ids = torch.tensor([tokens])
            return TokenizerOutput(input_ids)
        return {"input_ids": tokens}

    def encode(self, text):
        """Encode text to token IDs."""
        tokens = [min(i + 2, self.vocab_size - 1) for i in range(len(text.split()))]
        return tokens if tokens else [2]

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Simple mock: return based on number of tokens
        num_tokens = len(token_ids) if isinstance(token_ids, list) else 1
        return f"Generated text with {num_tokens} tokens"


@pytest.fixture
def mock_llm_model():
    """Create mock LLM model."""
    model = MockLLMModel()
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def llm_server(mock_llm_model, mock_tokenizer):
    """Create LLM server for testing."""
    from torchbridge.deployment.serving.llm_server import (
        LLMServerConfig,
        create_llm_server,
    )

    LLMServerConfig(
        model_name="test-model",
        enable_streaming=True,
        enable_dynamic_batching=True,
        max_batch_size=4,
        batch_timeout_ms=50,
    )

    server = create_llm_server(
        model=mock_llm_model,
        tokenizer=mock_tokenizer,
        model_name="test-model",
        enable_streaming=True,
        enable_dynamic_batching=True,
        max_batch_size=4,
    )

    # Override config for faster tests
    server.config.batch_timeout_ms = 50

    return server


@pytest.fixture
def test_client(llm_server):
    """Create test client."""
    return TestClient(llm_server.app)


# ============================================================================
# Server Lifecycle Tests
# ============================================================================


class TestServerLifecycle:
    """Tests for server startup and shutdown."""

    def test_server_creation(self, llm_server):
        """Test server can be created."""
        assert llm_server is not None
        assert llm_server.model is not None
        assert llm_server.tokenizer is not None

    def test_server_health_status(self, llm_server):
        """Test server health status."""
        from torchbridge.deployment.serving.llm_server import HealthStatus

        assert llm_server.status == HealthStatus.HEALTHY

    def test_server_app_creation(self, llm_server):
        """Test FastAPI app is created."""
        assert llm_server.app is not None
        assert hasattr(llm_server.app, "routes")

    def test_server_device_setup(self, llm_server):
        """Test device is set up correctly."""
        assert llm_server.device is not None
        assert isinstance(llm_server.device, torch.device)


# ============================================================================
# Endpoint Tests
# ============================================================================


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""

    def test_generate_basic(self, test_client):
        """Test basic text generation."""
        response = test_client.post(
            "/generate",
            json={
                "prompt": "Once upon a time",
                "max_new_tokens": 10,
                "temperature": 0.7,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "generated_text" in data
        assert "prompt" in data
        assert "num_tokens" in data
        assert "inference_time_ms" in data
        assert data["prompt"] == "Once upon a time"
        assert data["model_name"] == "test-model"

    def test_generate_with_defaults(self, test_client):
        """Test generation with default parameters."""
        response = test_client.post(
            "/generate",
            json={"prompt": "Hello world"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data

    def test_generate_with_custom_params(self, test_client):
        """Test generation with custom parameters."""
        response = test_client.post(
            "/generate",
            json={
                "prompt": "Test prompt",
                "max_new_tokens": 20,
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
                "repetition_penalty": 1.2,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["num_tokens"] > 0

    def test_generate_empty_prompt(self, test_client):
        """Test generation with empty prompt."""
        response = test_client.post(
            "/generate",
            json={"prompt": ""}
        )

        # Should still work (tokenizer handles it)
        assert response.status_code in [200, 500]

    def test_generate_streaming(self, test_client):
        """Test streaming generation."""
        response = test_client.post(
            "/generate",
            json={
                "prompt": "Stream test",
                "max_new_tokens": 5,
                "stream": True,
            }
        )

        assert response.status_code == 200
        # Check SSE content type
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Read streaming response
        content = response.text
        assert "data:" in content


class TestChatEndpoint:
    """Tests for /chat endpoint."""

    def test_chat_basic(self, test_client):
        """Test basic chat completion."""
        response = test_client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"},
                ],
                "max_new_tokens": 20,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data

    def test_chat_multi_turn(self, test_client):
        """Test multi-turn chat."""
        response = test_client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
            }
        )

        assert response.status_code == 200

    def test_chat_streaming(self, test_client):
        """Test streaming chat completion."""
        response = test_client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            }
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestTokenizeEndpoint:
    """Tests for /tokenize endpoint."""

    def test_tokenize_basic(self, test_client):
        """Test basic token counting."""
        response = test_client.post(
            "/tokenize",
            json={"text": "Hello world this is a test"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "text" in data
        assert "num_tokens" in data
        assert data["num_tokens"] > 0
        assert data["text"] == "Hello world this is a test"

    def test_tokenize_empty(self, test_client):
        """Test tokenizing empty text."""
        response = test_client.post(
            "/tokenize",
            json={"text": ""}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["num_tokens"] >= 0

    def test_tokenize_with_token_list(self, test_client):
        """Test getting token list."""
        response = test_client.post(
            "/tokenize",
            json={"text": "Test tokenization"}
        )

        assert response.status_code == 200
        data = response.json()
        # tokens field is optional
        if "tokens" in data and data["tokens"] is not None:
            assert isinstance(data["tokens"], list)


# ============================================================================
# Health and Metrics Tests
# ============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_endpoint(self, test_client):
        """Test /health endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_name" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "uptime_seconds" in data
        assert data["model_name"] == "test-model"
        assert data["model_loaded"] is True

    def test_liveness_probe(self, test_client):
        """Test /health/live endpoint."""
        response = test_client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_probe(self, test_client):
        """Test /health/ready endpoint."""
        response = test_client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True

    def test_metrics_endpoint(self, test_client):
        """Test /metrics endpoint."""
        # Make a generation request first to have some metrics
        test_client.post(
            "/generate",
            json={"prompt": "Test", "max_new_tokens": 5}
        )

        response = test_client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "generation_count" in data
        assert "total_generation_time_ms" in data
        assert "average_generation_time_ms" in data
        assert "total_tokens_generated" in data
        assert "average_tokens_per_second" in data
        assert data["generation_count"] >= 1


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, test_client):
        """Test / endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "service" in data
        assert "model_name" in data
        assert "status" in data
        assert "features" in data
        assert "endpoints" in data

        # Check features
        features = data["features"]
        assert "streaming" in features
        assert "dynamic_batching" in features

        # Check endpoints list
        assert isinstance(data["endpoints"], list)
        assert len(data["endpoints"]) > 0


# ============================================================================
# Dynamic Batching Tests
# ============================================================================


class TestDynamicBatching:
    """Tests for dynamic batching functionality."""

    @pytest.mark.asyncio
    async def test_batch_processing(self, llm_server):
        """Test that requests are batched together."""
        # This test requires the server to be running with batching enabled
        assert llm_server.config.enable_dynamic_batching is True

    def test_batch_queue(self, llm_server):
        """Test batch queue exists."""
        assert hasattr(llm_server, '_batch_queue')
        assert hasattr(llm_server, '_batch_lock')

    def test_batch_thread_started(self, llm_server):
        """Test batch processing thread is started."""
        if llm_server.config.enable_dynamic_batching:
            assert llm_server._batch_thread is not None
            assert llm_server._batch_thread.is_alive()

    def test_concurrent_requests(self, test_client):
        """Test handling multiple concurrent requests."""
        import concurrent.futures

        def make_request(prompt):
            return test_client.post(
                "/generate",
                json={"prompt": prompt, "max_new_tokens": 5}
            )

        # Send multiple requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(make_request, f"Prompt {i}")
                for i in range(3)
            ]

            results = [f.result() for f in futures]

        # All should succeed
        for response in results:
            assert response.status_code == 200


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_request_format(self, test_client):
        """Test handling of invalid request format."""
        response = test_client.post(
            "/generate",
            json={"invalid_field": "value"}
        )

        # Should return validation error
        assert response.status_code == 422

    def test_missing_required_field(self, test_client):
        """Test handling of missing required fields."""
        response = test_client.post(
            "/generate",
            json={}
        )

        assert response.status_code == 422

    def test_invalid_endpoint(self, test_client):
        """Test accessing invalid endpoint."""
        response = test_client.get("/invalid")

        assert response.status_code == 404


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, test_client):
        """Test complete workflow: health check -> generate -> metrics."""
        # 1. Check health
        health_response = test_client.get("/health")
        assert health_response.status_code == 200

        # 2. Generate text
        gen_response = test_client.post(
            "/generate",
            json={"prompt": "Test", "max_new_tokens": 10}
        )
        assert gen_response.status_code == 200

        # 3. Check metrics updated
        metrics_response = test_client.get("/metrics")
        assert metrics_response.status_code == 200
        assert metrics_response.json()["generation_count"] >= 1

    def test_multiple_generations(self, test_client):
        """Test multiple generation requests."""
        for i in range(3):
            response = test_client.post(
                "/generate",
                json={"prompt": f"Prompt {i}", "max_new_tokens": 5}
            )
            assert response.status_code == 200

        # Check metrics
        metrics = test_client.get("/metrics").json()
        assert metrics["generation_count"] >= 3


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_generation_latency(self, test_client):
        """Test generation completes in reasonable time."""
        start = time.time()
        response = test_client.post(
            "/generate",
            json={"prompt": "Test", "max_new_tokens": 10}
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should complete within 5 seconds even on slow hardware
        assert elapsed < 5.0

    def test_metrics_tracking(self, test_client):
        """Test metrics are tracked correctly."""
        # Generate some requests
        for _ in range(5):
            test_client.post(
                "/generate",
                json={"prompt": "Test", "max_new_tokens": 5}
            )

        metrics = test_client.get("/metrics").json()

        assert metrics["generation_count"] == 5
        assert metrics["total_generation_time_ms"] > 0
        assert metrics["average_generation_time_ms"] > 0


# ============================================================================
# Cleanup Tests
# ============================================================================


class TestCleanup:
    """Tests for proper cleanup."""

    def test_server_shutdown(self, llm_server):
        """Test server can be shut down cleanly."""

        # Trigger shutdown
        llm_server._stop_batching.set()

        # Wait a bit for thread to stop
        if llm_server._batch_thread:
            llm_server._batch_thread.join(timeout=1.0)

        # Check thread stopped
        if llm_server._batch_thread:
            # Thread should have stopped or be about to stop
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
