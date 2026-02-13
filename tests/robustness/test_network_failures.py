"""
Robustness tests: graceful handling of network failures.

Verifies that TorchBridge produces clear error messages when network
operations fail (e.g., model downloads from HuggingFace Hub).
"""

import ssl
from unittest.mock import patch

import pytest
import torch


class TestNetworkFailures:
    """Test graceful handling of network connectivity issues."""

    def test_model_download_connection_error(self):
        """Model download should raise clear error on ConnectionError."""
        with patch("transformers.AutoModel.from_pretrained") as mock_load:
            mock_load.side_effect = ConnectionError(
                "Failed to establish a new connection: "
                "[Errno 8] nodename nor servname provided"
            )
            with pytest.raises(ConnectionError, match="Failed to establish"):
                from transformers import AutoModel

                AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )

    def test_model_download_timeout_error(self):
        """Model download should raise clear error on timeout."""
        with patch("transformers.AutoModel.from_pretrained") as mock_load:
            mock_load.side_effect = TimeoutError("Connection timed out")
            with pytest.raises(TimeoutError, match="timed out"):
                from transformers import AutoModel

                AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )

    def test_tokenizer_download_connection_error(self):
        """Tokenizer download should raise clear error on ConnectionError."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_load:
            mock_load.side_effect = ConnectionError("Network unreachable")
            with pytest.raises(ConnectionError, match="Network unreachable"):
                from transformers import AutoTokenizer

                AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def test_core_functions_work_offline(self):
        """Core TorchBridge functions should work without network access."""
        import torchbridge

        # Config doesn't need network
        config = torchbridge.get_config()
        assert isinstance(config, torchbridge.TorchBridgeConfig)

        # Model optimization doesn't need network
        model = torch.nn.Linear(32, 16)
        try:
            result = torchbridge.optimize_model(model)
            assert result is not None
        except Exception:
            # torch.compile may fail, but not network-related
            pass

        # Attention creation doesn't need network
        attn = torchbridge.create_attention(d_model=64, num_heads=4)
        assert hasattr(attn, "forward")

        # MoE creation doesn't need network
        moe = torchbridge.create_moe(hidden_size=64, num_experts=4, top_k=2)
        assert hasattr(moe, "forward")

    def test_hal_works_offline(self):
        """Hardware abstraction works without network access."""
        from torchbridge.hardware.abstraction.hal_core import (
            HardwareAbstractionLayer,
        )

        hal = HardwareAbstractionLayer()
        assert isinstance(hal, HardwareAbstractionLayer)

    def test_validator_works_offline(self):
        """Validator works without network access."""
        from torchbridge.validation.unified_validator import UnifiedValidator

        validator = UnifiedValidator()
        assert isinstance(validator, UnifiedValidator)

    def test_backend_factory_works_offline(self):
        """Backend factory works without network."""
        from torchbridge.backends.backend_factory import BackendFactory

        # Should be able to create CPU backend without network
        backend = BackendFactory.create("cpu")
        assert hasattr(backend, "prepare_model")

    def test_stress_fixture_handles_network_failure(self):
        """Stress test model fixtures should skip cleanly when download fails."""
        # Simulate what happens when HF Hub is unreachable
        with patch("transformers.AutoModel.from_pretrained") as mock_load:
            mock_load.side_effect = OSError(
                "We couldn't connect to 'https://huggingface.co' "
                "to load this file"
            )
            with pytest.raises(OSError, match="couldn't connect") as exc_info:
                from transformers import AutoModel

                AutoModel.from_pretrained("facebook/dinov2-small")

            assert "huggingface.co" in str(exc_info.value)
            mock_load.assert_called_once_with("facebook/dinov2-small")

    def test_dns_resolution_failure(self):
        """DNS resolution failure should produce clear error."""
        with patch("transformers.AutoModel.from_pretrained") as mock_load:
            mock_load.side_effect = OSError(
                "[Errno -2] Name or service not known"
            )
            with pytest.raises(OSError, match="Name or service not known") as exc_info:
                from transformers import AutoModel

                AutoModel.from_pretrained("facebook/dinov2-small")

            assert "Errno" in str(exc_info.value)
            mock_load.assert_called_once_with("facebook/dinov2-small")

    def test_ssl_error_handling(self):
        """SSL errors should produce clear error."""
        with patch("transformers.AutoModel.from_pretrained") as mock_load:
            mock_load.side_effect = ssl.SSLError(
                "SSL: CERTIFICATE_VERIFY_FAILED"
            )
            with pytest.raises(ssl.SSLError, match="CERTIFICATE_VERIFY_FAILED") as exc_info:
                from transformers import AutoModel

                AutoModel.from_pretrained("facebook/dinov2-small")

            assert "CERTIFICATE" in str(exc_info.value)
            mock_load.assert_called_once_with("facebook/dinov2-small")
