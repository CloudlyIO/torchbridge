"""
Tests for Single Dispatch Path in Attention Registry

Tests that _select_best_implementation() no longer creates its own
AttentionDispatcher — dispatch is delegated to AttentionDispatcher.create_attention().
"""

from unittest.mock import patch

import pytest

from torchbridge.attention.core.registry import _select_best_implementation


class TestDispatchSinglePath:
    """Tests that registry uses heuristic path, not dispatcher."""

    def test_no_dispatcher_import_in_select_best(self):
        """_select_best_implementation should NOT import AttentionDispatcher."""
        # Patch the dispatcher import to track if it's called
        with patch(
            "torchbridge.attention.dispatch.AttentionDispatcher"
        ) as mock_dispatcher:
            try:
                from torchbridge.attention.core.registry import (
                    AttentionConfig,
                )

                config = AttentionConfig(
                    embed_dim=768,
                    num_heads=12,
                    max_sequence_length=512,
                )
                _select_best_implementation(config)
            except Exception:
                pass  # May fail if no implementations registered

            # The dispatcher should NOT have been instantiated
            mock_dispatcher.assert_not_called()

    def test_select_best_returns_string(self):
        """Should return an implementation name string."""
        from torchbridge.attention.core.registry import (
            _ATTENTION_REGISTRY,
            AttentionConfig,
        )

        if not _ATTENTION_REGISTRY:
            pytest.skip("No attention implementations registered")

        config = AttentionConfig(
            embed_dim=768,
            num_heads=12,
            max_sequence_length=512,
        )
        result = _select_best_implementation(config)
        assert isinstance(result, str)
        assert result in _ATTENTION_REGISTRY

    def test_heuristic_prefers_pattern_specific(self):
        """Ring pattern should select ring_attention if available."""
        from torchbridge.attention.core.registry import (
            _ATTENTION_REGISTRY,
            AttentionConfig,
            AttentionPatterns,
        )

        if "ring_attention" not in _ATTENTION_REGISTRY:
            pytest.skip("ring_attention not registered")

        config = AttentionConfig(
            embed_dim=768,
            num_heads=12,
            max_sequence_length=512,
            pattern=AttentionPatterns.RING,
        )
        result = _select_best_implementation(config)
        assert result == "ring_attention"

    def test_docstring_mentions_dispatcher(self):
        """Docstring should mention AttentionDispatcher as the dispatch path."""
        docstring = _select_best_implementation.__doc__
        assert "AttentionDispatcher" in docstring
        assert "heuristic" in docstring.lower()

    def test_fallback_returns_any_registered(self):
        """With no pattern/flash/memory_efficient flags, should still return something."""
        from torchbridge.attention.core.registry import (
            _ATTENTION_REGISTRY,
            AttentionConfig,
        )

        if not _ATTENTION_REGISTRY:
            pytest.skip("No attention implementations registered")

        config = AttentionConfig(
            embed_dim=768,
            num_heads=12,
            max_sequence_length=512,
            use_flash_attention=False,
            use_memory_efficient=False,
        )
        result = _select_best_implementation(config)
        assert result in _ATTENTION_REGISTRY
