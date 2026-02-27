"""
Tests for speculative decoding method clarity (v0.5.42).

Covers:
- requires_custom_arch flag set correctly on EAGLE, MEDUSA, LAYER_SKIP
- DRAFT_MODEL and PROMPT_LOOKUP do NOT have requires_custom_arch
- NotImplementedError messages are descriptive and suggest alternatives
- SpeculationEngine.get_available_methods() excludes custom-arch methods
"""

from __future__ import annotations

import pytest

from torchbridge.backends import BackendType
from torchbridge.inference.speculative.engine import (
    SpeculationConfig,
    SpeculationEngine,
)
from torchbridge.inference.speculative.methods import (
    SPECULATIVE_METHOD_SPECS,
    SpeculativeMethod,
)

# ---------------------------------------------------------------------------
# Tests: requires_custom_arch flag
# ---------------------------------------------------------------------------

class TestRequiresCustomArchFlag:
    @pytest.mark.parametrize("method", [
        SpeculativeMethod.EAGLE,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.LAYER_SKIP,
    ])
    def test_custom_arch_methods_flagged(self, method):
        """EAGLE, MEDUSA, LAYER_SKIP must have requires_custom_arch=True."""
        spec = SPECULATIVE_METHOD_SPECS[method]
        assert spec.requires_custom_arch is True, (
            f"{method.name} should have requires_custom_arch=True"
        )

    @pytest.mark.parametrize("method", [
        SpeculativeMethod.DRAFT_MODEL,
        SpeculativeMethod.PROMPT_LOOKUP,
        SpeculativeMethod.NONE,
    ])
    def test_standard_methods_not_flagged(self, method):
        """DRAFT_MODEL, PROMPT_LOOKUP, NONE must have requires_custom_arch=False."""
        spec = SPECULATIVE_METHOD_SPECS[method]
        assert spec.requires_custom_arch is False, (
            f"{method.name} should have requires_custom_arch=False"
        )

    @pytest.mark.parametrize("method", [
        SpeculativeMethod.EAGLE,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.LAYER_SKIP,
    ])
    def test_custom_arch_methods_not_generate_compatible(self, method):
        """Custom-arch methods must also be is_generate_compatible=False."""
        spec = SPECULATIVE_METHOD_SPECS[method]
        assert spec.is_generate_compatible is False


# ---------------------------------------------------------------------------
# Tests: NotImplementedError messages
# ---------------------------------------------------------------------------

def _engine_for_method(method: SpeculativeMethod) -> SpeculationEngine:
    """Create an engine with the given method forcefully set (bypasses compat matrix)."""
    config = SpeculationConfig(method=SpeculativeMethod.NONE)  # auto → NONE first
    engine = SpeculationEngine(config, BackendType.CPU)
    # Override internal resolved method to simulate forceful selection
    engine._resolved_method = method  # type: ignore[attr-defined]
    engine._config = SpeculationConfig(method=method)  # type: ignore[attr-defined]
    return engine


class TestNotImplementedErrorMessages:
    def test_eagle_error_mentions_custom_architecture(self):
        engine = _engine_for_method(SpeculativeMethod.EAGLE)
        with pytest.raises(NotImplementedError, match="custom model architecture"):
            engine.get_generation_kwargs()

    def test_eagle_error_suggests_alternatives(self):
        engine = _engine_for_method(SpeculativeMethod.EAGLE)
        with pytest.raises(NotImplementedError, match="DRAFT_MODEL|PROMPT_LOOKUP"):
            engine.get_generation_kwargs()

    def test_medusa_error_mentions_custom_architecture(self):
        engine = _engine_for_method(SpeculativeMethod.MEDUSA)
        with pytest.raises(NotImplementedError, match="custom model architecture"):
            engine.get_generation_kwargs()

    def test_layer_skip_error_mentions_custom_architecture(self):
        engine = _engine_for_method(SpeculativeMethod.LAYER_SKIP)
        with pytest.raises(NotImplementedError, match="early-exit"):
            engine.get_generation_kwargs()

    def test_layer_skip_error_suggests_prompt_lookup(self):
        engine = _engine_for_method(SpeculativeMethod.LAYER_SKIP)
        with pytest.raises(NotImplementedError, match="PROMPT_LOOKUP"):
            engine.get_generation_kwargs()


# ---------------------------------------------------------------------------
# Tests: get_available_methods()
# ---------------------------------------------------------------------------

class TestGetAvailableMethods:
    def test_cpu_available_methods_excludes_custom_arch(self):
        """CPU backend should return only PROMPT_LOOKUP (no custom-arch methods)."""
        config = SpeculationConfig(method=SpeculativeMethod.NONE)
        engine = SpeculationEngine(config, BackendType.CPU)
        available = engine.get_available_methods()
        custom_arch = {
            SpeculativeMethod.EAGLE,
            SpeculativeMethod.MEDUSA,
            SpeculativeMethod.LAYER_SKIP,
        }
        assert not any(m in custom_arch for m in available), (
            f"Custom-arch methods should not appear in get_available_methods(): {available}"
        )

    def test_cpu_available_methods_includes_prompt_lookup(self):
        """CPU backend must include PROMPT_LOOKUP in available methods."""
        config = SpeculationConfig(method=SpeculativeMethod.NONE)
        engine = SpeculationEngine(config, BackendType.CPU)
        available = engine.get_available_methods()
        assert SpeculativeMethod.PROMPT_LOOKUP in available

    def test_available_methods_are_all_generate_compatible(self):
        """Every method returned by get_available_methods() must be generate-compatible."""
        config = SpeculationConfig(method=SpeculativeMethod.NONE)
        engine = SpeculationEngine(config, BackendType.CPU)
        for method in engine.get_available_methods():
            spec = SPECULATIVE_METHOD_SPECS.get(method)
            if spec is not None:
                assert spec.is_generate_compatible, (
                    f"{method.name} is not generate-compatible but appeared in "
                    "get_available_methods()"
                )

    def test_available_methods_returns_list(self):
        """get_available_methods() must return a list."""
        config = SpeculationConfig(method=SpeculativeMethod.NONE)
        engine = SpeculationEngine(config, BackendType.CPU)
        result = engine.get_available_methods()
        assert isinstance(result, list)
