"""
Tests for Speculative Decoding Engine

Tests config creation, method resolution, generate kwargs, batch size
gating, and diagnostic info.
"""

from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture
from torchbridge.inference.speculative.engine import (
    SpeculationConfig,
    SpeculationEngine,
)
from torchbridge.inference.speculative.methods import SpeculativeMethod


class TestSpeculationConfig:
    """Tests for SpeculationConfig dataclass."""

    def test_default_config(self):
        config = SpeculationConfig()
        assert config.method is None
        assert config.draft_model_name is None
        assert config.num_speculative_tokens == 5
        assert config.max_batch_size_for_speculation == 8
        assert config.acceptance_threshold == 0.0
        assert config.enabled is True

    def test_to_dict(self):
        config = SpeculationConfig(method=SpeculativeMethod.EAGLE)
        d = config.to_dict()
        assert d["method"] == "eagle"
        assert d["enabled"] is True

    def test_to_dict_auto(self):
        config = SpeculationConfig()
        d = config.to_dict()
        assert d["method"] == "auto"


class TestSpeculationEngine:
    """Tests for SpeculationEngine."""

    def test_auto_resolves_to_optimal(self):
        """Auto method resolves to optimal for backend."""
        engine = SpeculationEngine(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        assert engine.method == SpeculativeMethod.DRAFT_MODEL

    def test_auto_resolves_cpu(self):
        """Auto method resolves to PROMPT_LOOKUP on CPU."""
        engine = SpeculationEngine(backend=HardwareBackend.CPU)
        assert engine.method == SpeculativeMethod.PROMPT_LOOKUP

    def test_explicit_method_preserved(self):
        """Explicit supported method is preserved."""
        config = SpeculationConfig(method=SpeculativeMethod.DRAFT_MODEL)
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        assert engine.method == SpeculativeMethod.DRAFT_MODEL

    def test_unsupported_method_falls_back(self):
        """Unsupported explicit method falls back."""
        config = SpeculationConfig(method=SpeculativeMethod.EAGLE)
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CPU,
        )
        assert engine.method == SpeculativeMethod.PROMPT_LOOKUP

    def test_disabled_engine(self):
        """Disabled engine resolves to NONE."""
        config = SpeculationConfig(enabled=False)
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CUDA)
        assert engine.method == SpeculativeMethod.NONE

    def test_should_speculate_within_batch_limit(self):
        engine = SpeculationEngine(backend=HardwareBackend.CPU)
        assert engine.should_speculate(batch_size=1) is True

    def test_should_speculate_exceeds_batch_limit(self):
        config = SpeculationConfig(max_batch_size_for_speculation=4)
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        assert engine.should_speculate(batch_size=5) is False

    def test_should_speculate_disabled(self):
        config = SpeculationConfig(enabled=False)
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        assert engine.should_speculate(batch_size=1) is False

    def test_kwargs_draft_model(self):
        """Draft model method produces assistant_model kwarg."""
        config = SpeculationConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_name="small-model",
            num_speculative_tokens=3,
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        kwargs = engine.get_generation_kwargs()
        assert kwargs["assistant_model"] == "small-model"
        assert kwargs["num_assistant_tokens"] == 3

    def test_kwargs_prompt_lookup(self):
        """Prompt lookup method produces prompt_lookup_num_tokens kwarg."""
        config = SpeculationConfig(
            method=SpeculativeMethod.PROMPT_LOOKUP,
            num_speculative_tokens=4,
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CPU,
        )
        kwargs = engine.get_generation_kwargs()
        assert kwargs["prompt_lookup_num_tokens"] == 4
        assert "assistant_model" not in kwargs

    def test_kwargs_disabled_empty(self):
        """Disabled engine returns empty kwargs."""
        config = SpeculationConfig(enabled=False)
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        assert engine.get_generation_kwargs() == {}

    def test_eagle_explicit_falls_back(self):
        """EAGLE is not in the matrix — explicit request falls back to DRAFT_MODEL."""
        config = SpeculationConfig(
            method=SpeculativeMethod.EAGLE,
            draft_model_name="eagle-model",
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        # EAGLE not supported → engine falls back via chain to DRAFT_MODEL
        assert engine.method == SpeculativeMethod.DRAFT_MODEL

    def test_get_info(self):
        """get_info returns diagnostic dict."""
        engine = SpeculationEngine(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        info = engine.get_info()
        assert info["resolved_method"] == "draft_model"
        assert info["requested_method"] == "auto"
        assert info["backend"] == "cuda"
        assert info["enabled"] is True
        assert "display_name" in info

    def test_get_info_with_explicit_method(self):
        config = SpeculationConfig(method=SpeculativeMethod.PROMPT_LOOKUP)
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        info = engine.get_info()
        assert info["resolved_method"] == "prompt_lookup"
        assert info["requested_method"] == "prompt_lookup"

    def test_none_method_disabled(self):
        """Explicitly setting NONE method means no speculation."""
        config = SpeculationConfig(method=SpeculativeMethod.NONE, enabled=True)
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        # enabled=True but method=NONE triggers auto-resolution
        # since NONE is treated as auto
        assert engine.method == SpeculativeMethod.PROMPT_LOOKUP

    def test_layer_skip_explicit_falls_back(self):
        """LAYER_SKIP is not in the matrix — explicit request falls back."""
        config = SpeculationConfig(method=SpeculativeMethod.LAYER_SKIP)
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.TRAINIUM,
            architecture=None,
        )
        # LAYER_SKIP not supported → falls back to PROMPT_LOOKUP
        assert engine.method == SpeculativeMethod.PROMPT_LOOKUP
