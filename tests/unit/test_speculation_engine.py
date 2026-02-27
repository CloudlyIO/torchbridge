"""
Tests for Speculative Decoding Engine

Tests config creation, method resolution, generate kwargs, batch size
gating, and diagnostic info.
"""

from unittest.mock import MagicMock, patch

import pytest

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.inference.speculative.compatibility import (
    SpeculationCompatibilityMatrix,
)
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
        """Draft model method produces assistant_model kwarg as a loaded model object."""
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
        mock_model = MagicMock()
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = mock_model
            kwargs = engine.get_generation_kwargs()
        # assistant_model must be the loaded model object, NOT the name string
        assert kwargs["assistant_model"] is mock_model
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


class TestDraftModelLoading:
    """Tests for lazy draft model loading in DRAFT_MODEL mode."""

    def _draft_engine(self, name: str = "gpt2-small") -> SpeculationEngine:
        config = SpeculationConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_name=name,
            num_speculative_tokens=4,
        )
        return SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )

    def test_is_draft_model_loaded_false_initially(self):
        """Draft model is not loaded until get_generation_kwargs() is called."""
        engine = self._draft_engine()
        assert engine.is_draft_model_loaded is False

    def test_get_generation_kwargs_returns_model_object(self):
        """get_generation_kwargs() must return a loaded model, not a string."""
        engine = self._draft_engine()
        mock_model = MagicMock()
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = mock_model
            kwargs = engine.get_generation_kwargs()
        assert kwargs["assistant_model"] is mock_model
        assert not isinstance(kwargs["assistant_model"], str)

    def test_is_draft_model_loaded_true_after_load(self):
        """is_draft_model_loaded is True after get_generation_kwargs() is called."""
        engine = self._draft_engine()
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = MagicMock()
            engine.get_generation_kwargs()
        assert engine.is_draft_model_loaded is True

    def test_draft_model_loaded_only_once(self):
        """from_pretrained is called only once even with repeated get_generation_kwargs()."""
        engine = self._draft_engine()
        mock_model = MagicMock()
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = mock_model
            engine.get_generation_kwargs()
            engine.get_generation_kwargs()
            engine.get_generation_kwargs()
        MockAuto.from_pretrained.assert_called_once()

    def test_explicit_load_draft_model(self):
        """load_draft_model() can be called explicitly before get_generation_kwargs()."""
        engine = self._draft_engine()
        mock_model = MagicMock()
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = mock_model
            engine.load_draft_model(device="cpu")
            assert engine.is_draft_model_loaded is True
            kwargs = engine.get_generation_kwargs()
        # from_pretrained should still only be called once
        MockAuto.from_pretrained.assert_called_once()
        # .to("cpu") must have been called — not some other device
        MockAuto.from_pretrained.return_value.to.assert_called_once_with("cpu")
        assert kwargs["assistant_model"] is mock_model

    def test_load_draft_model_raises_import_error_if_no_transformers(self):
        """ImportError with clear message when transformers is not installed."""
        engine = self._draft_engine()
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                engine.load_draft_model()

    def test_get_info_still_shows_draft_model_name_string(self):
        """get_info() returns the string name for introspection, not the model object."""
        engine = self._draft_engine("gpt2-small")
        info = engine.get_info()
        assert info["draft_model_name"] == "gpt2-small"
        assert isinstance(info["draft_model_name"], str)

    def test_load_draft_model_passes_device_to_to(self):
        """load_draft_model(device=X) calls .to(X) on the loaded model."""
        engine = self._draft_engine()
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = MagicMock()
            engine.load_draft_model(device="cpu")
        MockAuto.from_pretrained.return_value.to.assert_called_once_with("cpu")

    def test_get_generation_kwargs_uses_backend_device(self):
        """get_generation_kwargs() loads draft model onto the backend device (cuda for CUDA)."""
        engine = self._draft_engine()  # CUDA/AMPERE backend
        mock_model = MagicMock()
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = mock_model
            engine.get_generation_kwargs()
        # CUDA backend → _infer_device() → "cuda" → .to("cuda")
        MockAuto.from_pretrained.return_value.to.assert_called_once_with("cuda")

    def test_get_generation_kwargs_draft_model_no_name_raises(self):
        """ValueError when DRAFT_MODEL is selected but draft_model_name is None."""
        config = SpeculationConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_name=None,
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        with pytest.raises(ValueError, match="draft_model_name"):
            engine.get_generation_kwargs()

    def test_get_generation_kwargs_whitespace_name_raises(self):
        """ValueError when draft_model_name is whitespace-only."""
        config = SpeculationConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_name="   ",
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        with pytest.raises(ValueError, match="empty or whitespace"):
            engine.get_generation_kwargs()

    def test_get_generation_kwargs_explicit_device_overrides_inference(self):
        """Explicit device= arg overrides backend inference (e.g. force CPU on CUDA engine)."""
        engine = self._draft_engine()  # CUDA backend → would infer "cuda"
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = MagicMock()
            engine.get_generation_kwargs(device="cpu")  # explicit override
        # Must use "cpu", NOT "cuda", despite CUDA backend
        MockAuto.from_pretrained.return_value.to.assert_called_once_with("cpu")

    def test_get_generation_kwargs_multi_gpu_device(self):
        """device='cuda:1' routes draft model to second GPU."""
        engine = self._draft_engine()  # CUDA backend
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = MagicMock()
            engine.get_generation_kwargs(device="cuda:1")
        MockAuto.from_pretrained.return_value.to.assert_called_once_with("cuda:1")

    def test_get_generation_kwargs_device_none_uses_inference(self):
        """device=None (default) still uses _infer_device() — backward compatible."""
        engine = self._draft_engine()  # CUDA backend → infers "cuda"
        with patch("transformers.AutoModelForCausalLM") as MockAuto:
            MockAuto.from_pretrained.return_value.to.return_value = MagicMock()
            engine.get_generation_kwargs(device=None)  # explicit None == default
        MockAuto.from_pretrained.return_value.to.assert_called_once_with("cuda")


class TestNotImplementedMethodsAreUnreachable:
    """Proves EAGLE/MEDUSA/LAYER_SKIP NotImplementedError branches are dead code
    under normal usage — the compatibility matrix never resolves to them."""

    _ALL_BACKENDS = [
        (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
        (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
        (HardwareBackend.AMD, AMDArchitecture.CDNA3),
        (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
        (HardwareBackend.TPU, TPUVersion.V5E),
        (HardwareBackend.CPU, None),
    ]
    _NOT_IMPLEMENTED = [
        SpeculativeMethod.EAGLE,
        SpeculativeMethod.MEDUSA,
        SpeculativeMethod.LAYER_SKIP,
    ]

    def test_no_backend_resolves_to_eagle_as_optimal(self):
        """EAGLE is never the optimal method on any backend."""
        for backend, arch in self._ALL_BACKENDS:
            optimal = SpeculationCompatibilityMatrix.get_optimal_method(backend, arch)
            assert optimal != SpeculativeMethod.EAGLE, (
                f"{backend.value}/{arch} resolved to EAGLE — matrix should not include it"
            )

    def test_no_backend_resolves_to_medusa_as_optimal(self):
        """MEDUSA is never the optimal method on any backend."""
        for backend, arch in self._ALL_BACKENDS:
            optimal = SpeculationCompatibilityMatrix.get_optimal_method(backend, arch)
            assert optimal != SpeculativeMethod.MEDUSA

    def test_no_backend_resolves_to_layer_skip_as_optimal(self):
        """LAYER_SKIP is never the optimal method on any backend."""
        for backend, arch in self._ALL_BACKENDS:
            optimal = SpeculationCompatibilityMatrix.get_optimal_method(backend, arch)
            assert optimal != SpeculativeMethod.LAYER_SKIP

    def test_explicitly_requesting_not_implemented_methods_always_falls_back(self):
        """Requesting EAGLE/MEDUSA/LAYER_SKIP via config always falls back in __init__."""
        for method in self._NOT_IMPLEMENTED:
            for backend, arch in self._ALL_BACKENDS:
                config = SpeculationConfig(method=method)
                engine = SpeculationEngine(config=config, backend=backend, architecture=arch)
                assert engine.method not in self._NOT_IMPLEMENTED, (
                    f"{method.value} was NOT fallen back on {backend.value}/{arch}"
                )

    def test_not_implemented_branch_is_safety_net_only(self):
        """NotImplementedError IS raised when _resolved_method is forced — documents the guard."""
        config = SpeculationConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_name="gpt2-small",
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        # Force an unreachable state to confirm the safety net fires
        engine._resolved_method = SpeculativeMethod.EAGLE
        with pytest.raises(NotImplementedError, match="EAGLE"):
            engine.get_generation_kwargs()
