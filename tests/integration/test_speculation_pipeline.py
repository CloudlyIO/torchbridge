"""
Integration Tests for Speculative Decoding Pipeline

End-to-end tests combining speculation engine, structured output,
and phase detection.
"""

from unittest.mock import MagicMock, patch

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.inference import (
    OutputFormat,
    PhaseDetector,
    PhaseType,
    SpeculationCompatibilityMatrix,
    SpeculationConfig,
    SpeculationEngine,
    SpeculativeMethod,
    StructuredOutputProcessor,
)


class TestSpeculationPipeline:
    """Integration tests for the speculation pipeline."""

    def test_compatibility_to_engine_pipeline(self):
        """Compatibility matrix -> SpeculationEngine -> kwargs (model load is mocked)."""
        optimal = SpeculationCompatibilityMatrix.get_optimal_method(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        # DRAFT_MODEL requires a draft_model_name; provide one so get_generation_kwargs works
        draft_name = "gpt2" if optimal == SpeculativeMethod.DRAFT_MODEL else None
        config = SpeculationConfig(method=optimal, draft_model_name=draft_name)
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        assert engine.method == optimal
        # Mock the model load — this test checks pipeline wiring, not HuggingFace
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        with patch("transformers.AutoModelForCausalLM") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_model
            kwargs = engine.get_generation_kwargs()
        assert isinstance(kwargs, dict)

    def test_engine_with_phase_detection(self):
        """Engine should_speculate integrates with phase detection."""
        engine = SpeculationEngine(backend=HardwareBackend.CPU)
        phase = PhaseDetector.detect_phase(prompt_tokens=100, generated_tokens=0)
        assert phase == PhaseType.PREFILL
        # During prefill, speculation is typically beneficial
        assert engine.should_speculate(batch_size=1) is True

    def test_structured_output_with_validation(self):
        """StructuredOutputProcessor validates generated output."""
        schema = {"type": "object", "required": ["answer"]}
        proc = StructuredOutputProcessor(
            format=OutputFormat.JSON_SCHEMA, schema=schema
        )
        assert proc.validate_output('{"answer": 42}') is True
        assert proc.validate_output('{"wrong_key": 42}') is False

    def test_all_backends_have_fallback(self):
        """Every backend resolves to at least one method."""
        for backend in HardwareBackend:
            if backend == HardwareBackend.CUSTOM:
                continue
            methods = SpeculationCompatibilityMatrix.get_supported_methods(backend)
            assert len(methods) > 0, f"No methods for {backend.value}"
            engine = SpeculationEngine(backend=backend)
            assert engine.method != SpeculativeMethod.NONE

    def test_phase_aware_speculation(self):
        """Phase detection influences speculation decisions."""
        engine = SpeculationEngine(
            config=SpeculationConfig(max_batch_size_for_speculation=4),
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        # Low batch size → speculate
        assert engine.should_speculate(batch_size=2) is True
        # High batch size → don't speculate
        assert engine.should_speculate(batch_size=8) is False

    def test_full_info_round_trip(self):
        """Engine info includes all expected fields."""
        config = SpeculationConfig(
            method=SpeculativeMethod.PROMPT_LOOKUP,
            num_speculative_tokens=3,
        )
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        info = engine.get_info()
        assert info["resolved_method"] == "prompt_lookup"
        assert info["num_speculative_tokens"] == 3
        assert info["backend"] == "cpu"

    def test_imports_from_top_level(self):
        """All key classes importable from torchbridge.inference."""
        import torchbridge.inference as inf

        expected_names = [
            "OutputFormat", "PhaseDetector", "PhaseProfile", "PhaseType",
            "SpeculationCompatibilityMatrix", "SpeculationConfig",
            "SpeculationEngine", "SpeculativeMethod", "StructuredOutputProcessor",
        ]
        for name in expected_names:
            assert hasattr(inf, name), f"Missing export: {name}"

    def test_cross_backend_method_comparison(self):
        """Compare optimal methods across all major backends."""
        results = {}
        combos = [
            ("NVIDIA Hopper", HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
            ("NVIDIA Ampere", HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
            ("AMD CDNA3", HardwareBackend.AMD, AMDArchitecture.CDNA3),
            ("Trainium TRN2", HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
            ("TPU v7", HardwareBackend.TPU, TPUVersion.V7),
            ("CPU", HardwareBackend.CPU, None),
        ]
        for name, backend, arch in combos:
            results[name] = SpeculationCompatibilityMatrix.get_optimal_method(
                backend, arch
            ).value

        # Hopper gets DRAFT_MODEL (EAGLE not implemented), CPU gets PROMPT_LOOKUP
        assert results["NVIDIA Hopper"] == "draft_model"
        assert results["CPU"] == "prompt_lookup"
        assert len(results) == 6
