"""
Tests for Disaggregated Serving Phase Detection

Tests phase type detection, hardware profiling, and edge cases.
"""

from torchbridge.core.config import HardwareBackend
from torchbridge.inference.phase_detection import (
    PhaseDetector,
    PhaseProfile,
    PhaseType,
)


class TestPhaseType:
    """Tests for PhaseType enum."""

    def test_all_phases_exist(self):
        expected = {"prefill", "decode", "mixed"}
        actual = {p.value for p in PhaseType}
        assert actual == expected


class TestPhaseDetector:
    """Tests for PhaseDetector.detect_phase."""

    def test_zero_generated_is_prefill(self):
        """No generated tokens → prefill."""
        phase = PhaseDetector.detect_phase(prompt_tokens=100, generated_tokens=0)
        assert phase == PhaseType.PREFILL

    def test_zero_prompt_is_decode(self):
        """No prompt tokens → decode."""
        phase = PhaseDetector.detect_phase(prompt_tokens=0, generated_tokens=10)
        assert phase == PhaseType.DECODE

    def test_small_ratio_is_prefill(self):
        """gen/prompt < 0.1 → prefill."""
        phase = PhaseDetector.detect_phase(prompt_tokens=100, generated_tokens=5)
        assert phase == PhaseType.PREFILL

    def test_large_ratio_is_decode(self):
        """gen/prompt >= 1.0 → decode."""
        phase = PhaseDetector.detect_phase(prompt_tokens=100, generated_tokens=100)
        assert phase == PhaseType.DECODE

    def test_medium_ratio_is_mixed(self):
        """0.1 <= gen/prompt < 1.0 → mixed."""
        phase = PhaseDetector.detect_phase(prompt_tokens=100, generated_tokens=50)
        assert phase == PhaseType.MIXED

    def test_boundary_prefill(self):
        """Exactly at threshold boundary (gen/prompt = 0.1) is mixed."""
        phase = PhaseDetector.detect_phase(prompt_tokens=100, generated_tokens=10)
        assert phase == PhaseType.MIXED

    def test_boundary_decode(self):
        """Exactly at decode threshold (gen/prompt = 1.0) is decode."""
        phase = PhaseDetector.detect_phase(prompt_tokens=100, generated_tokens=100)
        assert phase == PhaseType.DECODE

    def test_very_long_generation(self):
        """Very long generation is decode."""
        phase = PhaseDetector.detect_phase(prompt_tokens=10, generated_tokens=1000)
        assert phase == PhaseType.DECODE


class TestHardwareProfile:
    """Tests for PhaseDetector.get_hardware_profile."""

    def test_prefill_is_compute_bound(self):
        profile = PhaseDetector.get_hardware_profile(PhaseType.PREFILL)
        assert profile.is_compute_bound is True
        assert profile.is_memory_bound is False

    def test_decode_is_memory_bound(self):
        profile = PhaseDetector.get_hardware_profile(PhaseType.DECODE)
        assert profile.is_compute_bound is False
        assert profile.is_memory_bound is True

    def test_mixed_is_both(self):
        profile = PhaseDetector.get_hardware_profile(PhaseType.MIXED)
        assert profile.is_compute_bound is True
        assert profile.is_memory_bound is True

    def test_prefill_recommendation_cuda(self):
        profile = PhaseDetector.get_hardware_profile(
            PhaseType.PREFILL, HardwareBackend.CUDA
        )
        assert isinstance(profile, PhaseProfile)
        assert profile.phase == PhaseType.PREFILL
        assert "FLOPS" in profile.recommended_hardware or "GPU" in profile.recommended_hardware

    def test_decode_recommendation_amd(self):
        profile = PhaseDetector.get_hardware_profile(
            PhaseType.DECODE, HardwareBackend.AMD
        )
        assert "bandwidth" in profile.recommended_hardware.lower() or "MI300X" in profile.recommended_hardware

    def test_profile_has_description(self):
        """All profiles have non-empty descriptions."""
        for phase in PhaseType:
            profile = PhaseDetector.get_hardware_profile(phase)
            assert len(profile.description) > 0
