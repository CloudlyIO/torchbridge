"""
Fuzz tests using Hypothesis for config validation and error serialization.

Property-based testing ensures edge cases are covered for all validated
configuration boundaries and error round-trip serialization.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from torchbridge.core.config import (
    AttentionConfig,
    DynamicSparseConfig,
    MemoryConfig,
    PrecisionConfig,
    ValidationConfig,
)
from torchbridge.core.errors import (
    ConfigValidationError,
    HardwareDetectionError,
    OptimizationError,
    TorchBridgeError,
    format_error_chain,
)

# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

unit_float = st.floats(min_value=0.0, max_value=1.0)
positive_float = st.floats(min_value=1e-9, max_value=1e6, allow_nan=False)
positive_int = st.integers(min_value=1, max_value=10_000)
non_negative_int = st.integers(min_value=0, max_value=10_000)

# Floats that are strictly outside [0, 1] (for rejection tests)
out_of_unit_float = st.one_of(
    st.floats(max_value=-1e-9),
    st.floats(min_value=1.0 + 1e-9),
).filter(lambda x: x == x)  # exclude NaN


# ===== PrecisionConfig =====


class TestPrecisionConfigFuzz:
    @given(
        entropy_threshold=unit_float,
        memory_budget=unit_float,
        quality_target=unit_float,
        fp8_interval=positive_int,
        calibration_samples=positive_int,
    )
    @settings(max_examples=50)
    def test_valid_precision_config(
        self, entropy_threshold, memory_budget, quality_target, fp8_interval, calibration_samples
    ):
        cfg = PrecisionConfig(
            entropy_threshold=entropy_threshold,
            memory_budget=memory_budget,
            quality_target=quality_target,
            fp8_interval=fp8_interval,
            calibration_samples=calibration_samples,
        )
        assert 0.0 <= cfg.entropy_threshold <= 1.0
        assert 0.0 <= cfg.memory_budget <= 1.0
        assert 0.0 <= cfg.quality_target <= 1.0
        assert cfg.fp8_interval >= 1
        assert cfg.calibration_samples >= 1

    @given(bad=out_of_unit_float)
    @settings(max_examples=20)
    def test_entropy_threshold_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="entropy_threshold"):
            PrecisionConfig(entropy_threshold=bad)

    @given(bad=out_of_unit_float)
    @settings(max_examples=20)
    def test_memory_budget_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="memory_budget"):
            PrecisionConfig(memory_budget=bad)

    @given(bad=out_of_unit_float)
    @settings(max_examples=20)
    def test_quality_target_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="quality_target"):
            PrecisionConfig(quality_target=bad)

    @given(bad=st.integers(max_value=0))
    @settings(max_examples=20)
    def test_fp8_interval_rejects_non_positive(self, bad):
        with pytest.raises(ValueError, match="fp8_interval"):
            PrecisionConfig(fp8_interval=bad)


# ===== MemoryConfig =====


class TestMemoryConfigFuzz:
    @given(
        memory_fraction=unit_float,
        fragmentation_threshold=unit_float,
        max_memory_gb=positive_float,
        sequence_length_threshold=positive_int,
    )
    @settings(max_examples=50)
    def test_valid_memory_config(
        self, memory_fraction, fragmentation_threshold, max_memory_gb, sequence_length_threshold
    ):
        cfg = MemoryConfig(
            memory_fraction=memory_fraction,
            fragmentation_threshold=fragmentation_threshold,
            max_memory_gb=max_memory_gb,
            sequence_length_threshold=sequence_length_threshold,
        )
        assert 0.0 <= cfg.memory_fraction <= 1.0
        assert 0.0 <= cfg.fragmentation_threshold <= 1.0
        assert cfg.max_memory_gb > 0
        assert cfg.sequence_length_threshold >= 1

    @given(bad=out_of_unit_float)
    @settings(max_examples=20)
    def test_memory_fraction_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="memory_fraction"):
            MemoryConfig(memory_fraction=bad)

    @given(bad=st.floats(max_value=0.0).filter(lambda x: x == x))
    @settings(max_examples=20)
    def test_max_memory_gb_rejects_non_positive(self, bad):
        with pytest.raises(ValueError, match="max_memory_gb"):
            MemoryConfig(max_memory_gb=bad)


# ===== AttentionConfig =====


class TestAttentionConfigFuzz:
    @given(
        sparsity_ratio=unit_float,
        max_sequence_length=positive_int,
        context_parallel_size=positive_int,
    )
    @settings(max_examples=50)
    def test_valid_attention_config(
        self, sparsity_ratio, max_sequence_length, context_parallel_size
    ):
        cfg = AttentionConfig(
            sparsity_ratio=sparsity_ratio,
            max_sequence_length=max_sequence_length,
            context_parallel_size=context_parallel_size,
        )
        assert 0.0 <= cfg.sparsity_ratio <= 1.0
        assert cfg.max_sequence_length >= 1
        assert cfg.context_parallel_size >= 1

    @given(bad=out_of_unit_float)
    @settings(max_examples=20)
    def test_sparsity_ratio_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="sparsity_ratio"):
            AttentionConfig(sparsity_ratio=bad)


# ===== DynamicSparseConfig =====


class TestDynamicSparseConfigFuzz:
    @given(
        sparsity_threshold=unit_float,
        efficiency_target=unit_float,
        min_max=st.tuples(unit_float, unit_float).map(lambda t: tuple(sorted(t))),
    )
    @settings(max_examples=50)
    def test_valid_dynamic_sparse_config(self, sparsity_threshold, efficiency_target, min_max):
        min_s, max_s = min_max
        cfg = DynamicSparseConfig(
            sparsity_threshold=sparsity_threshold,
            efficiency_target=efficiency_target,
            min_sparsity=min_s,
            max_sparsity=max_s,
        )
        assert cfg.min_sparsity <= cfg.max_sparsity

    @given(
        min_s=st.floats(min_value=0.5, max_value=1.0),
        max_s=st.floats(min_value=0.0, max_value=0.49),
    )
    @settings(max_examples=20)
    def test_min_greater_than_max_rejected(self, min_s, max_s):
        with pytest.raises(ValueError, match="min_sparsity"):
            DynamicSparseConfig(min_sparsity=min_s, max_sparsity=max_s)

    @given(bad=out_of_unit_float)
    @settings(max_examples=20)
    def test_sparsity_threshold_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="sparsity_threshold"):
            DynamicSparseConfig(sparsity_threshold=bad)


# ===== ValidationConfig =====


class TestValidationConfigFuzz:
    @given(
        accuracy_threshold=unit_float,
        performance_threshold=unit_float,
        memory_threshold_gb=positive_float,
        benchmark_iterations=positive_int,
        warmup_iterations=non_negative_int,
    )
    @settings(max_examples=50)
    def test_valid_validation_config(
        self,
        accuracy_threshold,
        performance_threshold,
        memory_threshold_gb,
        benchmark_iterations,
        warmup_iterations,
    ):
        cfg = ValidationConfig(
            accuracy_threshold=accuracy_threshold,
            performance_threshold=performance_threshold,
            memory_threshold_gb=memory_threshold_gb,
            benchmark_iterations=benchmark_iterations,
            warmup_iterations=warmup_iterations,
        )
        assert 0.0 <= cfg.accuracy_threshold <= 1.0
        assert 0.0 <= cfg.performance_threshold <= 1.0
        assert cfg.memory_threshold_gb > 0
        assert cfg.benchmark_iterations >= 1
        assert cfg.warmup_iterations >= 0

    @given(bad=out_of_unit_float)
    @settings(max_examples=20)
    def test_accuracy_threshold_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="accuracy_threshold"):
            ValidationConfig(accuracy_threshold=bad)

    @given(bad=st.integers(max_value=-1))
    @settings(max_examples=20)
    def test_warmup_iterations_rejects_negative(self, bad):
        with pytest.raises(ValueError, match="warmup_iterations"):
            ValidationConfig(warmup_iterations=bad)


# ===== Error serialization =====


class TestErrorSerializationFuzz:
    @given(
        message=st.text(min_size=0, max_size=200),
        hint=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    )
    @settings(max_examples=50)
    def test_to_dict_round_trip(self, message, hint):
        err = TorchBridgeError(message, hint=hint)
        d = err.to_dict()
        assert d["error_type"] == "TorchBridgeError"
        assert d["message"] == message
        assert d["hint"] == hint
        assert isinstance(d["details"], dict)

    @given(
        message=st.text(min_size=1, max_size=100),
        details=st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(str.isidentifier),
            values=st.one_of(st.integers(), st.text(max_size=50), st.floats(allow_nan=False)),
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_to_dict_preserves_details(self, message, details):
        err = TorchBridgeError(message, details=details)
        d = err.to_dict()
        assert d["details"] == details

    @given(depth=st.integers(min_value=0, max_value=8))
    @settings(max_examples=20)
    def test_format_error_chain_depth(self, depth):
        # Build a chain of the given depth
        current = None
        for i in range(depth + 1):
            current = TorchBridgeError(f"level-{i}", cause=current)

        result = format_error_chain(current, max_depth=5)
        assert "level-" in result
        # Should contain at most min(depth, 5) + 1 lines
        lines = result.strip().split("\n")
        assert len(lines) <= min(depth, 5) + 1

    @given(
        message=st.text(min_size=1, max_size=100),
        param=st.text(min_size=1, max_size=20).filter(str.isidentifier),
    )
    @settings(max_examples=20)
    def test_subclass_to_dict_preserves_type(self, message, param):
        # OptimizationError accepts (message, details, cause, hint) like base
        err_opt = OptimizationError(message)
        d = err_opt.to_dict()
        assert d["error_type"] == "OptimizationError"
        assert d["message"] == message

        # ConfigValidationError requires extra positional args
        err_cfg = ConfigValidationError(message, parameter=param, value="v", reason="r")
        d = err_cfg.to_dict()
        assert d["error_type"] == "ConfigValidationError"

        # HardwareDetectionError requires (hardware_type, reason)
        err_hw = HardwareDetectionError(hardware_type=message, reason="test")
        d = err_hw.to_dict()
        assert d["error_type"] == "HardwareDetectionError"
