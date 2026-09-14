"""
Unit tests for torchbridge.testing.tolerance_db (v0.5.62 expansion).

Covers:
  - ToleranceEntry source validation
  - Three-level fallback chain: (family, backend, dtype) → (backend, dtype) → default
  - Measured entries match cloud validation data (Qwen3-0.6B)
  - Derived entries are looser than their measured baselines
  - Table completeness: all 5 families × 5 backends × supported dtypes
  - API: families(), is_measured(), register(), register_family(), to_dict(), all_backends()
  - Backward compatibility: get() without model_family still works
"""

import pytest

from torchbridge.testing.tolerance_db import (
    _DEFAULT_TOLERANCE,
    _FAMILY_TOLERANCE_TABLE,
    MODEL_FAMILIES,
    ToleranceDB,
    ToleranceEntry,
    TolerancePair,
)

# ---------------------------------------------------------------------------
# ToleranceEntry
# ---------------------------------------------------------------------------


class TestToleranceEntry:
    def test_valid_measured_source(self):
        e = ToleranceEntry(atol=1e-4, rtol=1e-5, source="measured")
        assert e.source == "measured"

    def test_valid_derived_source(self):
        e = ToleranceEntry(atol=1e-3, rtol=1e-4, source="derived")
        assert e.source == "derived"

    def test_valid_fallback_source(self):
        e = ToleranceEntry(atol=1e-3, rtol=1e-4, source="fallback")
        assert e.source == "fallback"

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            ToleranceEntry(atol=1e-4, rtol=1e-5, source="guessed")

    def test_notes_default_empty(self):
        e = ToleranceEntry(atol=1e-4, rtol=1e-5, source="measured")
        assert e.notes == ""

    def test_notes_stored(self):
        e = ToleranceEntry(atol=1e-4, rtol=1e-5, source="derived", notes="test note")
        assert e.notes == "test note"

    def test_atol_and_rtol_accessible(self):
        e = ToleranceEntry(atol=2e-3, rtol=1e-4, source="measured")
        assert e.atol == 2e-3
        assert e.rtol == 1e-4


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_get_without_family_returns_entry(self):
        db = ToleranceDB()
        tol = db.get("cuda", "float32")
        assert isinstance(tol, ToleranceEntry)
        assert tol.atol > 0

    def test_get_cuda_float32_atol(self):
        db = ToleranceDB()
        tol = db.get("cuda", "float32")
        assert tol.atol == 1e-4

    def test_get_rocm_float32_atol(self):
        db = ToleranceDB()
        tol = db.get("rocm", "float32")
        assert tol.atol == 1e-3

    def test_get_case_insensitive(self):
        db = ToleranceDB()
        tol_lower = db.get("cuda", "float32")
        tol_upper = db.get("CUDA", "FLOAT32")
        assert tol_lower.atol == tol_upper.atol

    def test_unknown_backend_returns_default(self):
        db = ToleranceDB()
        tol = db.get("gaudi", "float32")
        assert tol.atol == _DEFAULT_TOLERANCE.atol

    def test_unknown_backend_source_is_fallback(self):
        db = ToleranceDB()
        tol = db.get("gaudi", "float32")
        assert tol.source == "fallback"

    def test_tolerancepair_still_exists(self):
        pair = TolerancePair(atol=1e-4, rtol=1e-5)
        assert pair.atol == 1e-4

    def test_tolerancepair_importable_from_package(self):
        from torchbridge.testing import TolerancePair as TP

        assert TP is TolerancePair

    def test_whitespace_backend_normalised(self):
        db = ToleranceDB()
        tol_clean = db.get("cuda", "float32")
        tol_padded = db.get(" cuda ", " float32 ")
        assert tol_clean.atol == tol_padded.atol

    def test_whitespace_family_normalised(self):
        db = ToleranceDB()
        tol_clean = db.get("cuda", "float32", model_family="decoder-large")
        tol_padded = db.get("cuda", "float32", model_family=" decoder-large ")
        assert tol_clean.atol == tol_padded.atol

    def test_empty_string_family_falls_back_to_base(self):
        db = ToleranceDB()
        tol_none = db.get("cuda", "float32", model_family=None)
        tol_empty = db.get("cuda", "float32", model_family="")
        assert tol_none.atol == tol_empty.atol

    def test_register_then_get_source_label(self):
        db = ToleranceDB()
        # Overriding a known backend keeps source="measured" (reflects original provenance).
        # This is documented behavior — use register_family() for custom source labels.
        db.register("cuda", "float32", atol=9e-9, rtol=9e-9)
        tol = db.get("cuda", "float32")
        assert tol.source == "measured"  # key still in _TOLERANCE_TABLE

    def test_extra_parameter_merged(self):
        extra = {("gaudi", "float32"): TolerancePair(atol=5e-3, rtol=5e-4)}
        db = ToleranceDB(extra=extra)
        tol = db.get("gaudi", "float32")
        assert tol.atol == 5e-3

    def test_register_family_new_family_appears_in_families(self):
        db = ToleranceDB()
        db.register_family(
            "custom-model", "cuda", "float32", 1e-5, 1e-6, source="measured"
        )
        assert "custom-model" in db.families()


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_family_hit_returns_family_entry(self):
        db = ToleranceDB()
        tol = db.get("cuda", "float32", model_family="decoder-small")
        assert tol.source == "measured"
        assert tol.atol == 1e-4

    def test_unknown_family_falls_back_to_base(self):
        db = ToleranceDB()
        tol_base = db.get("cuda", "float32")
        tol_family = db.get("cuda", "float32", model_family="unknown-family")
        assert tol_base.atol == tol_family.atol

    def test_none_family_falls_back_to_base(self):
        db = ToleranceDB()
        tol_none = db.get("cuda", "float32", model_family=None)
        tol_base = db.get("cuda", "float32")
        assert tol_none.atol == tol_base.atol

    def test_unknown_backend_with_family_falls_back_to_default(self):
        db = ToleranceDB()
        tol = db.get("gaudi", "float32", model_family="decoder-small")
        assert tol.atol == _DEFAULT_TOLERANCE.atol

    def test_unknown_backend_with_family_source_is_fallback(self):
        db = ToleranceDB()
        tol = db.get("gaudi", "float32", model_family="decoder-large")
        assert tol.source == "fallback"

    def test_xla_float16_absent_falls_back_to_base(self):
        # XLA has no float16 entry in the family table
        db = ToleranceDB()
        tol = db.get("xla", "float16", model_family="decoder-small")
        # Falls back — either base xla/float16 or default
        assert tol.atol > 0


# ---------------------------------------------------------------------------
# Measured entries (Qwen3-0.6B cloud validation data)
# ---------------------------------------------------------------------------


class TestMeasuredEntries:
    def test_decoder_small_cuda_float32_is_measured(self):
        db = ToleranceDB()
        tol = db.get("cuda", "float32", model_family="decoder-small")
        assert tol.source == "measured"

    def test_decoder_small_cuda_float32_atol(self):
        db = ToleranceDB()
        tol = db.get("cuda", "float32", model_family="decoder-small")
        # Must match cloud validation worst-case (≤ 1e-4)
        assert tol.atol <= 1e-4

    def test_decoder_small_rocm_float32_is_measured(self):
        db = ToleranceDB()
        tol = db.get("rocm", "float32", model_family="decoder-small")
        assert tol.source == "measured"

    def test_decoder_small_xla_bfloat16_is_flagged_unverified(self):
        """Was asserted as "measured". The measurement it referred to cannot be
        found: the v0.5.31 changelog claims a TPU v5e run and points at report
        files that are not in the repository, while cloud-validation.md records
        TPU as PENDING and hardware-matrix.md has no XLA row. The value is
        unchanged; only the label it carries is."""
        db = ToleranceDB()
        tol = db.get("xla", "bfloat16", model_family="decoder-small")
        assert tol.source == "fallback"
        assert tol.atol == 0.5

    def test_decoder_small_cpu_float32_is_tight(self):
        db = ToleranceDB()
        tol = db.get("cpu", "float32", model_family="decoder-small")
        assert tol.atol <= 1e-5  # CPU is reference backend


# ---------------------------------------------------------------------------
# Derived entries — larger models are looser
# ---------------------------------------------------------------------------


class TestDerivedEntries:
    def test_decoder_large_looser_than_small_cuda(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        large = db.get("cuda", "float32", model_family="decoder-large")
        assert large.atol > small.atol
        assert large.source == "derived"

    def test_decoder_medium_between_small_and_large(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        medium = db.get("cuda", "float32", model_family="decoder-medium")
        large = db.get("cuda", "float32", model_family="decoder-large")
        assert small.atol < medium.atol < large.atol

    def test_encoder_tighter_than_decoder_small(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        enc = db.get("cuda", "float32", model_family="encoder")
        assert enc.atol < small.atol
        assert enc.source == "derived"

    def test_vision_language_looser_than_decoder_small(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        vl = db.get("cuda", "float32", model_family="vision-language")
        assert vl.atol > small.atol
        assert vl.source == "derived"

    def test_decoder_large_rocm_is_derived(self):
        db = ToleranceDB()
        tol = db.get("rocm", "bfloat16", model_family="decoder-large")
        assert tol.source == "derived"

    def test_decoder_large_is_4x_decoder_small_cuda_float32(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        large = db.get("cuda", "float32", model_family="decoder-large")
        assert abs(large.atol / small.atol - 4.0) < 0.01

    def test_decoder_medium_is_2x_decoder_small_cuda_float32(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        medium = db.get("cuda", "float32", model_family="decoder-medium")
        assert abs(medium.atol / small.atol - 2.0) < 0.01

    def test_encoder_is_half_decoder_small_cuda_float32(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        enc = db.get("cuda", "float32", model_family="encoder")
        assert abs(enc.atol / small.atol - 0.5) < 0.01

    def test_vision_language_is_3x_decoder_small_cuda_float32(self):
        db = ToleranceDB()
        small = db.get("cuda", "float32", model_family="decoder-small")
        vl = db.get("cuda", "float32", model_family="vision-language")
        assert abs(vl.atol / small.atol - 3.0) < 0.01

    def test_register_family_invalid_source_raises(self):
        db = ToleranceDB()
        with pytest.raises(ValueError, match="source"):
            db.register_family(
                "decoder-small", "cuda", "float32", 1e-4, 1e-5, source="guessed"
            )

    def test_model_family_case_insensitive(self):
        db = ToleranceDB()
        lower = db.get("cuda", "float32", model_family="decoder-large")
        mixed = db.get("cuda", "float32", model_family="Decoder-Large")
        assert lower.atol == mixed.atol


# ---------------------------------------------------------------------------
# Table completeness
# ---------------------------------------------------------------------------


class TestTableCompleteness:
    # XLA and Trainium support float32 and bfloat16 only (no float16)
    _LIMITED_DTYPES = {"float32", "bfloat16"}
    _ALL_DTYPES = {"float32", "float16", "bfloat16"}
    _BACKENDS = {"cuda", "rocm", "mps", "xla", "cpu", "trainium"}

    def _expected_dtypes(self, backend: str) -> set[str]:
        return (
            self._LIMITED_DTYPES if backend in {"xla", "trainium"} else self._ALL_DTYPES
        )

    def test_all_families_present(self):
        db = ToleranceDB()
        assert set(db.families()) == set(MODEL_FAMILIES)

    def test_decoder_small_all_backends_float32(self):
        db = ToleranceDB()
        for backend in self._BACKENDS:
            tol = db.get(backend, "float32", model_family="decoder-small")
            assert tol.atol > 0, f"Missing decoder-small/{backend}/float32"

    def test_all_families_cuda_float32(self):
        db = ToleranceDB()
        for family in MODEL_FAMILIES:
            tol = db.get("cuda", "float32", model_family=family)
            assert tol.atol > 0, f"Missing {family}/cuda/float32"

    def test_no_xla_float16_family_entries(self):
        for family in MODEL_FAMILIES:
            assert (family, "xla", "float16") not in _FAMILY_TOLERANCE_TABLE

    def test_family_table_size(self):
        # 13 families × 34 entries each = 442 total (v0.5.100)
        # Per family breakdown:
        #   Core backends: cuda/rocm/mps/cpu × 3 dtypes = 12
        #   XLA: float32 + bfloat16 = 2
        #   Trainium: float32 + bfloat16 = 2
        #   Gen-scaling (v0.5.69): cuda_blackwell(3) + cuda_blackwell_consumer(3)
        #                          + rocm_cdna4(3) + xla_v7(2) + trainium_trn3(2) = 13
        #   Gen-scaling (v0.5.100): rocm_rdna4(3) + trainium2(2) = 5
        #   Subtotal per family: 34
        # 13 families × 34 = 442
        assert len(_FAMILY_TOLERANCE_TABLE) == 442

    def test_trainium_has_family_entries(self):
        db = ToleranceDB()
        for family in MODEL_FAMILIES:
            tol = db.get("trainium", "float32", model_family=family)
            assert tol.atol > 0, f"Missing trainium entry for {family}"


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class TestAPI:
    def test_families_returns_all_families(self):
        db = ToleranceDB()
        assert len(db.families()) == 13

    def test_families_sorted(self):
        db = ToleranceDB()
        f = db.families()
        assert f == sorted(f)

    def test_all_backends_returns_list(self):
        db = ToleranceDB()
        backends = db.all_backends()
        assert isinstance(backends, list)
        assert "cuda" in backends
        assert "rocm" in backends

    def test_is_measured_true_for_decoder_small(self):
        db = ToleranceDB()
        assert db.is_measured("cuda", "float32", model_family="decoder-small")

    def test_is_measured_false_for_derived(self):
        db = ToleranceDB()
        assert not db.is_measured("cuda", "float32", model_family="decoder-large")

    def test_is_measured_base_lookup(self):
        db = ToleranceDB()
        # Base (cuda, float32) is in _TOLERANCE_TABLE → measured
        assert db.is_measured("cuda", "float32")

    def test_is_measured_unknown_backend_false(self):
        db = ToleranceDB()
        # Unknown backend → _DEFAULT_TOLERANCE → source="fallback"
        assert not db.is_measured("gaudi", "float32")

    def test_register_overrides_base(self):
        db = ToleranceDB()
        db.register("cuda", "float32", atol=9.9e-9, rtol=9.9e-9)
        tol = db.get("cuda", "float32")
        assert tol.atol == 9.9e-9

    def test_register_family_overrides_family_entry(self):
        db = ToleranceDB()
        db.register_family(
            "decoder-small",
            "cuda",
            "float32",
            atol=9.9e-9,
            rtol=9.9e-9,
            source="measured",
            notes="custom",
        )
        tol = db.get("cuda", "float32", model_family="decoder-small")
        assert tol.atol == 9.9e-9
        assert tol.notes == "custom"

    def test_to_dict_contains_family_entries(self):
        db = ToleranceDB()
        d = db.to_dict()
        assert "decoder-small/cuda/float32" in d
        assert "decoder-large/rocm/bfloat16" in d

    def test_to_dict_has_source_field(self):
        db = ToleranceDB()
        d = db.to_dict()
        entry = d["decoder-small/cuda/float32"]
        assert "source" in entry
        assert entry["source"] == "measured"


# ── v0.5.69: bounds validation and fallback warning ───────────────────────────


class TestRegisterBoundsValidation:
    def test_register_negative_atol_raises(self):
        db = ToleranceDB()
        with pytest.raises(ValueError, match="atol must be >= 0"):
            db.register("cuda", "float32", atol=-0.1, rtol=0.0)

    def test_register_negative_rtol_raises(self):
        db = ToleranceDB()
        with pytest.raises(ValueError, match="rtol must be >= 0"):
            db.register("cuda", "float32", atol=0.0, rtol=-1e-5)

    def test_register_zero_values_ok(self):
        db = ToleranceDB()
        db.register("cuda", "float32", atol=0.0, rtol=0.0)
        tol = db.get("cuda", "float32")
        assert tol.atol == 0.0
        assert tol.rtol == 0.0

    def test_register_family_negative_atol_raises(self):
        db = ToleranceDB()
        with pytest.raises(ValueError, match="atol must be >= 0"):
            db.register_family("decoder-small", "cuda", "float32", atol=-1e-4, rtol=0.0)

    def test_register_family_negative_rtol_raises(self):
        db = ToleranceDB()
        with pytest.raises(ValueError, match="rtol must be >= 0"):
            db.register_family(
                "decoder-small", "cuda", "float32", atol=1e-4, rtol=-1e-6
            )

    def test_register_family_zero_values_ok(self):
        db = ToleranceDB()
        db.register_family(
            "decoder-small", "cpu", "float64", atol=0.0, rtol=0.0, source="measured"
        )
        tol = db.get("cpu", "float64", model_family="decoder-small")
        assert tol.atol == 0.0


class TestFallbackWarning:
    def test_unknown_backend_logs_warning(self, caplog):
        import logging

        db = ToleranceDB()
        with caplog.at_level(
            logging.WARNING, logger="torchbridge.testing.tolerance_db"
        ):
            tol = db.get("gaudi_v3", "float32")
        assert tol.source == "fallback"
        assert any("gaudi_v3" in msg for msg in caplog.messages)

    def test_known_backend_no_fallback_warning(self, caplog):
        import logging

        db = ToleranceDB()
        with caplog.at_level(
            logging.WARNING, logger="torchbridge.testing.tolerance_db"
        ):
            tol = db.get("cuda", "float32")
        assert tol.source in ("measured", "derived")
        assert not any("fallback" in msg.lower() for msg in caplog.messages)


# ── v0.5.69: new hardware generation backends ─────────────────────────────────

_NEW_BACKENDS_THREE_DTYPES = ["cuda_blackwell", "cuda_blackwell_consumer", "rocm_cdna4"]
_NEW_BACKENDS_TWO_DTYPES = ["xla_v7", "trainium_trn3"]  # float32 + bfloat16 only


@pytest.mark.parametrize(
    "backend", _NEW_BACKENDS_THREE_DTYPES + _NEW_BACKENDS_TWO_DTYPES
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_new_backend_decoder_small_entry_exists(backend, dtype):
    db = ToleranceDB()
    tol = db.get(backend, dtype, model_family="decoder-small")
    assert tol.atol > 0


@pytest.mark.parametrize(
    "backend", _NEW_BACKENDS_THREE_DTYPES + _NEW_BACKENDS_TWO_DTYPES
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_new_backend_source_is_derived(backend, dtype):
    db = ToleranceDB()
    tol = db.get(backend, dtype, model_family="decoder-small")
    assert tol.source == "derived"


@pytest.mark.parametrize(
    "backend", _NEW_BACKENDS_THREE_DTYPES + _NEW_BACKENDS_TWO_DTYPES
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_new_backend_notes_nonempty(backend, dtype):
    db = ToleranceDB()
    tol = db.get(backend, dtype, model_family="decoder-small")
    assert tol.notes != ""
