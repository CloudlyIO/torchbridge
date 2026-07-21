"""
tests/unit/test_tolerance_db_coverage.py

CI-visible assertions that enforce minimum DB coverage.
These fail loudly when someone accidentally deletes an entry or the DB regresses.

Run with:
    python -m pytest tests/unit/test_tolerance_db_coverage.py -q
"""

from torchbridge.testing.tolerance_db import (
    _FAMILY_TOLERANCE_TABLE,
    MODEL_FAMILIES,
    ToleranceDB,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORE_BACKENDS = ("cuda", "rocm", "mps", "cpu")
MEASURED_BACKENDS = ("cuda", "rocm", "mps", "cpu", "trainium")

# Backends added in v0.5.69 — derived from measured baselines
GEN_SCALING_BACKENDS = (
    "cuda_blackwell",
    "cuda_blackwell_consumer",
    "rocm_cdna4",
    "xla_v7",
    "trainium_trn3",
    # Added in v0.5.100
    "rocm_rdna4",
    "trainium2",
)

# dtype sets per backend (XLA / Trainium / Neuron don't expose float16)
FULL_DTYPES = ("float32", "float16", "bfloat16")
NO_FLOAT16_DTYPES = ("float32", "bfloat16")

# Minimum total entries guard — update this number if you intentionally add or
# remove entries, but never let it decrease without a corresponding PR review.
_MIN_ENTRY_COUNT = 442


# ---------------------------------------------------------------------------
# 1. decoder-small: every backend must have float32
# ---------------------------------------------------------------------------


def test_all_decoder_small_backends_have_float32():
    """Every backend that appears in _FAMILY_TOLERANCE_TABLE must have a
    float32 entry for decoder-small (the measured baseline family)."""
    backends_in_table = {b for (_, b, _) in _FAMILY_TOLERANCE_TABLE}
    missing = []
    for backend in backends_in_table:
        key = ("decoder-small", backend, "float32")
        if key not in _FAMILY_TOLERANCE_TABLE:
            missing.append(backend)
    assert not missing, (
        f"decoder-small/float32 entry missing for backends: {sorted(missing)}\n"
        "Add measured or derived entries to _FAMILY_TOLERANCE_TABLE in tolerance_db.py."
    )


# ---------------------------------------------------------------------------
# 2. Core backends must have at least one "measured" entry
# ---------------------------------------------------------------------------


def test_measured_entries_exist_for_core_backends():
    """cuda, rocm, mps, and cpu must all have at least one measured entry
    (source == 'measured') for the decoder-small family."""
    db = ToleranceDB()
    not_measured = []
    for backend in MEASURED_BACKENDS:
        entry = db.get(backend, "float32", model_family="decoder-small")
        if entry.source != "measured":
            not_measured.append(backend)
    assert not not_measured, (
        f"Expected source='measured' for decoder-small/float32 on: {not_measured}\n"
        "These backends were validated on real hardware — source label must not be changed."
    )


# ---------------------------------------------------------------------------
# 3. Entry count guard — protects against accidental deletion
# ---------------------------------------------------------------------------


def test_entry_count_at_least_442():
    """_FAMILY_TOLERANCE_TABLE must contain at least 442 entries.

    Breakdown (v0.5.100):
      Per family (34 entries each after v0.5.100):
        Core backends: cuda/rocm/mps/cpu × 3 dtypes = 12
        XLA: float32 + bfloat16 = 2
        Trainium: float32 + bfloat16 = 2
        Gen-scaling (v0.5.69): cuda_blackwell(3) + cuda_blackwell_consumer(3)
                                + rocm_cdna4(3) + xla_v7(2) + trainium_trn3(2) = 13
        Gen-scaling (v0.5.100): rocm_rdna4(3) + trainium2(2) = 5
        Subtotal per family: 34

      13 families × 34 = 442
    """
    count = len(_FAMILY_TOLERANCE_TABLE)
    assert count >= _MIN_ENTRY_COUNT, (
        f"_FAMILY_TOLERANCE_TABLE has only {count} entries; expected >= {_MIN_ENTRY_COUNT}.\n"
        "If entries were intentionally removed, update _MIN_ENTRY_COUNT in this file."
    )


# ---------------------------------------------------------------------------
# 4. All model families covered for every core backend × dtype
# ---------------------------------------------------------------------------


def test_all_model_families_covered_for_core_backends():
    """Every (model_family, core_backend, dtype) triple must be present."""
    missing = []
    for family in MODEL_FAMILIES:
        for backend in CORE_BACKENDS:
            dtypes = FULL_DTYPES if backend not in ("xla",) else NO_FLOAT16_DTYPES
            for dtype in dtypes:
                if (family, backend, dtype) not in _FAMILY_TOLERANCE_TABLE:
                    missing.append((family, backend, dtype))
    assert not missing, (
        f"Missing {len(missing)} entries in _FAMILY_TOLERANCE_TABLE:\n"
        + "\n".join(f"  {f}/{b}/{d}" for f, b, d in missing)
    )


# ---------------------------------------------------------------------------
# 5. gen-scaling backends have bfloat16 for all model families
# ---------------------------------------------------------------------------


def test_gen_scaling_backends_have_bfloat16():
    """All v0.5.69 gen-scaling backends must have bfloat16 entries for every
    model family (it is the primary low-precision dtype on those platforms)."""
    missing = []
    for family in MODEL_FAMILIES:
        for backend in GEN_SCALING_BACKENDS:
            key = (family, backend, "bfloat16")
            if key not in _FAMILY_TOLERANCE_TABLE:
                missing.append(key)
    assert not missing, (
        "Missing bfloat16 entries for gen-scaling backends:\n"
        + "\n".join(f"  {f}/{b}/{d}" for f, b, d in missing)
    )


# ---------------------------------------------------------------------------
# 6. ToleranceDB.get() never raises for any registered key
# ---------------------------------------------------------------------------


def test_tolerance_db_get_never_raises_for_registered_keys():
    """ToleranceDB.get() must return a valid ToleranceEntry (atol >= 0,
    rtol >= 0) for every key present in _FAMILY_TOLERANCE_TABLE."""
    db = ToleranceDB()
    errors = []
    for family, backend, dtype in _FAMILY_TOLERANCE_TABLE:
        try:
            entry = db.get(backend, dtype, model_family=family)
            if entry.atol < 0 or entry.rtol < 0:
                errors.append(
                    f"{family}/{backend}/{dtype}: atol={entry.atol}, rtol={entry.rtol}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{family}/{backend}/{dtype}: raised {exc!r}")
    assert not errors, f"{len(errors)} keys returned invalid entries:\n" + "\n".join(
        errors
    )


# ---------------------------------------------------------------------------
# 7. source labels are valid strings
# ---------------------------------------------------------------------------


def test_all_family_entries_have_valid_source():
    """Every ToleranceEntry in _FAMILY_TOLERANCE_TABLE must have source in
    {'measured', 'derived'} — 'fallback' is reserved for runtime miss paths."""
    valid = {"measured", "derived"}
    bad = [
        f"{fam}/{b}/{d} → source={entry.source!r}"
        for (fam, b, d), entry in _FAMILY_TOLERANCE_TABLE.items()
        if entry.source not in valid
    ]
    assert not bad, (
        "Invalid source labels found (only 'measured' or 'derived' allowed in table):\n"
        + "\n".join(bad)
    )
