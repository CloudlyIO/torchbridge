"""
Unit tests for torchbridge.inference.disaggregated.

Covers:
  - DisaggregatedRoleConfig / DisaggregatedFleetConfig dataclass fields
  - JSON serialisability of to_dict()
  - KV dtype matrix lookups (known entries + unknown-arch fallback)
  - Transfer format matrix
  - Memory and batch heuristics
  - Memory override respected
  - Notes non-empty
  - Large model triggers extra notes
"""

import json

from torchbridge.inference.disaggregated import (
    _TRANSFER_FORMAT_MATRIX,
    DisaggregatedFleetAdvisor,
    DisaggregatedFleetConfig,
    DisaggregatedRoleConfig,
    _lookup_kv_dtype,
)

# ---------------------------------------------------------------------------
# Dataclass field coverage
# ---------------------------------------------------------------------------


class TestRoleConfigFields:
    def test_role_config_fields_exist(self):
        cfg = DisaggregatedRoleConfig(
            role="prefill",
            backend="cuda",
            architecture="hopper",
            kv_dtype="bfloat16",
            kv_cache_budget_gb=16.0,
            max_batch_size=8,
            max_seq_len=32768,
            notes=["some note"],
        )
        assert cfg.role == "prefill"
        assert cfg.backend == "cuda"
        assert cfg.architecture == "hopper"
        assert cfg.kv_dtype == "bfloat16"
        assert cfg.kv_cache_budget_gb == 16.0
        assert cfg.max_batch_size == 8
        assert cfg.max_seq_len == 32768
        assert cfg.notes == ["some note"]

    def test_fleet_config_fields_exist(self):
        role = DisaggregatedRoleConfig(
            role="decode",
            backend="rocm",
            architecture="cdna3",
            kv_dtype="int8",
            kv_cache_budget_gb=64.0,
            max_batch_size=128,
            max_seq_len=2048,
        )
        fleet = DisaggregatedFleetConfig(
            model_params=7_000_000_000,
            prefill=role,
            decode=role,
            kv_transfer_format="float16",
            notes=["fleet note"],
        )
        assert fleet.model_params == 7_000_000_000
        assert fleet.prefill is role
        assert fleet.decode is role
        assert fleet.kv_transfer_format == "float16"
        assert fleet.notes == ["fleet note"]

    def test_to_dict_is_json_serialisable(self):
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            prefill_arch="hopper",
            decode_backend="rocm",
            decode_arch="cdna3",
        )
        d = cfg.to_dict()
        # Must not raise
        serialised = json.dumps(d)
        assert len(serialised) > 0

    def test_to_dict_roundtrips_through_json(self):
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="rocm",
        )
        parsed = json.loads(json.dumps(cfg.to_dict()))
        assert "prefill" in parsed
        assert "decode" in parsed


# ---------------------------------------------------------------------------
# KV dtype matrix
# ---------------------------------------------------------------------------


class TestKvDtypeMatrix:
    def test_prefill_nvidia_hopper_uses_bfloat16(self):
        dtype, _ = _lookup_kv_dtype("prefill", "cuda", "hopper")
        assert dtype == "bfloat16"

    def test_prefill_nvidia_ampere_uses_float16(self):
        dtype, _ = _lookup_kv_dtype("prefill", "cuda", "ampere")
        assert dtype == "float16"

    def test_prefill_amd_cdna3_uses_bfloat16(self):
        dtype, _ = _lookup_kv_dtype("prefill", "rocm", "cdna3")
        assert dtype == "bfloat16"

    def test_prefill_amd_cdna2_uses_float16(self):
        dtype, _ = _lookup_kv_dtype("prefill", "rocm", "cdna2")
        assert dtype == "float16"

    def test_decode_nvidia_hopper_uses_int8(self):
        dtype, _ = _lookup_kv_dtype("decode", "cuda", "hopper")
        assert dtype == "int8"

    def test_decode_amd_cdna3_uses_int8(self):
        dtype, _ = _lookup_kv_dtype("decode", "rocm", "cdna3")
        assert dtype == "int8"

    def test_decode_amd_cdna2_uses_float16(self):
        # CDNA2: int8 KV less stable
        dtype, _ = _lookup_kv_dtype("decode", "rocm", "cdna2")
        assert dtype == "float16"

    def test_unknown_arch_falls_back_to_none_key(self):
        # "volta" is not in the matrix; must fall back to (role, "cuda", None)
        dtype, note = _lookup_kv_dtype("prefill", "cuda", "volta")
        assert dtype == "float16"
        assert "unknown" in note.lower() or "conservative" in note.lower()

    def test_cpu_role_uses_float32(self):
        for role in ("prefill", "decode"):
            dtype, _ = _lookup_kv_dtype(role, "cpu", None)
            assert dtype == "float32", f"Expected float32 for cpu/{role}"

    def test_tpu_decode_uses_bfloat16(self):
        # XLA doesn't support int8 KV natively
        dtype, _ = _lookup_kv_dtype("decode", "tpu", None)
        assert dtype == "bfloat16"

    def test_prefill_nvidia_blackwell_uses_bfloat16(self):
        dtype, _ = _lookup_kv_dtype("prefill", "cuda", "blackwell")
        assert dtype == "bfloat16"


# ---------------------------------------------------------------------------
# Transfer format matrix
# ---------------------------------------------------------------------------


class TestTransferFormatMatrix:
    def test_nvidia_to_nvidia_transfer_bfloat16(self):
        fmt = _TRANSFER_FORMAT_MATRIX[("cuda", "cuda")]
        assert fmt == "bfloat16"

    def test_nvidia_to_amd_transfer_float16(self):
        fmt = _TRANSFER_FORMAT_MATRIX[("cuda", "rocm")]
        assert fmt == "float16"

    def test_amd_to_nvidia_transfer_float16(self):
        fmt = _TRANSFER_FORMAT_MATRIX[("rocm", "cuda")]
        assert fmt == "float16"

    def test_amd_to_amd_transfer_bfloat16(self):
        fmt = _TRANSFER_FORMAT_MATRIX[("rocm", "rocm")]
        assert fmt == "bfloat16"

    def test_same_cpu_transfer_float32(self):
        fmt = _TRANSFER_FORMAT_MATRIX[("cpu", "cpu")]
        assert fmt == "float32"

    def test_tpu_to_tpu_transfer_bfloat16(self):
        fmt = _TRANSFER_FORMAT_MATRIX[("tpu", "tpu")]
        assert fmt == "bfloat16"

    def test_unknown_pair_falls_back_to_float16_in_recommend(self):
        # cpu→tpu not in matrix; recommend() must not crash and use float16 fallback
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cpu",
            decode_backend="tpu",
        )
        assert isinstance(cfg.kv_transfer_format, str)
        assert len(cfg.kv_transfer_format) > 0


# ---------------------------------------------------------------------------
# Memory and batch heuristics
# ---------------------------------------------------------------------------


class TestMemoryAndBatchHeuristics:
    def _recommend(self, **kwargs):
        defaults = {
            "model_params": 7_000_000_000,
            "prefill_backend": "cuda",
            "prefill_arch": "hopper",
            "decode_backend": "rocm",
            "decode_arch": "cdna3",
        }
        defaults.update(kwargs)
        return DisaggregatedFleetAdvisor.recommend(**defaults)

    def test_decode_kv_budget_larger_than_prefill(self):
        cfg = self._recommend()
        # decode allocates 80% of memory to KV cache; prefill only 20%
        assert cfg.decode.kv_cache_budget_gb > cfg.prefill.kv_cache_budget_gb

    def test_decode_batch_larger_than_prefill(self):
        cfg = self._recommend()
        assert cfg.decode.max_batch_size > cfg.prefill.max_batch_size

    def test_prefill_seq_len_larger_than_decode(self):
        cfg = self._recommend()
        assert cfg.prefill.max_seq_len > cfg.decode.max_seq_len

    def test_memory_override_respected(self):
        # 160 GB prefill memory → 20% = 32 GB KV budget
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
            prefill_memory_gb=160.0,
        )
        assert abs(cfg.prefill.kv_cache_budget_gb - 32.0) < 0.1

    def test_large_model_decode_memory_override(self):
        # 192 GB decode memory → 80% = 153.6 GB KV budget
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="rocm",
            decode_backend="rocm",
            decode_memory_gb=192.0,
        )
        assert abs(cfg.decode.kv_cache_budget_gb - 153.6) < 0.5

    def test_larger_memory_scales_max_batch(self):
        cfg_80 = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
            decode_memory_gb=80.0,
        )
        cfg_160 = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
            decode_memory_gb=160.0,
        )
        assert cfg_160.decode.max_batch_size > cfg_80.decode.max_batch_size


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_nonempty(self):
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="rocm",
        )
        assert len(cfg.notes) > 0
        assert len(cfg.prefill.notes) > 0
        assert len(cfg.decode.notes) > 0

    def test_large_model_gets_fleet_note(self):
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=70_000_000_000,
            prefill_backend="cuda",
            decode_backend="rocm",
        )
        combined = " ".join(cfg.notes)
        assert "70B" in combined or "tensor-parallel" in combined

    def test_fleet_note_mentions_runtime(self):
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
        )
        combined = " ".join(cfg.notes).lower()
        assert any(rt in combined for rt in ("vllm", "sglang", "dynamo", "runtime"))

    def test_cpu_cpu_works_and_uses_float32(self):
        cfg = DisaggregatedFleetAdvisor.recommend(
            model_params=1_000_000_000,
            prefill_backend="cpu",
            decode_backend="cpu",
        )
        assert cfg.prefill.kv_dtype == "float32"
        assert cfg.decode.kv_dtype == "float32"
        assert cfg.kv_transfer_format == "float32"


# ---------------------------------------------------------------------------
# Package-level exports and edge cases
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_importable_from_torchbridge_inference(self):
        from torchbridge.inference import (
            DisaggregatedFleetAdvisor,
            DisaggregatedFleetConfig,
            DisaggregatedRoleConfig,
        )

        assert DisaggregatedFleetAdvisor is not None
        assert DisaggregatedFleetConfig is not None
        assert DisaggregatedRoleConfig is not None

    def test_memory_override_none_vs_explicit_zero_differ(self):
        # None → uses default (80 GB for cuda); 0.0 → 0 GB explicitly
        # This verifies the `is not None` guard is correct
        cfg_default = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
            prefill_memory_gb=None,
        )
        cfg_zero = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
            prefill_memory_gb=0.0,
        )
        # Default uses 80 GB → 20% = 16 GB KV budget
        assert cfg_default.prefill.kv_cache_budget_gb == 16.0
        # Explicit 0.0 → 0% = 0.0 GB KV budget
        assert cfg_zero.prefill.kv_cache_budget_gb == 0.0

    def test_small_memory_scales_batch_down(self):
        # 16 GB decode (e.g. T4) should give smaller max_batch than 80 GB default
        cfg_small = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
            decode_memory_gb=16.0,
        )
        cfg_large = DisaggregatedFleetAdvisor.recommend(
            model_params=7_000_000_000,
            prefill_backend="cuda",
            decode_backend="cuda",
            decode_memory_gb=80.0,
        )
        assert cfg_small.decode.max_batch_size < cfg_large.decode.max_batch_size
