"""
Unit tests for torchbridge.inference.kv_handoff.

Covers:
  - KVHandoffSpec dataclass fields and serialisation
  - _KV_HANDOFF_MATRIX known entries (page_size, alignment, layout)
  - Fallback chain: arch → None → safe default
  - KVHandoffNegotiator.negotiate() negotiation rules:
      page_size = max(prefill, decode)
      alignment = min(prefill, decode)
      layout = "interleaved" only if both prefer it
  - Package-level imports from torchbridge.inference
"""

import json

from torchbridge.inference.kv_handoff import (
    _KV_HANDOFF_MATRIX,
    _SAFE_DEFAULT,
    KVHandoffNegotiator,
    KVHandoffSpec,
    _lookup_hw_spec,
)

# ---------------------------------------------------------------------------
# Dataclass fields and serialisation
# ---------------------------------------------------------------------------

class TestKVHandoffSpecFields:
    def test_fields_exist(self):
        spec = KVHandoffSpec(
            dtype="float16",
            layout="separate",
            page_size_tokens=16,
            alignment_bytes=128,
            notes=["a note"],
        )
        assert spec.dtype == "float16"
        assert spec.layout == "separate"
        assert spec.page_size_tokens == 16
        assert spec.alignment_bytes == 128
        assert spec.notes == ["a note"]

    def test_to_dict_is_json_serialisable(self):
        spec = KVHandoffSpec(
            dtype="float16",
            layout="separate",
            page_size_tokens=16,
            alignment_bytes=128,
        )
        serialised = json.dumps(spec.to_dict())
        assert len(serialised) > 0

    def test_to_dict_has_all_keys(self):
        spec = KVHandoffSpec(
            dtype="bfloat16",
            layout="interleaved",
            page_size_tokens=32,
            alignment_bytes=64,
        )
        d = spec.to_dict()
        for key in ("dtype", "layout", "page_size_tokens", "alignment_bytes", "notes"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_roundtrips_through_json(self):
        spec = KVHandoffSpec(dtype="float16", layout="separate",
                             page_size_tokens=16, alignment_bytes=128)
        parsed = json.loads(json.dumps(spec.to_dict()))
        assert parsed["page_size_tokens"] == 16


# ---------------------------------------------------------------------------
# Matrix lookups via _lookup_hw_spec
# ---------------------------------------------------------------------------

class TestMatrixLookups:
    def test_hopper_page_size_16(self):
        page, _, _ = _lookup_hw_spec("cuda", "hopper")
        assert page == 16

    def test_hopper_alignment_128(self):
        _, align, _ = _lookup_hw_spec("cuda", "hopper")
        assert align == 128

    def test_hopper_layout_separate(self):
        _, _, layout = _lookup_hw_spec("cuda", "hopper")
        assert layout == "separate"

    def test_ada_alignment_64(self):
        _, align, _ = _lookup_hw_spec("cuda", "ada")
        assert align == 64

    def test_cdna3_alignment_64(self):
        _, align, _ = _lookup_hw_spec("rocm", "cdna3")
        assert align == 64

    def test_cdna3_layout_separate(self):
        _, _, layout = _lookup_hw_spec("rocm", "cdna3")
        assert layout == "separate"

    def test_tpu_page_size_32(self):
        page, _, _ = _lookup_hw_spec("tpu", None)
        assert page == 32

    def test_tpu_layout_interleaved(self):
        _, _, layout = _lookup_hw_spec("tpu", None)
        assert layout == "interleaved"

    def test_cpu_page_size_1(self):
        page, _, _ = _lookup_hw_spec("cpu", None)
        assert page == 1

    def test_unknown_arch_falls_back_to_none_key(self):
        # "volta" not in matrix — falls back to ("cuda", None)
        result = _lookup_hw_spec("cuda", "volta")
        expected = _KV_HANDOFF_MATRIX[("cuda", None)]
        assert result == expected

    def test_unknown_backend_uses_safe_default(self):
        result = _lookup_hw_spec("gaudi", "gaudi2")
        assert result == _SAFE_DEFAULT


# ---------------------------------------------------------------------------
# Negotiation rules
# ---------------------------------------------------------------------------

class TestNegotiation:
    def _negotiate(self, pb, db, pa=None, da=None, dtype="float16"):
        return KVHandoffNegotiator.negotiate(
            prefill_backend=pb,
            decode_backend=db,
            kv_dtype=dtype,
            prefill_arch=pa,
            decode_arch=da,
        )

    def test_negotiate_preserves_dtype(self):
        spec = self._negotiate("cuda", "rocm", dtype="bfloat16")
        assert spec.dtype == "bfloat16"

    def test_negotiate_cuda_cuda_page_size_16(self):
        spec = self._negotiate("cuda", "cuda", pa="hopper", da="ampere")
        assert spec.page_size_tokens == 16

    def test_negotiate_page_size_is_max_of_two(self):
        # TPU (32) vs CUDA (16) → 32
        spec = self._negotiate("tpu", "cuda")
        assert spec.page_size_tokens == 32

    def test_negotiate_alignment_is_min_of_two(self):
        # CUDA Hopper (128) vs ROCm CDNA3 (64) → 64
        spec = self._negotiate("cuda", "rocm", pa="hopper", da="cdna3")
        assert spec.alignment_bytes == 64

    def test_negotiate_same_cuda_alignment_preserved(self):
        spec = self._negotiate("cuda", "cuda", pa="hopper", da="hopper")
        assert spec.alignment_bytes == 128

    def test_negotiate_tpu_tpu_layout_interleaved(self):
        spec = self._negotiate("tpu", "tpu")
        assert spec.layout == "interleaved"

    def test_negotiate_cuda_tpu_layout_separate(self):
        # Only TPU prefers interleaved — CUDA does not → separate
        spec = self._negotiate("cuda", "tpu")
        assert spec.layout == "separate"

    def test_negotiate_tpu_cuda_layout_separate(self):
        spec = self._negotiate("tpu", "cuda")
        assert spec.layout == "separate"

    def test_negotiate_cuda_rocm_layout_separate(self):
        spec = self._negotiate("cuda", "rocm")
        assert spec.layout == "separate"

    def test_negotiate_cpu_cpu_page_size_1(self):
        spec = self._negotiate("cpu", "cpu")
        assert spec.page_size_tokens == 1

    def test_negotiate_notes_nonempty(self):
        spec = self._negotiate("cuda", "rocm", pa="hopper", da="cdna3")
        assert len(spec.notes) > 0

    def test_negotiate_notes_mention_runtime(self):
        spec = self._negotiate("cuda", "rocm")
        combined = " ".join(spec.notes).lower()
        assert any(rt in combined for rt in ("vllm", "sglang", "runtime", "page_size"))

    def test_negotiate_unknown_pair_uses_safe_defaults(self):
        # Both fallback to _SAFE_DEFAULT → no crash
        spec = self._negotiate("gaudi", "gaudi", dtype="float16")
        assert isinstance(spec.page_size_tokens, int)
        assert isinstance(spec.alignment_bytes, int)

    def test_negotiate_to_dict_is_json_serialisable(self):
        spec = self._negotiate("cuda", "rocm", pa="hopper", da="cdna3")
        serialised = json.dumps(spec.to_dict())
        assert len(serialised) > 0


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_importable_from_torchbridge_inference(self):
        from torchbridge.inference import KVHandoffNegotiator, KVHandoffSpec
        assert KVHandoffNegotiator is not None
        assert KVHandoffSpec is not None
