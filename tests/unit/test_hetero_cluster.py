"""
Unit tests for torchbridge.distributed.hetero.

Tests define the contract (TDD — written RED before implementation):
- _COLLECTIVE_BRIDGE_MATRIX maps validated (nvidia_arch, amd_arch) pairs to bridge names
- Unknown pairs fall back to _COLLECTIVE_BRIDGE_DEFAULT
- _PARTITION_THRESHOLDS maps AMD/NVIDIA memory ratio to partition strategy
- HeterogeneousClusterAdvisor.recommend() returns a HeterogeneousClusterConfig
- Collective bridge is populated from matrix
- FSDP strategy is hybrid_shard when vendor GPU count > 8, full_shard otherwise
- to_dict() contains all required keys
- notes list is populated
"""

from torchbridge.core.config import AMDArchitecture, NVIDIAArchitecture

# ── Matrix content tests ───────────────────────────────────────────────────


class TestCollectiveBridgeMatrix:
    def test_hopper_cdna3_returns_hetccl(self):
        from torchbridge.distributed.hetero import _COLLECTIVE_BRIDGE_MATRIX

        assert (
            _COLLECTIVE_BRIDGE_MATRIX.get(
                (NVIDIAArchitecture.HOPPER, AMDArchitecture.CDNA3)
            )
            == "hetccl"
        )

    def test_hopper_cdna4_returns_hetccl(self):
        from torchbridge.distributed.hetero import _COLLECTIVE_BRIDGE_MATRIX

        assert (
            _COLLECTIVE_BRIDGE_MATRIX.get(
                (NVIDIAArchitecture.HOPPER, AMDArchitecture.CDNA4)
            )
            == "hetccl"
        )

    def test_blackwell_dc_cdna3_returns_hetccl(self):
        from torchbridge.distributed.hetero import _COLLECTIVE_BRIDGE_MATRIX

        assert (
            _COLLECTIVE_BRIDGE_MATRIX.get(
                (NVIDIAArchitecture.BLACKWELL_DC, AMDArchitecture.CDNA3)
            )
            == "hetccl"
        )

    def test_ampere_cdna3_returns_ucc(self):
        from torchbridge.distributed.hetero import _COLLECTIVE_BRIDGE_MATRIX

        assert (
            _COLLECTIVE_BRIDGE_MATRIX.get(
                (NVIDIAArchitecture.AMPERE, AMDArchitecture.CDNA3)
            )
            == "ucc"
        )

    def test_unknown_pair_uses_default(self):
        from torchbridge.distributed.hetero import (
            _COLLECTIVE_BRIDGE_DEFAULT,
            _COLLECTIVE_BRIDGE_MATRIX,
        )

        # None, None is not in the matrix → fallback
        result = _COLLECTIVE_BRIDGE_MATRIX.get((None, None), _COLLECTIVE_BRIDGE_DEFAULT)
        assert result == _COLLECTIVE_BRIDGE_DEFAULT

    def test_default_is_ucc(self):
        from torchbridge.distributed.hetero import _COLLECTIVE_BRIDGE_DEFAULT

        assert _COLLECTIVE_BRIDGE_DEFAULT == "ucc"

    def test_all_values_are_valid_bridge_names(self):
        from torchbridge.distributed.hetero import _COLLECTIVE_BRIDGE_MATRIX

        valid = {"hetccl", "ucc", "gloo"}
        for pair, bridge in _COLLECTIVE_BRIDGE_MATRIX.items():
            assert bridge in valid, f"Invalid bridge {bridge!r} for pair {pair}"


# ── Partition threshold tests ──────────────────────────────────────────────


class TestPartitionThresholds:
    def test_thresholds_is_list_of_tuples(self):
        from torchbridge.distributed.hetero import _PARTITION_THRESHOLDS

        assert isinstance(_PARTITION_THRESHOLDS, list)
        for item in _PARTITION_THRESHOLDS:
            assert isinstance(item, tuple) and len(item) == 2

    def test_balanced_memory_ratio_returns_vendor_isolated(self):
        """ratio = 1.0 → vendor_isolated (no memory advantage to cross-vendor)."""
        from torchbridge.core.config import AMDArchitecture, NVIDIAArchitecture
        from torchbridge.distributed.hetero import HeterogeneousClusterAdvisor

        # HOPPER (80 GB) × 4 = 320 GB vs CDNA3 (192 GB) × 1 = 192 GB → ratio < 2
        cfg = HeterogeneousClusterAdvisor.recommend(
            nvidia_count=4,
            nvidia_arch=NVIDIAArchitecture.HOPPER,
            amd_count=1,
            amd_arch=AMDArchitecture.CDNA3,
            model_params=7_000_000_000,
        )
        assert cfg.partition_strategy == "vendor_isolated"

    def test_amd_dominates_memory_returns_memory_balanced(self):
        """AMD total memory ≥ 2× NVIDIA total → memory_balanced."""
        from torchbridge.distributed.hetero import HeterogeneousClusterAdvisor

        # HOPPER (80 GB) × 1 = 80 GB vs CDNA3 (192 GB) × 2 = 384 GB → ratio = 4.8
        cfg = HeterogeneousClusterAdvisor.recommend(
            nvidia_count=1,
            nvidia_arch=NVIDIAArchitecture.HOPPER,
            amd_count=2,
            amd_arch=AMDArchitecture.CDNA3,
            model_params=7_000_000_000,
        )
        assert cfg.partition_strategy == "memory_balanced"


# ── Advisor tests ──────────────────────────────────────────────────────────


class TestHeterogeneousClusterAdvisor:
    def _make_cfg(self, **kwargs):
        from torchbridge.distributed.hetero import HeterogeneousClusterAdvisor

        defaults = {
            "nvidia_count": 4,
            "nvidia_arch": NVIDIAArchitecture.HOPPER,
            "amd_count": 4,
            "amd_arch": AMDArchitecture.CDNA3,
            "model_params": 7_000_000_000,
        }
        defaults.update(kwargs)
        return HeterogeneousClusterAdvisor.recommend(**defaults)

    def test_returns_hetero_cluster_config(self):
        from torchbridge.distributed.hetero import HeterogeneousClusterConfig

        cfg = self._make_cfg()
        assert isinstance(cfg, HeterogeneousClusterConfig)

    def test_hopper_cdna3_gets_hetccl(self):
        cfg = self._make_cfg(
            nvidia_arch=NVIDIAArchitecture.HOPPER,
            amd_arch=AMDArchitecture.CDNA3,
        )
        assert cfg.collective_bridge == "hetccl"

    def test_ampere_cdna2_gets_ucc(self):
        cfg = self._make_cfg(
            nvidia_arch=NVIDIAArchitecture.AMPERE,
            amd_arch=AMDArchitecture.CDNA2,
        )
        assert cfg.collective_bridge == "ucc"

    def test_none_arch_gets_default_bridge(self):
        from torchbridge.distributed.hetero import _COLLECTIVE_BRIDGE_DEFAULT

        cfg = self._make_cfg(nvidia_arch=None, amd_arch=None)
        assert cfg.collective_bridge == _COLLECTIVE_BRIDGE_DEFAULT

    def test_hybrid_shard_for_large_nvidia_group(self):
        """> 8 GPUs on one vendor side → hybrid_shard for that vendor."""
        cfg = self._make_cfg(nvidia_count=16, amd_count=4)
        assert cfg.nvidia_fsdp_strategy == "hybrid_shard"

    def test_full_shard_for_small_cluster(self):
        """≤ 8 GPUs per vendor → full_shard."""
        cfg = self._make_cfg(nvidia_count=4, amd_count=4)
        assert cfg.nvidia_fsdp_strategy == "full_shard"
        assert cfg.amd_fsdp_strategy == "full_shard"

    def test_mixed_precision_strings_are_valid(self):
        cfg = self._make_cfg()
        valid = {"bf16", "fp16", "fp32", "fp8"}
        assert cfg.nvidia_mixed_precision in valid
        assert cfg.amd_mixed_precision in valid

    def test_cross_vendor_comm_is_positive(self):
        cfg = self._make_cfg()
        assert cfg.estimated_cross_vendor_comm_gb > 0.0

    def test_notes_not_empty(self):
        cfg = self._make_cfg()
        assert len(cfg.notes) > 0

    def test_to_dict_has_required_keys(self):
        required = {
            "nvidia_count",
            "nvidia_arch",
            "amd_count",
            "amd_arch",
            "model_params",
            "collective_bridge",
            "partition_strategy",
            "nvidia_fsdp_strategy",
            "amd_fsdp_strategy",
            "nvidia_mixed_precision",
            "amd_mixed_precision",
            "estimated_cross_vendor_comm_gb",
            "notes",
        }
        cfg = self._make_cfg()
        d = cfg.to_dict()
        assert required.issubset(d.keys())

    def test_to_dict_values_are_json_serialisable(self):
        import json

        cfg = self._make_cfg()
        # Must not raise
        json.dumps(cfg.to_dict())
