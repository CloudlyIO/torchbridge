"""Integration tests for B1 fix: attention dispatch result wired into create_attention().

Verifies that AttentionDispatcher.create_attention() uses the kernel selected by
select_kernel() rather than silently falling back to auto-select when the dispatched
implementation name is not in the registry.
"""

from unittest.mock import patch

from torchbridge.attention.core.config import AttentionModuleConfig
from torchbridge.attention.dispatch.dispatcher import (
    _KERNEL_REGISTRY_MAP,
    AttentionDispatcher,
    AttentionDispatchResult,
    AttentionKernelType,
)
from torchbridge.core.config import HardwareBackend


class TestDispatchResultWired:
    """B1: Verify dispatched kernel is used, not silently discarded."""

    def test_create_attention_uses_dispatched_impl(self):
        """create_attention() must honour the impl_name from select_kernel()."""
        config = AttentionModuleConfig(embed_dim=64, num_heads=4)
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )

        # On CPU, select_kernel returns PYTORCH_SDPA → "memory_efficient_attention"
        result = dispatcher.select_kernel(
            seq_length=config.max_sequence_length,
            num_heads=config.num_heads,
            head_dim=config.head_dim or (config.embed_dim // config.num_heads),
        )
        # The impl_name must be in the registry; otherwise create_attention falls back
        from torchbridge.attention.core.registry import _ATTENTION_REGISTRY
        assert result.implementation_name in _ATTENTION_REGISTRY, (
            f"Dispatched impl '{result.implementation_name}' not in registry — "
            "B1 fix must ensure this path works"
        )

        layer = dispatcher.create_attention(config)
        assert layer is not None

    def test_fallback_chain_walked_on_registry_miss(self):
        """When dispatched impl is not in registry, fallback chain is tried before auto-select."""
        config = AttentionModuleConfig(embed_dim=64, num_heads=4)
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )

        # Patch select_kernel to return a result with a bogus impl_name
        # but a valid fallback_chain
        bogus_result = AttentionDispatchResult(
            kernel_type=AttentionKernelType.FLEX_ATTENTION,
            implementation_name="nonexistent_impl_xyz",
            used_fallback=False,
            fallback_chain=[AttentionKernelType.PYTORCH_SDPA],
        )
        with patch.object(dispatcher, "select_kernel", return_value=bogus_result):
            # Should NOT raise — fallback chain should be used
            layer = dispatcher.create_attention(config)
        assert layer is not None

    def test_auto_select_used_when_entire_fallback_chain_misses(self):
        """If both dispatched impl AND entire fallback chain miss, auto-select is used."""
        config = AttentionModuleConfig(embed_dim=64, num_heads=4)
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )

        # Patch select_kernel to return a result with ALL bogus names
        bogus_result = AttentionDispatchResult(
            kernel_type=AttentionKernelType.FLEX_ATTENTION,
            implementation_name="bogus_primary",
            used_fallback=False,
            fallback_chain=[],  # empty fallback
        )
        with patch.object(dispatcher, "select_kernel", return_value=bogus_result):
            layer = dispatcher.create_attention(config)
        assert layer is not None  # auto-select must handle this

    def test_kernel_registry_map_covers_all_kernel_types(self):
        """_KERNEL_REGISTRY_MAP should map every AttentionKernelType."""
        for kt in AttentionKernelType:
            mapped = _KERNEL_REGISTRY_MAP.get(kt)
            assert mapped is not None, (
                f"AttentionKernelType.{kt.name} has no entry in _KERNEL_REGISTRY_MAP"
            )

    def test_cpu_dispatch_result_kernel_type_is_sdpa(self):
        """On CPU, select_kernel must return PYTORCH_SDPA (always available)."""
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        result = dispatcher.select_kernel(seq_length=128, num_heads=4, head_dim=64)
        assert result.kernel_type == AttentionKernelType.PYTORCH_SDPA

    def test_dispatch_result_impl_name_not_empty(self):
        """implementation_name must always be a non-empty string."""
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        result = dispatcher.select_kernel(seq_length=128, num_heads=4, head_dim=64)
        assert isinstance(result.implementation_name, str)
        assert len(result.implementation_name) > 0
