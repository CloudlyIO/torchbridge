"""
Input bounds validation tests.

Verifies that invalid inputs raise clear errors rather than producing silently
wrong results. These tests FAIL until validation is added to:
- dispatcher.select_kernel()
- DistributedConfig.auto()
- QuantizationEngine.quantize()
"""

import pytest


class TestSelectKernelValidation:
    """AttentionDispatcher.select_kernel() must reject invalid dimensions."""

    def test_select_kernel_negative_seq_length(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        d = AttentionDispatcher(use_benchmark_cache=False)
        with pytest.raises(ValueError, match="seq_length"):
            d.select_kernel(seq_length=-1, num_heads=4, head_dim=32)

    def test_select_kernel_zero_seq_length(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        d = AttentionDispatcher(use_benchmark_cache=False)
        with pytest.raises(ValueError, match="seq_length"):
            d.select_kernel(seq_length=0, num_heads=4, head_dim=32)

    def test_select_kernel_zero_num_heads(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        d = AttentionDispatcher(use_benchmark_cache=False)
        with pytest.raises(ValueError, match="num_heads"):
            d.select_kernel(seq_length=128, num_heads=0, head_dim=32)

    def test_select_kernel_zero_head_dim(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        d = AttentionDispatcher(use_benchmark_cache=False)
        with pytest.raises(ValueError, match="head_dim"):
            d.select_kernel(seq_length=128, num_heads=4, head_dim=0)


class TestDistributedConfigValidation:
    """DistributedConfig.auto() must reject invalid world_size and model_params."""

    def test_distributed_config_zero_world_size(self):
        from torchbridge.distributed.config import DistributedConfig

        with pytest.raises(ValueError, match="world_size"):
            DistributedConfig.auto(model_params=1_000_000, world_size=0)

    def test_distributed_config_negative_world_size(self):
        from torchbridge.distributed.config import DistributedConfig

        with pytest.raises(ValueError, match="world_size"):
            DistributedConfig.auto(model_params=1_000_000, world_size=-1)

    def test_distributed_config_negative_model_params(self):
        from torchbridge.distributed.config import DistributedConfig

        with pytest.raises(ValueError, match="model_params"):
            DistributedConfig.auto(model_params=-1, world_size=1)

    def test_distributed_config_gpus_exceeds_world_size(self):
        from torchbridge.distributed.config import DistributedConfig

        with pytest.raises(ValueError, match="gpus_per_node"):
            DistributedConfig.auto(
                model_params=1_000_000,
                world_size=4,
                gpus_per_node=8,
            )


class TestQuantizationEngineValidation:
    """QuantizationEngine.quantize() must reject non-nn.Module inputs."""

    def test_quantize_non_module_raises(self):
        from torchbridge.precision.engine import QuantizationEngine

        engine = QuantizationEngine()
        with pytest.raises(TypeError, match="nn.Module"):
            engine.quantize({"weight": [1, 2, 3]})  # type: ignore[arg-type]

    def test_quantize_string_raises(self):
        from torchbridge.precision.engine import QuantizationEngine

        engine = QuantizationEngine()
        with pytest.raises(TypeError, match="nn.Module"):
            engine.quantize("not_a_module")  # type: ignore[arg-type]

    def test_quantize_valid_module_does_not_raise(self):
        """Sanity: a real nn.Module must still work."""
        import torch.nn as nn

        from torchbridge.precision.engine import QuantizationEngine

        engine = QuantizationEngine()
        model = nn.Linear(4, 4)
        result = engine.quantize(model, format="none")
        assert result.success
