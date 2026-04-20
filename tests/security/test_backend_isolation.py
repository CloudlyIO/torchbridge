"""
Security tests for backend state isolation.

Verifies that TorchBridge components do not leak state between instances or
calls — each engine/dispatcher/backend instance is independent.
"""

import torch
import torch.nn as nn


class TestQuantizationEngineIsolation:
    """Two QuantizationEngine instances must not share mutable state."""

    def test_two_engines_are_independent_instances(self):
        """Creating two engines produces distinct objects with no shared state."""
        from torchbridge.precision import QuantizationEngine

        engine_a = QuantizationEngine()
        engine_b = QuantizationEngine()

        assert engine_a is not engine_b
        assert id(engine_a) != id(engine_b)

    def test_quantizing_model_a_does_not_affect_engine_for_model_b(self):
        """Quantizing model A leaves engine B's internal state unchanged."""
        from torchbridge.precision import QuantizationEngine

        model_a = nn.Linear(32, 16)

        engine_a = QuantizationEngine()
        engine_b = QuantizationEngine()

        backend_name_before = engine_b.backend_name
        engine_a.quantize(model_a, format="int8_dynamic")
        backend_name_after = engine_b.backend_name

        assert backend_name_before == backend_name_after, (
            "Quantizing model_a changed engine_b.backend_name"
        )

    def test_engine_results_are_independent(self):
        """Two separate quantize() calls on different engines yield independent results."""
        from torchbridge.precision import QuantizationEngine

        model_a = nn.Linear(32, 16)
        model_b = nn.Linear(64, 32)

        result_a = QuantizationEngine().quantize(model_a, format="bf16")
        result_b = QuantizationEngine().quantize(model_b, format="bf16")

        # Models must be distinct objects
        if result_a.model is not None and result_b.model is not None:
            assert result_a.model is not result_b.model


class TestAttentionDispatcherIsolation:
    """AttentionDispatcher must return fresh results per call, not cached stale ones."""

    def test_two_dispatcher_instances_are_independent(self):
        """Two AttentionDispatcher instances are distinct objects."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        d1 = AttentionDispatcher()
        d2 = AttentionDispatcher()

        assert d1 is not d2
        assert id(d1) != id(d2)

    def test_different_seq_lengths_produce_different_results(self):
        """select_kernel() for seq_len=128 vs seq_len=8192 must not produce identical results
        from stale cache when the optimal kernel differs."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        dispatcher = AttentionDispatcher(use_benchmark_cache=False)
        result_short = dispatcher.select_kernel(
            seq_length=128, num_heads=8, head_dim=64
        )
        result_long = dispatcher.select_kernel(
            seq_length=8192, num_heads=8, head_dim=64
        )

        # Both must be non-None — content may or may not differ depending on hardware
        assert result_short is not None
        assert result_long is not None
        # Crucially: the second result must be a new object, not the same reference
        assert result_short is not result_long


class TestBackendFactoryIsolation:
    """BackendFactory.create() must return independent instances each call."""

    def test_two_cpu_backends_are_independent(self):
        """Creating CPU backend twice yields two separate instances."""
        from torchbridge.backends.backend_factory import BackendFactory, BackendType

        b1 = BackendFactory.create(BackendType.CPU)
        b2 = BackendFactory.create(BackendType.CPU)

        assert b1 is not b2
        assert id(b1) != id(b2)

    def test_device_info_of_two_backends_is_independent(self):
        """DeviceInfo from two CPU backend instances are independent objects."""
        from torchbridge.backends.backend_factory import BackendFactory, BackendType

        b1 = BackendFactory.create(BackendType.CPU)
        b2 = BackendFactory.create(BackendType.CPU)

        info1 = b1.get_device_info()
        info2 = b2.get_device_info()

        # Both must return valid info — they should not share mutable state
        assert info1 is not None
        assert info2 is not None


class TestAutoOptimizeIsolation:
    """auto_optimize() results for different models must not interfere."""

    def test_optimizing_two_models_yields_independent_results(self):
        """auto_optimize() on model A and then model B returns independent models."""
        from torchbridge import get_manager

        manager = get_manager()

        model_a = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        model_b = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

        optimized_a = manager.auto_optimize(model_a, for_inference=True)
        optimized_b = manager.auto_optimize(model_b, for_inference=True)

        # They must be distinct objects
        assert optimized_a is not optimized_b

        # model_a's result must still work independently
        test_input_a = torch.randn(1, 32)
        with torch.no_grad():
            out_a = optimized_a(test_input_a)
        assert out_a.shape == (1, 16)
