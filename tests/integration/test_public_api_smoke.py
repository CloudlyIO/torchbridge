"""Public API smoke tests — verify top-level imports and basic calls work on CPU."""


class TestPublicAPISmoke:
    def test_import_top_level(self):
        import torchbridge

        assert hasattr(torchbridge, "__version__")

    def test_detect_best_backend_returns_string(self):
        from torchbridge.backends import detect_best_backend

        result = detect_best_backend()
        # detect_best_backend returns a BackendType enum; its .value is the string
        assert hasattr(result, "value")
        assert result.value in ("cuda", "hip", "neuron", "xla", "mps", "cpu")

    def test_hardware_config_instantiates(self):
        from torchbridge.core.config import HardwareConfig

        hw = HardwareConfig()
        assert hw.backend is not None

    def test_quantization_engine_get_optimal_format(self):
        from torchbridge.precision.engine import QuantizationEngine

        engine = QuantizationEngine()
        fmt = engine.get_optimal_format()
        assert fmt is not None

    def test_attention_dispatcher_select_kernel(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        dispatcher = AttentionDispatcher()
        result = dispatcher.select_kernel(seq_length=128, num_heads=4, head_dim=64)
        assert result.kernel_type is not None
        assert isinstance(result.implementation_name, str)

    def test_adapter_compatibility_matrix(self):
        from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
        from torchbridge.core.config import HardwareBackend

        method = AdapterCompatibilityMatrix.get_optimal(backend=HardwareBackend.CPU)
        assert method is not None

    def test_manager_optimize_cpu_model(self):
        import torch.nn as nn

        from torchbridge.core.management import get_manager

        manager = get_manager()
        model = nn.Linear(16, 16)
        optimized = manager.optimize(model)
        assert optimized is not None

    def test_version_is_string(self):
        import torchbridge

        assert isinstance(torchbridge.__version__, str)
        parts = torchbridge.__version__.split(".")
        assert len(parts) >= 3
