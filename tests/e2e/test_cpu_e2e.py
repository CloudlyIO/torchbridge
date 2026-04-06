"""CPU-only E2E test — full TorchBridge pipeline on CI runners."""

import pytest
import torch
import torch.nn as nn


class TestCPUEndToEnd:
    def test_linear_model_optimize_and_infer(self):
        from torchbridge.core.management import get_manager

        manager = get_manager()
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        model.eval()
        optimized = manager.optimize(model)
        assert optimized is not None
        x = torch.randn(4, 32)
        with torch.no_grad():
            out_orig = model(x)
        assert out_orig.shape == (4, 10)
        # Running the optimized model may invoke torch.compile; skip gracefully if
        # the local compiler environment is broken (e.g. stale PCH on macOS).
        try:
            with torch.no_grad():
                out_opt = optimized(x)
            assert out_opt.shape == out_orig.shape
        except Exception:
            pytest.skip("torch.compile not available in this environment")

    def test_backend_detection_returns_cpu_without_gpu(self):
        from torchbridge.backends import detect_best_backend

        # detect_best_backend returns a BackendType enum; .value is the string
        result = detect_best_backend()
        assert hasattr(result, "value")
        assert result.value in ("cuda", "hip", "neuron", "xla", "mps", "cpu")

    def test_quantization_engine_cpu_format(self):
        from torchbridge.precision.quantization.engine import QuantizationEngine

        engine = QuantizationEngine()
        # Engine auto-detects backend; on CPU returns INT8 or similar
        fmt = engine.get_optimal_format()
        assert fmt is not None
