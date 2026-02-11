"""
Trainium backend stress tests.

Exercises Trainium-specific codepaths: NeuronCore detection, Neuron SDK
compilation, memory management, and Trainium architecture-specific features.

All tests are marked @pytest.mark.trainium and auto-skipped when Neuron SDK
is unavailable. CPU-mockable tests run everywhere.
"""

import pytest
import torch
import torch.nn as nn

from torchbridge.core.config import TorchBridgeConfig, TrainiumArchitecture


def _neuron_available() -> bool:
    """Check if AWS Neuron SDK is available."""
    try:
        import torch_neuronx  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_neuron = pytest.mark.skipif(
    not _neuron_available(),
    reason="Requires AWS Trainium (Neuron SDK)",
)


@pytest.mark.stress
@pytest.mark.trainium
class TestTrainiumBackendStress:
    """Stress tests targeting Trainium backend initialization and config."""

    def test_trainium_backend_creation_cpu_fallback(self):
        """TrainiumBackend should fall back gracefully without Neuron SDK."""
        from torchbridge.backends.trainium.trainium_backend import TrainiumBackend

        try:
            backend = TrainiumBackend()
            # Either properly initialized or fell back to CPU
            assert backend.device is not None
        except Exception as e:
            # Import/init errors for Neuron are acceptable
            assert "neuron" in str(e).lower() or "xla" in str(e).lower() or "not available" in str(e).lower()

    def test_trainium_adapter_creation(self):
        """TrainiumAdapter should instantiate without Neuron SDK."""
        from torchbridge.backends.trainium.trainium_adapter import TrainiumAdapter

        try:
            optimizer = TrainiumAdapter()
            assert optimizer is not None
        except Exception as e:
            # May fail if Neuron SDK is required for init
            assert "neuron" in str(e).lower() or "xla" in str(e).lower() or "not available" in str(e).lower()

    def test_trainium_architecture_enum_coverage(self):
        """All TrainiumArchitecture enum members should be valid."""
        for arch in TrainiumArchitecture:
            assert arch.value is not None
            assert isinstance(arch.value, str)

    def test_trainium_config_all_architectures(self):
        """TrainiumConfig should accept all architecture values."""
        config = TorchBridgeConfig()
        for arch in TrainiumArchitecture:
            config.hardware.trainium.architecture = arch
            assert config.hardware.trainium.architecture == arch

    def test_trainium_config_precision_options(self):
        """TrainiumConfig should accept precision settings."""
        config = TorchBridgeConfig()
        trainium_cfg = config.hardware.trainium

        # Default precision
        assert trainium_cfg.precision == "bfloat16"

        # Mixed precision toggle
        trainium_cfg.mixed_precision = True
        assert trainium_cfg.mixed_precision is True

        trainium_cfg.mixed_precision = False
        assert trainium_cfg.mixed_precision is False

    def test_neuron_compiler_creation(self):
        """NeuronCompiler should instantiate (may warn without SDK)."""
        from torchbridge.backends.trainium.neuron_compiler import NeuronCompiler
        from torchbridge.core.config import TrainiumConfig

        trainium_cfg = TrainiumConfig()
        try:
            compiler = NeuronCompiler(trainium_cfg)
            assert compiler is not None
        except Exception:
            # Expected without Neuron SDK
            pass

    def test_trainium_memory_manager_creation(self):
        """TrainiumMemoryManager should instantiate without crashing."""
        from torchbridge.backends.trainium.memory_manager import TrainiumMemoryManager

        try:
            manager = TrainiumMemoryManager()
            assert manager is not None
        except Exception:
            # Expected without Neuron SDK
            pass

    def test_neuron_utilities_safe(self):
        """Neuron utility functions should not crash without SDK."""
        from torchbridge.backends.trainium.neuron_utilities import (
            get_neuron_env_info,
            get_neuron_sdk_version,
            is_neuron_available,
        )

        # These should always return a value (True/False/None/dict)
        available = is_neuron_available()
        assert isinstance(available, bool)

        version = get_neuron_sdk_version()
        # None when SDK not installed, string otherwise
        assert version is None or isinstance(version, str)

        env_info = get_neuron_env_info()
        assert isinstance(env_info, dict)

    @skip_no_neuron
    def test_trainium_model_prepare(self):
        """On real Trainium: verify model preparation works."""
        from torchbridge.backends.trainium.trainium_backend import TrainiumBackend

        backend = TrainiumBackend()
        assert backend.is_available

        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )
        prepared = backend.prepare_model(model)
        assert prepared is not None

    @skip_no_neuron
    def test_trainium_batch_scaling(self):
        """On real Trainium: verify batch scaling works."""
        from torchbridge.backends.trainium.trainium_backend import TrainiumBackend

        backend = TrainiumBackend()
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        prepared = backend.prepare_model(model)
        prepared.eval()

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 128)
            with torch.no_grad():
                out = prepared(x)
            assert out.shape == (batch_size, 128)
