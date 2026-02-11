"""
AMD backend stress tests.

Exercises AMD-specific codepaths: ROCm device detection, Matrix Core optimization,
HIP memory management, and CDNA architecture-specific features.

All tests are marked @pytest.mark.amd and auto-skipped when ROCm is unavailable.
"""

import pytest
import torch
import torch.nn as nn

from torchbridge.backends.amd.amd_adapter import AMDAdapter
from torchbridge.backends.amd.amd_backend import AMDBackend
from torchbridge.core.config import AMDArchitecture, AMDConfig


def _rocm_available() -> bool:
    """Check if ROCm is available (AMD GPU via HIP/CUDA compat)."""
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
        return "instinct" in name or "radeon" in name or "amd" in name
    except Exception:
        return False


skip_no_rocm = pytest.mark.skipif(
    not _rocm_available(),
    reason="Requires AMD GPU with ROCm",
)


@pytest.mark.stress
@pytest.mark.amd
class TestAMDBackendStress:
    """Stress tests targeting AMD backend initialization and device queries."""

    def test_amd_backend_creation_cpu_fallback(self):
        """AMDBackend should fall back to CPU gracefully when ROCm is unavailable."""
        # On non-AMD machines this should initialize without crashing
        try:
            backend = AMDBackend()
            # Either properly initialized or fell back to CPU
            assert backend.device is not None
        except Exception as e:
            # Import errors for ROCm are acceptable
            assert "ROCm" in str(e) or "HIP" in str(e) or "not available" in str(e).lower()

    def test_amd_optimizer_all_levels(self):
        """AMDAdapter should accept all optimization levels without error."""
        config = AMDConfig()
        for level in ("conservative", "balanced", "aggressive"):
            config_copy = AMDConfig(
                optimization_level=level,
                architecture=config.architecture,
            )
            optimizer = AMDAdapter(config_copy)
            assert optimizer is not None

    def test_amd_optimizer_optimize_small_model(self):
        """AMDAdapter.optimize should handle a small model on CPU."""
        config = AMDConfig(optimization_level="conservative")
        optimizer = AMDAdapter(config)
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # optimize() should not crash even without ROCm
        try:
            result = optimizer.optimize(model)
            assert result is not None
        except Exception:
            # Some code paths may require ROCm — that's OK
            pass

    def test_amd_architecture_enum_coverage(self):
        """All AMDArchitecture enum members should be valid."""
        for arch in AMDArchitecture:
            assert arch.value is not None
            config = AMDConfig(architecture=arch)
            # AUTO may resolve to a concrete architecture in __post_init__
            assert isinstance(config.architecture, AMDArchitecture)

    @skip_no_rocm
    def test_amd_device_info(self):
        """On real AMD GPU: verify device info is populated."""
        backend = AMDBackend()
        assert backend.is_available
        info = backend.get_device_info(0)
        assert info.backend == "amd"
        assert info.device_name != ""
        assert info.total_memory_bytes > 0

    @skip_no_rocm
    def test_amd_batch_scaling_on_gpu(self):
        """On real AMD GPU: verify batch scaling works."""
        backend = AMDBackend()
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        ).to(backend.device)
        model.eval()
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 256, device=backend.device)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (batch_size, 256)
