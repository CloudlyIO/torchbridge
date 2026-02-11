"""
Shared fixtures for real-model HAL tests.

These tests validate TorchBridge behavior with real pretrained models.
All tests are marked @pytest.mark.real_model and skipped by default in CI.

To run:
    PYTHONPATH=src pytest tests/models/ -v -m real_model
"""

import pytest
import torch
import torch.nn.functional as F


def _check_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


requires_transformers = pytest.mark.skipif(
    not _check_transformers(),
    reason="Requires HuggingFace transformers library",
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA GPU",
)


def assert_cross_backend_consistency(
    cpu_output: torch.Tensor,
    gpu_output: torch.Tensor,
    max_diff_threshold: float = 1e-4,
    cosine_sim_threshold: float = 0.9999,
    label: str = "",
) -> dict[str, float]:
    """Assert that CPU and GPU outputs are consistent."""
    cpu_flat = cpu_output.detach().float().cpu().flatten()
    gpu_flat = gpu_output.detach().float().cpu().flatten()

    max_diff = torch.abs(cpu_flat - gpu_flat).max().item()
    cosine_sim = F.cosine_similarity(
        cpu_flat.unsqueeze(0), gpu_flat.unsqueeze(0)
    ).item()

    prefix = f"[{label}] " if label else ""
    assert max_diff < max_diff_threshold, (
        f"{prefix}Max diff {max_diff:.2e} exceeds threshold {max_diff_threshold:.2e}"
    )
    assert cosine_sim > cosine_sim_threshold, (
        f"{prefix}Cosine sim {cosine_sim:.6f} below threshold {cosine_sim_threshold}"
    )

    return {"max_diff": max_diff, "cosine_sim": cosine_sim}
