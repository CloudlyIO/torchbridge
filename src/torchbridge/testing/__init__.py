"""
TorchBridge Cross-Backend Testing Utilities

Provides decorators, base classes, and reporting tools for validating
PyTorch model correctness across hardware backends.

Example usage::

    from torchbridge.testing import cross_backend, CrossBackendTestSuite, BackendTolerance

    @cross_backend(min_backends=1)
    def test_linear_forward(backend):
        import torch
        import torch.nn as nn
        model = nn.Linear(64, 32).to(backend.device)
        x = torch.randn(4, 64).to(backend.device)
        out = model(x)
        assert out.shape == (4, 32)
"""

from torchbridge.testing.decorator import cross_backend
from torchbridge.testing.divergence import DivergenceTracer, LayerDivergence
from torchbridge.testing.report import QualificationReport
from torchbridge.testing.suite import BackendTolerance, CrossBackendTestSuite
from torchbridge.testing.tolerance_db import ToleranceDB

__all__ = [
    "BackendTolerance",
    "CrossBackendTestSuite",
    "DivergenceTracer",
    "LayerDivergence",
    "QualificationReport",
    "ToleranceDB",
    "cross_backend",
]
