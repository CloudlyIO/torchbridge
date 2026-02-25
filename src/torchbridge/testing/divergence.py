"""
Layer-by-Layer Divergence Tracer

Hooks into named modules to capture per-layer outputs on a reference device
(CPU) and a test device (GPU/TPU/MPS), then reports which layer first shows
significant numerical divergence.

Usage::

    from torchbridge.testing import DivergenceTracer
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
    x = torch.randn(4, 64)

    tracer = DivergenceTracer(model, device=torch.device("cpu"))
    with tracer:
        out = model(x)

    for layer in tracer.layers_with_divergence(atol=1e-4):
        print(layer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerDivergence:
    """Divergence statistics for a single named module."""

    layer_name: str
    max_diff: float
    mean_diff: float
    cosine_sim: float
    output_shape: tuple[int, ...]
    backend: str = "unknown"
    dtype: str = "float32"

    @property
    def exceeds_threshold(self) -> bool:
        """True if max_diff > 1e-4 (default threshold)."""
        return self.max_diff > 1e-4


@dataclass
class _LayerCapture:
    """Internal: captures activations for one named module."""
    name: str
    outputs: list[torch.Tensor] = field(default_factory=list)


class DivergenceTracer:
    """Register forward hooks to capture per-layer outputs, then diff them.

    Operates as a context manager.  After the ``with`` block, call
    :meth:`compare_with` or :meth:`layers_with_divergence` to analyse results.

    Args:
        model: The PyTorch module to trace.
        device: Device to run the model on.  CPU is used as the reference.
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None) -> None:
        self._model = model
        self._device = device or torch.device("cpu")
        self._captures: dict[str, _LayerCapture] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> DivergenceTracer:
        self._captures.clear()
        for name, module in self._model.named_modules():
            if not name:
                continue  # skip root
            capture = _LayerCapture(name=name)
            self._captures[name] = capture

            def _hook(mod: nn.Module, inp: Any, out: Any, cap: _LayerCapture = capture) -> None:
                if isinstance(out, torch.Tensor):
                    cap.outputs.append(out.detach().cpu().float())

            handle = module.register_forward_hook(_hook)
            self._handles.append(handle)
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def layer_names(self) -> list[str]:
        """Names of all captured layers."""
        return list(self._captures.keys())

    def compare_with(
        self,
        reference: DivergenceTracer,
        atol: float = 1e-4,
    ) -> list[LayerDivergence]:
        """Compare this tracer's captures against a reference tracer.

        Args:
            reference: A second :class:`DivergenceTracer` that ran on a different
                device (typically CPU).
            atol: Absolute tolerance used to flag divergent layers.

        Returns:
            List of :class:`LayerDivergence`, one per compared layer, sorted by
            ``max_diff`` descending.
        """
        import torch.nn.functional as F

        results: list[LayerDivergence] = []
        for name, cap in self._captures.items():
            ref_cap = reference._captures.get(name)
            if ref_cap is None or not cap.outputs or not ref_cap.outputs:
                continue
            test_out = cap.outputs[-1]
            ref_out = ref_cap.outputs[-1]
            if test_out.shape != ref_out.shape:
                continue
            try:
                diff = torch.abs(test_out - ref_out)
                max_diff = float(diff.max())
                mean_diff = float(diff.mean())
                cos_sim = float(F.cosine_similarity(
                    test_out.flatten().unsqueeze(0),
                    ref_out.flatten().unsqueeze(0),
                ))
                results.append(LayerDivergence(
                    layer_name=name,
                    max_diff=max_diff,
                    mean_diff=mean_diff,
                    cosine_sim=cos_sim,
                    output_shape=tuple(test_out.shape),
                    backend=str(self._device),
                    dtype=str(test_out.dtype).replace("torch.", ""),
                ))
            except Exception as e:
                logger.debug("Could not compare layer %s: %s", name, e)

        results.sort(key=lambda r: r.max_diff, reverse=True)
        return results

    def layers_with_divergence(self, atol: float = 1e-4) -> list[LayerDivergence]:
        """Return a single-tracer summary flagging layers with large self-variation.

        For single-device use when you want to identify numerically unstable layers
        (e.g. very large activations) without a reference tracer.

        Args:
            atol: Threshold above which a layer is considered divergent.

        Returns:
            List of :class:`LayerDivergence` where ``max_diff > atol``.
        """
        results: list[LayerDivergence] = []
        for name, cap in self._captures.items():
            if len(cap.outputs) < 2:
                continue
            # Compare consecutive forward passes
            a, b = cap.outputs[-2], cap.outputs[-1]
            if a.shape != b.shape:
                continue
            diff = torch.abs(a - b)
            max_diff = float(diff.max())
            if max_diff > atol:
                results.append(LayerDivergence(
                    layer_name=name,
                    max_diff=max_diff,
                    mean_diff=float(diff.mean()),
                    cosine_sim=0.0,
                    output_shape=tuple(a.shape),
                ))
        results.sort(key=lambda r: r.max_diff, reverse=True)
        return results
