"""
Adapter Engine for TorchBridge

Injects LoRA/DoRA/QLoRA/QDoRA adapter layers into an nn.Module, replacing
targeted nn.Linear layers in-place.

Usage::

    from torchbridge.adapters.engine import AdapterEngine
    from torchbridge.adapters.config import AdapterConfig, AdapterMethod
    from torchbridge.core.config import HardwareBackend

    config = AdapterConfig(method=AdapterMethod.QLORA, rank=16, alpha=32.0)
    engine = AdapterEngine(config=config, backend=HardwareBackend.CUDA)
    result = engine.inject(model)
    # result.base_quantized == True
    # result.trainable_params / result.total_params ≈ 0.01
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn

from torchbridge.adapters import layers as _layers_module
from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
from torchbridge.adapters.config import AdapterConfig, AdapterMethod, InitMethod

if TYPE_CHECKING:
    from torchbridge.core.config import HardwareBackend
    from torchbridge.precision.formats import QuantizationFormat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Summary of adapter injection.

    Attributes:
        method_applied: The adapter method actually applied (may differ from
            requested method if a fallback occurred, e.g., QLoRA → LoRA
            when torchao is unavailable).
        layers_modified: Number of nn.Linear layers replaced.
        trainable_params: Total trainable parameter count after injection.
        total_params: Total parameter count (trainable + frozen) after injection.
        base_quantized: True if base weights were quantized (QLoRA/QDoRA).
        base_quant_format: The QuantizationFormat used for base weights, or
            None if base is not quantized.
    """

    method_applied: AdapterMethod
    layers_modified: int
    trainable_params: int
    total_params: int
    base_quantized: bool
    base_quant_format: QuantizationFormat | None  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AdapterEngine:
    """Injects parameter-efficient adapter layers into an nn.Module.

    Args:
        config: Adapter configuration (method, rank, alpha, target_modules).
        backend: Hardware backend — used to select the appropriate
            quantization format for QLoRA/QDoRA.
    """

    def __init__(
        self,
        config: AdapterConfig,
        backend: HardwareBackend | None = None,
    ) -> None:
        from torchbridge.core.config import HardwareBackend

        self._config = config
        self._backend = backend if backend is not None else HardwareBackend.CPU

    def inject(self, model: nn.Module) -> AdapterResult:
        """Replace targeted nn.Linear layers with adapter layers in-place.

        Args:
            model: The nn.Module to inject adapters into.

        Returns:
            AdapterResult with injection statistics.

        Raises:
            TypeError: If model is not an nn.Module.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")

        method = self._config.method
        layers_modified = 0
        _base_quantized = False
        _base_quant_format = None
        _method_applied = method

        # Collect replacements first to avoid modifying the module dict during iteration
        replacements: list[tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self._matches_target(name):
                continue
            new_layer = self._create_adapter_layer(module, method)
            replacements.append((name, new_layer))

        for name, new_layer in replacements:
            self._rsetattr(model, name, new_layer)
            layers_modified += 1

            # Capture quant metadata from the first quantized layer
            if not _base_quantized and isinstance(
                new_layer, (_layers_module.QLoRALinear, _layers_module.QDoRALinear)
            ):
                _base_quantized = True
                _base_quant_format = getattr(new_layer, "_quant_format", None)

            # Detect if a QLoRA fallback occurred (fell back to LoRALinear)
            if (
                method in (AdapterMethod.QLORA, AdapterMethod.QDORA)
                and isinstance(new_layer, _layers_module.LoRALinear)
                and not isinstance(
                    new_layer, (_layers_module.QLoRALinear, _layers_module.QDoRALinear)
                )
            ):
                _method_applied = AdapterMethod.LORA

        # Count params on the modified model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        return AdapterResult(
            method_applied=_method_applied,
            layers_modified=layers_modified,
            trainable_params=trainable,
            total_params=total,
            base_quantized=_base_quantized,
            base_quant_format=_base_quant_format,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _matches_target(self, name: str) -> bool:
        """Return True if module `name` matches any target_modules pattern."""
        for target in self._config.target_modules:
            if name == target or name.endswith("." + target):
                return True
        return False

    def _create_adapter_layer(
        self, base_linear: nn.Linear, method: AdapterMethod
    ) -> nn.Module:
        """Dispatch to the correct adapter layer constructor."""
        if method == AdapterMethod.QLORA:
            return self._create_qlora_layer(base_linear)
        if method == AdapterMethod.QDORA:
            return self._create_qdora_layer(base_linear)
        if method == AdapterMethod.DORA:
            return _layers_module.DoRALinear(
                base_linear,
                rank=self._config.rank,
                alpha=self._config.alpha,
                dropout=self._config.dropout,
                init_method=InitMethod(self._config.init_method.value),
            )
        return _layers_module.LoRALinear(
            base_linear,
            rank=self._config.rank,
            alpha=self._config.alpha,
            dropout=self._config.dropout,
            init_method=InitMethod(self._config.init_method.value),
        )

    def _create_qlora_layer(self, base_linear: nn.Linear) -> nn.Module:
        """Create QLoRALinear, falling back to LoRALinear if torchao is unavailable."""
        quant_format = AdapterCompatibilityMatrix.get_base_quant_format(self._backend)

        if quant_format is None or not _layers_module._TORCHAO_AVAILABLE:
            if quant_format is None:
                reason = f"QLoRA not supported on {self._backend.value}"
            else:
                reason = "torchao not installed"
            logger.warning("QLoRA unavailable (%s) — falling back to LoRA", reason)
            return _layers_module.LoRALinear(
                base_linear,
                rank=self._config.rank,
                alpha=self._config.alpha,
                dropout=self._config.dropout,
            )

        return _layers_module.QLoRALinear(
            base_linear,
            rank=self._config.rank,
            alpha=self._config.alpha,
            dropout=self._config.dropout,
            quant_format=quant_format,
        )

    def _create_qdora_layer(self, base_linear: nn.Linear) -> nn.Module:
        """Create QDoRALinear, falling back to DoRALinear if torchao is unavailable."""
        quant_format = AdapterCompatibilityMatrix.get_base_quant_format(self._backend)

        if quant_format is None or not _layers_module._TORCHAO_AVAILABLE:
            if quant_format is None:
                reason = f"QDoRA not supported on {self._backend.value}"
            else:
                reason = "torchao not installed"
            logger.warning("QDoRA unavailable (%s) — falling back to DoRA", reason)
            return _layers_module.DoRALinear(
                base_linear,
                rank=self._config.rank,
                alpha=self._config.alpha,
                dropout=self._config.dropout,
            )

        return _layers_module.QDoRALinear(
            base_linear,
            rank=self._config.rank,
            alpha=self._config.alpha,
            dropout=self._config.dropout,
            quant_format=quant_format,
        )

    @staticmethod
    def _rsetattr(model: nn.Module, path: str, value: nn.Module) -> None:
        """Set a (possibly nested) attribute on the model by dotted path."""
        parts = path.rsplit(".", 1)
        if len(parts) == 1:
            setattr(model, parts[0], value)
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            setattr(parent, parts[1], value)
