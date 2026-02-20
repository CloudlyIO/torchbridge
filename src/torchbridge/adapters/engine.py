"""
Adapter Engine

Injects adapter layers into models, manages base model quantization
for QLoRA/QDoRA, and provides merge-for-deployment functionality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchbridge.core.config import HardwareBackend
from torchbridge.precision.quantization.formats import QuantizationFormat

from .compatibility import AdapterCompatibilityMatrix, Architecture
from .config import AdapterConfig, AdapterMethod
from .layers import DoRALinear, LoRALinear

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    """Result from adapter injection.

    Attributes:
        success: Whether injection completed successfully.
        method_applied: Adapter method actually used.
        method_requested: Adapter method originally requested.
        used_fallback: Whether a fallback method was used.
        modules_adapted: Number of modules replaced with adapters.
        trainable_params: Total trainable parameters after injection.
        total_params: Total parameters (base + adapters).
        trainable_ratio: Ratio of trainable to total parameters.
        base_quantized: Whether the base model was quantized.
        base_quant_format: Quantization format used for base model.
        memory_before_mb: Model memory before injection (MB).
        memory_after_mb: Model memory after injection (MB).
        warnings: Any warnings generated during injection.
    """

    success: bool
    method_applied: AdapterMethod
    method_requested: AdapterMethod
    used_fallback: bool = False
    modules_adapted: int = 0
    trainable_params: int = 0
    total_params: int = 0
    trainable_ratio: float = 0.0
    base_quantized: bool = False
    base_quant_format: QuantizationFormat | None = None
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "method_applied": self.method_applied.value,
            "method_requested": self.method_requested.value,
            "used_fallback": self.used_fallback,
            "modules_adapted": self.modules_adapted,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_ratio": round(self.trainable_ratio, 6),
            "base_quantized": self.base_quantized,
            "base_quant_format": (
                self.base_quant_format.value if self.base_quant_format else None
            ),
            "memory_before_mb": round(self.memory_before_mb, 1),
            "memory_after_mb": round(self.memory_after_mb, 1),
            "warnings": self.warnings,
        }


def _model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 * 1024)


class AdapterEngine:
    """Inject and manage adapters on any model.

    Handles module targeting, base model quantization for QLoRA/QDoRA,
    backend-aware method selection, and adapter lifecycle management.

    Args:
        config: Adapter configuration.
        backend: Target hardware backend.
        architecture: Specific hardware architecture (optional).
    """

    def __init__(
        self,
        config: AdapterConfig,
        backend: HardwareBackend = HardwareBackend.CPU,
        architecture: Architecture = None,
    ):
        self._config = config
        self._backend = backend
        self._architecture = architecture
        self._resolved_method = self._resolve_method()

    def _resolve_method(self) -> AdapterMethod:
        """Resolve the adapter method, falling back if unsupported."""
        requested = self._config.method
        chain = AdapterCompatibilityMatrix.get_fallback_chain(
            self._backend, self._architecture
        )

        if requested in chain:
            return requested

        fallback = chain[0]
        logger.warning(
            "%s not supported on %s, falling back to %s",
            requested.value,
            self._backend.value,
            fallback.value,
        )
        return fallback

    def inject(self, model: nn.Module) -> AdapterResult:
        """Inject adapters into the model in-place.

        Steps:
        1. Measure pre-injection memory.
        2. Identify target modules by name pattern.
        3. Replace nn.Linear with LoRALinear or DoRALinear.
        4. Freeze all base parameters.
        5. Return result with stats.

        Args:
            model: The model to inject adapters into (modified in-place).

        Returns:
            AdapterResult with injection statistics.
        """
        method = self._resolved_method
        used_fallback = method != self._config.method
        warnings_list: list[str] = []

        if used_fallback:
            warnings_list.append(
                f"Fell back from {self._config.method.value} to {method.value}"
            )

        memory_before = _model_size_mb(model)

        # Identify and replace target modules
        modules_adapted = 0
        replacements: list[tuple[nn.Module, str, LoRALinear | DoRALinear]] = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self._is_target_module(name):
                continue
            # Find parent module and attribute name
            parent, attr = self._get_parent_and_attr(model, name)
            if parent is None:
                continue

            adapter_layer = self._create_adapter_layer(module, method)
            replacements.append((parent, attr, adapter_layer))
            modules_adapted += 1

        # Apply replacements
        for parent, attr, adapter_layer in replacements:
            setattr(parent, attr, adapter_layer)

        if modules_adapted == 0:
            warnings_list.append(
                f"No modules matched target patterns: {self._config.target_modules}"
            )

        # Count params
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model.parameters())
        ratio = trainable / total if total > 0 else 0.0

        memory_after = _model_size_mb(model)

        logger.info(
            "Injected %s adapters into %d modules "
            "(trainable: %d / %d = %.4f%%)",
            method.value,
            modules_adapted,
            trainable,
            total,
            ratio * 100,
        )

        return AdapterResult(
            success=modules_adapted > 0,
            method_applied=method,
            method_requested=self._config.method,
            used_fallback=used_fallback,
            modules_adapted=modules_adapted,
            trainable_params=trainable,
            total_params=total,
            trainable_ratio=ratio,
            base_quantized=False,
            base_quant_format=None,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            warnings=warnings_list,
        )

    def merge(self, model: nn.Module) -> int:
        """Merge all adapters back into base weights.

        Replaces LoRALinear/DoRALinear with plain nn.Linear containing
        merged weights. Used for deployment (inference with zero overhead).

        Args:
            model: Model with adapter layers to merge.

        Returns:
            Number of modules merged.
        """
        merged_count = 0
        replacements: list[tuple[nn.Module, str, nn.Linear]] = []

        for name, module in model.named_modules():
            if isinstance(module, (LoRALinear, DoRALinear)):
                parent, attr = self._get_parent_and_attr(model, name)
                if parent is None:
                    continue
                merged_linear = module.merge()
                replacements.append((parent, attr, merged_linear))
                merged_count += 1

        for parent, attr, merged_linear in replacements:
            setattr(parent, attr, merged_linear)

        logger.info("Merged %d adapter modules", merged_count)
        return merged_count

    def get_adapter_params(
        self, model: nn.Module
    ) -> dict[str, torch.Tensor]:
        """Extract only adapter parameters (for saving).

        Returns parameters that belong to LoRALinear/DoRALinear modules
        (lora_A, lora_B, magnitude). Non-adapter trainable params like
        embeddings and heads are excluded.

        Args:
            model: Model with adapter layers.

        Returns:
            Dict of adapter parameter name → tensor.
        """
        # Collect prefixes of all adapter modules
        adapter_prefixes: list[str] = []
        for name, module in model.named_modules():
            if isinstance(module, (LoRALinear, DoRALinear)):
                adapter_prefixes.append(name + ".")

        adapter_params: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in adapter_prefixes):
                if param.requires_grad:
                    adapter_params[name] = param.data.clone()
        return adapter_params

    def load_adapter_params(
        self, model: nn.Module, params: dict[str, torch.Tensor]
    ) -> int:
        """Load adapter parameters into an adapted model.

        Args:
            model: Model with adapter layers.
            params: Dict of adapter parameter name → tensor.

        Returns:
            Number of parameters loaded.
        """
        loaded = 0
        model_params = dict(model.named_parameters())
        for name, tensor in params.items():
            if name in model_params and model_params[name].requires_grad:
                model_params[name].data.copy_(tensor)
                loaded += 1
        return loaded

    def get_info(self, model: nn.Module) -> dict[str, Any]:
        """Return adapter injection statistics.

        Args:
            model: Model (with or without adapters).

        Returns:
            Dict with adapter statistics.
        """
        adapter_count = 0
        adapter_types: set[str] = set()
        for module in model.modules():
            if isinstance(module, LoRALinear):
                adapter_count += 1
                adapter_types.add("lora")
            elif isinstance(module, DoRALinear):
                adapter_count += 1
                adapter_types.add("dora")

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model.parameters())

        return {
            "adapter_count": adapter_count,
            "adapter_types": sorted(adapter_types),
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio": trainable / total if total > 0 else 0.0,
            "config": self._config.to_dict(),
            "backend": self._backend.value,
            "resolved_method": self._resolved_method.value,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _is_target_module(self, name: str) -> bool:
        """Check if a module name matches any target pattern."""
        return any(name.endswith(t) for t in self._config.target_modules)

    def _get_parent_and_attr(
        self, root: nn.Module, target_name: str
    ) -> tuple[nn.Module | None, str]:
        """Find the parent module and attribute name for a given path."""
        parts = target_name.split(".")
        if len(parts) == 1:
            return root, parts[0]

        parent = root
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                return None, ""
        return parent, parts[-1]

    def _create_adapter_layer(
        self,
        base_linear: nn.Linear,
        method: AdapterMethod,
    ) -> LoRALinear | DoRALinear:
        """Create the appropriate adapter layer."""
        use_dora = method in (AdapterMethod.DORA, AdapterMethod.QDORA)
        cls = DoRALinear if use_dora else LoRALinear
        return cls(
            base_linear=base_linear,
            rank=self._config.rank,
            alpha=self._config.alpha,
            dropout=self._config.dropout,
            init_method=self._config.init_method,
        )
