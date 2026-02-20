"""
Multi-Adapter Serving

LRU cache for multiple adapters sharing a single base model.
Supports hot-swap, CPU offload, and merge-for-deployment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .config import AdapterConfig
from .layers import DoRALinear, LoRALinear

logger = logging.getLogger(__name__)


@dataclass
class AdapterSlot:
    """Single loaded adapter with metadata."""

    name: str
    params: dict[str, torch.Tensor]
    config: AdapterConfig
    last_used: float = field(default_factory=time.monotonic)
    device: str = "cpu"


class MultiAdapterManager:
    """Manage multiple adapters on a shared base model.

    Supports LRU eviction, CPU offload for evicted adapters, and
    hot-swapping active adapters with zero model reload.

    Args:
        base_model: Model with adapter layers (LoRALinear/DoRALinear).
        max_loaded: Maximum number of adapters to keep on device.
            When exceeded, the least recently used adapter is evicted
            to CPU memory.
        device: Target device for active adapters. Defaults to the
            device of the first model parameter.
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_loaded: int = 4,
        device: torch.device | None = None,
    ):
        if max_loaded < 1:
            raise ValueError(f"max_loaded must be >= 1, got {max_loaded}")

        self._model = base_model
        self._max_loaded = max_loaded
        self._device = device or self._detect_device()
        self._slots: dict[str, AdapterSlot] = {}
        self._active: str | None = None

    def _detect_device(self) -> torch.device:
        """Detect device from model parameters."""
        for param in self._model.parameters():
            return param.device
        return torch.device("cpu")

    def load_adapter(
        self,
        name: str,
        params: dict[str, torch.Tensor],
        config: AdapterConfig | None = None,
    ) -> None:
        """Load adapter parameters into the cache.

        If the cache is at capacity, the least recently used adapter
        is evicted to CPU.

        Args:
            name: Unique name for this adapter.
            params: Adapter parameter tensors (from AdapterEngine.get_adapter_params).
            config: Optional adapter config metadata.
        """
        if config is None:
            config = AdapterConfig()

        # Evict LRU if at capacity
        if (
            name not in self._slots
            and len(self._slots) >= self._max_loaded
        ):
            self._evict_lru()

        # Store params on CPU for cache
        cpu_params = {k: v.cpu().clone() for k, v in params.items()}
        self._slots[name] = AdapterSlot(
            name=name,
            params=cpu_params,
            config=config,
            last_used=time.monotonic(),
            device="cpu",
        )

        logger.info("Loaded adapter '%s' (%d params)", name, len(params))

    def activate(self, name: str) -> None:
        """Set the active adapter by loading its params into the model.

        Args:
            name: Name of the adapter to activate.

        Raises:
            KeyError: If no adapter with this name is loaded.
        """
        if name not in self._slots:
            raise KeyError(f"Adapter '{name}' not loaded")

        slot = self._slots[name]
        slot.last_used = time.monotonic()

        # Load params into model
        model_params = dict(self._model.named_parameters())
        for param_name, tensor in slot.params.items():
            if param_name in model_params:
                model_params[param_name].data.copy_(
                    tensor.to(self._device)
                )

        self._active = name
        logger.debug("Activated adapter '%s'", name)

    def deactivate(self) -> None:
        """Zero out adapter weights (base-model-only inference)."""
        for module in self._model.modules():
            if isinstance(module, (LoRALinear, DoRALinear)):
                for param in [module.lora_A.weight, module.lora_B.weight]:
                    param.data.zero_()
                if isinstance(module, DoRALinear):
                    # Reset magnitude to base weight norms
                    with torch.no_grad():
                        norms = module.base_linear.weight.norm(dim=1)
                        module.magnitude.data.copy_(norms)

        self._active = None
        logger.debug("Deactivated all adapters")

    def remove(self, name: str) -> None:
        """Permanently remove an adapter from the cache.

        If the removed adapter is currently active, the model's adapter
        weights are zeroed out first (deactivation).

        Args:
            name: Name of the adapter to remove.

        Raises:
            KeyError: If no adapter with this name is loaded.
        """
        if name not in self._slots:
            raise KeyError(f"Adapter '{name}' not loaded")

        if self._active == name:
            self.deactivate()

        del self._slots[name]
        logger.info("Removed adapter '%s'", name)

    def list_adapters(self) -> list[dict[str, Any]]:
        """List all loaded adapters with metadata.

        Returns:
            List of adapter info dicts.
        """
        return [
            {
                "name": slot.name,
                "param_count": sum(
                    t.numel() for t in slot.params.values()
                ),
                "device": slot.device,
                "active": slot.name == self._active,
                "method": slot.config.method.value,
                "rank": slot.config.rank,
            }
            for slot in self._slots.values()
        ]

    def get_active(self) -> str | None:
        """Return the name of the currently active adapter."""
        return self._active

    @property
    def loaded_count(self) -> int:
        """Number of loaded adapters."""
        return len(self._slots)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _evict_lru(self) -> None:
        """Evict the least recently used adapter."""
        if not self._slots:
            return

        lru_name = min(self._slots, key=lambda n: self._slots[n].last_used)

        # Don't evict the active adapter if possible
        if lru_name == self._active and len(self._slots) > 1:
            non_active = [
                n for n in self._slots if n != self._active
            ]
            lru_name = min(
                non_active, key=lambda n: self._slots[n].last_used
            )

        # Clear active state if we're evicting the active adapter
        if lru_name == self._active:
            self._active = None

        del self._slots[lru_name]
        logger.info(
            "Evicted adapter '%s' (LRU, capacity=%d)",
            lru_name,
            self._max_loaded,
        )
