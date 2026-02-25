"""
Speculative Decoding Engine

Core engine that resolves speculative decoding methods via the compatibility
matrix and produces kwargs for HuggingFace model.generate().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from torchbridge.core.config import HardwareBackend

from .compatibility import Architecture, SpeculationCompatibilityMatrix
from .methods import SPECULATIVE_METHOD_SPECS, SpeculativeMethod

logger = logging.getLogger(__name__)


@dataclass
class SpeculationConfig:
    """Configuration for speculative decoding."""

    method: SpeculativeMethod | None = None  # None = auto-select
    draft_model_name: str | None = None
    num_speculative_tokens: int = 5
    max_batch_size_for_speculation: int = 8
    acceptance_threshold: float = 0.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value if self.method else "auto",
            "draft_model_name": self.draft_model_name,
            "num_speculative_tokens": self.num_speculative_tokens,
            "max_batch_size_for_speculation": self.max_batch_size_for_speculation,
            "acceptance_threshold": self.acceptance_threshold,
            "enabled": self.enabled,
        }


class SpeculationEngine:
    """Resolves and configures speculative decoding for model.generate().

    Selects the optimal speculative method based on backend/architecture
    and produces the appropriate kwargs for HuggingFace ``model.generate()``.

    Supported methods that produce valid generate() kwargs:

    - ``PROMPT_LOOKUP``: N-gram matching; produces ``prompt_lookup_num_tokens``.
      No extra dependencies required.
    - ``DRAFT_MODEL``: Standard assistant-model speculation; produces
      ``assistant_model`` (a *name/path string* for lazy loading) and
      ``num_assistant_tokens``.  Requires ``draft_model_name`` to be set.

    Methods that require custom model architectures and are **not** wired to
    standard HuggingFace generate() kwargs (``EAGLE``, ``MEDUSA``,
    ``LAYER_SKIP``) will raise ``NotImplementedError`` if selected via
    ``get_generation_kwargs()``.  Use ``get_info()`` to inspect the resolved
    method without generating kwargs.

    Args:
        config: Speculation configuration.
        backend: Hardware backend enum.
        architecture: Architecture enum for the current hardware.
    """

    def __init__(
        self,
        config: SpeculationConfig | None = None,
        backend: HardwareBackend = HardwareBackend.CPU,
        architecture: Architecture = None,
    ):
        self._config = config or SpeculationConfig()
        self._backend = backend
        self._architecture = architecture

        # Resolve method
        if self._config.method is None or self._config.method == SpeculativeMethod.NONE:
            if self._config.enabled:
                self._resolved_method = (
                    SpeculationCompatibilityMatrix.get_optimal_method(
                        backend, architecture
                    )
                )
            else:
                self._resolved_method = SpeculativeMethod.NONE
        else:
            # Validate requested method is supported, fall back if not
            if SpeculationCompatibilityMatrix.is_method_supported(
                self._config.method, backend, architecture
            ):
                self._resolved_method = self._config.method
            else:
                chain = SpeculationCompatibilityMatrix.get_fallback_chain(
                    self._config.method, backend, architecture
                )
                self._resolved_method = chain[0] if chain else SpeculativeMethod.NONE
                logger.warning(
                    f"Requested method {self._config.method.value} not supported "
                    f"on {backend.value}/{architecture}; falling back to "
                    f"{self._resolved_method.value}"
                )

    @property
    def method(self) -> SpeculativeMethod:
        """Return the resolved speculative method."""
        return self._resolved_method

    def should_speculate(self, batch_size: int = 1) -> bool:
        """Check if speculation should be used for the given batch size.

        Speculative decoding typically loses efficiency at high batch sizes
        because verification cost scales with batch size.
        """
        if not self._config.enabled:
            return False
        if self._resolved_method == SpeculativeMethod.NONE:
            return False
        return batch_size <= self._config.max_batch_size_for_speculation

    def get_generation_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass to ``model.generate()`` for speculative decoding.

        Returns:
            Dict of kwargs. Empty dict if speculation is disabled or method is NONE.

        Raises:
            NotImplementedError: If the resolved method (``EAGLE``, ``MEDUSA``,
                ``LAYER_SKIP``) does not map to standard HuggingFace generate() kwargs.
                These methods require custom model architectures and cannot be applied
                through generate() alone.
            ValueError: If ``DRAFT_MODEL`` is selected but ``draft_model_name`` is
                not set in the config.
        """
        if not self._config.enabled:
            return {}
        if self._resolved_method == SpeculativeMethod.NONE:
            return {}

        method = self._resolved_method
        kwargs: dict[str, Any] = {}

        if method == SpeculativeMethod.DRAFT_MODEL:
            if not self._config.draft_model_name:
                raise ValueError(
                    "SpeculativeMethod.DRAFT_MODEL requires draft_model_name to be set "
                    "in SpeculationConfig. Provide a HuggingFace model name or local path."
                )
            kwargs["assistant_model"] = self._config.draft_model_name
            kwargs["num_assistant_tokens"] = self._config.num_speculative_tokens

        elif method == SpeculativeMethod.PROMPT_LOOKUP:
            kwargs["prompt_lookup_num_tokens"] = self._config.num_speculative_tokens

        elif method in (SpeculativeMethod.EAGLE, SpeculativeMethod.MEDUSA):
            raise NotImplementedError(
                f"SpeculativeMethod.{method.name} requires a custom model architecture "
                "and does not map to standard HuggingFace generate() kwargs. "
                "Use SpeculativeMethod.DRAFT_MODEL or SpeculativeMethod.PROMPT_LOOKUP "
                "for generate()-compatible speculation."
            )

        elif method == SpeculativeMethod.LAYER_SKIP:
            raise NotImplementedError(
                "SpeculativeMethod.LAYER_SKIP requires a model with early-exit support "
                "and does not have standard HuggingFace generate() kwargs. "
                "Use SpeculativeMethod.PROMPT_LOOKUP for generate()-compatible speculation."
            )

        return kwargs

    def get_info(self) -> dict[str, Any]:
        """Return diagnostic info about the engine configuration."""
        spec = SPECULATIVE_METHOD_SPECS.get(self._resolved_method)
        return {
            "resolved_method": self._resolved_method.value,
            "requested_method": (
                self._config.method.value if self._config.method else "auto"
            ),
            "backend": self._backend.value,
            "architecture": (
                self._architecture.value if self._architecture else None
            ),
            "enabled": self._config.enabled,
            "num_speculative_tokens": self._config.num_speculative_tokens,
            "draft_model_name": self._config.draft_model_name,
            "max_batch_size_for_speculation": (
                self._config.max_batch_size_for_speculation
            ),
            "display_name": spec.display_name if spec else "Unknown",
            "requires_draft_model": spec.requires_draft_model if spec else False,
        }
