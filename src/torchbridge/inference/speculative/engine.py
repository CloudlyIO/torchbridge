"""
Speculative Decoding Engine

Core engine that resolves speculative decoding methods via the compatibility
matrix and produces kwargs for HuggingFace model.generate().
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from torchbridge.core.config import HardwareBackend

from .compatibility import Architecture, SpeculationCompatibilityMatrix
from .methods import SPECULATIVE_METHOD_SPECS, SpeculativeMethod, SpeculativeMethodSpec

logger = logging.getLogger(__name__)

# Maps HardwareBackend values to the PyTorch device string used by that backend.
# CUDA and AMD (ROCm) both use "cuda" as the PyTorch device name.
# DRAFT_MODEL is not supported on TPU/Trainium (their compatibility matrices route
# to PROMPT_LOOKUP), but default to "cpu" as a safe fallback if ever reached.
_BACKEND_TO_DEVICE: dict[HardwareBackend, str] = {
    HardwareBackend.CUDA: "cuda",
    HardwareBackend.AMD: "cuda",  # ROCm-enabled PyTorch uses torch.device("cuda")
    HardwareBackend.TPU: "xla",
    HardwareBackend.TRAINIUM: "cpu",  # Neuron SDK device naming is complex; safe fallback
    HardwareBackend.CPU: "cpu",
    HardwareBackend.CUSTOM: "cpu",
}


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
      ``assistant_model`` (a loaded ``PreTrainedModel``) and
      ``num_assistant_tokens``.  Requires ``draft_model_name`` to be set.
      The draft model is loaded lazily on the first ``get_generation_kwargs()``
      call and cached for subsequent calls.

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
        self._loaded_draft_model: Any = None  # lazily loaded for DRAFT_MODEL
        self._draft_model_lock = threading.Lock()  # guards concurrent load_draft_model calls

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

    @property
    def is_draft_model_loaded(self) -> bool:
        """True if the draft model has been loaded into memory."""
        return self._loaded_draft_model is not None

    def _infer_device(self) -> str:
        """Return the PyTorch device string appropriate for the current backend."""
        return _BACKEND_TO_DEVICE.get(self._backend, "cpu")

    def load_draft_model(self, device: str = "cpu") -> None:
        """Load the draft model into memory for DRAFT_MODEL speculation.

        Called automatically by ``get_generation_kwargs()`` the first time it
        is invoked for DRAFT_MODEL. Can be called explicitly to pre-load the
        model before the inference loop begins.

        Thread-safe: concurrent callers block until the first load completes;
        subsequent calls are no-ops.

        Args:
            device: PyTorch device string to load the model on (default: ``"cpu"``).
                When called from ``get_generation_kwargs()``, the device is inferred
                automatically from the backend via ``_infer_device()``.

        Raises:
            ValueError: If ``draft_model_name`` is not set or is empty/whitespace.
            ImportError: If ``transformers`` is not installed.

        Note for tests: patch ``"transformers.AutoModelForCausalLM"`` (not the
        engine-module path) because the import occurs inside this method at runtime.
        """
        # Fast path: already loaded — no lock needed (reference read is atomic in CPython)
        if self._loaded_draft_model is not None:
            return

        if not self._config.draft_model_name:
            raise ValueError(
                "draft_model_name must be set in SpeculationConfig to load the draft model."
            )
        name = self._config.draft_model_name.strip()
        if not name:
            raise ValueError(
                "draft_model_name cannot be empty or whitespace. "
                "Provide a HuggingFace model name or local path."
            )

        with self._draft_model_lock:
            # Double-check: another thread may have loaded while we waited for the lock
            if self._loaded_draft_model is not None:
                return

            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise ImportError(
                    "DRAFT_MODEL requires the 'transformers' package. "
                    "Install it with: pip install transformers"
                ) from exc

            logger.info(
                "Loading draft model '%s' on device '%s'",
                name,
                device,
            )
            self._loaded_draft_model = AutoModelForCausalLM.from_pretrained(name).to(device)

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

    def get_generation_kwargs(self, device: str | None = None) -> dict[str, Any]:
        """Return kwargs to pass to ``model.generate()`` for speculative decoding.

        For ``DRAFT_MODEL``, the draft model is loaded lazily on the first call
        (via ``load_draft_model()``) and cached. Subsequent calls reuse the cached
        model. The returned ``assistant_model`` value is a loaded ``PreTrainedModel``
        instance, as required by HuggingFace ``model.generate()``.

        Args:
            device: Target device for the draft model (``DRAFT_MODEL`` only).
                If ``None``, the device is inferred from the backend via
                ``_infer_device()`` (CUDA/AMD → ``"cuda"``, CPU → ``"cpu"``,
                TPU → ``"xla"``).  For multi-GPU setups, pass the explicit device
                string (e.g. ``device="cuda:1"``) so the draft model is placed on
                the same device as the main model.

        Returns:
            Dict of kwargs. Empty dict if speculation is disabled or method is NONE.

        Raises:
            NotImplementedError: If the resolved method (``EAGLE``, ``MEDUSA``,
                ``LAYER_SKIP``) does not map to standard HuggingFace generate() kwargs.
                These methods require custom model architectures and cannot be applied
                through generate() alone.  Under normal usage these methods are
                excluded from the compatibility matrix and will never be resolved.
            ValueError: If ``DRAFT_MODEL`` is selected but ``draft_model_name`` is
                not set in the config.
            ImportError: If ``DRAFT_MODEL`` is selected and ``transformers`` is not
                installed.
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
            name = self._config.draft_model_name.strip()
            if not name:
                raise ValueError(
                    "draft_model_name cannot be empty or whitespace. "
                    "Provide a HuggingFace model name or local path."
                )
            # Load the draft model if not already cached. HuggingFace model.generate()
            # requires assistant_model to be a loaded PreTrainedModel instance, not a
            # path string. Device is inferred from the backend so the draft model is
            # placed on the same device as the main model (e.g. "cuda" for CUDA/AMD).
            # Callers can override via the `device` argument for multi-GPU setups.
            target_device = device if device is not None else self._infer_device()
            self.load_draft_model(device=target_device)
            kwargs["assistant_model"] = self._loaded_draft_model
            kwargs["num_assistant_tokens"] = self._config.num_speculative_tokens

        elif method == SpeculativeMethod.PROMPT_LOOKUP:
            kwargs["prompt_lookup_num_tokens"] = self._config.num_speculative_tokens

        elif method in (SpeculativeMethod.EAGLE, SpeculativeMethod.MEDUSA):
            # Safety net: EAGLE and MEDUSA are excluded from the compatibility matrix
            # and never appear as _resolved_method in normal usage. This branch guards
            # against future matrix additions where the method is listed but the
            # generate() wiring is not yet implemented.
            raise NotImplementedError(
                f"SpeculativeMethod.{method.name} requires a custom model architecture "
                "and does not map to standard HuggingFace generate() kwargs. "
                "Use SpeculativeMethod.DRAFT_MODEL or SpeculativeMethod.PROMPT_LOOKUP "
                "for generate()-compatible speculation."
            )

        elif method == SpeculativeMethod.LAYER_SKIP:
            # Safety net: LAYER_SKIP is excluded from the compatibility matrix and
            # never appears as _resolved_method in normal usage. See EAGLE/MEDUSA above.
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

    def get_available_methods(self) -> list[SpeculativeMethod]:
        """Return methods usable on the current backend without custom architectures.

        Filters the backend's supported methods to those that are compatible
        with standard HuggingFace ``model.generate()`` kwargs (i.e., excludes
        EAGLE, MEDUSA, and LAYER_SKIP which require custom model architectures).

        Returns:
            List of ``SpeculativeMethod`` values that can be passed to
            ``SpeculationConfig(method=...)`` without raising ``NotImplementedError``.
        """
        supported = SpeculationCompatibilityMatrix.get_supported_methods(
            self._backend, self._architecture
        )
        result = []
        for m in supported:
            spec: SpeculativeMethodSpec | None = SPECULATIVE_METHOD_SPECS.get(m)
            if spec is None or not spec.requires_custom_arch:
                result.append(m)
        return result
