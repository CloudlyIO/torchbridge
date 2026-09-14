"""Reach torchao's quantization API across the rename that split it in two.

torchao replaced its quantization factory functions with config classes:

===================================== =====================================
old (torchao < ~0.14)                 new
===================================== =====================================
``int4_weight_only``                  ``Int4WeightOnlyConfig``
``int8_dynamic_activation_int8_weight`` ``Int8DynamicActivationInt8WeightConfig``
===================================== =====================================

``pyproject.toml`` declares ``torchao>=0.4.0,<1.0.0``, a range that spans the
rename, so either spelling can legitimately turn up in an install that
satisfies the declared dependency. Both have to be accepted.

The reason this is a module rather than a ``try``/``except`` at each call site
is that it *was* a ``try``/``except`` at each call site — one in
``adapters/layers.py``, one in ``precision/torchao_integration.py`` — and both
caught ``ImportError`` and concluded "torchao is not installed". With torchao
0.18 installed, every QLoRA and QDoRA construction failed with:

    RuntimeError: torchao is required for QDoRALinear.
                  Install it with: pip install torchao

which names the one action that cannot help, because the package is already
there. Ten tests covered this and every one of them was skipped, because they
are guarded by ``importorskip("torchao")`` and CI never installs the ``all``
extra. Keeping the knowledge in one place is what stops the two copies drifting
apart again.

:func:`unavailable_reason` exists for the same reason: a caller that refuses to
run should be able to say *why* it refused, and "absent" and "present but
renamed" need different advice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "TORCHAO_AVAILABLE",
    "fp8_dynamic_config",
    "int4_config",
    "int8_dynamic_config",
    "quantize_model",
    "unavailable_reason",
]


_quantize_: Callable[..., Any] | None = None
_int4: Callable[..., Any] | None = None
_int8: Callable[..., Any] | None = None
#: FP8 is probed separately: older torchao builds omit it entirely, and that is
#: a legitimate "this version cannot do FP8" rather than a broken install.
_fp8: Callable[..., Any] | None = None

#: Why torchao could not be used, or None when it can. Recorded at import time
#: because that is the only moment the distinction is visible.
_UNAVAILABLE_REASON: str | None = None


def _probe() -> None:
    """Work out, once, which torchao API is present — or why neither is."""
    global _quantize_, _int4, _int8, _UNAVAILABLE_REASON

    try:
        import torchao
    except ImportError as exc:
        _UNAVAILABLE_REASON = (
            f"torchao is not installed ({exc}). Install it with: pip install torchao"
        )
        return

    version = getattr(torchao, "__version__", "unknown")

    try:
        from torchao.quantization import quantize_ as _q
    except ImportError as exc:
        # torchao imported but its quantization entry point did not. Naming the
        # installed version is the difference between an actionable report and
        # a wrong one.
        _UNAVAILABLE_REASON = (
            f"torchao {version} is installed but 'torchao.quantization.quantize_' "
            f"could not be imported ({exc}). This is not a missing package — "
            f"reinstalling will not change it. The installed build is "
            f"incompatible; pin a different torchao version."
        )
        return

    new_api: tuple[Any, Any] | None = None
    old_api: tuple[Any, Any] | None = None
    try:
        from torchao.quantization import (  # type: ignore[attr-defined]
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
        )

        new_api = (Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig)
    except ImportError:
        try:
            from torchao.quantization import (  # type: ignore[attr-defined]
                int4_weight_only,
                int8_dynamic_activation_int8_weight,
            )

            old_api = (int4_weight_only, int8_dynamic_activation_int8_weight)
        except ImportError as exc:
            _UNAVAILABLE_REASON = (
                f"torchao {version} is installed, but it exposes neither the "
                f"current quantization config classes (Int4WeightOnlyConfig, "
                f"Int8DynamicActivationInt8WeightConfig) nor the older factory "
                f"functions (int4_weight_only, "
                f"int8_dynamic_activation_int8_weight): {exc}. Again, not a "
                f"missing package — this build is unsupported."
            )
            return

    chosen = new_api or old_api
    assert chosen is not None  # one of the two branches assigned it
    _int4, _int8 = chosen
    _quantize_ = _q

    # FP8 renamed alongside the others. Its absence is not an error — plenty of
    # torchao builds have no FP8 — so this does not set _UNAVAILABLE_REASON.
    # Without this branch FP8 fell back to the native path even where torchao
    # supported it, silently and for no reason but the new spelling.
    global _fp8
    for name in (
        "Float8DynamicActivationFloat8WeightConfig",
        "float8_dynamic_activation_float8_weight",
    ):
        try:
            module = __import__("torchao.quantization", fromlist=[name])
            _fp8 = getattr(module, name)
            break
        except (ImportError, AttributeError):
            continue


_probe()

#: True when a usable torchao quantization API was found, in either spelling.
TORCHAO_AVAILABLE: bool = _UNAVAILABLE_REASON is None


def unavailable_reason(context: str = "") -> str:
    """Explain why torchao cannot be used, accurately.

    Args:
        context: What was being attempted, e.g. ``"QLoRALinear"``. Prefixed to
            the reason when given.

    Returns:
        A sentence naming the real cause. Distinguishes an absent package from
        one that is installed but exposes an API this code cannot use, because
        only the first is fixed by installing anything.
    """
    reason = _UNAVAILABLE_REASON or "torchao is available"
    return f"torchao is required for {context}. {reason}" if context else reason


def int4_config(group_size: int = 128) -> Any:
    """The torchao argument for INT4 weight-only quantization.

    Args:
        group_size: Quantization group size. Both API generations spell this
            kwarg the same way, so it passes through unchanged.

    Raises:
        RuntimeError: If torchao is unusable. The message says which of the two
            cases applies.
    """
    if _int4 is None:
        raise RuntimeError(unavailable_reason("INT4 weight-only quantization"))
    return _int4(group_size=group_size)


def int8_dynamic_config() -> Any:
    """The torchao argument for INT8 dynamic-activation quantization.

    Raises:
        RuntimeError: If torchao is unusable, with the reason.
    """
    if _int8 is None:
        raise RuntimeError(unavailable_reason("INT8 dynamic-activation quantization"))
    return _int8()


def _linear_weight_types(model: Any) -> dict[str, str]:
    """Type name of every ``nn.Linear`` weight in ``model``, keyed by path."""
    import torch.nn as nn

    out: dict[str, str] = {}
    if isinstance(model, nn.Linear):
        out[""] = type(model.weight.data).__name__
        return out
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out[name] = type(mod.weight.data).__name__
    return out


def quantize_model(model: Any, config: Any, verify: bool = True) -> Any:
    """Apply ``config`` to ``model`` via torchao's ``quantize_``.

    ``quantize_`` mutates in place and returns None in some torchao versions, so
    the model is returned here rather than whatever ``quantize_`` gives back.

    Args:
        model: Module to quantize in place.
        config: A config from :func:`int4_config`, :func:`int8_dynamic_config`
            or :func:`fp8_dynamic_config`.
        verify: Check that the weights actually changed representation. On by
            default, because torchao can decline silently.

    Raises:
        RuntimeError: If torchao is unusable, or — when ``verify`` — if nothing
            was quantized.

    Note:
        The verification is not defensive padding. ``quantize_`` with
        ``Int4WeightOnlyConfig(group_size=128)`` on an ``nn.Linear(64, 64)``
        returns normally, raises nothing, logs nothing, and leaves the weight a
        plain ``torch.Tensor``: the group size exceeds ``in_features``, so there
        is no group to quantize. Reproduced directly::

            in_features=  64 group_size=128 -> Tensor      (unchanged)
            in_features= 256 group_size=128 -> ImportError: Requires mslk

        A caller that asked for INT4 and silently got float32 back believes it
        has a quantized model and has not. Catching a silent no-op is the whole
        job of this project, so it is not something to pass along quietly.
    """
    if _quantize_ is None:
        raise RuntimeError(unavailable_reason("quantization"))

    before = _linear_weight_types(model) if verify else {}
    _quantize_(model, config)
    if not verify:
        return model

    after = _linear_weight_types(model)
    changed = [k for k in after if before.get(k) != after[k]]
    if before and not changed:
        group_size = getattr(config, "group_size", None)
        hint = ""
        if group_size is not None:
            import torch.nn as nn

            widths = {
                m.in_features
                for m in ([model] if isinstance(model, nn.Linear) else model.modules())
                if isinstance(m, nn.Linear)
            }
            too_narrow = sorted(w for w in widths if w < group_size)
            if too_narrow:
                hint = (
                    f" The config's group_size is {group_size} and these layers "
                    f"are narrower than that: in_features={too_narrow}. torchao "
                    f"skips a layer with no complete group, without saying so."
                )
        raise RuntimeError(
            f"torchao ran but quantized nothing — every nn.Linear weight is "
            f"still {sorted(set(after.values()))}.{hint} Pass verify=False only "
            f"if a no-op is genuinely acceptable here."
        )
    return model


def fp8_dynamic_config() -> Any:
    """The torchao argument for FP8 dynamic-activation quantization.

    Raises:
        RuntimeError: If torchao is unusable, or if this torchao build has no
            FP8 support at all — a real possibility, and distinct from the
            rename, so the two say different things.
    """
    if not TORCHAO_AVAILABLE:
        raise RuntimeError(unavailable_reason("FP8 quantization"))
    if _fp8 is None:
        raise RuntimeError(
            "torchao is installed and usable, but this build exposes no FP8 "
            "quantization entry point (neither "
            "Float8DynamicActivationFloat8WeightConfig nor "
            "float8_dynamic_activation_float8_weight). FP8 falls back to the "
            "TorchBridge native path."
        )
    return _fp8()
