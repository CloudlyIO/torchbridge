# SPDX-License-Identifier: Apache-2.0
"""Load a model, build an input for it, and read a tensor back out.

Every one of these three steps was written assuming a text decoder, in three
different places, and the assumption is invisible until a model that is not one
arrives. ``tb-validate --model facebook/dinov2-small`` failed with::

    Unrecognized configuration class Dinov2Config for this kind of AutoModel:
    AutoModelForCausalLM

which names the symptom, at the first of the three. Fixing only that pushes the
failure one layer down: the model then loads and is handed a tensor of token
IDs, because the input builder is equally hardcoded.

So the three steps live together here. A model kind is decided once, from the
config, and the loader, the input and the output extractor all follow from it.
Adding a family means adding one entry, not hunting three call sites.

The four kinds:

``tensor``
    Not a transformers model at all — the smoke model, or a file on disk.
    Takes a float tensor positionally.
``causal-lm``
    A text decoder. Takes ``input_ids``, returns ``.logits``.
``vision``
    An image backbone: DINOv2, ViT. Takes ``pixel_values``, returns
    ``.last_hidden_state`` — it has no vocabulary and so no logits.
``vision-language``
    Takes an image *and* text together. Returns ``.logits`` over the text
    vocabulary.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "CAUSAL_LM",
    "TENSOR",
    "VISION",
    "VISION_LANGUAGE",
    "build_inputs",
    "default_input_shape",
    "extract_output_tensor",
    "infer_model_kind",
    "load_for_validation",
]

TENSOR = "tensor"
CAUSAL_LM = "causal-lm"
VISION = "vision"
VISION_LANGUAGE = "vision-language"

#: Kinds that take an image. Kept as a set rather than repeated ``or`` chains
#: so a new image-taking family is one entry, not a scattered edit.
_IMAGE_KINDS = frozenset({VISION, VISION_LANGUAGE})


def infer_model_kind(config: Any) -> str:
    """Decide what kind of model a config describes.

    Order matters. A vision-language config carries ``vision_config`` *and* a
    text ``vocab_size``, so it has to be tested before the vision case or it
    reads as an image-only backbone and loses its text half.

    Args:
        config: A transformers ``PretrainedConfig``.

    Returns:
        One of :data:`CAUSAL_LM`, :data:`VISION`, :data:`VISION_LANGUAGE`.
    """
    if getattr(config, "vision_config", None) is not None:
        return VISION_LANGUAGE
    # A vision backbone describes its input in pixels and has no token
    # vocabulary. An absent vocab_size is the discriminator: a text encoder
    # such as BERT has image_size unset and vocab_size set.
    if (
        getattr(config, "image_size", None) is not None
        and getattr(config, "vocab_size", None) is None
    ):
        return VISION
    return CAUSAL_LM


def load_for_validation(model_path: str, dtype: torch.dtype) -> tuple[Any, str]:
    """Load a HuggingFace model with the AutoModel class that fits it.

    Args:
        model_path: A HuggingFace model ID.
        dtype: Dtype to load the weights in.

    Returns:
        ``(model, kind)`` — the loaded model and which of the four kinds it is.

    Raises:
        Whatever transformers raises. The caller reports it; wrapping it here
        would hide which of loading, config reading or download failed.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)  # nosec B615
    kind = infer_model_kind(config)

    if kind == VISION_LANGUAGE:
        # AutoModelForVision2Seq does not exist in every transformers release,
        # and the concrete class name differs per model. Resolving a list means
        # the first spelling this install actually has is the one used, rather
        # than an ImportError naming a class nobody needs.
        import transformers

        for name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            cls = getattr(transformers, name, None)
            if cls is not None:
                return cls.from_pretrained(model_path, dtype=dtype), kind  # nosec B615
        raise ImportError(
            f"transformers {transformers.__version__} exposes neither "
            f"AutoModelForImageTextToText nor AutoModelForVision2Seq, so "
            f"{model_path} cannot be loaded as a vision-language model."
        )

    if kind == VISION:
        from transformers import AutoModel

        return AutoModel.from_pretrained(model_path, dtype=dtype), kind  # nosec B615

    from transformers import AutoModelForCausalLM

    return (
        AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype),  # nosec B615
        kind,
    )


def default_input_shape(
    kind: str, config: Any, requested: tuple[int, ...] | None
) -> tuple[int, ...]:
    """The input shape to use, derived from the config when it can be.

    An image model needs ``(batch, channels, H, W)``; the CLI default is
    ``1,64``, which is token-shaped. Handing that to DINOv2 fails deep inside
    the patch embedding rather than at the command line, so the shape is taken
    from the config unless the caller asked for a specific one.

    DINOv2-small's ``image_size`` is 518, not the 224 that ViT habits suggest —
    another reason to read it rather than assume it.

    Args:
        kind: A model kind from this module.
        config: The model's config, or None.
        requested: Shape the user asked for, or None.

    Returns:
        The shape to build an input with.
    """
    if kind not in _IMAGE_KINDS:
        return requested or (1, 64)
    # A 4-D request is already image-shaped and the user meant it.
    if requested is not None and len(requested) == 4:
        return requested
    vision = getattr(config, "vision_config", None) or config
    size = getattr(vision, "image_size", None) or 224
    channels = getattr(vision, "num_channels", None) or 3
    batch = requested[0] if requested else 1
    return (batch, channels, size, size)


def build_inputs(
    kind: str, shape: tuple[int, ...], dtype: torch.dtype, config: Any = None
) -> dict[str, torch.Tensor]:
    """Build a synthetic input, as the keyword arguments the model expects.

    Returned as a dict rather than a bare tensor because the keyword name is
    part of what differs between kinds — ``input_ids`` and ``pixel_values`` are
    not interchangeable, and passing the wrong one to a model that accepts
    ``**kwargs`` fails somewhere unhelpful.

    Args:
        kind: A model kind from this module.
        shape: Input shape, from :func:`default_input_shape`.
        dtype: Dtype for float inputs. Token IDs ignore it — they are indices,
            and a float index is not a low-precision index, it is a crash.
        config: The model's config, used for the vision-language text half.

    Returns:
        Keyword arguments to call the model with.
    """
    if kind == VISION:
        return {"pixel_values": torch.randn(*shape, dtype=dtype)}

    if kind == VISION_LANGUAGE:
        return _vision_language_inputs(shape, dtype, config)

    if kind == CAUSAL_LM:
        return {"input_ids": torch.ones(*shape, dtype=torch.long)}

    return {"": torch.randn(*shape, dtype=dtype)}  # tensor kind, passed positionally


def _vision_language_inputs(
    shape: tuple[int, ...], dtype: torch.dtype, config: Any
) -> dict[str, torch.Tensor]:
    """Build a vision-language input through the model's own processor.

    A hand-built ``pixel_values`` tensor is not enough here, and the failure is
    not obvious. Qwen2.5-VL uses dynamic resolution: it wants a *flattened
    patch sequence* of shape ``(num_patches, patch_dim)`` together with an
    ``image_grid_thw`` describing the grid those patches came from. Handing it
    a plain ``(1, 3, H, W)`` image fails with::

        AttributeError: 'NoneType' object has no attribute 'tolist'

    which says nothing about the real cause. The processor is the only thing
    that knows the packing, so it does the packing.

    The text half goes through the chat template for the same reason. Without
    the template the prompt has no image placeholder token, the image is never
    attended to, and the vision tower's contribution quietly drops out of the
    comparison — a run that passes while measuring the wrong thing.

    The image is a fixed pattern, not noise: both machines in a cross-backend
    comparison must see byte-identical input, and a seeded RNG is one more
    thing that can differ between two installs.
    """
    import numpy as np
    from transformers import AutoProcessor

    name = getattr(config, "_name_or_path", None)
    if not name:
        raise ValueError(
            "Cannot build a vision-language input: the config does not record "
            "which model it came from, so its processor cannot be loaded."
        )
    processor = AutoProcessor.from_pretrained(name)  # nosec B615

    height, width = (shape[2], shape[3]) if len(shape) == 4 else (224, 224)
    # A smooth deterministic gradient. Zeros would work too, but an all-zero
    # image drives parts of the tower to constant activations, and a constant
    # output makes a divergence comparison vacuous rather than clean.
    row = np.linspace(0, 255, width, dtype=np.uint8)
    plane = np.tile(row, (height, 1))
    image = np.stack([plane, plane.T, (plane // 2)], axis=-1)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    batch = processor(text=[text], images=[image], return_tensors="pt")
    inputs = dict(batch)
    # Float tensors follow the run's dtype; index tensors must stay integral.
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            inputs[key] = value.to(dtype)
    return inputs


def extract_output_tensor(output: Any, kind: str) -> torch.Tensor:
    """Read the tensor to compare out of a model's output.

    A vision backbone has no ``logits`` — it has no vocabulary to score over —
    so ``last_hidden_state`` is the comparable tensor. Reaching for ``logits``
    on one raises, and falling back to "the whole output object" silently
    compares something else.

    Args:
        output: Whatever the model returned.
        kind: A model kind from this module.

    Returns:
        A tensor.

    Raises:
        TypeError: If no tensor can be found, naming the type that was seen.
    """
    if isinstance(output, torch.Tensor):
        return output

    if kind in (CAUSAL_LM, VISION_LANGUAGE):
        logits = getattr(output, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits[:, -1, :] if logits.ndim >= 2 else logits

    for attr in ("last_hidden_state", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value

    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]

    raise TypeError(
        f"Cannot extract a tensor from a {type(output).__name__} for model kind "
        f"'{kind}'. Expected .logits, .last_hidden_state or a tensor."
    )
