"""Unit tests for torchbridge.cli.model_io.

The bug this module fixes was three hardcoded assumptions that a model is a
text decoder — the AutoModel class, the input, and the output extraction —
sitting in three places. ``tb-validate --model facebook/dinov2-small`` failed
with "Unrecognized configuration class Dinov2Config for this kind of AutoModel:
AutoModelForCausalLM", and fixing only the loader moves the failure one layer
down rather than removing it.

So these tests check the three steps agree with each other for each kind, not
just that each works alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchbridge.cli import model_io as M


def _cfg(**kw):
    """A stand-in config. Absent attributes must read as absent, so the
    defaults are None rather than being left off — getattr with a default is
    what the module uses, and SimpleNamespace would otherwise raise."""
    base = {"vision_config": None, "image_size": None, "vocab_size": None}
    base.update(kw)
    return SimpleNamespace(**base)


class TestModelKindInference:
    def test_a_text_decoder_is_a_causal_lm(self):
        assert M.infer_model_kind(_cfg(vocab_size=151936)) == M.CAUSAL_LM

    def test_a_vision_backbone_is_vision(self):
        """DINOv2 and ViT: an image size and no vocabulary."""
        assert M.infer_model_kind(_cfg(image_size=518)) == M.VISION

    def test_a_vision_language_model_is_not_read_as_vision(self):
        """A VL config carries a vision_config AND a text vocab_size. Tested
        before the vision case or it reads as image-only and loses its text
        half — which changes both the loader and the input."""
        cfg = _cfg(
            vision_config=_cfg(image_size=224), image_size=224, vocab_size=151936
        )
        assert M.infer_model_kind(cfg) == M.VISION_LANGUAGE

    def test_a_text_encoder_is_not_read_as_vision(self):
        """BERT has a vocabulary and no image size. The discriminator for
        'vision' is an absent vocab_size, not a present image_size."""
        assert M.infer_model_kind(_cfg(vocab_size=30522)) == M.CAUSAL_LM


class TestInputShape:
    def test_a_text_model_keeps_the_requested_shape(self):
        assert M.default_input_shape(M.CAUSAL_LM, _cfg(), (1, 64)) == (1, 64)

    def test_a_vision_model_ignores_a_token_shaped_default(self):
        """1,64 is the CLI default and is token-shaped. Passing it to a vision
        model fails deep inside the patch embedding, not at the command line."""
        shape = M.default_input_shape(
            M.VISION, _cfg(image_size=518, num_channels=3), (1, 64)
        )
        assert shape == (1, 3, 518, 518)

    def test_the_size_comes_from_the_config_not_a_habit(self):
        """dinov2-small is 518, not the 224 that ViT familiarity suggests. A
        hardcoded 224 would run and produce numbers for the wrong input."""
        assert M.default_input_shape(M.VISION, _cfg(image_size=518), None)[-1] == 518
        assert M.default_input_shape(M.VISION, _cfg(image_size=224), None)[-1] == 224

    def test_an_explicit_image_shape_is_respected(self):
        """A 4-D request is already image-shaped and the user meant it."""
        got = M.default_input_shape(M.VISION, _cfg(image_size=518), (2, 3, 64, 64))
        assert got == (2, 3, 64, 64)

    def test_a_vision_language_size_comes_from_its_vision_config(self):
        cfg = _cfg(vision_config=_cfg(image_size=336, num_channels=3), vocab_size=32000)
        assert M.default_input_shape(M.VISION_LANGUAGE, cfg, None) == (1, 3, 336, 336)


class TestInputBuilding:
    def test_a_decoder_gets_integer_token_ids(self):
        """Token IDs are indices. A float index is not a low-precision index,
        it is a crash — so dtype must not reach them."""
        got = M.build_inputs(M.CAUSAL_LM, (1, 64), torch.bfloat16)
        assert set(got) == {"input_ids"}
        assert got["input_ids"].dtype == torch.long

    def test_a_vision_model_gets_pixels_under_the_right_name(self):
        """The keyword name is part of what differs between kinds. A model
        accepting **kwargs will take input_ids and fail somewhere unhelpful."""
        got = M.build_inputs(M.VISION, (1, 3, 224, 224), torch.float32)
        assert set(got) == {"pixel_values"}
        assert got["pixel_values"].shape == (1, 3, 224, 224)

    def test_vision_pixels_follow_the_run_dtype(self):
        got = M.build_inputs(M.VISION, (1, 3, 8, 8), torch.bfloat16)
        assert got["pixel_values"].dtype == torch.bfloat16


class TestOutputExtraction:
    def test_a_decoder_output_gives_last_position_logits(self):
        out = SimpleNamespace(logits=torch.zeros(1, 7, 100))
        assert M.extract_output_tensor(out, M.CAUSAL_LM).shape == (1, 100)

    def test_a_vision_output_gives_the_hidden_state(self):
        """A vision backbone has no logits — no vocabulary to score over.
        Reaching for .logits raises; falling back to the whole output object
        silently compares something else."""
        out = SimpleNamespace(last_hidden_state=torch.zeros(1, 1370, 384))
        assert M.extract_output_tensor(out, M.VISION).shape == (1, 1370, 384)

    def test_a_bare_tensor_passes_through(self):
        t = torch.zeros(2, 3)
        assert M.extract_output_tensor(t, M.TENSOR) is t

    def test_an_unreadable_output_names_the_type_it_saw(self):
        with pytest.raises(TypeError, match="Cannot extract a tensor"):
            M.extract_output_tensor(object(), M.VISION)


class TestTheThreeStepsAgree:
    """The original bug was three steps disagreeing, so agreement is the test.

    Each kind is walked end to end with a stub model that asserts it received
    the keyword it expects. A loader fix that left the input builder behind
    passes every test above and fails here.
    """

    def test_a_vision_pipeline_feeds_pixels_and_reads_hidden_state(self):
        cfg = _cfg(image_size=32, num_channels=3)
        kind = M.infer_model_kind(cfg)
        shape = M.default_input_shape(kind, cfg, (1, 64))
        inputs = M.build_inputs(kind, shape, torch.float32, cfg)

        seen = {}

        def model(**kw):
            seen.update(kw)
            return SimpleNamespace(last_hidden_state=torch.zeros(1, 5, 384))

        tensor = M.extract_output_tensor(model(**inputs), kind)
        assert list(seen) == ["pixel_values"], "vision model was not given pixels"
        assert seen["pixel_values"].shape == (1, 3, 32, 32)
        assert tensor.shape == (1, 5, 384)

    def test_a_decoder_pipeline_feeds_token_ids_and_reads_logits(self):
        cfg = _cfg(vocab_size=100)
        kind = M.infer_model_kind(cfg)
        shape = M.default_input_shape(kind, cfg, (1, 8))
        inputs = M.build_inputs(kind, shape, torch.float32, cfg)

        seen = {}

        def model(**kw):
            seen.update(kw)
            return SimpleNamespace(logits=torch.zeros(1, 8, 100))

        tensor = M.extract_output_tensor(model(**inputs), kind)
        assert list(seen) == ["input_ids"]
        assert tensor.shape == (1, 100)
