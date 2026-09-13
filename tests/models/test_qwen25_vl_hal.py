"""
Qwen2.5-VL-3B HAL Tests

Validates TorchBridge HAL behavior with a multimodal VLM.
Model: Qwen/Qwen2.5-VL-3B-Instruct (3B params, ~6GB FP16)

Marked @slow due to model size.
"""

import pytest
import torch

from .conftest import (
    assert_cross_backend_consistency,
    requires_gpu,
    requires_transformers,
)


@pytest.mark.real_model
@pytest.mark.slow
@requires_transformers
class TestQwen25VLHAL:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover - guarded by the marker
            pytest.skip(f"transformers unavailable: {e}")

        # Qwen2.5-VL is a vision-language model, so AutoModelForCausalLM does
        # not map to it:
        #
        #   Unrecognized configuration class Qwen2_5_VLConfig for this kind of
        #   AutoModel: AutoModelForCausalLM
        #
        # That is raised at load time, caught by the blanket `except Exception`
        # that used to be here, and turned into a skip — so the only
        # vision-language test in the repository never ran, and reported
        # nothing more than "Cannot load model". The vision-language row in
        # ToleranceDB has never been exercised against a real model.
        #
        # The right class has been spelled differently across transformers
        # releases (AutoModelForVision2Seq in older ones,
        # AutoModelForImageTextToText since), so try each and fall back to the
        # model's own class.
        import transformers

        auto_cls = None
        for name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
        ):
            auto_cls = getattr(transformers, name, None)
            if auto_cls is not None:
                break
        if auto_cls is None:  # pragma: no cover - very old transformers
            pytest.skip(
                "this transformers build exposes no vision-language auto class "
                "(tried AutoModelForImageTextToText, AutoModelForVision2Seq, "
                "Qwen2_5_VLForConditionalGeneration)"
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            model = auto_cls.from_pretrained(self.MODEL_ID)
        except Exception as e:
            # Weights absent or no network is a legitimate skip. A wrong model
            # class is not, and is no longer possible to reach this way.
            pytest.skip(f"Cannot load {self.MODEL_ID} weights: {type(e).__name__}: {e}")
        model.eval()
        return model, tokenizer

    def test_cpu_forward_pass(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        # Text-only forward pass
        inputs = tokenizer("Describe this image:", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape[0] == 1
        # A vision-language config keeps vocab_size on its text sub-config, not
        # at the top level: Qwen2_5_VLConfig has no `vocab_size` attribute, and
        # reading one raises AttributeError. This assertion had never run —
        # the fixture used to fail earlier, on the wrong AutoModel class — so
        # the mistake was invisible. `get_text_config()` is the accessor
        # transformers provides for exactly this, and it works for plain
        # decoder configs too.
        vocab_size = model.config.get_text_config().vocab_size
        assert outputs.logits.shape[-1] == vocab_size

    def test_cpu_generation(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("What is in this picture?", return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        assert generated.shape[1] > inputs["input_ids"].shape[1]

    @requires_gpu
    def test_cross_backend_consistency(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Analyze the following:", return_tensors="pt")

        with torch.no_grad():
            cpu_out = model(**inputs)

        device = torch.device("cuda")
        model_gpu = model.to(device)
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gpu_out = model_gpu(**inputs_gpu)

        assert_cross_backend_consistency(
            cpu_out.logits, gpu_out.logits, label="Qwen2.5-VL-3B"
        )

        model.to("cpu")
