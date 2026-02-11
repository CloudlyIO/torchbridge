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
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID)
            model.eval()
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Cannot load model: {e}")

    def test_cpu_forward_pass(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        # Text-only forward pass
        inputs = tokenizer("Describe this image:", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[-1] == model.config.vocab_size

    def test_cpu_generation(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("What is in this picture?", return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=10, do_sample=False
            )
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
