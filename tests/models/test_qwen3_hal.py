"""
Qwen3-0.6B HAL Tests

Validates TorchBridge HAL behavior with the smallest modern decoder LLM.
Model: Qwen/Qwen3-0.6B (600M params, ~1.2GB FP16)
"""

import pytest
import torch

from .conftest import (
    assert_cross_backend_consistency,
    requires_gpu,
    requires_transformers,
)


@pytest.mark.real_model
@requires_transformers
class TestQwen3HAL:
    MODEL_ID = "Qwen/Qwen3-0.6B"

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
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[-1] == model.config.vocab_size

    def test_cpu_generation(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=10, do_sample=False
            )
        assert generated.shape[1] > inputs["input_ids"].shape[1]
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        assert len(text) > 0

    @requires_gpu
    def test_cross_backend_consistency(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("What is machine learning?", return_tensors="pt")

        with torch.no_grad():
            cpu_out = model(**inputs)

        device = torch.device("cuda")
        model_gpu = model.to(device)
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gpu_out = model_gpu(**inputs_gpu)

        assert_cross_backend_consistency(
            cpu_out.logits, gpu_out.logits, label="Qwen3-0.6B"
        )

        # Move back to CPU for other tests
        model.to("cpu")

    def test_output_dtype(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Test", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
