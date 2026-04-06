"""
DeepSeek-R1-Distill-Qwen-1.5B HAL Tests

Validates TorchBridge HAL behavior with a reasoning-optimized distilled model.
Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (1.5B params, ~3GB FP16)
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
class TestDeepSeekR1HAL:
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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
        inputs = tokenizer("Solve: 2 + 2 = ?", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[-1] == model.config.vocab_size

    def test_cpu_generation(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("What is 15 * 7?", return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        assert generated.shape[1] > inputs["input_ids"].shape[1]

    @requires_gpu
    def test_cross_backend_consistency(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Explain neural networks.", return_tensors="pt")

        with torch.no_grad():
            cpu_out = model(**inputs)

        device = torch.device("cuda")
        model_gpu = model.to(device)
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gpu_out = model_gpu(**inputs_gpu)

        assert_cross_backend_consistency(
            cpu_out.logits, gpu_out.logits, label="DeepSeek-R1-1.5B"
        )

        model.to("cpu")
