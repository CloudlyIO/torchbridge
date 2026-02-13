"""
MiniLM-L6-v2 HAL Tests

Validates TorchBridge HAL behavior with an embedding model.
Model: sentence-transformers/all-MiniLM-L6-v2 (22M params, ~90MB)
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
class TestMiniLMHAL:
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        try:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            model = AutoModel.from_pretrained(self.MODEL_ID)
            model.eval()
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Cannot load model: {e}")

    def test_cpu_forward_pass(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.last_hidden_state.ndim == 3
        assert outputs.last_hidden_state.shape[2] == 384  # MiniLM hidden dim

    def test_batch_embedding(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        sentences = [
            "TorchBridge provides hardware abstraction.",
            "Models run on NVIDIA, AMD, Trainium, and TPU.",
            "Cross-backend validation is important.",
        ]
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embeddings
        embeddings = outputs.last_hidden_state[:, 0]
        assert embeddings.shape == (3, 384)

    def test_embedding_similarity(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        similar = ["The cat sat on the mat.", "A cat was sitting on a mat."]
        different = ["The cat sat on the mat.", "Stock prices rose sharply today."]

        for pair, expected_higher in [(similar, True), (different, False)]:
            inputs = tokenizer(pair, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0]
            sim = torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2]).item()
            if expected_higher:
                assert sim > 0.5, f"Similar sentences should have high similarity, got {sim}"

    @requires_gpu
    def test_cross_backend_consistency(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer(
            "Hardware abstraction for PyTorch.",
            return_tensors="pt",
        )

        with torch.no_grad():
            cpu_out = model(**inputs)

        device = torch.device("cuda")
        model_gpu = model.to(device)
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gpu_out = model_gpu(**inputs_gpu)

        assert_cross_backend_consistency(
            cpu_out.last_hidden_state,
            gpu_out.last_hidden_state,
            label="MiniLM-L6-v2",
        )

        model.to("cpu")
