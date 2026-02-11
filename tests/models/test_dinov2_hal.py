"""
DINOv2-small HAL Tests

Validates TorchBridge HAL behavior with a vision encoder.
Model: facebook/dinov2-small (22M params, ~44MB)
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
class TestDINOv2HAL:
    MODEL_ID = "facebook/dinov2-small"

    @pytest.fixture(scope="class")
    def model_and_processor(self):
        try:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            model = AutoModel.from_pretrained(self.MODEL_ID)
            model.eval()
            return model, processor
        except Exception as e:
            pytest.skip(f"Cannot load model: {e}")

    def test_cpu_forward_pass(self, model_and_processor):
        model, processor = model_and_processor
        dummy_image = torch.randn(1, 3, 224, 224)
        inputs = processor(images=dummy_image, return_tensors="pt", do_rescale=False)
        with torch.no_grad():
            outputs = model(**inputs)
        # DINOv2 outputs: last_hidden_state [batch, num_patches+1, hidden_dim]
        assert outputs.last_hidden_state.ndim == 3
        assert outputs.last_hidden_state.shape[0] == 1
        assert outputs.last_hidden_state.shape[2] == 384  # dinov2-small hidden dim

    def test_batch_forward(self, model_and_processor):
        model, processor = model_and_processor
        dummy_images = torch.randn(4, 3, 224, 224)
        inputs = processor(images=dummy_images, return_tensors="pt", do_rescale=False)
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.last_hidden_state.shape[0] == 4

    @requires_gpu
    def test_cross_backend_consistency(self, model_and_processor):
        model, processor = model_and_processor
        dummy_image = torch.randn(2, 3, 224, 224)
        inputs = processor(images=dummy_image, return_tensors="pt", do_rescale=False)

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
            label="DINOv2-small",
        )

        model.to("cpu")

    def test_feature_norm(self, model_and_processor):
        model, processor = model_and_processor
        dummy_image = torch.randn(1, 3, 224, 224)
        inputs = processor(images=dummy_image, return_tensors="pt", do_rescale=False)
        with torch.no_grad():
            outputs = model(**inputs)
        # Features should have reasonable norms (not all zeros or exploding)
        norms = outputs.last_hidden_state.norm(dim=-1)
        assert norms.mean() > 0.1
        assert norms.mean() < 1000.0
