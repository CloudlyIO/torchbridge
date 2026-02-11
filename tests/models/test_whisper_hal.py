"""
Whisper-tiny HAL Tests

Validates TorchBridge HAL behavior with a speech model.
Model: openai/whisper-tiny (39M params, ~78MB)
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
class TestWhisperHAL:
    MODEL_ID = "openai/whisper-tiny"

    @pytest.fixture(scope="class")
    def model_and_processor(self):
        try:
            from transformers import AutoProcessor, WhisperForConditionalGeneration

            processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            model = WhisperForConditionalGeneration.from_pretrained(self.MODEL_ID)
            model.eval()
            return model, processor
        except Exception as e:
            pytest.skip(f"Cannot load model: {e}")

    def test_cpu_encoder_forward(self, model_and_processor):
        model, processor = model_and_processor
        # 5 seconds of audio at 16kHz
        dummy_audio = torch.randn(16000 * 5).numpy()
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            encoder_out = model.get_encoder()(**inputs)
        assert encoder_out.last_hidden_state.ndim == 3
        assert encoder_out.last_hidden_state.shape[0] == 1

    def test_cpu_generation(self, model_and_processor):
        model, processor = model_and_processor
        dummy_audio = torch.randn(16000 * 5).numpy()
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
        assert generated_ids.shape[1] > 0
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        assert isinstance(text, list)
        assert len(text) == 1

    @requires_gpu
    def test_cross_backend_consistency(self, model_and_processor):
        model, processor = model_and_processor
        dummy_audio = torch.randn(16000 * 3).numpy()
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            cpu_out = model.get_encoder()(**inputs)

        device = torch.device("cuda")
        model_gpu = model.to(device)
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gpu_out = model_gpu.get_encoder()(**inputs_gpu)

        assert_cross_backend_consistency(
            cpu_out.last_hidden_state,
            gpu_out.last_hidden_state,
            max_diff_threshold=1e-3,
            cosine_sim_threshold=0.999,
            label="Whisper-tiny encoder",
        )

        model.to("cpu")

    def test_encoder_output_shape(self, model_and_processor):
        model, processor = model_and_processor
        # Whisper expects 30s chunks by default; shorter audio gets padded
        dummy_audio = torch.randn(16000 * 10).numpy()
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            encoder_out = model.get_encoder()(**inputs)
        # Whisper tiny: hidden_size = 384
        assert encoder_out.last_hidden_state.shape[2] == 384
