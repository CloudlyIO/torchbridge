import pytest
import torch


@pytest.mark.stress
@pytest.mark.real_model
class TestConcurrentInference:
    def test_two_models_same_device_cpu(self, minilm_model_and_tokenizer, dinov2_model_for_stress):
        """Two different models coexist and produce correct output on CPU."""
        text_model, tokenizer = minilm_model_and_tokenizer
        vision_model = dinov2_model_for_stress

        text_inputs = tokenizer("Concurrent test", return_tensors="pt")
        image_inputs = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            text_out = text_model(**text_inputs)
            vision_out = vision_model(image_inputs)

        assert text_out.last_hidden_state.shape[-1] == 384  # MiniLM dim
        assert vision_out.last_hidden_state.shape[-1] == 384  # DINOv2-small dim

    def test_interleaved_inference(self, minilm_model_and_tokenizer, dinov2_model_for_stress):
        """Interleaved forward passes don't corrupt state."""
        text_model, tokenizer = minilm_model_and_tokenizer
        vision_model = dinov2_model_for_stress

        results_text = []
        results_vision = []

        for i in range(5):
            with torch.no_grad():
                text_inputs = tokenizer(f"Iteration {i}", return_tensors="pt")
                text_out = text_model(**text_inputs).last_hidden_state
                results_text.append(text_out.clone())

                vision_out = vision_model(
                    torch.randn(1, 3, 224, 224)
                ).last_hidden_state
                results_vision.append(vision_out.clone())

        # Same input should give same output (determinism check)
        same_input = tokenizer("Iteration 0", return_tensors="pt")
        with torch.no_grad():
            check = text_model(**same_input).last_hidden_state
        assert torch.allclose(results_text[0], check, atol=1e-6)

    @pytest.mark.gpu
    def test_two_models_same_gpu(self, minilm_model_and_tokenizer, dinov2_model_for_stress):
        """Two models on same GPU, both produce valid output."""
        device = torch.device("cuda")
        text_model, tokenizer = minilm_model_and_tokenizer
        text_gpu = text_model.to(device)
        vision_gpu = dinov2_model_for_stress.to(device)

        text_inputs = tokenizer("GPU concurrent", return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        image_inputs = torch.randn(1, 3, 224, 224, device=device)

        with torch.no_grad():
            text_out = text_gpu(**text_inputs)
            vision_out = vision_gpu(image_inputs)

        assert not torch.isnan(text_out.last_hidden_state).any()
        assert not torch.isnan(vision_out.last_hidden_state).any()
        text_model.cpu()
        dinov2_model_for_stress.cpu()
