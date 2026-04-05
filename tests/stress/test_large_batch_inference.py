import pytest
import torch


@pytest.mark.stress
@pytest.mark.real_model
class TestLargeBatchInference:
    BATCH_SIZES = [1, 8, 32, 64, 128]

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_minilm_batch_scaling(self, minilm_model_and_tokenizer, batch_size):
        """Verify MiniLM handles increasing batch sizes without error."""
        model, tokenizer = minilm_model_and_tokenizer
        texts = ["TorchBridge hardware abstraction test."] * batch_size
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            output = model(**inputs)
        assert output.last_hidden_state.shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_dinov2_batch_scaling(self, dinov2_model_for_stress, batch_size):
        """Verify DINOv2 handles increasing batch sizes without error."""
        images = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            output = dinov2_model_for_stress(images)
        assert output.last_hidden_state.shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_qwen3_batch_generation(self, qwen3_model, batch_size):
        """Verify Qwen3 autoregressive generation scales across batch sizes."""
        model, tokenizer = qwen3_model
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = ["Explain quantum computing in one sentence."] * batch_size
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        assert generated.shape[0] == batch_size
        assert generated.shape[1] > inputs["input_ids"].shape[1]

    @pytest.mark.gpu
    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_gpu_batch_scaling(self, minilm_model_and_tokenizer, batch_size):
        """Verify GPU batch inference scales correctly."""
        model, tokenizer = minilm_model_and_tokenizer
        device = torch.device("cuda")
        model_gpu = model.to(device)
        texts = ["HAL cross-backend test."] * batch_size
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model_gpu(**inputs)
        assert output.last_hidden_state.shape[0] == batch_size
        model.cpu()  # cleanup
