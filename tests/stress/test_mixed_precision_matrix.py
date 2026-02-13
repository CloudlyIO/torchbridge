import pytest
import torch


@pytest.mark.stress
@pytest.mark.real_model
class TestMixedPrecisionMatrix:
    DTYPES_CPU = [torch.float32, torch.float16, torch.bfloat16]

    @pytest.mark.parametrize("dtype", DTYPES_CPU, ids=["fp32", "fp16", "bf16"])
    def test_minilm_precision(self, minilm_model_and_tokenizer, dtype):
        """MiniLM produces valid output at each precision."""
        model, tokenizer = minilm_model_and_tokenizer
        model_cast = model.to(dtype=dtype)
        inputs = tokenizer("Precision test", return_tensors="pt")
        with torch.no_grad():
            output = model_cast(**inputs)
        assert not torch.isnan(output.last_hidden_state).any()
        assert not torch.isinf(output.last_hidden_state).any()
        model.float()  # restore

    @pytest.mark.parametrize("dtype", DTYPES_CPU, ids=["fp32", "fp16", "bf16"])
    def test_dinov2_precision(self, dinov2_model_for_stress, dtype):
        """DINOv2 produces valid output at each precision."""
        model = dinov2_model_for_stress.to(dtype=dtype)
        images = torch.randn(2, 3, 224, 224, dtype=dtype)
        with torch.no_grad():
            output = model(images)
        assert not torch.isnan(output.last_hidden_state).any()
        model.float()

    @pytest.mark.parametrize("dtype", DTYPES_CPU, ids=["fp32", "fp16", "bf16"])
    def test_cross_precision_consistency(self, minilm_model_and_tokenizer, dtype):
        """Output at lower precision is close to FP32 baseline."""
        model, tokenizer = minilm_model_and_tokenizer
        inputs = tokenizer("Consistency test", return_tensors="pt")

        with torch.no_grad():
            fp32_out = model(**inputs).last_hidden_state.float()
            model_cast = model.to(dtype=dtype)
            cast_out = model_cast(**inputs).last_hidden_state.float()

        # FP16/BF16 should be close but not exact
        if dtype == torch.float32:
            assert torch.allclose(fp32_out, cast_out, atol=1e-6)
        else:
            cos_sim = torch.nn.functional.cosine_similarity(
                fp32_out.flatten().unsqueeze(0),
                cast_out.flatten().unsqueeze(0),
            ).item()
            assert cos_sim > 0.99, f"Cosine sim too low: {cos_sim}"
        model.float()

    @pytest.mark.parametrize("dtype", DTYPES_CPU, ids=["fp32", "fp16", "bf16"])
    def test_qwen3_generation_precision(self, qwen3_model, dtype):
        """LLM generation produces valid tokens at each precision."""
        model, tokenizer = qwen3_model
        model_cast = model.to(dtype=dtype)
        inputs = tokenizer("The meaning of life is", return_tensors="pt")
        with torch.no_grad():
            generated = model_cast.generate(
                **inputs, max_new_tokens=10, do_sample=False
            )
        assert generated.shape[1] > inputs["input_ids"].shape[1]
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        assert len(text) > 0
        model.float()

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16]
    )
    def test_gpu_precision_matrix(self, minilm_model_and_tokenizer, dtype):
        """GPU inference at each precision produces valid output."""
        model, tokenizer = minilm_model_and_tokenizer
        device = torch.device("cuda")
        model_gpu = model.to(device=device, dtype=dtype)
        inputs = tokenizer("GPU precision test", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model_gpu(**inputs)
        assert not torch.isnan(output.last_hidden_state).any()
        model.cpu().float()
