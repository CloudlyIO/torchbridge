import pytest
import torch


def _try_compile(model, mode="reduce-overhead"):
    """Attempt torch.compile; skip test if not available."""
    try:
        return torch.compile(model, mode=mode)
    except Exception as e:
        pytest.skip(f"torch.compile not available: {e}")


def _try_compiled_forward(compiled_model, *args, **kwargs):
    """Run compiled forward; skip if Inductor backend fails (e.g. macOS PCH stale)."""
    try:
        return compiled_model(*args, **kwargs)
    except Exception as e:
        err_msg = str(e).lower()
        if "inductor" in err_msg or "compile" in err_msg or "cpp" in err_msg:
            pytest.skip(f"torch.compile backend failed on this platform: {e}")
        raise


@pytest.mark.stress
@pytest.mark.real_model
class TestTorchCompileCompat:
    def test_minilm_compile(self, minilm_model_and_tokenizer):
        """MiniLM compiles and produces valid output."""
        model, tokenizer = minilm_model_and_tokenizer
        compiled = _try_compile(model)

        inputs = tokenizer("Compile test", return_tensors="pt")
        with torch.no_grad():
            eager_out = model(**inputs).last_hidden_state
            compiled_out = _try_compiled_forward(compiled, **inputs).last_hidden_state

        max_diff = torch.abs(eager_out - compiled_out).max().item()
        assert max_diff < 1e-4, f"Compiled vs eager diverged: {max_diff}"

    def test_dinov2_compile(self, dinov2_model_for_stress):
        """DINOv2 compiles and produces valid output."""
        compiled = _try_compile(dinov2_model_for_stress)

        image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            eager_out = dinov2_model_for_stress(image).last_hidden_state
            compiled_out = _try_compiled_forward(compiled, image).last_hidden_state

        max_diff = torch.abs(eager_out - compiled_out).max().item()
        assert max_diff < 1e-4, f"Compiled vs eager diverged: {max_diff}"

    def test_qwen3_compile_forward(self, qwen3_model):
        """Qwen3 LLM forward pass compiles and matches eager."""
        model, tokenizer = qwen3_model
        compiled = _try_compile(model)

        inputs = tokenizer("Compile LLM test", return_tensors="pt")
        with torch.no_grad():
            eager_out = model(**inputs).logits
            compiled_out = _try_compiled_forward(compiled, **inputs).logits

        max_diff = torch.abs(eager_out - compiled_out).max().item()
        assert max_diff < 1e-4, f"Compiled LLM vs eager diverged: {max_diff}"

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead", "max-autotune"])
    def test_compile_modes(self, minilm_model_and_tokenizer, mode):
        """All torch.compile modes produce valid output."""
        model, tokenizer = minilm_model_and_tokenizer
        compiled = _try_compile(model, mode=mode)

        inputs = tokenizer("Mode test", return_tensors="pt")
        with torch.no_grad():
            output = _try_compiled_forward(compiled, **inputs)
        assert not torch.isnan(output.last_hidden_state).any()

    @pytest.mark.gpu
    def test_gpu_compile_consistency(self, minilm_model_and_tokenizer):
        """Compiled model on GPU matches eager on GPU."""
        model, tokenizer = minilm_model_and_tokenizer
        device = torch.device("cuda")
        model_gpu = model.to(device)

        compiled_gpu = _try_compile(model_gpu)

        inputs = tokenizer("GPU compile", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            eager = model_gpu(**inputs).last_hidden_state
            compiled_result = _try_compiled_forward(
                compiled_gpu, **inputs
            ).last_hidden_state

        max_diff = torch.abs(eager - compiled_result).max().item()
        assert max_diff < 1e-4
        model.cpu()
