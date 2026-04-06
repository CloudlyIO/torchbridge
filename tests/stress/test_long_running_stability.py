import pytest
import torch


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.real_model
class TestLongRunningStability:
    ITERATIONS = 1000
    CHECK_INTERVAL = 100

    def test_minilm_1000_iterations_no_drift(self, minilm_model_and_tokenizer):
        """1000 forward passes produce identical output (no drift)."""
        model, tokenizer = minilm_model_and_tokenizer
        inputs = tokenizer("Stability test sentence.", return_tensors="pt")

        with torch.no_grad():
            baseline = model(**inputs).last_hidden_state.clone()

        for i in range(self.ITERATIONS):
            with torch.no_grad():
                output = model(**inputs).last_hidden_state
            if (i + 1) % self.CHECK_INTERVAL == 0:
                max_diff = torch.abs(baseline - output).max().item()
                assert max_diff < 1e-6, (
                    f"Drift detected at iteration {i + 1}: max_diff={max_diff}"
                )

    def test_dinov2_1000_iterations_no_drift(self, dinov2_model_for_stress):
        """1000 forward passes on vision model produce identical output."""
        image = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            baseline = dinov2_model_for_stress(image).last_hidden_state.clone()

        for i in range(self.ITERATIONS):
            with torch.no_grad():
                output = dinov2_model_for_stress(image).last_hidden_state
            if (i + 1) % self.CHECK_INTERVAL == 0:
                max_diff = torch.abs(baseline - output).max().item()
                assert max_diff < 1e-6, (
                    f"Drift at iteration {i + 1}: max_diff={max_diff}"
                )

    def test_qwen3_100_generations_no_drift(self, qwen3_model):
        """100 autoregressive generations produce consistent output."""
        model, tokenizer = qwen3_model
        inputs = tokenizer("The capital of France is", return_tensors="pt")

        with torch.no_grad():
            baseline = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        baseline_text = tokenizer.decode(baseline[0], skip_special_tokens=True)

        for i in range(100):
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            if (i + 1) % 25 == 0:
                text = tokenizer.decode(output[0], skip_special_tokens=True)
                assert text == baseline_text, (
                    f"Generation drift at iteration {i + 1}: "
                    f"got '{text}' vs baseline '{baseline_text}'"
                )

    def test_1000_iterations_no_memory_leak(
        self, minilm_model_and_tokenizer, memory_tracker
    ):
        """1000 iterations don't accumulate memory."""
        model, tokenizer = minilm_model_and_tokenizer
        inputs = tokenizer("Leak test", return_tensors="pt")

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(**inputs)

        memory_tracker.snapshot("post_warmup")

        for i in range(self.ITERATIONS):
            with torch.no_grad():
                output = model(**inputs)
            del output
            if (i + 1) % self.CHECK_INTERVAL == 0:
                memory_tracker.snapshot(f"iter_{i + 1}")

        memory_tracker.assert_no_leak(tolerance_mb=50)

    @pytest.mark.gpu
    def test_gpu_1000_iterations_stable(self, minilm_model_and_tokenizer):
        """1000 GPU iterations: no drift, no VRAM growth."""
        model, tokenizer = minilm_model_and_tokenizer
        device = torch.device("cuda")
        model_gpu = model.to(device)
        inputs = tokenizer("GPU stability", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            baseline = model_gpu(**inputs).last_hidden_state.cpu().clone()

        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated()

        for i in range(self.ITERATIONS):
            with torch.no_grad():
                output = model_gpu(**inputs)
            if (i + 1) % self.CHECK_INTERVAL == 0:
                max_diff = (
                    torch.abs(baseline - output.last_hidden_state.cpu()).max().item()
                )
                assert max_diff < 1e-5, f"GPU drift at iter {i + 1}: {max_diff}"
            del output

        torch.cuda.synchronize()
        final_mem = torch.cuda.memory_allocated()
        leak_mb = (final_mem - initial_mem) / 1024**2
        assert leak_mb < 10, f"GPU memory leak: {leak_mb:.1f} MB"
        model.cpu()
