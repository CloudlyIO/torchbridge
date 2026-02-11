import gc

import pytest
import torch


@pytest.mark.stress
@pytest.mark.real_model
class TestMultiModelSequential:
    MODELS = [
        ("Qwen/Qwen3-0.6B", "llm"),
        ("facebook/dinov2-small", "vision"),
        ("sentence-transformers/all-MiniLM-L6-v2", "embedding"),
    ]

    def test_sequential_load_unload_no_leak(self, memory_tracker):
        """Load/unload 3 models sequentially, verify no memory leak."""
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        memory_tracker.snapshot("start")

        for model_id, model_type in self.MODELS:
            if model_type == "llm":
                model = AutoModelForCausalLM.from_pretrained(model_id)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                inputs = tokenizer("Memory test", return_tensors="pt")
                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                del tokenizer
            elif model_type == "vision":
                model = AutoModel.from_pretrained(model_id)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.randn(1, 3, 224, 224))
            else:  # embedding
                model = AutoModel.from_pretrained(model_id)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                inputs = tokenizer("Memory leak test", return_tensors="pt")
                with torch.no_grad():
                    _ = model(**inputs)
                del tokenizer

            del model
            gc.collect()

        memory_tracker.snapshot("end")
        # 200MB tolerance: Qwen3-0.6B (600M params) + HF tokenizer caches
        # leave residual memory after gc.collect() on CPU
        memory_tracker.assert_no_leak(tolerance_mb=200)

    @pytest.mark.gpu
    def test_sequential_gpu_load_unload(self, memory_tracker):
        """Load/unload models on GPU, verify VRAM is freed."""
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda")

        torch.cuda.empty_cache()
        memory_tracker.snapshot("start")

        for model_id, _ in self.MODELS:
            model = AutoModel.from_pretrained(model_id).to(device)
            model.eval()
            with torch.no_grad():
                if "dinov2" in model_id:
                    _ = model(torch.randn(1, 3, 224, 224, device=device))
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    inputs = tokenizer("GPU leak test", return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    _ = model(**inputs)
                    del tokenizer, inputs

            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        memory_tracker.snapshot("end")
        memory_tracker.assert_no_leak(tolerance_mb=50)
