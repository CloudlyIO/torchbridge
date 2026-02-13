import gc

import pytest
import torch

# ── Skip helpers ──
requires_transformers = pytest.importorskip(
    "transformers", reason="transformers required"
)


def pytest_collection_modifyitems(items):
    """Auto-skip @pytest.mark.gpu tests when CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="Requires CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ── Memory tracking fixture ──
@pytest.fixture
def memory_tracker():
    """Track memory before/after a test to detect leaks."""

    class MemoryTracker:
        def __init__(self):
            self.snapshots = []

        def snapshot(self, label=""):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
            else:
                import psutil

                allocated = psutil.Process().memory_info().rss
            self.snapshots.append({"label": label, "bytes": allocated})
            return allocated

        def assert_no_leak(self, tolerance_mb=50):
            """Assert final memory <= initial + tolerance."""
            if len(self.snapshots) < 2:
                return
            initial = self.snapshots[0]["bytes"]
            final = self.snapshots[-1]["bytes"]
            leak_mb = (final - initial) / 1024**2
            assert leak_mb < tolerance_mb, (
                f"Memory leak detected: {leak_mb:.1f} MB "
                f"(tolerance: {tolerance_mb} MB)"
            )

    return MemoryTracker()


# ── Small model fixtures (for stress tests — need to be small) ──
@pytest.fixture(scope="module")
def minilm_model_and_tokenizer():
    """MiniLM-L6-v2 -- 22M params, ideal for stress tests. Returns (model, tokenizer)."""
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model.eval()
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Cannot load MiniLM: {e}")


@pytest.fixture(scope="module")
def dinov2_model_for_stress():
    """DINOv2-small -- 22M params, vision model for stress tests."""
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained("facebook/dinov2-small")
        model.eval()
        return model
    except Exception as e:
        pytest.skip(f"Cannot load DINOv2: {e}")


@pytest.fixture(scope="module")
def qwen3_model():
    """Qwen3-0.6B -- 600M params, modern decoder LLM for generation stress tests."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        model.eval()
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Cannot load Qwen3: {e}")
