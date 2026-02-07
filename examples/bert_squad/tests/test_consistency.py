"""
Cross-Backend Consistency Tests for BERT SQuAD

Pytest-based tests to validate numerical consistency across backends.
Run with: pytest tests/test_consistency.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


# Test fixtures
@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer once for all tests."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def base_model():
    """Load model once for all tests."""
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    model.eval()
    return model


@pytest.fixture
def sample_input(tokenizer):
    """Create sample input for testing."""
    question = "What is the capital of France?"
    context = "Paris is the capital and largest city of France."
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return inputs


def get_available_devices():
    """Get list of available devices for testing."""
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append(torch.device("xpu"))

    return devices


def synchronize_device(device: torch.device):
    """Synchronize device."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


@pytest.fixture(params=get_available_devices())
def device(request):
    """Parametrized fixture for all available devices."""
    return request.param


class TestCrossBackendConsistency:
    """Tests for cross-backend numerical consistency."""

    def test_model_loads_on_device(self, base_model, device):
        """Test that model can be loaded on each device."""
        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        model = model.to(device)
        model.eval()

        # Check model is on correct device
        param = next(model.parameters())
        assert param.device.type == device.type

    def test_inference_runs_on_device(self, base_model, sample_input, device):
        """Test that inference runs on each device."""
        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        model = model.to(device)
        model.eval()

        inputs = {k: v.to(device) for k, v in sample_input.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.start_logits is not None
        assert outputs.end_logits is not None
        assert outputs.start_logits.shape == (1, 384)
        assert outputs.end_logits.shape == (1, 384)

    def test_output_consistency_vs_cpu(self, base_model, sample_input, device):
        """Test that outputs match CPU within tolerance."""
        tolerance = 1e-4

        # Get CPU reference
        cpu_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        cpu_model.eval()
        with torch.no_grad():
            cpu_outputs = cpu_model(**sample_input)
        cpu_start = cpu_outputs.start_logits
        cpu_end = cpu_outputs.end_logits

        # Get device outputs
        device_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        device_model = device_model.to(device)
        device_model.eval()

        inputs = {k: v.to(device) for k, v in sample_input.items()}
        with torch.no_grad():
            device_outputs = device_model(**inputs)
        device_start = device_outputs.start_logits.cpu()
        device_end = device_outputs.end_logits.cpu()

        # Compare
        start_diff = torch.abs(cpu_start - device_start).max().item()
        end_diff = torch.abs(cpu_end - device_end).max().item()

        assert start_diff < tolerance, f"Start logits diff {start_diff} exceeds tolerance"
        assert end_diff < tolerance, f"End logits diff {end_diff} exceeds tolerance"

    def test_cosine_similarity_vs_cpu(self, base_model, sample_input, device):
        """Test that outputs have high cosine similarity to CPU."""
        min_similarity = 0.9999

        # Get CPU reference
        cpu_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        cpu_model.eval()
        with torch.no_grad():
            cpu_outputs = cpu_model(**sample_input)
        cpu_start = cpu_outputs.start_logits.flatten()
        cpu_end = cpu_outputs.end_logits.flatten()

        # Get device outputs
        device_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        device_model = device_model.to(device)
        device_model.eval()

        inputs = {k: v.to(device) for k, v in sample_input.items()}
        with torch.no_grad():
            device_outputs = device_model(**inputs)
        device_start = device_outputs.start_logits.cpu().flatten()
        device_end = device_outputs.end_logits.cpu().flatten()

        # Compute cosine similarity
        start_sim = F.cosine_similarity(cpu_start.unsqueeze(0), device_start.unsqueeze(0)).item()
        end_sim = F.cosine_similarity(cpu_end.unsqueeze(0), device_end.unsqueeze(0)).item()

        assert start_sim >= min_similarity, f"Start cosine sim {start_sim} below threshold"
        assert end_sim >= min_similarity, f"End cosine sim {end_sim} below threshold"

    def test_answer_prediction_matches(self, base_model, sample_input, device, tokenizer):
        """Test that predicted answer is the same across backends."""
        # Get CPU answer
        cpu_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        cpu_model.eval()
        with torch.no_grad():
            cpu_outputs = cpu_model(**sample_input)
        cpu_start_idx = cpu_outputs.start_logits.argmax().item()
        cpu_end_idx = cpu_outputs.end_logits.argmax().item()

        # Get device answer
        device_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        device_model = device_model.to(device)
        device_model.eval()

        inputs = {k: v.to(device) for k, v in sample_input.items()}
        with torch.no_grad():
            device_outputs = device_model(**inputs)
        device_start_idx = device_outputs.start_logits.argmax().item()
        device_end_idx = device_outputs.end_logits.argmax().item()

        assert cpu_start_idx == device_start_idx, "Start position mismatch"
        assert cpu_end_idx == device_end_idx, "End position mismatch"


class TestBatchConsistency:
    """Test consistency with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_size_consistency(self, tokenizer, device, batch_size):
        """Test outputs are consistent across batch sizes."""
        question = "What is the capital?"
        context = "Paris is the capital of France."

        # Single sample
        single_input = tokenizer(
            question, context,
            max_length=384, truncation=True, padding="max_length",
            return_tensors="pt"
        )

        # Batched
        batch_input = tokenizer(
            [question] * batch_size, [context] * batch_size,
            max_length=384, truncation=True, padding="max_length",
            return_tensors="pt"
        )

        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        model = model.to(device)
        model.eval()

        single_input = {k: v.to(device) for k, v in single_input.items()}
        batch_input = {k: v.to(device) for k, v in batch_input.items()}

        with torch.no_grad():
            single_output = model(**single_input)
            batch_output = model(**batch_input)

        # Check that each batch item matches single output
        for i in range(batch_size):
            start_diff = torch.abs(
                single_output.start_logits[0] - batch_output.start_logits[i]
            ).max().item()
            end_diff = torch.abs(
                single_output.end_logits[0] - batch_output.end_logits[i]
            ).max().item()

            assert start_diff < 1e-5, f"Batch item {i} start logits differ"
            assert end_diff < 1e-5, f"Batch item {i} end logits differ"


class TestDeterminism:
    """Test deterministic behavior."""

    def test_multiple_runs_same_output(self, base_model, sample_input, device):
        """Test that multiple runs produce identical outputs."""
        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        model = model.to(device)
        model.eval()

        inputs = {k: v.to(device) for k, v in sample_input.items()}

        outputs = []
        for _ in range(5):
            with torch.no_grad():
                out = model(**inputs)
            outputs.append((out.start_logits.cpu(), out.end_logits.cpu()))

        # All outputs should be identical
        for i in range(1, len(outputs)):
            start_diff = torch.abs(outputs[0][0] - outputs[i][0]).max().item()
            end_diff = torch.abs(outputs[0][1] - outputs[i][1]).max().item()
            assert start_diff == 0, f"Run {i} start logits differ"
            assert end_diff == 0, f"Run {i} end logits differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
