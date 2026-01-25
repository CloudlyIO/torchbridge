"""
Cross-Backend End-to-End Tests for Real BERT Model

Tests that BERT optimization works consistently across all 4 hardware backends
(NVIDIA, AMD, TPU, Intel) with:
- Consistent output correctness
- Measurable speedups on each platform
- Same predictions regardless of backend

Markers:
    - @pytest.mark.e2e: End-to-end test
    - @pytest.mark.slow: Long-running test
    - @pytest.mark.real_model: Uses real model weights
    - @pytest.mark.requires_transformers: Requires HuggingFace transformers

Success Criteria:
    - BERT runs on each available backend
    - Output matches baseline within tolerance
    - Speedup achieved on each backend
"""

import copy
import pytest
import torch

from .conftest import (
    requires_transformers,
    benchmark_function,
    assert_output_close,
)


# =============================================================================
# Backend Availability Checks
# =============================================================================

def is_nvidia_available():
    """Check if NVIDIA/CUDA backend is available."""
    return torch.cuda.is_available() and not _is_rocm()


def is_amd_available():
    """Check if AMD/ROCm backend is available."""
    return torch.cuda.is_available() and _is_rocm()


def is_tpu_available():
    """Check if TPU/XLA backend is available."""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        _ = xm.xla_device()
        return True
    except Exception:
        return False


def is_intel_available():
    """Check if Intel/XPU backend is available."""
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def _is_rocm():
    """Check if running on ROCm instead of CUDA."""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


# Skip decorators
requires_nvidia = pytest.mark.skipif(
    not is_nvidia_available(),
    reason="Requires NVIDIA CUDA GPU"
)

requires_amd = pytest.mark.skipif(
    not is_amd_available(),
    reason="Requires AMD ROCm GPU"
)

requires_tpu = pytest.mark.skipif(
    not is_tpu_available(),
    reason="Requires TPU with torch_xla"
)

requires_intel = pytest.mark.skipif(
    not is_intel_available(),
    reason="Requires Intel XPU"
)


# =============================================================================
# Backend Helper Functions
# =============================================================================

def get_nvidia_backend():
    """Create and return NVIDIA backend."""
    from kernel_pytorch.backends.nvidia import NVIDIABackend
    return NVIDIABackend()


def get_amd_backend():
    """Create and return AMD backend."""
    from kernel_pytorch.backends.amd import AMDBackend
    return AMDBackend()


def get_tpu_backend():
    """Create and return TPU backend."""
    from kernel_pytorch.backends.tpu import TPUBackend
    return TPUBackend()


def get_intel_backend():
    """Create and return Intel backend."""
    from kernel_pytorch.backends.intel import IntelBackend
    return IntelBackend()


# =============================================================================
# Cross-Backend BERT Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
class TestCrossBackendBERT:
    """Test BERT optimization across all backends."""

    @pytest.fixture
    def bert_baseline(self):
        """Get baseline BERT outputs on CPU."""
        from transformers import AutoModel, AutoTokenizer

        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Test texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming technology.",
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )

        # Get CPU baseline output
        with torch.no_grad():
            baseline_output = model(**inputs).last_hidden_state

        return {
            "model": model,
            "tokenizer": tokenizer,
            "inputs": inputs,
            "baseline_output": baseline_output,
            "texts": texts,
        }

    @requires_nvidia
    def test_bert_nvidia_backend(self, bert_baseline):
        """Test BERT on NVIDIA backend matches baseline."""
        from kernel_pytorch.backends.nvidia import NVIDIABackend

        backend = NVIDIABackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]
        baseline_output = bert_baseline["baseline_output"]

        # Prepare model for NVIDIA
        prepared_model = backend.prepare_model(model)

        # Move inputs to CUDA
        cuda_inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            nvidia_output = prepared_model(**cuda_inputs).last_hidden_state

        # Move back to CPU for comparison
        nvidia_output_cpu = nvidia_output.cpu()

        # Assert outputs match
        assert_output_close(
            baseline_output,
            nvidia_output_cpu,
            atol=0.05,  # Realistic for cross-device FP32
            rtol=0.01,
            message="BERT NVIDIA output vs CPU baseline"
        )

        print(f"\nBERT NVIDIA Backend Test:")
        print(f"  Device: {backend.device}")
        print(f"  Output shape: {nvidia_output.shape}")
        print(f"  Output matches baseline: PASS")

    @requires_nvidia
    def test_bert_nvidia_speedup(self, bert_baseline):
        """Test BERT speedup on NVIDIA backend."""
        import copy
        from kernel_pytorch.backends.nvidia import NVIDIABackend
        from kernel_pytorch.backends.base_backend import OptimizationLevel

        backend = NVIDIABackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]

        # CPU baseline benchmark - use deepcopy to avoid device conflicts
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model(**cpu_inputs)

        # NVIDIA optimized benchmark
        prepared_model = backend.prepare_model(model, optimization_level=OptimizationLevel.O2)
        cuda_inputs = {k: v.cuda() for k, v in inputs.items()}

        def run_nvidia():
            with torch.no_grad():
                return prepared_model(**cuda_inputs)

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        nvidia_result = benchmark_function(run_nvidia, warmup_runs=5, benchmark_runs=10, sync_cuda=True)

        speedup = cpu_result.mean_time_ms / nvidia_result.mean_time_ms

        print(f"\nBERT NVIDIA Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  NVIDIA: {nvidia_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # NVIDIA should be faster than CPU
        assert speedup > 1.0, f"Expected NVIDIA speedup over CPU, got {speedup:.2f}x"

    @requires_amd
    def test_bert_amd_backend(self, bert_baseline):
        """Test BERT on AMD backend matches baseline."""
        from kernel_pytorch.backends.amd import AMDBackend

        backend = AMDBackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]
        baseline_output = bert_baseline["baseline_output"]

        # Prepare model for AMD
        prepared_model = backend.prepare_model(model)

        # Move inputs to ROCm device
        device = backend.device
        rocm_inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            amd_output = prepared_model(**rocm_inputs).last_hidden_state

        # Move back to CPU for comparison
        amd_output_cpu = amd_output.cpu()

        # Assert outputs match
        assert_output_close(
            baseline_output,
            amd_output_cpu,
            atol=0.05,  # Realistic for cross-device FP32
            rtol=0.01,
            message="BERT AMD output vs CPU baseline"
        )

        print(f"\nBERT AMD Backend Test:")
        print(f"  Device: {backend.device}")
        print(f"  Output shape: {amd_output.shape}")
        print(f"  Output matches baseline: PASS")

    @requires_amd
    def test_bert_amd_speedup(self, bert_baseline):
        """Test BERT speedup on AMD backend."""
        from kernel_pytorch.backends.amd import AMDBackend
        from kernel_pytorch.backends.base_backend import OptimizationLevel

        backend = AMDBackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]

        # CPU baseline
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model(**cpu_inputs)

        # AMD optimized
        prepared_model = backend.prepare_model(model, optimization_level=OptimizationLevel.O2)
        device = backend.device
        rocm_inputs = {k: v.to(device) for k, v in inputs.items()}

        def run_amd():
            with torch.no_grad():
                return prepared_model(**rocm_inputs)

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        amd_result = benchmark_function(run_amd, warmup_runs=5, benchmark_runs=10, sync_cuda=True)

        speedup = cpu_result.mean_time_ms / amd_result.mean_time_ms

        print(f"\nBERT AMD Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  AMD: {amd_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert speedup > 1.0, f"Expected AMD speedup over CPU, got {speedup:.2f}x"

    @requires_tpu
    def test_bert_tpu_backend(self, bert_baseline):
        """Test BERT on TPU backend matches baseline."""
        from kernel_pytorch.backends.tpu import TPUBackend
        import torch_xla.core.xla_model as xm

        backend = TPUBackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]
        baseline_output = bert_baseline["baseline_output"]

        # Prepare model for TPU
        prepared_model = backend.prepare_model(model)

        # Move inputs to TPU
        device = xm.xla_device()
        tpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            tpu_output = prepared_model(**tpu_inputs).last_hidden_state

        # Sync and move back to CPU
        xm.mark_step()
        tpu_output_cpu = tpu_output.cpu()

        # Assert outputs match (TPU may have slightly different precision)
        assert_output_close(
            baseline_output,
            tpu_output_cpu,
            atol=1e-2,  # Slightly looser tolerance for TPU
            rtol=1e-2,
            message="BERT TPU output vs CPU baseline"
        )

        print(f"\nBERT TPU Backend Test:")
        print(f"  Device: {device}")
        print(f"  Output shape: {tpu_output.shape}")
        print(f"  Output matches baseline: PASS")

    @requires_tpu
    def test_bert_tpu_speedup(self, bert_baseline):
        """Test BERT speedup on TPU backend."""
        from kernel_pytorch.backends.tpu import TPUBackend
        import torch_xla.core.xla_model as xm

        backend = TPUBackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]

        # CPU baseline
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model(**cpu_inputs)

        # TPU optimized
        prepared_model = backend.prepare_model(model)
        device = xm.xla_device()
        tpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        def run_tpu():
            with torch.no_grad():
                output = prepared_model(**tpu_inputs)
            xm.mark_step()
            return output

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        tpu_result = benchmark_function(run_tpu, warmup_runs=5, benchmark_runs=10, sync_cuda=False)

        speedup = cpu_result.mean_time_ms / tpu_result.mean_time_ms

        print(f"\nBERT TPU Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  TPU: {tpu_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert speedup > 1.0, f"Expected TPU speedup over CPU, got {speedup:.2f}x"

    @requires_intel
    def test_bert_intel_backend(self, bert_baseline):
        """Test BERT on Intel backend matches baseline."""
        from kernel_pytorch.backends.intel import IntelBackend

        backend = IntelBackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]
        baseline_output = bert_baseline["baseline_output"]

        # Prepare model for Intel XPU
        prepared_model = backend.prepare_model(model)

        # Move inputs to XPU
        device = backend.device
        xpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            intel_output = prepared_model(**xpu_inputs).last_hidden_state

        # Move back to CPU for comparison
        intel_output_cpu = intel_output.cpu()

        # Assert outputs match
        assert_output_close(
            baseline_output,
            intel_output_cpu,
            atol=0.05,  # Realistic for cross-device FP32
            rtol=0.01,
            message="BERT Intel output vs CPU baseline"
        )

        print(f"\nBERT Intel Backend Test:")
        print(f"  Device: {backend.device}")
        print(f"  Output shape: {intel_output.shape}")
        print(f"  Output matches baseline: PASS")

    @requires_intel
    def test_bert_intel_speedup(self, bert_baseline):
        """Test BERT speedup on Intel backend."""
        from kernel_pytorch.backends.intel import IntelBackend
        from kernel_pytorch.backends.base_backend import OptimizationLevel

        backend = IntelBackend()
        model = bert_baseline["model"]
        inputs = bert_baseline["inputs"]

        # CPU baseline
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model(**cpu_inputs)

        # Intel XPU optimized
        prepared_model = backend.prepare_model(model, optimization_level=OptimizationLevel.O2)
        device = backend.device
        xpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        def run_intel():
            with torch.no_grad():
                return prepared_model(**xpu_inputs)

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        intel_result = benchmark_function(run_intel, warmup_runs=5, benchmark_runs=10, sync_cuda=False)

        speedup = cpu_result.mean_time_ms / intel_result.mean_time_ms

        print(f"\nBERT Intel Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  Intel XPU: {intel_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert speedup > 1.0, f"Expected Intel speedup over CPU, got {speedup:.2f}x"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
class TestCrossBackendBERTConsistency:
    """Test that BERT produces consistent predictions across backends."""

    @pytest.fixture
    def classification_setup(self):
        """Set up BERT for classification task."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        model.eval()

        texts = [
            "This movie was absolutely fantastic!",
            "I really hated this product.",
            "The weather is nice today.",
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )

        # Get CPU baseline predictions
        with torch.no_grad():
            baseline_logits = model(**inputs).logits
        baseline_predictions = baseline_logits.argmax(dim=-1)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "inputs": inputs,
            "baseline_predictions": baseline_predictions,
        }

    @requires_nvidia
    def test_classification_consistency_nvidia(self, classification_setup):
        """Test classification predictions match on NVIDIA."""
        from kernel_pytorch.backends.nvidia import NVIDIABackend

        backend = NVIDIABackend()
        model = classification_setup["model"]
        inputs = classification_setup["inputs"]
        baseline_predictions = classification_setup["baseline_predictions"]

        # Run on NVIDIA
        prepared_model = backend.prepare_model(model)
        cuda_inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            nvidia_logits = prepared_model(**cuda_inputs).logits
        nvidia_predictions = nvidia_logits.argmax(dim=-1).cpu()

        # Predictions should match exactly
        assert torch.equal(baseline_predictions, nvidia_predictions), \
            f"NVIDIA predictions {nvidia_predictions.tolist()} != baseline {baseline_predictions.tolist()}"

        print(f"\nBERT Classification Consistency (NVIDIA):")
        print(f"  Baseline predictions: {baseline_predictions.tolist()}")
        print(f"  NVIDIA predictions: {nvidia_predictions.tolist()}")
        print(f"  Match: PASS")

    @requires_amd
    def test_classification_consistency_amd(self, classification_setup):
        """Test classification predictions match on AMD."""
        from kernel_pytorch.backends.amd import AMDBackend

        backend = AMDBackend()
        model = classification_setup["model"]
        inputs = classification_setup["inputs"]
        baseline_predictions = classification_setup["baseline_predictions"]

        # Run on AMD
        prepared_model = backend.prepare_model(model)
        device = backend.device
        amd_inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            amd_logits = prepared_model(**amd_inputs).logits
        amd_predictions = amd_logits.argmax(dim=-1).cpu()

        assert torch.equal(baseline_predictions, amd_predictions), \
            f"AMD predictions {amd_predictions.tolist()} != baseline {baseline_predictions.tolist()}"

        print(f"\nBERT Classification Consistency (AMD):")
        print(f"  Baseline predictions: {baseline_predictions.tolist()}")
        print(f"  AMD predictions: {amd_predictions.tolist()}")
        print(f"  Match: PASS")

    @requires_tpu
    def test_classification_consistency_tpu(self, classification_setup):
        """Test classification predictions match on TPU."""
        from kernel_pytorch.backends.tpu import TPUBackend
        import torch_xla.core.xla_model as xm

        backend = TPUBackend()
        model = classification_setup["model"]
        inputs = classification_setup["inputs"]
        baseline_predictions = classification_setup["baseline_predictions"]

        # Run on TPU
        prepared_model = backend.prepare_model(model)
        device = xm.xla_device()
        tpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            tpu_logits = prepared_model(**tpu_inputs).logits
        xm.mark_step()
        tpu_predictions = tpu_logits.argmax(dim=-1).cpu()

        assert torch.equal(baseline_predictions, tpu_predictions), \
            f"TPU predictions {tpu_predictions.tolist()} != baseline {baseline_predictions.tolist()}"

        print(f"\nBERT Classification Consistency (TPU):")
        print(f"  Baseline predictions: {baseline_predictions.tolist()}")
        print(f"  TPU predictions: {tpu_predictions.tolist()}")
        print(f"  Match: PASS")

    @requires_intel
    def test_classification_consistency_intel(self, classification_setup):
        """Test classification predictions match on Intel."""
        from kernel_pytorch.backends.intel import IntelBackend

        backend = IntelBackend()
        model = classification_setup["model"]
        inputs = classification_setup["inputs"]
        baseline_predictions = classification_setup["baseline_predictions"]

        # Run on Intel XPU
        prepared_model = backend.prepare_model(model)
        device = backend.device
        xpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            intel_logits = prepared_model(**xpu_inputs).logits
        intel_predictions = intel_logits.argmax(dim=-1).cpu()

        assert torch.equal(baseline_predictions, intel_predictions), \
            f"Intel predictions {intel_predictions.tolist()} != baseline {baseline_predictions.tolist()}"

        print(f"\nBERT Classification Consistency (Intel):")
        print(f"  Baseline predictions: {baseline_predictions.tolist()}")
        print(f"  Intel predictions: {intel_predictions.tolist()}")
        print(f"  Match: PASS")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
class TestBackendAutoSelection:
    """Test automatic backend selection with BERT."""

    def test_auto_backend_selection(self):
        """Test that BackendFactory auto-selects best available backend."""
        from kernel_pytorch.backends import BackendFactory, list_available_backends, detect_best_backend

        available = list_available_backends()
        best = detect_best_backend()

        print(f"\nBackend Auto-Selection:")
        print(f"  Available backends: {available}")
        print(f"  Best backend: {best}")

        # Should always have at least CPU
        assert len(available) > 0
        assert "cpu" in [b.lower() for b in available]

    def test_bert_on_best_backend(self):
        """Test BERT runs on auto-selected best backend."""
        from transformers import AutoModel, AutoTokenizer
        from kernel_pytorch.backends import BackendFactory, detect_best_backend

        # Load BERT
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Get best backend
        best_backend_type = detect_best_backend()
        backend = BackendFactory.create(best_backend_type)

        print(f"\nBERT on Best Backend:")
        print(f"  Selected backend: {best_backend_type}")
        print(f"  Device: {backend.device}")

        # Prepare and run
        prepared_model = backend.prepare_model(model)

        texts = ["Testing auto backend selection"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)

        # Move inputs to backend device
        device = backend.device
        device_inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = prepared_model(**device_inputs).last_hidden_state

        print(f"  Output shape: {output.shape}")
        assert output.shape[0] == 1  # batch size
        assert output.shape[2] == 768  # hidden size
