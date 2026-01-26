"""
End-to-End Tests for Quantization Quality

Tests that quantization modes (INT8, INT4, FP8) preserve model quality
within acceptable thresholds on real models.

v0.4.21 - Quantization Quality Validation

Markers:
    - @pytest.mark.e2e: End-to-end test
    - @pytest.mark.slow: Long-running test
    - @pytest.mark.quantization: Quantization-specific test

Quality Targets:
    - INT8: <2% perplexity increase
    - INT4: <5% perplexity increase
    - FP8: <1% perplexity increase
"""

import copy
import pytest
import torch
import torch.nn as nn
import warnings

from .conftest import requires_transformers, requires_cuda


# =============================================================================
# Quality Thresholds
# =============================================================================

QUALITY_THRESHOLDS = {
    "int8": {
        "max_perplexity_increase_pct": 2.0,
        "max_accuracy_drop_pct": 2.0,
        "min_memory_reduction_pct": 30.0,  # At least 30% memory savings
    },
    "int4": {
        "max_perplexity_increase_pct": 5.0,
        "max_accuracy_drop_pct": 5.0,
        "min_memory_reduction_pct": 60.0,
    },
    "fp8": {
        "max_perplexity_increase_pct": 1.0,
        "max_accuracy_drop_pct": 1.0,
        "min_memory_reduction_pct": 40.0,
    },
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gpt2_for_quantization():
    """Load GPT-2 model for quantization testing."""
    pytest.importorskip("transformers")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@pytest.fixture
def bert_for_quantization():
    """Load BERT model for quantization testing."""
    pytest.importorskip("transformers")
    from transformers import BertForSequenceClassification, BertTokenizer

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer


@pytest.fixture
def sample_texts():
    """Sample texts for evaluation."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the way we build software.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformer architectures have revolutionized NLP tasks.",
    ]


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def calculate_perplexity_simple(
    model: nn.Module,
    tokenizer,
    texts: list,
    device: torch.device
) -> float:
    """Calculate perplexity on sample texts."""
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        with torch.no_grad():
            try:
                outputs = model(**inputs)
                loss = outputs.loss
                num_tokens = (inputs["attention_mask"].sum()).item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            except Exception as e:
                warnings.warn(f"Error computing loss: {e}")
                continue

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def quantize_int8(model: nn.Module) -> nn.Module:
    """Apply INT8 dynamic quantization."""
    model_cpu = copy.deepcopy(model).cpu()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized


# =============================================================================
# INT8 Quantization Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.quantization
@requires_transformers
class TestINT8Quantization:
    """Test INT8 quantization quality."""

    def test_gpt2_int8_perplexity(self, gpt2_for_quantization, sample_texts):
        """Test GPT-2 perplexity degradation with INT8."""
        model, tokenizer = gpt2_for_quantization
        device = torch.device("cpu")  # INT8 typically runs on CPU

        # Baseline perplexity
        baseline_model = copy.deepcopy(model)
        baseline_ppl = calculate_perplexity_simple(
            baseline_model, tokenizer, sample_texts, device
        )

        # INT8 quantized perplexity
        quantized_model = quantize_int8(model)
        quantized_ppl = calculate_perplexity_simple(
            quantized_model, tokenizer, sample_texts, device
        )

        # Calculate degradation
        ppl_increase_pct = (quantized_ppl - baseline_ppl) / baseline_ppl * 100

        print(f"\nGPT-2 INT8 Quantization Results:")
        print(f"  Baseline perplexity: {baseline_ppl:.2f}")
        print(f"  INT8 perplexity: {quantized_ppl:.2f}")
        print(f"  Perplexity increase: {ppl_increase_pct:.2f}%")
        print(f"  Target: <{QUALITY_THRESHOLDS['int8']['max_perplexity_increase_pct']}%")

        # Assert within threshold
        assert ppl_increase_pct < QUALITY_THRESHOLDS["int8"]["max_perplexity_increase_pct"], \
            f"INT8 perplexity increase {ppl_increase_pct:.2f}% exceeds threshold"

    def test_gpt2_int8_memory_reduction(self, gpt2_for_quantization):
        """Test GPT-2 memory reduction with INT8."""
        model, _ = gpt2_for_quantization

        baseline_size = get_model_size_mb(model)
        quantized_model = quantize_int8(model)
        quantized_size = get_model_size_mb(quantized_model)

        memory_reduction_pct = (baseline_size - quantized_size) / baseline_size * 100

        print(f"\nGPT-2 INT8 Memory Results:")
        print(f"  Baseline size: {baseline_size:.1f} MB")
        print(f"  INT8 size: {quantized_size:.1f} MB")
        print(f"  Memory reduction: {memory_reduction_pct:.1f}%")
        print(f"  Target: >{QUALITY_THRESHOLDS['int8']['min_memory_reduction_pct']}%")

        # INT8 should reduce memory by at least 30%
        assert memory_reduction_pct >= QUALITY_THRESHOLDS["int8"]["min_memory_reduction_pct"], \
            f"INT8 memory reduction {memory_reduction_pct:.1f}% below threshold"

    def test_bert_int8_output_similarity(self, bert_for_quantization, sample_texts):
        """Test BERT output similarity with INT8."""
        model, tokenizer = bert_for_quantization
        device = torch.device("cpu")

        # Get baseline outputs
        baseline_model = copy.deepcopy(model).to(device)
        baseline_model.eval()

        inputs = tokenizer(
            sample_texts[0],
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            baseline_output = baseline_model(**inputs).logits

        # Get INT8 outputs
        quantized_model = quantize_int8(model)
        quantized_model.eval()

        with torch.no_grad():
            quantized_output = quantized_model(**inputs).logits

        # Check output similarity (cosine similarity)
        baseline_flat = baseline_output.flatten().float()
        quantized_flat = quantized_output.flatten().float()

        cosine_sim = torch.nn.functional.cosine_similarity(
            baseline_flat.unsqueeze(0),
            quantized_flat.unsqueeze(0)
        ).item()

        print(f"\nBERT INT8 Output Similarity:")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Target: >0.95")

        # Outputs should be highly similar
        assert cosine_sim > 0.95, f"INT8 output similarity {cosine_sim:.4f} too low"


# =============================================================================
# FP8 Quantization Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.quantization
@requires_transformers
class TestFP8Quantization:
    """Test FP8 quantization quality."""

    def test_fp8_availability(self):
        """Test FP8 support detection."""
        from kernel_pytorch.precision.fp8_native import (
            FP8_NATIVE_AVAILABLE,
            FP8_DTYPES_AVAILABLE,
            is_fp8_available,
            get_fp8_info
        )

        print(f"\nFP8 Availability Check:")
        print(f"  Native FP8 available: {FP8_NATIVE_AVAILABLE}")
        print(f"  FP8 dtypes available: {FP8_DTYPES_AVAILABLE}")
        print(f"  is_fp8_available(): {is_fp8_available()}")

        info = get_fp8_info()
        print(f"  PyTorch version: {info.get('pytorch_version', 'unknown')}")
        print(f"  Hardware support: {info.get('hardware_support', 'unknown')}")

        # Test should pass regardless of availability (informational)
        assert True

    def test_fp8_quantize_dequantize_accuracy(self):
        """Test FP8 quantization preserves values."""
        from kernel_pytorch.precision.fp8_native import (
            quantize_to_fp8,
            dequantize_from_fp8,
            FP8_NATIVE_AVAILABLE
        )

        if not FP8_NATIVE_AVAILABLE:
            pytest.skip("FP8 not available on this system")

        # Create test tensor
        original = torch.randn(100, 100)

        # Quantize and dequantize
        quantized = quantize_to_fp8(original)
        recovered = dequantize_from_fp8(quantized)

        # Check reconstruction error
        mse = torch.mean((original - recovered) ** 2).item()
        max_error = torch.max(torch.abs(original - recovered)).item()

        print(f"\nFP8 Quantization Accuracy:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Max error: {max_error:.6f}")

        # FP8 should have reasonable reconstruction
        assert mse < 0.01, f"FP8 MSE {mse:.6f} too high"
        assert max_error < 0.5, f"FP8 max error {max_error:.6f} too high"

    @requires_cuda
    def test_fp8_linear_layer(self):
        """Test FP8 linear layer functionality."""
        from kernel_pytorch.precision.fp8_native import NativeFP8Linear, FP8_NATIVE_AVAILABLE

        if not FP8_NATIVE_AVAILABLE:
            pytest.skip("FP8 not available on this system")

        device = torch.device("cuda")

        # Create FP8 linear layer
        fp8_linear = NativeFP8Linear(256, 128).to(device)

        # Test forward pass
        x = torch.randn(4, 256, device=device)
        output = fp8_linear(x)

        print(f"\nFP8 Linear Layer Test:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")

        assert output.shape == (4, 128)
        assert not torch.isnan(output).any()

    @requires_cuda
    def test_gpt2_fp8_conversion(self, gpt2_for_quantization, sample_texts):
        """Test GPT-2 FP8 conversion and quality."""
        from kernel_pytorch.precision.fp8_native import (
            convert_model_to_native_fp8,
            FP8_NATIVE_AVAILABLE
        )

        if not FP8_NATIVE_AVAILABLE:
            pytest.skip("FP8 not available on this system")

        model, tokenizer = gpt2_for_quantization
        device = torch.device("cuda")

        # Baseline
        baseline_model = copy.deepcopy(model).to(device)
        baseline_ppl = calculate_perplexity_simple(
            baseline_model, tokenizer, sample_texts, device
        )

        # FP8 conversion
        fp8_model = convert_model_to_native_fp8(model, device=device)
        fp8_ppl = calculate_perplexity_simple(
            fp8_model, tokenizer, sample_texts, device
        )

        ppl_increase_pct = (fp8_ppl - baseline_ppl) / baseline_ppl * 100

        print(f"\nGPT-2 FP8 Quantization Results:")
        print(f"  Baseline perplexity: {baseline_ppl:.2f}")
        print(f"  FP8 perplexity: {fp8_ppl:.2f}")
        print(f"  Perplexity increase: {ppl_increase_pct:.2f}%")
        print(f"  Target: <{QUALITY_THRESHOLDS['fp8']['max_perplexity_increase_pct']}%")

        # FP8 should have minimal quality loss
        assert ppl_increase_pct < QUALITY_THRESHOLDS["fp8"]["max_perplexity_increase_pct"], \
            f"FP8 perplexity increase {ppl_increase_pct:.2f}% exceeds threshold"


# =============================================================================
# Quantization Integration Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.quantization
@requires_transformers
class TestQuantizationIntegration:
    """Test quantization integration with model optimizers."""

    def test_text_model_int8_optimization(self, gpt2_for_quantization, sample_texts):
        """Test INT8 quantization via TextModelOptimizer."""
        from kernel_pytorch.models.text import TextModelOptimizer, TextModelConfig, OptimizationMode

        model, tokenizer = gpt2_for_quantization
        device = torch.device("cpu")

        # Baseline
        baseline_model = copy.deepcopy(model).to(device)
        baseline_model.eval()

        inputs = tokenizer(sample_texts[0], return_tensors="pt").to(device)
        with torch.no_grad():
            baseline_output = baseline_model(**inputs).logits

        # Optimized with quantization
        config = TextModelConfig(
            optimization_mode=OptimizationMode.MEMORY,
            use_torch_compile=False,
            device="cpu",
        )
        optimizer = TextModelOptimizer(config)
        optimized_model = optimizer.optimize(model, task="causal-lm")

        # Apply INT8 quantization
        quantized_model = quantize_int8(optimized_model)
        quantized_model.eval()

        with torch.no_grad():
            quantized_output = quantized_model(**inputs).logits

        # Verify outputs are similar
        max_diff = (baseline_output - quantized_output).abs().max().item()

        print(f"\nTextModelOptimizer + INT8 Test:")
        print(f"  Max output difference: {max_diff:.4f}")

        # Allow reasonable difference for quantized model
        assert max_diff < 1.0, f"Output difference {max_diff:.4f} too large"

    def test_quantization_preserves_generation(self, gpt2_for_quantization):
        """Test that quantized model can still generate text."""
        model, tokenizer = gpt2_for_quantization
        device = torch.device("cpu")

        prompt = "The future of AI is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Quantize
        quantized_model = quantize_int8(model)

        # Generate
        with torch.no_grad():
            outputs = quantized_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nQuantized Generation Test:")
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {generated}")

        # Verify generation worked
        assert len(generated) > len(prompt), "Quantized model failed to generate"
        assert generated.startswith(prompt[:10]), "Generation doesn't match prompt"


# =============================================================================
# Memory and Performance Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.quantization
@requires_transformers
class TestQuantizationMemory:
    """Test quantization memory benefits."""

    def test_int8_memory_footprint(self, gpt2_for_quantization):
        """Test INT8 reduces memory footprint."""
        model, _ = gpt2_for_quantization

        # Baseline
        baseline_params = sum(p.numel() for p in model.parameters())
        baseline_size = get_model_size_mb(model)

        # Quantized
        quantized = quantize_int8(model)
        quantized_size = get_model_size_mb(quantized)

        reduction = (baseline_size - quantized_size) / baseline_size * 100

        print(f"\nINT8 Memory Footprint:")
        print(f"  Parameters: {baseline_params:,}")
        print(f"  FP32 size: {baseline_size:.1f} MB")
        print(f"  INT8 size: {quantized_size:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")

        # Expect at least 2x reduction (INT8 is 1/4 the size of FP32)
        # But dynamic quantization only quantizes weights, not all params
        assert reduction > 20, f"Memory reduction {reduction:.1f}% below expected"

    def test_quantization_inference_works(self, bert_for_quantization, sample_texts):
        """Test quantized model performs inference correctly."""
        model, tokenizer = bert_for_quantization

        # Quantize
        quantized = quantize_int8(model)

        # Run inference
        inputs = tokenizer(
            sample_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )

        with torch.no_grad():
            outputs = quantized(**inputs)

        logits = outputs.logits
        predictions = logits.argmax(dim=-1)

        print(f"\nQuantized Inference Test:")
        print(f"  Batch size: {len(sample_texts)}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Predictions: {predictions.tolist()}")

        assert logits.shape == (len(sample_texts), 2)
        assert not torch.isnan(logits).any()
