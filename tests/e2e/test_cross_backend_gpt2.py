"""
Cross-Backend End-to-End Tests for Real GPT-2 Model

Tests that GPT-2 text generation works consistently across all 4 hardware
backends (NVIDIA, AMD, TPU, Intel) with:
- Consistent text generation output
- Matching logits across backends
- Measurable speedups on each platform

Markers:
    - @pytest.mark.e2e: End-to-end test
    - @pytest.mark.slow: Long-running test
    - @pytest.mark.real_model: Uses real model weights
    - @pytest.mark.requires_transformers: Requires HuggingFace transformers

Success Criteria:
    - GPT-2 generates identical text on each backend (greedy decoding)
    - Logits match within tolerance
    - Speedup achieved on each backend
"""

import copy

import pytest
import torch

from .conftest import (
    assert_output_close,
    benchmark_function,
    requires_transformers,
)
from .test_cross_backend_bert import (
    requires_amd,
    requires_intel,
    requires_nvidia,
    requires_tpu,
)

# =============================================================================
# Cross-Backend GPT-2 Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
class TestCrossBackendGPT2:
    """Test GPT-2 optimization across all backends."""

    @pytest.fixture
    def gpt2_baseline(self):
        """Get baseline GPT-2 outputs on CPU."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()

        # Test prompts
        prompts = [
            "The future of artificial intelligence is",
            "In the year 2050, technology will",
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        )

        # Get CPU baseline logits
        with torch.no_grad():
            baseline_logits = model(**inputs).logits

        # Get CPU baseline generation (greedy)
        with torch.no_grad():
            baseline_generated = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        baseline_text = tokenizer.batch_decode(baseline_generated, skip_special_tokens=True)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "inputs": inputs,
            "baseline_logits": baseline_logits,
            "baseline_generated": baseline_generated,
            "baseline_text": baseline_text,
            "prompts": prompts,
        }

    @requires_nvidia
    def test_gpt2_nvidia_logits_match(self, gpt2_baseline):
        """Test GPT-2 logits match baseline on NVIDIA."""
        from kernel_pytorch.backends.nvidia import NVIDIABackend

        backend = NVIDIABackend()
        model = gpt2_baseline["model"]
        inputs = gpt2_baseline["inputs"]
        baseline_logits = gpt2_baseline["baseline_logits"]

        # Prepare model for NVIDIA
        prepared_model = backend.prepare_model(model)

        # Move inputs to CUDA
        cuda_inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            nvidia_logits = prepared_model(**cuda_inputs).logits

        # Move back to CPU for comparison
        nvidia_logits_cpu = nvidia_logits.cpu()

        # Assert logits match
        assert_output_close(
            baseline_logits,
            nvidia_logits_cpu,
            atol=0.1,   # Realistic for cross-device FP32
            rtol=0.01,
            message="GPT-2 NVIDIA logits vs CPU baseline"
        )

        print("\nGPT-2 NVIDIA Logits Test:")
        print(f"  Device: {backend.device}")
        print(f"  Logits shape: {nvidia_logits.shape}")
        print("  Logits match baseline: PASS")

    @requires_nvidia
    def test_gpt2_nvidia_generation_match(self, gpt2_baseline):
        """Test GPT-2 generation matches baseline on NVIDIA."""
        from kernel_pytorch.backends.nvidia import NVIDIABackend

        backend = NVIDIABackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]
        baseline_text = gpt2_baseline["baseline_text"]

        # Prepare model for NVIDIA
        prepared_model = backend.prepare_model(model)

        # Move inputs to CUDA
        cuda_inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            nvidia_generated = prepared_model.generate(
                **cuda_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        nvidia_text = tokenizer.batch_decode(nvidia_generated, skip_special_tokens=True)

        # Compare generated text
        print("\nGPT-2 NVIDIA Generation Test:")
        for i, (baseline, nvidia) in enumerate(zip(baseline_text, nvidia_text)):
            print(f"  Sample {i+1}:")
            print(f"    Baseline: {baseline}")
            print(f"    NVIDIA:   {nvidia}")
            assert baseline == nvidia, f"Generation mismatch at sample {i+1}"

        print("  Generation match: PASS")

    @requires_nvidia
    def test_gpt2_nvidia_speedup(self, gpt2_baseline):
        """Test GPT-2 speedup on NVIDIA backend."""
        from kernel_pytorch.backends.nvidia import NVIDIABackend

        backend = NVIDIABackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]

        # CPU baseline
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model.generate(
                    **cpu_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # NVIDIA optimized
        prepared_model = backend.prepare_model(model)
        cuda_inputs = {k: v.cuda() for k, v in inputs.items()}

        def run_nvidia():
            with torch.no_grad():
                return prepared_model.generate(
                    **cuda_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        nvidia_result = benchmark_function(run_nvidia, warmup_runs=5, benchmark_runs=10, sync_cuda=True)

        speedup = cpu_result.mean_time_ms / nvidia_result.mean_time_ms

        print("\nGPT-2 NVIDIA Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  NVIDIA: {nvidia_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert speedup > 1.0, f"Expected NVIDIA speedup over CPU, got {speedup:.2f}x"

    @requires_amd
    def test_gpt2_amd_generation_match(self, gpt2_baseline):
        """Test GPT-2 generation matches baseline on AMD."""
        from kernel_pytorch.backends.amd import AMDBackend

        backend = AMDBackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]
        baseline_text = gpt2_baseline["baseline_text"]

        # Prepare model for AMD
        prepared_model = backend.prepare_model(model)

        # Move inputs to ROCm device
        device = backend.device
        amd_inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            amd_generated = prepared_model.generate(
                **amd_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        amd_text = tokenizer.batch_decode(amd_generated, skip_special_tokens=True)

        # Compare
        print("\nGPT-2 AMD Generation Test:")
        for i, (baseline, amd) in enumerate(zip(baseline_text, amd_text)):
            print(f"  Sample {i+1}:")
            print(f"    Baseline: {baseline}")
            print(f"    AMD:      {amd}")
            assert baseline == amd, f"Generation mismatch at sample {i+1}"

        print("  Generation match: PASS")

    @requires_amd
    def test_gpt2_amd_speedup(self, gpt2_baseline):
        """Test GPT-2 speedup on AMD backend."""
        from kernel_pytorch.backends.amd import AMDBackend

        backend = AMDBackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]

        # CPU baseline
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model.generate(
                    **cpu_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # AMD optimized
        prepared_model = backend.prepare_model(model)
        device = backend.device
        amd_inputs = {k: v.to(device) for k, v in inputs.items()}

        def run_amd():
            with torch.no_grad():
                return prepared_model.generate(
                    **amd_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        amd_result = benchmark_function(run_amd, warmup_runs=5, benchmark_runs=10, sync_cuda=True)

        speedup = cpu_result.mean_time_ms / amd_result.mean_time_ms

        print("\nGPT-2 AMD Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  AMD: {amd_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert speedup > 1.0, f"Expected AMD speedup over CPU, got {speedup:.2f}x"

    @requires_tpu
    def test_gpt2_tpu_generation_match(self, gpt2_baseline):
        """Test GPT-2 generation matches baseline on TPU."""
        import torch_xla.core.xla_model as xm

        from kernel_pytorch.backends.tpu import TPUBackend

        backend = TPUBackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]
        baseline_text = gpt2_baseline["baseline_text"]

        # Prepare model for TPU
        prepared_model = backend.prepare_model(model)

        # Move inputs to TPU
        xm.xla_device()
        tpu_inputs = backend.prepare_data(inputs)

        # Generate text
        with torch.no_grad():
            tpu_generated = prepared_model.generate(
                **tpu_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        xm.mark_step()
        tpu_text = tokenizer.batch_decode(tpu_generated.cpu(), skip_special_tokens=True)

        # Compare
        print("\nGPT-2 TPU Generation Test:")
        for i, (baseline, tpu) in enumerate(zip(baseline_text, tpu_text)):
            print(f"  Sample {i+1}:")
            print(f"    Baseline: {baseline}")
            print(f"    TPU:      {tpu}")
            assert baseline == tpu, f"Generation mismatch at sample {i+1}"

        print("  Generation match: PASS")

    @requires_tpu
    def test_gpt2_tpu_speedup(self, gpt2_baseline):
        """Test GPT-2 speedup on TPU backend."""
        import torch_xla.core.xla_model as xm

        from kernel_pytorch.backends.tpu import TPUBackend

        backend = TPUBackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]

        # CPU baseline
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model.generate(
                    **cpu_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # TPU optimized
        prepared_model = backend.prepare_model(model)
        xm.xla_device()
        tpu_inputs = backend.prepare_data(inputs)

        def run_tpu():
            with torch.no_grad():
                output = prepared_model.generate(
                    **tpu_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            xm.mark_step()
            return output

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        tpu_result = benchmark_function(run_tpu, warmup_runs=5, benchmark_runs=10, sync_cuda=False)

        speedup = cpu_result.mean_time_ms / tpu_result.mean_time_ms

        print("\nGPT-2 TPU Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  TPU: {tpu_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert speedup > 1.0, f"Expected TPU speedup over CPU, got {speedup:.2f}x"

    @requires_intel
    def test_gpt2_intel_generation_match(self, gpt2_baseline):
        """Test GPT-2 generation matches baseline on Intel."""
        from kernel_pytorch.backends.intel import IntelBackend

        backend = IntelBackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]
        baseline_text = gpt2_baseline["baseline_text"]

        # Prepare model for Intel XPU
        prepared_model = backend.prepare_model(model)

        # Move inputs to XPU
        device = backend.device
        xpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            intel_generated = prepared_model.generate(
                **xpu_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        intel_text = tokenizer.batch_decode(intel_generated.cpu(), skip_special_tokens=True)

        # Compare
        print("\nGPT-2 Intel Generation Test:")
        for i, (baseline, intel) in enumerate(zip(baseline_text, intel_text)):
            print(f"  Sample {i+1}:")
            print(f"    Baseline: {baseline}")
            print(f"    Intel:    {intel}")
            assert baseline == intel, f"Generation mismatch at sample {i+1}"

        print("  Generation match: PASS")

    @requires_intel
    def test_gpt2_intel_speedup(self, gpt2_baseline):
        """Test GPT-2 speedup on Intel backend."""
        from kernel_pytorch.backends.intel import IntelBackend

        backend = IntelBackend()
        model = gpt2_baseline["model"]
        tokenizer = gpt2_baseline["tokenizer"]
        inputs = gpt2_baseline["inputs"]

        # CPU baseline
        cpu_model = copy.deepcopy(model).cpu()
        cpu_model.eval()
        cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

        def run_cpu():
            with torch.no_grad():
                return cpu_model.generate(
                    **cpu_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # Intel XPU optimized
        prepared_model = backend.prepare_model(model)
        device = backend.device
        xpu_inputs = {k: v.to(device) for k, v in inputs.items()}

        def run_intel():
            with torch.no_grad():
                return prepared_model.generate(
                    **xpu_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # Benchmark
        cpu_result = benchmark_function(run_cpu, warmup_runs=2, benchmark_runs=5, sync_cuda=False)
        intel_result = benchmark_function(run_intel, warmup_runs=5, benchmark_runs=10, sync_cuda=False)

        speedup = cpu_result.mean_time_ms / intel_result.mean_time_ms

        print("\nGPT-2 Intel Speedup:")
        print(f"  CPU: {cpu_result.mean_time_ms:.2f}ms")
        print(f"  Intel XPU: {intel_result.mean_time_ms:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert speedup > 1.0, f"Expected Intel speedup over CPU, got {speedup:.2f}x"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.real_model
@requires_transformers
class TestCrossBackendGPT2Perplexity:
    """Test GPT-2 perplexity consistency across backends."""

    @pytest.fixture
    def perplexity_setup(self):
        """Set up GPT-2 for perplexity calculation."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()

        # Test text for perplexity
        test_text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(test_text, return_tensors="pt")

        # Calculate baseline perplexity on CPU
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            baseline_loss = outputs.loss.item()
            baseline_perplexity = torch.exp(torch.tensor(baseline_loss)).item()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "inputs": inputs,
            "baseline_loss": baseline_loss,
            "baseline_perplexity": baseline_perplexity,
            "test_text": test_text,
        }

    @requires_nvidia
    def test_gpt2_perplexity_nvidia(self, perplexity_setup):
        """Test GPT-2 perplexity matches on NVIDIA."""
        from kernel_pytorch.backends.nvidia import NVIDIABackend

        backend = NVIDIABackend()
        model = perplexity_setup["model"]
        inputs = perplexity_setup["inputs"]
        baseline_perplexity = perplexity_setup["baseline_perplexity"]

        # Run on NVIDIA
        prepared_model = backend.prepare_model(model)
        cuda_inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = prepared_model(**cuda_inputs, labels=cuda_inputs["input_ids"])
            nvidia_loss = outputs.loss.item()
            nvidia_perplexity = torch.exp(torch.tensor(nvidia_loss)).item()

        # Perplexity should be very close
        perplexity_diff = abs(nvidia_perplexity - baseline_perplexity) / baseline_perplexity

        print("\nGPT-2 Perplexity Consistency (NVIDIA):")
        print(f"  Baseline perplexity: {baseline_perplexity:.4f}")
        print(f"  NVIDIA perplexity: {nvidia_perplexity:.4f}")
        print(f"  Difference: {perplexity_diff*100:.2f}%")

        assert perplexity_diff < 0.01, f"Perplexity differs by {perplexity_diff*100:.2f}% (max 1%)"

    @requires_amd
    def test_gpt2_perplexity_amd(self, perplexity_setup):
        """Test GPT-2 perplexity matches on AMD."""
        from kernel_pytorch.backends.amd import AMDBackend

        backend = AMDBackend()
        model = perplexity_setup["model"]
        inputs = perplexity_setup["inputs"]
        baseline_perplexity = perplexity_setup["baseline_perplexity"]

        # Run on AMD
        prepared_model = backend.prepare_model(model)
        device = backend.device
        amd_inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = prepared_model(**amd_inputs, labels=amd_inputs["input_ids"])
            amd_loss = outputs.loss.item()
            amd_perplexity = torch.exp(torch.tensor(amd_loss)).item()

        perplexity_diff = abs(amd_perplexity - baseline_perplexity) / baseline_perplexity

        print("\nGPT-2 Perplexity Consistency (AMD):")
        print(f"  Baseline perplexity: {baseline_perplexity:.4f}")
        print(f"  AMD perplexity: {amd_perplexity:.4f}")
        print(f"  Difference: {perplexity_diff*100:.2f}%")

        assert perplexity_diff < 0.01, f"Perplexity differs by {perplexity_diff*100:.2f}% (max 1%)"
