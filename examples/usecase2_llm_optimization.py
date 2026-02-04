"""
Use Case 2: LLM Optimization for Inference

Demonstrates TorchBridge's LLM optimizer: load a HuggingFace model,
apply optimizations (BetterTransformer, torch.compile, quantization),
and measure the impact on latency and throughput.

Run: PYTHONPATH=src python3 examples/usecase2_llm_optimization.py
"""
# ruff: noqa: E402

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load Baseline (vanilla HuggingFace) ─────────────────────
print("=" * 60)
print("STEP 1: Load GPT-2 Baseline (vanilla HuggingFace)")
print("=" * 60)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
baseline_model.eval()

param_count = sum(p.numel() for p in baseline_model.parameters())
param_mb = sum(p.numel() * p.element_size() for p in baseline_model.parameters()) / 1e6
print(f"  Model       : {model_name} ({param_count:,} params, {param_mb:.1f} MB)")
print(f"  Device      : {next(baseline_model.parameters()).device}")

# ── Step 2: Baseline Inference ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Baseline Inference Benchmark")
print("=" * 60)

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = baseline_model.generate(
            **inputs, max_new_tokens=50, do_sample=False, use_cache=True
        )

# Benchmark
n_runs = 5
baseline_times = []
for _ in range(n_runs):
    start = time.perf_counter()
    with torch.no_grad():
        output = baseline_model.generate(
            **inputs, max_new_tokens=50, do_sample=False, use_cache=True
        )
    baseline_times.append(time.perf_counter() - start)

baseline_text = tokenizer.decode(output[0], skip_special_tokens=True)
baseline_avg_ms = (sum(baseline_times) / n_runs) * 1000
tokens_generated = output.shape[1] - inputs["input_ids"].shape[1]

print(f"  Prompt      : \"{prompt}\"")
print(f"  Generated   : {tokens_generated} tokens")
print(f"  Avg latency : {baseline_avg_ms:.1f} ms ({n_runs} runs)")
print(f"  Throughput  : {tokens_generated / (baseline_avg_ms / 1000):.1f} tokens/s")
print(f"  Output      : \"{baseline_text[:80]}...\"")

# ── Step 3: TorchBridge Optimized ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: TorchBridge LLM Optimizer")
print("=" * 60)

from torchbridge.models.llm import LLMConfig, LLMOptimizer, QuantizationMode

config = LLMConfig(
    model_name=model_name,
    quantization=QuantizationMode.NONE,
    max_sequence_length=1024,
    use_flash_attention=False,      # No FlashAttention on CPU
    use_torch_compile=False,        # Skip compile for fair comparison
    use_bettertransformer=True,     # SDPA optimization
    device="cpu",
    device_map=None,                # Disable accelerate auto device mapping
    dtype=torch.float32,
    use_kv_cache=True,
)

print("  Config:")
print(f"    quantization       : {config.quantization.value}")
print(f"    use_bettertransformer : {config.use_bettertransformer}")
print(f"    use_kv_cache       : {config.use_kv_cache}")
print(f"    max_sequence_length: {config.max_sequence_length}")

optimizer = LLMOptimizer(config)

print("\n  Optimizing...")
opt_model, opt_tokenizer = optimizer.optimize(model_name)
opt_model.eval()

opt_info = optimizer.get_optimization_info()
print("\n  Optimizations applied:")
for k, v in opt_info.items():
    print(f"    {k:20s}: {v}")

# ── Step 4: Optimized Inference ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Optimized Inference Benchmark")
print("=" * 60)

opt_inputs = opt_tokenizer(prompt, return_tensors="pt")

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = opt_model.generate(
            **opt_inputs, max_new_tokens=50, do_sample=False, use_cache=True
        )

# Benchmark
opt_times = []
for _ in range(n_runs):
    start = time.perf_counter()
    with torch.no_grad():
        opt_output = opt_model.generate(
            **opt_inputs, max_new_tokens=50, do_sample=False, use_cache=True
        )
    opt_times.append(time.perf_counter() - start)

opt_text = opt_tokenizer.decode(opt_output[0], skip_special_tokens=True)
opt_avg_ms = (sum(opt_times) / n_runs) * 1000
opt_tokens = opt_output.shape[1] - opt_inputs["input_ids"].shape[1]

print(f"  Avg latency : {opt_avg_ms:.1f} ms ({n_runs} runs)")
print(f"  Throughput  : {opt_tokens / (opt_avg_ms / 1000):.1f} tokens/s")
print(f"  Output      : \"{opt_text[:80]}...\"")

# ── Step 5: torch.compile Optimization ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: torch.compile Optimization")
print("=" * 60)

compile_config = LLMConfig(
    model_name=model_name,
    quantization=QuantizationMode.NONE,
    max_sequence_length=1024,
    use_flash_attention=False,
    use_torch_compile=True,         # Enable torch.compile
    compile_mode="reduce-overhead",
    use_bettertransformer=True,
    device="cpu",
    device_map=None,
    use_kv_cache=True,
)

print(f"  torch.compile : True (mode={compile_config.compile_mode})")
compile_optimizer = LLMOptimizer(compile_config)
compile_model, compile_tokenizer = compile_optimizer.optimize(model_name)
compile_model.eval()

compile_inputs = compile_tokenizer(prompt, return_tensors="pt")

# Warmup (compile triggers on first runs)
print("  Warming up (compilation happens here)...")
with torch.no_grad():
    for _ in range(3):
        _ = compile_model.generate(
            **compile_inputs, max_new_tokens=50, do_sample=False, use_cache=True
        )

# Benchmark
compile_times = []
for _ in range(n_runs):
    start = time.perf_counter()
    with torch.no_grad():
        compile_output = compile_model.generate(
            **compile_inputs, max_new_tokens=50, do_sample=False, use_cache=True
        )
    compile_times.append(time.perf_counter() - start)

compile_text = compile_tokenizer.decode(compile_output[0], skip_special_tokens=True)
compile_avg_ms = (sum(compile_times) / n_runs) * 1000
compile_tokens = compile_output.shape[1] - compile_inputs["input_ids"].shape[1]

print(f"  Avg latency : {compile_avg_ms:.1f} ms ({n_runs} runs)")
print(f"  Throughput  : {compile_tokens / (compile_avg_ms / 1000):.1f} tokens/s")
print(f"  Output      : \"{compile_text[:80]}...\"")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — GPT-2 (124M params) on CPU")
print("=" * 60)

speedup_opt = baseline_avg_ms / opt_avg_ms
speedup_compile = baseline_avg_ms / compile_avg_ms

print(f"\n  {'Config':<25s} {'Latency':>10s} {'Tokens/s':>10s} {'Speedup':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'Baseline (vanilla HF)':<25s} {baseline_avg_ms:>9.1f}ms {tokens_generated / (baseline_avg_ms / 1000):>9.1f} {'1.00x':>10s}")
print(f"  {'TorchBridge optimized':<25s} {opt_avg_ms:>9.1f}ms {opt_tokens / (opt_avg_ms / 1000):>9.1f} {speedup_opt:>9.2f}x")
print(f"  {'TorchBridge + compile':<25s} {compile_avg_ms:>9.1f}ms {compile_tokens / (compile_avg_ms / 1000):>9.1f} {speedup_compile:>9.2f}x")
print()
print(f"  Outputs match: {baseline_text[:50] == opt_text[:50] == compile_text[:50]}")
print()
print("  NOTE: On GPU (CUDA), additional optimizations apply:")
print("    - FlashAttention-2 (Ampere+): ~2x attention speedup")
print("    - INT8/FP8 quantization (H100+): ~50% memory reduction")
print("    - torch.compile with Triton: kernel fusion for ~1.3-1.5x speedup")
