# Why Your Model Outputs Differ Between NVIDIA and AMD: SDPA Divergence in Cross-Backend Validation

When validating transformer models across GPU vendors, you may encounter output differences that look alarming but are actually expected. This post explains why Scaled Dot-Product Attention (SDPA) produces divergent results between NVIDIA CUDA and AMD ROCm backends, how the divergence compounds through transformer layers, and what you can do about it.

## The Discovery

When running BERT and GPT-2 through TorchBridge's cross-backend validation, we observed that model outputs on AMD ROCm GPUs diverged significantly from CPU and NVIDIA baselines. Embedding layers produced perfect matches, but after passing through 12 transformer layers, cosine similarity dropped to approximately 0.82.

This was not a bug in TorchBridge or ROCm -- it was a fundamental numerical difference in how attention is computed across hardware backends.

The divergence was first surfaced by TorchBridge's `UnifiedValidator` during end-to-end testing on an AMD MI300X system. The validator flagged cross-backend cosine similarity below the default threshold, which prompted a deeper investigation into which layers were responsible.

## Root Cause: Online Softmax in Flash Attention

PyTorch's `scaled_dot_product_attention` (SDPA) dispatches to different backend implementations depending on hardware:

- **CPU**: Math-only kernel (standard matrix multiply followed by a full softmax pass)
- **NVIDIA CUDA**: Flash Attention v2 or memory-efficient attention (via cuDNN or the built-in SDPA kernels)
- **AMD ROCm**: Flash attention via Composable Kernel (CK), which uses online softmax with a different accumulation order

The critical difference is in the softmax computation. The standard (math-only) approach computes softmax in two passes: one pass to find the row maximum and compute the denominator, and a second pass to normalize. Flash Attention, on both NVIDIA and AMD, uses an **online softmax** algorithm that fuses these passes for memory efficiency. However, the specific tiling strategy and accumulation order differ between the NVIDIA and AMD implementations.

In infinite-precision arithmetic, all three approaches produce identical results. In finite-precision FP32 and FP16 arithmetic, the different accumulation orders produce small rounding differences on the order of 1e-5 to 1e-4 per operation. These differences are individually negligible, but they do not stay negligible as they propagate through the model.

## Per-Layer Divergence Profile

The following table shows the cosine similarity between AMD ROCm and CPU outputs at different points in a 12-layer BERT model, measured during TorchBridge cross-backend validation:

| Layer Range | Cosine Similarity | Notes |
|---|---|---|
| Embeddings | 1.0000 | Perfect match -- no attention involved |
| Layers 1-3 | 0.9997 | Minimal divergence |
| Layers 4-8 | 0.9812 | Divergence accumulating |
| Layers 9-12 | 0.8200 | Significant divergence in final layers |

Each layer's small error (1e-5 to 1e-4) compounds multiplicatively through residual connections and layer normalization. The residual connection adds the attention output back to the input, preserving the error. Layer normalization then rescales the activations, which can amplify small absolute differences into larger relative differences. After 12 layers, these compounded errors produce outputs that are directionally similar but numerically quite different.

The same pattern was observed with GPT-2, with the divergence profile being nearly identical due to the shared transformer architecture.

## The Fix: Force Math-Only SDPA

For cross-backend comparison, the solution is to disable the hardware-specific SDPA backends and force the math-only kernel on all devices:

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

With these settings, ROCm falls back to the same math kernel as CPU, and outputs match within 1e-5 tolerance. This is exactly what TorchBridge does in its cross-backend validation tests.

The relevant implementation can be found in the test suite:

- `tests/e2e/test_cross_backend_bert.py` (lines 255-262): SDPA backend configuration for BERT cross-backend validation
- `tests/e2e/test_cross_backend_gpt2.py` (lines 228-239): Equivalent configuration for GPT-2, including the TF32 flags

Note that disabling flash attention and memory-efficient attention will increase memory usage and reduce throughput. This configuration is intended for **validation and testing only**, not for production inference. In production, you should use the native SDPA backends for each platform and accept that outputs will differ within floating-point tolerance.

## Implications for Multi-Vendor Deployment

This SDPA divergence has practical consequences for several real-world scenarios:

- **Training on NVIDIA, inference on AMD**: Teams that train models on NVIDIA GPUs and deploy to AMD GPUs for cost-optimized inference will see output differences. These differences do not indicate model corruption or incorrect porting. However, any test that naively compares outputs with tight tolerances will fail.

- **CI/CD pipelines that validate across hardware**: If your continuous integration pipeline runs model validation on both NVIDIA and AMD runners, you need backend-aware tolerance thresholds. A single fixed tolerance will either be too tight (causing false failures on cross-backend comparisons) or too loose (missing genuine regressions on same-backend comparisons).

- **Reproducibility research across GPU vendors**: Any research that depends on bitwise or near-bitwise reproducibility across GPU vendors must account for SDPA backend differences. Without forcing the math-only kernel, results will diverge at the attention layers regardless of how carefully other sources of nondeterminism are controlled.

The attention divergence is **not** a correctness bug -- both results are valid within floating-point precision. The online softmax used by flash attention is a mathematically equivalent reformulation of the standard softmax. The numerical differences arise solely from finite-precision arithmetic and different accumulation orders. But if you naively compare outputs across backends, you will see failures that look like model corruption.

## How TorchBridge Catches This

TorchBridge's `tb-validate` command and `UnifiedValidator` automatically handle this divergence by:

1. **Detecting which backends are available**: At startup, TorchBridge probes for CUDA, ROCm, and CPU backends and records which SDPA implementations are present on each.

2. **Setting SDPA to math-only mode for cross-backend comparisons**: When comparing outputs across different hardware backends, TorchBridge automatically disables flash attention and memory-efficient attention to ensure the same computation path is used on all devices.

3. **Using appropriate tolerance thresholds**: TorchBridge applies tighter tolerances for same-backend comparisons (where outputs should match closely) and looser tolerances for cross-backend comparisons (where SDPA divergence is expected). This eliminates false positives without masking genuine issues.

4. **Reporting per-layer divergence**: The validation output includes per-layer cosine similarity scores, so you can identify exactly where precision loss originates and whether it follows the expected compounding pattern or indicates an actual problem.

The `tb-migrate` command (new in v0.5.0) also flags `torch.cuda.amp` usage and suggests TorchBridge's precision config, which handles these cross-backend precision differences automatically. This is particularly useful when migrating existing NVIDIA-only codebases to multi-vendor deployments, where mixed-precision training interacts with the SDPA divergence to produce even larger output differences if not configured correctly.
