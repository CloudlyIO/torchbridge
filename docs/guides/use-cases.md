# TorchBridge Use Cases

Real-world scenarios where TorchBridge saves time and reduces friction when working
with PyTorch across different hardware backends.

---

## 1. Quick GPU Backend Check

**Problem:** You have a new machine (cloud instance, workstation, dev box) and need to
know if PyTorch can use the GPU, which driver version is installed, and whether
everything is compatible — before wasting time on a failed training run.

**Command:**
```bash
tb-doctor
```

**Expected output:**
```
TorchBridge Hardware Diagnostic
================================
PyTorch:    2.7.0+cu126
Backend:    NVIDIA CUDA
GPU:        NVIDIA A10G (24GB)
Driver:     570.86.15
CUDA:       12.6
Status:     READY

Checks:
  [PASS] CUDA runtime available
  [PASS] GPU memory accessible (23.5 GB free)
  [PASS] cuDNN detected (v9.1.0)
  [PASS] NCCL available for distributed
  [PASS] torch.compile backend functional
```

**When to use:** First thing after SSH-ing into a new cloud instance, after driver
updates, or when debugging "CUDA not available" errors.

---

## 2. One-Command Model Preparation

**Problem:** You have a trained model and want to prepare it for inference on the
current hardware — applying backend-specific optimizations like TensorCore layout,
mixed precision, or operator fusion — without writing backend-specific code.

**Command:**
```bash
tb-optimize model.pt --output optimized_model.pt
```

**Expected output:**
```
Loading model from model.pt...
Detected backend: NVIDIA CUDA (A10G, sm_86)
Applying optimizations:
  [1/3] Mixed precision (FP16) for Ampere TensorCores
  [2/3] Operator fusion (attention + linear)
  [3/3] Memory layout optimization (channels-last)
Saved optimized model to optimized_model.pt

Optimization summary:
  Original size:  440 MB
  Optimized size: 224 MB
  Expected speedup: ~1.8x on current hardware
```

**When to use:** Before deploying a model to production, when moving a model to
new hardware, or when you want hardware-specific optimizations without manual tuning.

---

## 3. Cross-Backend Performance Comparison

**Problem:** You need to decide which GPU/accelerator to use for your workload.
Running the same benchmark manually on each backend is tedious and error-prone.

**Command:**
```bash
tb-benchmark --model Qwen/Qwen3-0.6B --batch-sizes 1,8,32 --iterations 100
```

**Expected output:**
```
TorchBridge Benchmark: Qwen/Qwen3-0.6B
=========================================
Backend: NVIDIA CUDA (A10G)

Batch Size | Throughput (tok/s) | Latency (ms) | Memory (MB)
-----------|-------------------|--------------|------------
         1 |             1,240 |          8.1 |        1,847
         8 |             5,680 |         14.1 |        2,234
        32 |            12,450 |         25.7 |        3,891

Results saved to: benchmarks/results/qwen3_0.6b_a10g_20260211.json
```

**When to use:** When choosing between cloud instance types, comparing hardware
generations (e.g., T4 vs A10G), or establishing performance baselines before
optimization.

---

## 4. Production Model Export

**Problem:** You need to export a PyTorch model to ONNX or TorchScript for
deployment in a serving framework, with hardware-specific optimizations baked in.

**Command:**
```bash
tb-export model.pt --format onnx --optimize --output model_serving.onnx
```

**Expected output:**
```
Loading model from model.pt...
Detected backend: NVIDIA CUDA (A10G, sm_86)
Exporting to ONNX with optimizations:
  [1/3] Graph optimization (constant folding, dead code elimination)
  [2/3] Operator fusion for target hardware
  [3/3] Dynamic axis configuration (batch dimension)
Exported to model_serving.onnx

Export summary:
  Format:    ONNX (opset 18)
  Size:      218 MB
  Inputs:    input_ids [batch, seq_len], attention_mask [batch, seq_len]
  Outputs:   logits [batch, seq_len, vocab_size]
  Validated: inference matches PyTorch reference (max_diff < 1e-5)
```

**When to use:** When deploying models to ONNX Runtime, TensorRT, or other serving
frameworks. The export includes validation that outputs match the original PyTorch model.

---

## 5. Hardware Migration Validation

**Problem:** You're moving a model from one GPU type to another (e.g., T4 → A10G,
NVIDIA → AMD) and need to verify that inference outputs are numerically identical
across backends.

**Command:**
```bash
tb-validate --model model.pt --reference-device cpu --target-device cuda
```

**Expected output:**
```
TorchBridge Cross-Backend Validation
======================================
Model:      model.pt
Reference:  CPU (PyTorch 2.7.0)
Target:     NVIDIA CUDA (A10G)

Running validation with 10 random inputs...

Results:
  Max absolute diff:  2.34e-06
  Cosine similarity:  1.000000
  All outputs match:  YES

Latency comparison:
  CPU:    45.2 ms/inference
  CUDA:   3.1 ms/inference
  Speedup: 14.6x

Status: PASSED — outputs are numerically consistent across backends.
```

**When to use:** When migrating workloads between cloud providers, upgrading GPU
hardware, or validating that a model produces identical results on AMD ROCm vs
NVIDIA CUDA.
