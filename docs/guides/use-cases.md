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

## 2. Cross-Backend Performance Comparison

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

## 3. Hardware Migration Validation

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
