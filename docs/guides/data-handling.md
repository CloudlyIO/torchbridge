# Data Handling and Privacy

TorchBridge collects **no telemetry** and makes **no network requests**. All data stays on your machine.

## What Data Is Collected

When metrics collection is enabled, TorchBridge records local performance data:

- **Latency**: per-inference and per-batch timing (milliseconds)
- **Throughput**: samples per second for training and inference loops
- **Memory usage**: peak GPU/CPU memory allocation per run
- **Backend selection**: which hardware backend was chosen and why

This data is intended to help you profile and optimize your workloads locally.

## Where Data Is Stored

All metrics are written to a single JSON file:

```
~/.torchbridge/performance_metrics.json
```

No data is sent to any remote server, API, or analytics service. There is no phone-home behavior of any kind.

## Enabling Metrics Collection

Metrics collection is **disabled by default**. To enable it, set the environment variable before running your workload:

```bash
export TORCHBRIDGE_METRICS=1
```

To disable it again, unset the variable or set it to `0`:

```bash
unset TORCHBRIDGE_METRICS
# or
export TORCHBRIDGE_METRICS=0
```

## Clearing Stored Data

To remove all locally stored metrics, delete the TorchBridge data directory:

```bash
rm -rf ~/.torchbridge
```

This removes all cached performance data. TorchBridge will recreate the directory the next time metrics collection is enabled.

## Summary

| Question | Answer |
|----------|--------|
| Does TorchBridge phone home? | No |
| Is telemetry sent anywhere? | No |
| What is collected? | Local performance metrics only |
| Enabled by default? | No -- requires `TORCHBRIDGE_METRICS=1` |
| Where is data stored? | `~/.torchbridge/performance_metrics.json` |
| How to clear data? | `rm -rf ~/.torchbridge` |

## Analyzing Performance Metrics

When metrics collection is enabled, TorchBridge writes a JSON file you can inspect to understand performance characteristics. A typical entry looks like:

```json
{
  "timestamp": "2026-02-13T14:30:00Z",
  "backend": "nvidia_cuda",
  "device": "NVIDIA A10G",
  "operation": "optimize",
  "latency_ms": {
    "p50": 8.2,
    "p95": 12.4,
    "p99": 15.1
  },
  "memory_peak_mb": 1843,
  "batch_size": 32,
  "precision": "float16"
}
```

**Key metrics to examine:**

- **Latency p50 vs p95**: A large gap suggests occasional stalls (memory pressure, GC pauses, or thermal throttling). A p95 within 2x of p50 is healthy.
- **Memory peak**: Compare against your GPU's total VRAM. If peak is above 80%, consider reducing batch size or enabling gradient checkpointing.
- **Backend selection**: Confirms the detected backend matches your hardware. If you see `cpu` on a GPU machine, run `tb-doctor`.

**Comparing across runs:**

```bash
# View metrics file directly
cat ~/.torchbridge/performance_metrics.json | python3 -m json.tool

# Or use tb-benchmark for structured comparison
tb-benchmark --model model.pt --batch-sizes 1,8,32 --output results.json
```

## Privacy Best Practices

### Verifying metrics are disabled in production

Before deploying, confirm that metrics collection is off:

```bash
echo $TORCHBRIDGE_METRICS  # Should be empty or "0"
```

In container deployments, do not set `TORCHBRIDGE_METRICS=1` in your Dockerfile or orchestration config unless you explicitly want local profiling.

### What is safe to share

If you need to share performance data for debugging or benchmarking:

- **Safe to share**: Backend type, latency numbers, memory usage, batch sizes, precision settings, PyTorch version, GPU model name.
- **Avoid sharing**: Model file paths (may reveal internal project structure), custom kernel names, anything from proprietary model configurations.

### Data retention

Old metrics accumulate in `~/.torchbridge/performance_metrics.json` over time. Periodically clear them:

```bash
rm ~/.torchbridge/performance_metrics.json
```

TorchBridge recreates the file on the next metrics-enabled run.

### Corporate and compliance notes

- TorchBridge makes **zero network requests** -- safe for air-gapped environments.
- All metrics remain on the local filesystem. No data leaves the machine.
- For SOC 2 or HIPAA environments, the default (metrics disabled) requires no additional configuration.
