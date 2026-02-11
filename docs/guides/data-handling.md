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
