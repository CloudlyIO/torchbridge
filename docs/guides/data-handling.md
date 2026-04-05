# Data Handling and Privacy

TorchBridge collects **no telemetry** and makes **no network requests**. All data stays on your machine.

## What Data TorchBridge Writes Locally

TorchBridge writes one file to your local filesystem during normal operation:

### Kernel Benchmark Cache

```
~/.torchbridge/kernel_benchmarks.json
```

This cache stores attention kernel latency measurements so repeated hardware configuration lookups are fast. It is keyed by a hardware fingerprint (GPU model, PyTorch version, CUDA version) and is automatically invalidated when your hardware or software environment changes.

The cache contains only timing data (latency in milliseconds) and hardware metadata. It contains no model weights, inputs, outputs, or personally identifiable information.

## Privacy Properties

| Question | Answer |
|----------|--------|
| Does TorchBridge phone home? | No |
| Is telemetry sent anywhere? | No |
| What is written locally? | Kernel benchmark cache (`~/.torchbridge/kernel_benchmarks.json`) |
| Contains model data? | No — only kernel latency timings and hardware metadata |
| Safe for air-gapped environments? | Yes — zero network requests |

## Clearing Stored Data

To remove all locally stored data, delete the TorchBridge data directory:

```bash
rm -rf ~/.torchbridge
```

TorchBridge will recreate the directory and repopulate the benchmark cache on the next use.

## Corporate and Compliance Notes

- TorchBridge makes **zero network requests** — safe for air-gapped environments.
- All data remains on the local filesystem. No data leaves the machine.
- The OTel exporter (`--otel` flag on `tb-validate`) sends validation spans to an OTLP endpoint **only when explicitly configured**. No endpoint is configured by default; if none is set, spans are emitted to the console only.
- For SOC 2 or HIPAA environments, the default configuration requires no additional steps.

## What Is Safe to Share

If you need to share diagnostic output for debugging:

- **Safe to share**: Backend type, latency numbers, hardware model name, PyTorch version, validation pass/fail results, max_diff, cosine_sim values.
- **Avoid sharing**: Model file paths (may reveal internal project structure), custom model configurations, or any output derived from proprietary model weights.
