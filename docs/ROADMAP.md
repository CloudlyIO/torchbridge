# TorchBridge Roadmap

TorchBridge is the **numerical truth layer** for heterogeneous AI — the only tool
in the ML stack that addresses compounding divergence in multi-step agentic reasoning
chains across PyTorch backends. This document describes where the project is heading.

---

## Project Philosophy

**We validate. We don't wrap.**

Every TorchBridge feature answers one question: *is this model producing the same
answer across different hardware?* We do not abstract PyTorch, replace training loops,
or build a serving runtime. We sit between you and your hardware and tell you what's
actually happening numerically.

**Backend-aware, not backend-agnostic.**

Every compatibility matrix, fallback chain, and configuration recommendation is
grounded in real architectural differences between NVIDIA, AMD, Trainium, and TPU.
We don't treat hardware as interchangeable — we surface what makes each accelerator
different and help you exploit or compensate for those differences.

**Agentic reasoning is the frontier.**

Single-forward-pass validation is solved. The hard problem is multi-step divergence:
when a model generates 50 tokens on CUDA and 50 tokens on ROCm, does the reasoning
chain stay coherent? Does divergence at step 3 compound into nonsense by step 30?
That's where TorchBridge is headed.

---

## Current State: Community Launch

TorchBridge v0.5.x is production-ready for its core use cases:

- **Cross-backend output validation** — compare model outputs numerically across any two backends with per-layer attribution, tolerance DB lookup, and compliance certificates
- **Multi-step trace validation** — autoregressive drift detection across N generation steps, with first-divergence-step and amplification factor reporting
- **Configuration intelligence** — backend-aware quantization, attention dispatch, KV-cache dtype, speculative decoding, adapter method, and distributed topology selection — all via compatibility matrices with fallback chains
- **Observability** — OpenTelemetry span export for validation results, structured exit codes for CI/CD
- **11 CLI tools** — `tb-validate`, `tb-benchmark`, `tb-doctor`, `tb-advisor`, `tb-migrate`, `tb-quantize`, `tb-cache`, `tb-speculate`, `tb-checkpoint`, `tb-adapter`, and `torchbridge` dispatcher

Current test coverage: 2,139 tests, 0 ruff violations, 0 mypy errors, Apache 2.0.

---

## Phase 1 — Validation Depth

*Goal: make TorchBridge the definitive answer to "does my model work on this hardware?"*

### Divergence Attribution

Today TorchBridge reports per-layer max_diff and cosine similarity. The next step is
**causal attribution** — given a final-layer divergence of 0.003, which upstream layer
caused it and why? Attribution turns a flag into a fix.

- Layer-contribution scoring (sensitivity analysis)
- Dtype-cast divergence isolation (FP32 vs BF16 vs FP8 contribution)
- Attention pattern diff visualization output

### Tolerance DB Expansion

The current tolerance DB covers 5 model families × 3 backends × 3 dtypes = 80 entries.
Real-world coverage should extend to:

- 10+ model families (diffusion, audio, video, code generation, multimodal)
- Community-contributed measured tolerances (vs. current derived/fallback labels)
- Per-layer tolerance (some layers diverge more than others by design)

### Benchmark Regression Detection

`tb-benchmark` currently compares a run against a baseline JSON. Planned extensions:

- Statistical significance testing (not just threshold crossing)
- Multi-run aggregation and confidence intervals
- Integration with GitHub Actions as a first-class PR check

---

## Phase 2 — Agentic Trace Intelligence

*Goal: become the validation standard for multi-step agentic AI systems.*

The core insight: in a 50-step reasoning chain, divergence doesn't stay bounded. A
0.1% output difference at step 1 can compound into a completely different answer at
step 30. No other tool measures this. TorchBridge does.

### Divergence Amplification Analysis

- Amplification factor (divergence at step N / divergence at step 1) as a first-class metric
- Early-exit detection: identify the step at which two backends' chains become semantically distinct
- Threshold-based alerting: "backend A and B are safe for agentic use up to N steps at this model size"

### Cross-Framework Trace Comparison

Today: CUDA vs ROCm. Planned: any two inference frameworks (vLLM, TGI, llama.cpp, transformers) against each other — not just backend-level, but framework-level drift.

### Tool-Call Divergence

For tool-using agents, validate that function calls and JSON-structured outputs remain
identical across backends. A model that calls `search("Paris")` on CUDA but `search("paris")` on ROCm produces different downstream behavior even if raw logits look similar.

---

## Phase 3 — Fleet & Deployment Intelligence

*Goal: make mixed-hardware production deployments safe and observable.*

### Heterogeneous Fleet Validation

Production deployments increasingly mix hardware — NVIDIA for prefill, AMD or Inferentia
for decode, TPU for batch inference. TorchBridge already generates configuration for
these topologies (`tb-advisor --mode heterogeneous`). Phase 3 adds runtime validation:
confirm the actual running deployment matches its configuration certificate.

### KV-Cache Handoff Validation

When prefill runs on H100 and decode runs on MI300X, KV tensors must be transferred
with dtype and layout contracts intact. TorchBridge already generates compliance
certificates for KV handoff specs. Phase 3 extends this to:

- Runtime certificate verification (not just pre-deployment)
- Drift detection when cluster topology changes
- Compliance audit trail for regulated environments

### CI/CD Validation Workflows

- GitHub Actions workflows for automatic cross-backend validation on PR
- Pre-built Docker images for validation pipelines
- Hardware matrix reporting integrated with pull request checks

---

## Phase 4 — Community & Ecosystem

*Goal: make TorchBridge the default validation step before any model goes to production.*

### Tolerance DB Community Contributions

Open a process for contributors to submit measured tolerances from real hardware.
Community measurements are more trustworthy than derived tolerances and cover hardware
configurations we can't access directly (enterprise-only accelerators, exotic configurations).

### Integrations

- **vLLM** — validate that a vLLM deployment produces the same outputs as a baseline transformers run
- **TGI (Hugging Face)** — same as vLLM
- **ONNX Runtime** — validate exported ONNX models against PyTorch reference
- **CI providers** — first-class plugins for GitHub Actions, GitLab CI, Buildkite

### New Backends

Hardware support follows hardware market adoption:

- **Intel Gaudi 3** — if adoption recovers post-Falcon Shores cancellation
- **AWS Trainium 3** — when generally available
- **Apple Silicon (inference)** — MPS already detectable; deeper per-layer support
- **NVIDIA GB300 NVL72** — when available for validation

---

## What We Won't Build

**Serving runtime.** vLLM, TGI, and TorchServe are mature. We don't replace them.
We validate what they produce.

**Training loop orchestration.** FSDP, DeepSpeed, and Megatron exist. We generate
configuration for them. We don't wrap them.

**Model optimization.** We surface compatibility matrices so you can make informed
choices about quantization, attention kernels, and adapter methods. The actual
optimization work is done by torchao, PEFT, Flash Attention, etc. We integrate with
those libraries, not compete with them.

**Dashboard / UI.** The output of TorchBridge is structured JSON and CLI text, designed
to feed into whatever dashboard you already have. We don't build one.

---

## Contributing

We welcome contributions in any phase — especially:

- **Tolerance DB measurements** from hardware we can't access
- **Backend-specific bugs** — if validation fails on your hardware and we call it a pass, that's our highest priority bug
- **Compatibility matrix corrections** — if a kernel compatibility claim is wrong, open an issue with the exact hardware model and PyTorch version

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the development workflow.

Issues and discussion: [github.com/CloudlyIO/torchbridge](https://github.com/CloudlyIO/torchbridge)
