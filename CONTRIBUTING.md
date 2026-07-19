# Contributing to TorchBridge

TorchBridge is the numerical truth layer for heterogeneous AI — validating that models
produce consistent outputs across NVIDIA, AMD, Trainium, TPU, and Apple Silicon.

This guide follows the contributor journey from first install to merged PR.

---

## Quick Dev Setup

If you just want to run the tests and start hacking — three commands:

```bash
git clone https://github.com/CloudlyIO/torchbridge.git && cd torchbridge
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest tests/ -q -m "not gpu and not slow"   # ~2 min, all green
```

Lint and type check:

```bash
ruff check src tests   # must be clean
mypy src               # must be clean
```

That's it. The rest of this guide covers the full contributor journey.

---

## The Contributor Journey

```
First Impression
      ↓
Day 1 — Install & Test
      ↓
Day 2 — Use in Your Project
      ↓
Find a Bug or Feature to Contribute
      ↓
Make the Changes
      ↓
Upstream via PR
```

Each stage has clear entry points. You don't need to complete earlier stages
before contributing — start wherever you are.

---

## Stage 1 — First Impression

You've heard about TorchBridge and want to understand what it does.

**Recommended path:**

1. Read the [README](README.md) — especially the identity statement and the CLI overview
2. Browse the [docs/](docs/) directory for architecture context and guides
3. Scan the [public ROADMAP](docs/ROADMAP.md) to see what's been built and what's next

**No code needed here.** If something is confusing or missing, open a
[Discussion](https://github.com/CloudlyIO/torchbridge/discussions) — that's a valid
contribution.

---

## Stage 2 — Day 1: Install and Test

You've installed TorchBridge and are validating it works on your hardware.

```bash
pip install torchbridge-ml
tb-doctor          # detects your backend + reports health
tb-validate        # runs full numerical validation
```

**What we need from Day 1 contributors:**

| What you found | Where to report |
|---|---|
| Wrong recommendation from `tb-doctor` | [Matrix Correction](https://github.com/CloudlyIO/torchbridge/issues/new?template=matrix_correction.yml) issue |
| Measured tolerance from `tb-validate` | [Tolerance DB Measurement](https://github.com/CloudlyIO/torchbridge/issues/new?template=tolerance_measurement.yml) issue |
| Install error or crash | [Bug Report](https://github.com/CloudlyIO/torchbridge/issues/new?template=bug_report.yml) issue |

**Tolerance measurements** are the highest-signal contribution you can make as a Day 1
user. Run `tb-validate --compare`, paste the output in a Tolerance DB Measurement issue —
your hardware is now in the DB.

---

## Stage 3 — Day 2: Use in Your Project

You're integrating TorchBridge into your own model pipeline and hitting real-world
behavior.

**Common Day 2 contributions:**

- **Compatibility matrix corrections**: The matrix claims something is supported but
  it fails on your hardware. File a [Matrix Correction](https://github.com/CloudlyIO/torchbridge/issues/new?template=matrix_correction.yml)
  issue — or fix it directly and send a PR (see Stage 5 below).

- **Missing backend support**: Your hardware isn't listed. Open a
  [Feature Request](https://github.com/CloudlyIO/torchbridge/issues/new?template=feature_request.yml).

- **Documentation gaps**: A doc led you down the wrong path. A one-line doc fix is a
  fully valid PR.

---

## Stage 4 — Find a Bug or Feature to Contribute

You've identified something concrete to fix or add.

**Before writing code:**

1. Search [existing issues](https://github.com/CloudlyIO/torchbridge/issues) — someone
   may already be working on it.
2. Open an issue if one doesn't exist. Every PR must link to an issue — no exceptions.
3. Comment on the issue to claim it — prevents duplicate work.

**Good first contributions** (tagged `good first issue`):

- Compatibility matrix corrections (`good first issue` + `compatibility-matrix`)
- Tolerance DB measurements (`good first issue` + `tolerance-db`)
- Doc fixes and test additions

---

## Stage 5 — Make the Changes

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Git

Optional for GPU backend development: CUDA 12.0+ (NVIDIA), ROCm 6.2+ (AMD), PyTorch/XLA (TPU)

### Setup

```bash
git clone https://github.com/CloudlyIO/torchbridge.git
cd torchbridge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -c "import torchbridge; print(f'TorchBridge v{torchbridge.__version__} ready')"
```

To install all optional GPU/cloud extras (heavier, may require CUDA/ROCm):

```bash
pip install -e ".[dev,all]"
```

### Branch

Always branch from `main`. Use a descriptive prefix:

```bash
git checkout -b fix/description   # bug fix
git checkout -b feat/description  # new capability
git checkout -b docs/description  # documentation
git checkout -b test/description  # tests only
```

### Project structure

```
src/torchbridge/
├── backends/     # Vendor backends (NVIDIA, AMD, Trainium, TPU + factory)
├── core/         # Hardware detection, config, architecture enums
├── precision/    # Quantization compatibility matrix + torchao dispatch
├── attention/    # Attention kernel compatibility matrix + dispatcher
├── distributed/  # FSDP config advisor, heterogeneous cluster advisor, topology
├── inference/    # Disaggregated fleet, KV handoff, speculative, phase detection
├── models/       # LLM KV cache advisor
├── checkpoint/   # DCP wrapper with cross-backend metadata normalization
├── adapters/     # Adapter compatibility matrix + config (LoRA/QLoRA/DoRA)
├── benchmarks/   # 5 claim benchmarks with real measured results
├── testing/      # DivergenceTracer, ToleranceDB, OTel exporter, trace validator
├── validation/   # UnifiedValidator — model structure, hardware, numerical stability
├── cli/          # 11 CLI entry points (torchbridge + 10 tb-* commands)
└── utils/        # Shared utilities
```

### Code conventions

- Python files, packages, test dirs: `snake_case`
- Classes: `PascalCase` — Functions: `snake_case` — Constants: `UPPER_SNAKE_CASE`
- Type hints on all public APIs
- **No hardcoded version strings anywhere** — version lives in `pyproject.toml` only
- No new dependencies without discussion in an issue first

### Adding tolerance data (hardware-gated, high value)

If you have access to uncommon hardware (AMD MI300X, Trainium2, TPU v7 Ironwood), add
measured entries to `src/torchbridge/testing/tolerance_db.py`:

```python
# Use _m() for measured, _d() for derived from a measured baseline
_MY_NOTE = "measured on Qwen3-0.6B; <your hardware> <date>"
("decoder-small", "rocm", "float32"): _m(1e-3, 1e-4, _MY_NOTE),
```

Labels: `"measured"` (use `_m()`), `"derived"` (use `_d()`), `"fallback"` (auto-generated — do not add manually).

### Tests

```bash
pytest tests/ -q -m "not gpu and not slow"
```

Or with the project Makefile (from repo root, after `pip install -e ".[dev]"`):

```bash
make test       # pytest, skips gpu/slow
make lint       # ruff check src tests
make typecheck  # mypy src
```

Every behavior change needs a test. A PR without tests for the changed behavior will
not be merged.

### Lint

```bash
ruff check src tests
```

Must be clean — zero violations.

### Type check

```bash
mypy src
```

CI runs mypy — a PR that passes tests but fails type check will not merge.

### Version consistency check

TorchBridge enforces that version strings appear only in `pyproject.toml`. If you add a
new file that references a version number, the pre-commit hook will catch it:

```bash
python scripts/ci/check_version_consistency.py
```

If it reports a drift, the fix is to remove the hardcoded version and derive it from
`importlib.metadata.version("torchbridge-ml")` instead.

---

## Stage 6 — Upstream via PR

### Pre-flight checklist

- [ ] Tests pass: `pytest tests/ -q -m "not gpu and not slow"`
- [ ] Lint clean: `ruff check src tests`
- [ ] Issue linked (every PR must close or reference an issue — no exceptions)
- [ ] `CHANGELOG.md` entry added if this changes user-visible behavior

### What the PR template asks for

1. One-sentence description of what the PR does
2. Which stage of the contributor journey this came from
3. The linked issue number
4. The pytest summary line (paste directly — do not omit)
5. The ruff output (paste directly — blank means clean)

**Why so strict?** TorchBridge is validated across 8 hardware backends. A change that
looks correct on CPU can diverge silently on TPU. The paper trail in the PR is the
evidence that the fix is real.

### After merge

- Version is bumped by a maintainer
- Your fix is attributed in `CHANGELOG.md`
- Tolerance measurements are cited in `tb-validate` output for all users on that hardware

---

## Getting Help

- **Discussions** — [github.com/CloudlyIO/torchbridge/discussions](https://github.com/CloudlyIO/torchbridge/discussions)
- **Documentation** — [docs/](docs/) (online docs at docs.torchbridge.ml coming soon)
- **Stack Overflow** — [stackoverflow.com/questions/tagged/torchbridge](https://stackoverflow.com/questions/tagged/torchbridge)
