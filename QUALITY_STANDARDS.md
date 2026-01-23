# KernelPyTorch Quality Standards

> **Version**: 1.0 | **Applies to**: v0.4.17+ | **Last Updated**: Jan 2026

This document defines the quality gates for KernelPyTorch releases.

---

## Table of Contents

1. [Patch Release Quality Bar (0.0.x)](#patch-release-quality-bar-00x)
2. [Minor Release Quality Bar (0.y.0)](#minor-release-quality-bar-0y0)
3. [Quality Metrics Dashboard](#quality-metrics-dashboard)
4. [Automated Enforcement](#automated-enforcement)

---

## Patch Release Quality Bar (0.0.x)

**Purpose**: Every patch commit/push must meet these standards. These are automated and enforced via pre-commit hooks and CI.

### 1. Version Consistency (CRITICAL)

All version strings MUST match across:
- [ ] `pyproject.toml` → `version = "X.Y.Z"`
- [ ] `src/kernel_pytorch/__init__.py` → fallback `__version__ = "X.Y.Z"`
- [ ] `src/kernel_pytorch/cli/__init__.py` → `version='%(prog)s X.Y.Z'`
- [ ] `CHANGELOG.md` → `## [X.Y.Z]` entry exists

**Automated Check**: `scripts/check_versions.py`

### 2. Test Suite (REQUIRED)

| Test Category | Requirement | Command |
|---------------|-------------|---------|
| Unit Tests | 100% pass | `pytest tests/unit/ -x` |
| Backend Tests | 100% pass | `pytest tests/backends/ -x` |
| Feature Tests | 100% pass | `pytest tests/features/ -x` |
| Integration Tests | 95%+ pass | `pytest tests/integration/` |

**Minimum Coverage**: 70% line coverage on changed files

**Automated Check**: CI pipeline, pre-push hook

```bash
# Quick validation (run before every commit)
pytest tests/unit/ tests/backends/ tests/features/ -x -q --tb=line

# Full validation (run before push)
pytest tests/ -x --tb=short
```

### 3. Code Quality (REQUIRED)

| Check | Tool | Threshold |
|-------|------|-----------|
| Linting | Ruff | 0 errors |
| Type Hints | MyPy | 0 errors (strict mode) |
| Formatting | Ruff format | Compliant |
| Import Sorting | isort (via Ruff) | Sorted |

**Automated Check**: Pre-commit hooks

```bash
# Run all code quality checks
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/kernel_pytorch/
```

### 4. Import Integrity (REQUIRED)

All public imports must work without errors:

```bash
# Core package import test
python -c "import kernel_pytorch; print(kernel_pytorch.__version__)"

# Public API smoke test
python -c "
from kernel_pytorch import (
    KernelPyTorchConfig, get_config, set_config,
    FlashLightKernelCompiler, FusedGELU,
    UnifiedManager, get_manager,
    UnifiedAttentionFusion, RingAttentionLayer,
    UltraPrecisionModule, FP8TrainingEngine,
    HardwareAbstractionLayer,
    MoELayer, create_moe_layer,
)
print('All public imports OK')
"
```

### 5. No Regressions (REQUIRED)

- [ ] No new `# type: ignore` without justification
- [ ] No new `noqa` comments without justification
- [ ] No increase in `TODO`/`FIXME` count (must resolve as many as added)
- [ ] No new `NotImplementedError` without corresponding issue

### 6. Changelog Entry (REQUIRED)

Every patch must have a CHANGELOG.md entry:
- [ ] Version number matches
- [ ] Date is correct
- [ ] Change category is appropriate (feat/fix/refactor/docs/test/chore)
- [ ] Description is clear and concise

### 7. Commit Message (REQUIRED)

Follow Conventional Commits:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

---

## Minor Release Quality Bar (0.y.0)

**Purpose**: Comprehensive quality gate for public releases. Manual review required.

### 1. All Patch Requirements (PREREQUISITE)

All patch release requirements must be met first.

### 2. Full Test Matrix (REQUIRED)

| Test Suite | Requirement | Time |
|------------|-------------|------|
| Unit Tests | 100% pass | ~30s |
| Backend Tests | 100% pass | ~60s |
| Feature Tests | 100% pass | ~90s |
| Integration Tests | 100% pass | ~5min |
| E2E Tests | 95%+ pass | ~10min |

**Total Coverage**: 80%+ line coverage

```bash
# Full test suite with coverage
pytest tests/ --cov=src/kernel_pytorch --cov-report=html --cov-fail-under=80
```

### 3. Benchmark Validation (REQUIRED)

- [ ] No performance regressions >5% on core operations
- [ ] Benchmark suite runs without errors
- [ ] Results documented in `BENCHMARKS.md`

```bash
# Run benchmark validation
python -m kernel_pytorch.benchmarks.run_all --quick
```

### 4. Documentation Completeness (REQUIRED)

| Item | Requirement |
|------|-------------|
| README.md | Version badge matches, Quick Start works |
| CHANGELOG.md | Complete entry with all changes |
| API Docs | All public classes/functions documented |
| Guides | Updated for any new features |
| Migration | Breaking changes documented |

### 5. Demo Validation (REQUIRED)

All demos must run without errors:

```bash
# Validate all demos
for demo in demos/*.py; do
    python "$demo" --dry-run || exit 1
done
```

### 6. Cross-Platform Validation (RECOMMENDED)

| Platform | Status Required |
|----------|-----------------|
| Linux (Ubuntu 22.04+) | Must pass |
| macOS (13+) | Must pass |
| Windows (10+) | Should pass |

### 7. Hardware Backend Validation (RECOMMENDED)

| Backend | Requirement |
|---------|-------------|
| CPU (fallback) | Must pass |
| NVIDIA CUDA | Should pass |
| AMD ROCm | Should pass |
| Intel XPU | Should pass |
| TPU | Best effort |

### 8. Security Review (REQUIRED)

- [ ] No hardcoded secrets/credentials
- [ ] No vulnerable dependencies (`pip-audit`)
- [ ] No command injection vectors
- [ ] Safe tensor operations (no arbitrary code execution)

```bash
pip-audit --strict
```

### 9. API Stability (REQUIRED)

- [ ] No breaking changes to public API without deprecation
- [ ] Deprecated APIs have `FutureWarning`
- [ ] Breaking changes documented in CHANGELOG

### 10. Release Checklist

- [ ] All CI checks pass
- [ ] Version bumped correctly (0.y.0)
- [ ] CHANGELOG.md has release notes
- [ ] Git tag created (`v0.y.0`)
- [ ] PyPI package builds successfully
- [ ] Documentation site updated
- [ ] Release announcement drafted

---

## Quality Metrics Dashboard

### Current Baseline (v0.4.17)

| Metric | Value | Target |
|--------|-------|--------|
| Source Files | 188 | - |
| Lines of Code | 75,444 | - |
| Test Files | 67 | - |
| Test Functions | 1,408 | - |
| Test Pass Rate | 99%+ | 99%+ |
| Type Hint Coverage | 1,619 functions | 95%+ |
| TODO/FIXME Count | 20 | <25 |
| NotImplementedError | 5 | <10 |

### Health Thresholds

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Test Pass Rate | >99% | 95-99% | <95% |
| Coverage | >80% | 70-80% | <70% |
| Lint Errors | 0 | 1-5 | >5 |
| Type Errors | 0 | 1-10 | >10 |
| TODO Count | <25 | 25-50 | >50 |

---

## Automated Enforcement

### Pre-Commit Hooks

Located in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: version-check
        name: Version Consistency Check
        entry: python scripts/check_versions.py
        language: python
        pass_filenames: false

      - id: ruff-lint
        name: Ruff Linting
        entry: ruff check --fix
        language: python
        types: [python]

      - id: ruff-format
        name: Ruff Formatting
        entry: ruff format
        language: python
        types: [python]

      - id: import-check
        name: Import Smoke Test
        entry: python -c "import kernel_pytorch"
        language: python
        pass_filenames: false
```

### CI Pipeline Checks

GitHub Actions (`.github/workflows/ci.yml`):

1. **Quick Check** (every push): Lint + Unit tests
2. **Full Check** (PR merge): All tests + Coverage
3. **Release Check** (tags): Full matrix + Benchmarks

### Quick Reference Commands

```bash
# Before committing
ruff check src/ tests/ --fix
ruff format src/ tests/
python scripts/check_versions.py

# Before pushing
pytest tests/unit/ tests/backends/ tests/features/ -x

# Before release
pytest tests/ --cov=src/kernel_pytorch --cov-fail-under=80
python -m kernel_pytorch.benchmarks.run_all --quick
```

---

## Appendix: Quality Gate Decision Tree

```
Is this a patch release (0.0.x)?
├── YES → Run Patch Quality Bar
│   ├── All checks pass? → OK to push
│   └── Any check fails? → Fix before push
│
└── NO (minor release 0.y.0)
    ├── All Patch checks pass?
    │   ├── NO → Fix first
    │   └── YES → Continue to Minor checks
    │
    └── All Minor checks pass?
        ├── YES → OK to release
        └── NO → Fix or document exceptions
```

---

**Maintainer**: KernelPyTorch Team
**Questions**: Open an issue at https://github.com/KernelPyTorch/kernel-pytorch/issues
