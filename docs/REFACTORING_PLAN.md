# KernelPyTorch Refactoring Plan (v0.3.7 → v0.3.8)

**Created**: 2025-01-13
**Status**: Pending Implementation
**Estimated Impact**: 15-20% codebase reduction (~4,500-5,000 lines)

---

## Executive Summary

| Category | Issues Found | Severity | Lines Affected |
|----------|-------------|----------|----------------|
| Duplicate Modules | 2 critical | HIGH | ~2,150 lines |
| Backend Duplication | 3 areas | HIGH | ~3,000 lines |
| Overlapping Functionality | 4 systems | MEDIUM | ~1,500 lines |
| Dead Code / TODOs | 35+ locations | MEDIUM | ~350 lines |
| Structure Issues | 5 god classes | MEDIUM | Refactor needed |
| Legacy Support | 4 modules | LOW | Technical debt |

---

## Phase 1: Critical Duplicates (Priority: IMMEDIATE)

### 1.1 Merge Ultra Precision Modules

**Problem**: Two implementations of ultra-precision quantization

| File | Lines | Features |
|------|-------|----------|
| `src/kernel_pytorch/precision/ultra_precision.py` | 1,105 | Entropy-based allocation |
| `src/kernel_pytorch/optimizations/next_gen/ultra_precision.py` | 753 | NVFP4 with double quantization |

**Action**:
1. Merge both implementations into `precision/ultra_precision.py`
2. Keep both strategies as `AllocationStrategy` enum values
3. Delete `optimizations/next_gen/ultra_precision.py`
4. Update imports across codebase

**Files to Update**:
- `src/kernel_pytorch/optimizations/next_gen/__init__.py`
- Any files importing from `optimizations.next_gen.ultra_precision`

---

### 1.2 Consolidate PyGraph Optimizer

**Problem**: Duplicate CUDA graph optimization implementations

| File | Classes |
|------|---------|
| `src/kernel_pytorch/core/compilers/pygraph_optimizer.py` | `GraphDeploymentStrategy`, `WorkloadAnalysis`, `CUDAGraphManager` |
| `src/kernel_pytorch/optimizations/next_gen/pygraph_optimizer.py` | Same classes duplicated |

**Action**:
1. Keep `core/compilers/pygraph_optimizer.py` as authoritative
2. Delete `optimizations/next_gen/pygraph_optimizer.py`
3. Update `optimizations/next_gen/__init__.py` to re-export from `core.compilers`

---

### 1.3 Unify LRU Cache Implementations

**Problem**: Two LRU cache implementations with slightly different features

| File | Implementation |
|------|----------------|
| `src/kernel_pytorch/backends/amd/rocm_compiler.py` | Simple LRUCache (line 46+) |
| `src/kernel_pytorch/backends/tpu/cache_utils.py` | Generic LRUCache with statistics |

**Action**:
1. Create `src/kernel_pytorch/utils/cache.py`
2. Implement generic `LRUCache[K, V]` with statistics
3. Update AMD and TPU backends to use shared implementation
4. Remove duplicate implementations from backend files

---

## Phase 2: Backend Base Classes (Priority: HIGH)

### 2.1 Create Abstract Memory Manager

**Problem**: 80% code duplication across backend memory managers

| Backend | File | Common Methods |
|---------|------|----------------|
| NVIDIA | `backends/nvidia/memory_manager.py` | `allocate_tensor`, `optimize_memory_usage`, `get_memory_stats` |
| AMD | `backends/amd/memory_manager.py` | Same methods |
| TPU | `backends/tpu/memory_manager.py` | Same methods |

**Action**:
1. Create `src/kernel_pytorch/backends/base_memory_manager.py`
2. Define abstract base class with template methods
3. Backend-specific classes inherit and override only device-specific methods
4. Estimated reduction: 40-50% of memory manager code

---

### 2.2 Unify Backend Exceptions

**Problem**: Each backend defines its own exception hierarchy with overlaps

| Backend | File | Exception Count |
|---------|------|-----------------|
| NVIDIA | `backends/nvidia/nvidia_exceptions.py` | 11 classes |
| AMD | `backends/amd/amd_exceptions.py` | 14 classes |
| TPU | `backends/tpu/tpu_exceptions.py` | 15 classes |

**Common Duplicates**: `OutOfMemoryError`, `CompilationError`, `UnsupportedOperationError`

**Action**:
1. Create `src/kernel_pytorch/backends/base_exceptions.py`
2. Define shared exception hierarchy
3. Backend-specific exceptions inherit from base
4. Reduce from 40+ exceptions to ~25

---

### 2.3 Abstract Optimizer Pattern

**Problem**: Similar optimization patterns repeated across backends

| Backend | File | Duplicated Logic |
|---------|------|------------------|
| NVIDIA | `backends/nvidia/nvidia_optimizer.py` | `optimize_for_training`, `optimize_for_inference` |
| AMD | `backends/amd/amd_optimizer.py` | Same patterns |
| TPU | `backends/tpu/tpu_optimizer.py` | Same patterns |

**Action**:
1. Create `src/kernel_pytorch/backends/base_optimizer.py`
2. Define optimization template with hooks for backend-specific passes
3. Reduce duplicate fusion logic

---

## Phase 3: Configuration Consolidation (Priority: MEDIUM)

### 3.1 Merge Config Systems

**Problem**: Fragmented configuration across modules

| File | Config Classes |
|------|----------------|
| `src/kernel_pytorch/core/config.py` (928 lines) | `PrecisionConfig`, `MemoryConfig`, etc. |
| `src/kernel_pytorch/attention/core/config.py` (80+ lines) | `AttentionConfig`, `FP8AttentionConfig`, etc. |

**Action**:
1. Merge attention config classes into `core/config.py`
2. Use inheritance for attention-specific extensions
3. Remove `attention/core/config.py`
4. Update all imports

---

### 3.2 Remove Legacy Import Support

**Problem**: Technical debt from maintaining old import paths

**Files with Legacy Support**:
- `src/kernel_pytorch/core/__init__.py` (lines 109-135)
- `src/kernel_pytorch/attention/compatibility/__init__.py`
- `src/kernel_pytorch/hardware/__init__.py`
- `src/kernel_pytorch/optimizations/__init__.py`

**Action**:
1. Document deprecation in CHANGELOG (target: v0.4.0)
2. Create migration guide
3. Remove `_LegacyImportHelper` classes
4. Clean up `sys.modules` manipulation

---

## Phase 4: Dead Code Cleanup (Priority: MEDIUM)

### 4.1 Unimplemented Functions (35+ locations)

| File | Line | Issue |
|------|------|-------|
| `precision/ultra_precision.py` | 845-849 | Empty `pass` for dynamic precision |
| `precision/fp8_training_engine.py` | 568-569 | Empty context cleanup |
| `backends/amd/amd_optimizer.py` | 278, 302, 460, 487 | Multiple TODO items |
| `backends/amd/rocm_compiler.py` | 211 | HIP compilation TODO |
| `hardware/gpu/custom_kernels.py` | 346, 429-431 | CUDA kernel TODO |
| `backends/tpu/xla_compat.py` | 55, 87 | Empty exception handlers |

**Action**: Either implement or remove with clear error/skip messages

---

### 4.2 Bare Exception Handlers (15+ locations)

**Problem**: Bare `except:` or broad `except Exception:` hiding bugs

**Locations**:
- `attention/fusion/neural_operator.py` (~line 825)
- `hardware/__init__.py`
- Multiple files in `hardware/abstraction/`

**Action**:
1. Replace with specific exception types
2. Add proper logging
3. Re-raise where appropriate

---

### 4.3 Backup Directories

**Directories to Remove**:
```
.archive/                           # Empty
local/backups/demos_backup/         # 22 directories, duplicated demos
.github-workflows-backup/           # Outdated workflow backups
```

---

## Phase 5: Structure Improvements (Priority: LOW)

### 5.1 Break Down God Classes

| Class | File | Lines | Refactor Into |
|-------|------|-------|---------------|
| `HardwareAbstractionLayer` | `hardware/abstraction/hal_core.py` | 1,789 | Per-vendor adapter plugins |
| `UnifiedValidator` | `validation/unified_validator.py` | 1,327 | `ModuleValidator`, `ModelValidator`, `HardwareValidator` |
| `NeuralOperatorFusion` | `attention/fusion/neural_operator.py` | 1,058 | Separate fusion strategy classes |
| `DynamicShapesOptimizer` | `optimizations/patterns/dynamic_shapes.py` | 1,366 | Smaller focused optimizers |

---

### 5.2 Test Organization

**Current**: 47 test files in flat structure
**Proposed**:
```
tests/
├── unit/                    # Mirror source structure
│   ├── backends/
│   ├── core/
│   └── precision/
├── integration/             # Cross-module tests
└── regression/              # Performance regression tests
```

---

### 5.3 Remove setup.py

**Problem**: Both `pyproject.toml` and `setup.py` define project metadata

**Action**: Remove `setup.py`, use only `pyproject.toml` (modern standard)

---

## Implementation Checklist

### Phase 1 (Immediate - 2-3 days)
- [ ] Merge ultra_precision modules
- [ ] Consolidate pygraph_optimizer
- [ ] Create shared LRU cache utility
- [ ] Update all affected imports
- [ ] Run full test suite

### Phase 2 (High Priority - 1 week)
- [ ] Create base_memory_manager.py
- [ ] Create base_exceptions.py
- [ ] Refactor NVIDIA memory manager
- [ ] Refactor AMD memory manager
- [ ] Refactor TPU memory manager
- [ ] Run backend tests

### Phase 3 (Medium Priority - 1 week)
- [ ] Merge config systems
- [ ] Document legacy deprecation
- [ ] Create migration guide
- [ ] Clean dead code / TODOs

### Phase 4 (Low Priority - Ongoing)
- [ ] Break down god classes
- [ ] Reorganize tests
- [ ] Remove backup directories
- [ ] Remove setup.py

---

## Metrics

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Total Lines | 54,822 | ~50,000 |
| Duplicate Lines | ~4,500 | ~500 |
| God Classes (1000+ lines) | 5 | 0 |
| Exception Classes | 40+ | ~25 |
| Config Files | 2+ | 1 |

---

## Notes

- All changes should maintain backward compatibility until v0.4.0
- Each phase should include test updates
- CI should pass after each phase
- Performance benchmarks should be run before/after Phase 2
