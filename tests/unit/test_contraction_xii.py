"""
Regression tests for v0.5.78 Contraction XII — full cleanup pass.

Verifies:
- Rule 1 violations removed from attention/dispatch (10 methods)
- Rule 1 violation removed from precision/quantization/engine (_apply_bf16)
- TTLCache removed from utils.__all__
- Rule 1 violations removed from benchmarks/claim_benchmarks (add, benchmarks, to_json, save)
- Rule 2 violations deleted from claim_registry (3 benchmarks removed, 1 fixed)
- Attention dispatch benchmark has pre-created tensors (not inside timed loops)
- Top-level benchmarks/ directory deleted
- Broken demo files deleted
"""

import os

# ---------------------------------------------------------------------------
# Track 1a — attention/dispatch/benchmark_cache.py
# ---------------------------------------------------------------------------


class TestBenchmarkCacheRemovedMethods:
    """get_cached_latency and _make_key must be deleted."""

    def test_no_get_cached_latency(self):
        from torchbridge.attention.dispatch.benchmark_cache import KernelBenchmarkCache

        assert not hasattr(KernelBenchmarkCache, "get_cached_latency"), (
            "get_cached_latency() is a one-liner getter (Rule 1) — must be deleted"
        )

    def test_no_make_key(self):
        from torchbridge.attention.dispatch.benchmark_cache import KernelBenchmarkCache

        assert not hasattr(KernelBenchmarkCache, "_make_key"), (
            "_make_key() returns a single f-string (Rule 1) — must be deleted/inlined"
        )

    def test_run_benchmark_still_exists(self):
        """Core benchmark logic must be preserved."""
        from torchbridge.attention.dispatch.benchmark_cache import KernelBenchmarkCache

        assert hasattr(KernelBenchmarkCache, "run_benchmark")

    def test_warm_cache_still_exists(self):
        from torchbridge.attention.dispatch.benchmark_cache import KernelBenchmarkCache

        assert hasattr(KernelBenchmarkCache, "warm_cache")


# ---------------------------------------------------------------------------
# Track 1b — attention/dispatch/compatibility.py
# ---------------------------------------------------------------------------


class TestCompatibilityMatrixRemovedMethods:
    """get_optimal_kernel and is_kernel_supported must be deleted."""

    def test_no_get_optimal_kernel(self):
        from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix

        assert not hasattr(AttentionDispatchMatrix, "get_optimal_kernel"), (
            "get_optimal_kernel() returns kernels[0] (Rule 1) — must be deleted"
        )

    def test_no_is_kernel_supported(self):
        from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix

        assert not hasattr(AttentionDispatchMatrix, "is_kernel_supported"), (
            "is_kernel_supported() is a one-line boolean check (Rule 1) — must be deleted"
        )

    def test_get_supported_kernels_still_exists(self):
        """Primary selection logic must be preserved."""
        from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix

        assert hasattr(AttentionDispatchMatrix, "get_supported_kernels")

    def test_get_fallback_chain_still_exists(self):
        from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix

        assert hasattr(AttentionDispatchMatrix, "get_fallback_chain")

    def test_get_supported_kernels_returns_nonempty_list_for_cpu(self):
        """After deletion, get_supported_kernels must still work for CPU."""
        from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix
        from torchbridge.core.config import HardwareBackend

        kernels = AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.CPU)
        assert len(kernels) >= 1


# ---------------------------------------------------------------------------
# Track 1c — attention/dispatch/dispatcher.py
# ---------------------------------------------------------------------------


class TestDispatcherRemovedPropertiesAndMethods:
    """backend_name, architecture_name, and 4 try/except availability methods must be deleted."""

    def test_no_backend_name_property(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert not hasattr(AttentionDispatcher, "backend_name"), (
            "backend_name returns self._backend.value (Rule 1) — must be deleted"
        )

    def test_no_architecture_name_property(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert not hasattr(AttentionDispatcher, "architecture_name"), (
            "architecture_name returns value or 'unknown' (Rule 1) — must be deleted"
        )

    def test_no_check_flex_attention(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert not hasattr(AttentionDispatcher, "_check_flex_attention"), (
            "_check_flex_attention is a try/except import wrapper (Rule 1) — consolidated"
        )

    def test_no_check_flash_attention(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert not hasattr(AttentionDispatcher, "_check_flash_attention"), (
            "_check_flash_attention is a try/except import wrapper (Rule 1) — consolidated"
        )

    def test_no_check_neuronx(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert not hasattr(AttentionDispatcher, "_check_neuronx"), (
            "_check_neuronx is a try/except import wrapper (Rule 1) — consolidated"
        )

    def test_no_check_pallas(self):
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert not hasattr(AttentionDispatcher, "_check_pallas"), (
            "_check_pallas is a try/except import wrapper (Rule 1) — consolidated"
        )

    def test_select_kernel_still_works(self):
        """Core dispatch logic must remain functional."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        dispatcher = AttentionDispatcher(use_benchmark_cache=False)
        result = dispatcher.select_kernel(seq_length=128, num_heads=4, head_dim=32)
        assert result.kernel_type is not None

    def test_check_flash_attention_ck_still_exists(self):
        """CK check has ROCm-specific logic — must NOT be deleted."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert hasattr(AttentionDispatcher, "_check_flash_attention_ck")

    def test_check_kernel_availability_still_exists(self):
        """Router method must be preserved."""
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert hasattr(AttentionDispatcher, "_check_kernel_availability")


# ---------------------------------------------------------------------------
# Track 1d — precision/quantization/engine.py
# ---------------------------------------------------------------------------


class TestQuantizationEngineRemovedMethods:
    """_apply_bf16 must be deleted; call site inlines model.to(dtype=torch.bfloat16)."""

    def test_no_apply_bf16(self):
        from torchbridge.precision.quantization.engine import QuantizationEngine

        assert not hasattr(QuantizationEngine, "_apply_bf16"), (
            "_apply_bf16 is `return model.to(dtype=torch.bfloat16)` (Rule 1) — must be inlined"
        )

    def test_quantize_still_callable(self):
        """quantize() entry point must still work."""
        from torchbridge.precision.quantization.engine import QuantizationEngine

        assert hasattr(QuantizationEngine, "quantize")

    def test_no_dead_fp4_native_import(self):
        """_apply_nvfp4 must NOT import from the deleted fp4_native module."""
        import inspect

        from torchbridge.precision.quantization.engine import QuantizationEngine

        src = inspect.getsource(QuantizationEngine._apply_nvfp4)
        assert "fp4_native" not in src, (
            "_apply_nvfp4 still imports from deleted torchbridge.precision.fp4_native — "
            "remove the dead try block"
        )


# ---------------------------------------------------------------------------
# Track 1e — utils/__init__.py
# ---------------------------------------------------------------------------


class TestUtilsExports:
    """TTLCache must be removed from __all__; LRUCache must stay."""

    def test_ttlcache_not_in_all(self):
        import torchbridge.utils as u

        assert "TTLCache" not in u.__all__, (
            "TTLCache is never used in src/ — must be removed from __all__"
        )

    def test_lrucache_still_exported(self):
        import torchbridge.utils as u

        assert "LRUCache" in u.__all__

    def test_lrucache_importable(self):
        from torchbridge.utils import LRUCache  # noqa: F401


# ---------------------------------------------------------------------------
# Track 1f — benchmarks/claim_benchmarks.py
# ---------------------------------------------------------------------------


class TestClaimBenchmarksRemovedMethods:
    """add, benchmarks property, to_json, save must be deleted from BenchmarkSuite/Report."""

    def test_no_add_method_on_suite(self):
        from torchbridge.benchmarks.claim_benchmarks import BenchmarkSuite

        assert not hasattr(BenchmarkSuite, "add"), (
            "BenchmarkSuite.add() is self._benchmarks.append() (Rule 1) — must be deleted"
        )

    def test_no_benchmarks_property_on_suite(self):
        from torchbridge.benchmarks.claim_benchmarks import BenchmarkSuite

        assert not hasattr(BenchmarkSuite, "benchmarks"), (
            "BenchmarkSuite.benchmarks returns list(self._benchmarks) (Rule 1) — must be deleted"
        )

    def test_no_to_json_on_report(self):
        from torchbridge.benchmarks.claim_benchmarks import BenchmarkReport

        assert not hasattr(BenchmarkReport, "to_json"), (
            "BenchmarkReport.to_json() is json.dumps(self.to_dict()) (Rule 1) — must be deleted"
        )

    def test_no_save_on_report(self):
        from torchbridge.benchmarks.claim_benchmarks import BenchmarkReport

        assert not hasattr(BenchmarkReport, "save"), (
            "BenchmarkReport.save() writes to_json() to file (Rule 1) — must be deleted"
        )

    def test_run_all_still_exists(self):
        from torchbridge.benchmarks.claim_benchmarks import BenchmarkSuite

        assert hasattr(BenchmarkSuite, "run_all")

    def test_to_dict_still_exists(self):
        from torchbridge.benchmarks.claim_benchmarks import BenchmarkReport

        assert hasattr(BenchmarkReport, "to_dict")

    def test_summary_still_exists(self):
        from torchbridge.benchmarks.claim_benchmarks import BenchmarkReport

        assert hasattr(BenchmarkReport, "summary")


# ---------------------------------------------------------------------------
# Track 2 — claim_registry.py Rule 2 deletions + attention dispatch fix
# ---------------------------------------------------------------------------


class TestClaimRegistryCleanup:
    """3 benchmarks deleted, 1 fixed; registry now returns exactly 2."""

    def test_get_all_claim_benchmarks_returns_two(self):
        from torchbridge.benchmarks.claim_registry import get_all_claim_benchmarks

        benchmarks = get_all_claim_benchmarks()
        assert len(benchmarks) == 2, (
            f"Expected 2 claim benchmarks (quantization + attention_dispatch), got {len(benchmarks)}"
        )

    def test_tensor_core_alignment_deleted(self):
        from torchbridge.benchmarks import claim_registry

        assert not hasattr(claim_registry, "build_tensor_core_alignment_benchmark"), (
            "build_tensor_core_alignment_benchmark claims literature speedup, not measured — delete"
        )

    def test_channels_last_deleted(self):
        from torchbridge.benchmarks import claim_registry

        assert not hasattr(claim_registry, "build_channels_last_benchmark"), (
            "build_channels_last_benchmark claims literature speedup, not measured — delete"
        )

    def test_batch_throughput_deleted(self):
        from torchbridge.benchmarks import claim_registry

        assert not hasattr(claim_registry, "build_batch_throughput_benchmark"), (
            "build_batch_throughput_benchmark measures vanilla HuggingFace, not TorchBridge — delete"
        )

    def test_quantization_benchmark_still_registered(self):
        from torchbridge.benchmarks.claim_registry import get_all_claim_benchmarks

        names = {b.name for b in get_all_claim_benchmarks()}
        assert "quantization_int8_dynamic" in names

    def test_attention_dispatch_benchmark_still_registered(self):
        from torchbridge.benchmarks.claim_registry import get_all_claim_benchmarks

        names = {b.name for b in get_all_claim_benchmarks()}
        assert "attention_dispatch_overhead" in names

    def test_attention_dispatch_benchmark_tensors_precreated(self):
        """Tensors must be created outside the timed baseline/optimized functions.

        The old design created tensors inside both functions, so tensor creation
        noise (~5ms) dominated dispatch overhead (<0.1ms), making the measurement
        invalid. The fixed version pre-creates tensors in the outer scope.
        """
        import inspect

        from torchbridge.benchmarks.claim_registry import (
            build_attention_dispatch_benchmark,
        )

        bench = build_attention_dispatch_benchmark()

        # The closure source for baseline_fn should NOT contain torch.randn
        baseline_src = inspect.getsource(bench._baseline_fn)
        assert "torch.randn" not in baseline_src, (
            "baseline_fn must not create tensors internally — pre-create in outer scope"
        )
        optimized_src = inspect.getsource(bench._optimized_fn)
        assert "torch.randn" not in optimized_src, (
            "optimized_fn must not create tensors internally — pre-create in outer scope"
        )


# ---------------------------------------------------------------------------
# Track 3 — Top-level directory deletions
# ---------------------------------------------------------------------------


class TestTopLevelDeletions:
    """Broken top-level directories and demo files must be removed."""

    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    def test_benchmarks_directory_deleted(self):
        """Top-level benchmarks/ is entirely dead — 0/10 files measure TorchBridge value."""
        path = os.path.join(self._repo_root, "benchmarks")
        assert not os.path.isdir(path), (
            "Top-level benchmarks/ must be deleted — no file measures TorchBridge value; "
            "3 files have broken imports from deleted classes"
        )

    def test_amd_backend_demo_deleted(self):
        path = os.path.join(self._repo_root, "demos", "amd_backend_demo.py")
        assert not os.path.exists(path), (
            "demos/amd_backend_demo.py imports deleted AMDConfig, AMDArchitecture — delete"
        )

    def test_nvidia_integration_demo_deleted(self):
        path = os.path.join(self._repo_root, "demos", "nvidia_integration_demo.py")
        assert not os.path.exists(path), (
            "demos/nvidia_integration_demo.py imports deleted FlashAttention3, FP8Compiler — delete"
        )

    def test_auto_backend_selection_demo_deleted(self):
        path = os.path.join(self._repo_root, "demos", "auto_backend_selection_demo.py")
        assert not os.path.exists(path), (
            "demos/auto_backend_selection_demo.py imports deleted get_manager() — delete"
        )

    def test_run_all_demos_deleted(self):
        path = os.path.join(self._repo_root, "demos", "run_all_demos.py")
        assert not os.path.exists(path), (
            "demos/run_all_demos.py is a dead incomplete orchestrator — delete"
        )

    def test_rebuild_deck_script_deleted(self):
        path = os.path.join(self._repo_root, "scripts", "rebuild_deck_pass2.py")
        assert not os.path.exists(path), (
            "scripts/rebuild_deck_pass2.py was a one-time v0.5.72 task — delete"
        )

    def test_update_deck_script_deleted(self):
        path = os.path.join(self._repo_root, "scripts", "update_deck_v0572.py")
        assert not os.path.exists(path), (
            "scripts/update_deck_v0572.py was a one-time v0.5.72 task — delete"
        )

    def test_broken_benchmark_suite_script_deleted(self):
        path = os.path.join(self._repo_root, "scripts", "benchmarks", "benchmark_suite.py")
        assert not os.path.exists(path), (
            "scripts/benchmarks/benchmark_suite.py imports deleted LLMConfig, LLMOptimizer — delete"
        )

    def test_regression_tests_for_deleted_benchmarks_framework_deleted(self):
        """tests/regression/ tested the deleted benchmarks/ framework internals."""
        for fname in (
            "test_baseline_manager.py",
            "test_regression_detector.py",
            "test_threshold_manager.py",
        ):
            path = os.path.join(self._repo_root, "tests", "regression", fname)
            assert not os.path.exists(path), (
                f"tests/regression/{fname} imports deleted benchmarks.regression module — delete"
            )

    def test_benchmark_tests_for_deleted_cli_performance_benchmark_deleted(self):
        """tests/benchmark/test_cli_benchmarks.py tested the deleted benchmarks module."""
        path = os.path.join(self._repo_root, "tests", "benchmark", "test_cli_benchmarks.py")
        assert not os.path.exists(path), (
            "tests/benchmark/test_cli_benchmarks.py imports deleted benchmarks.cli_performance_benchmark — delete"
        )
