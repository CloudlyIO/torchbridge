"""
Integration tests for the disaggregated fleet advisor CLI pipeline.

Covers:
  - CLI argument registration (--mode, --prefill, --decode, --prefill-memory, --decode-memory)
  - Backward compatibility: default mode is training
  - Guards: disaggregated requires both --prefill and --decode
  - Smoke runs: nvidia:hopper + amd:cdna3, cpu+cpu
  - CI JSON schema and field presence
  - Memory override accepted
  - Training mode unchanged
"""

import json
import sys
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(**kwargs):
    """Build a SimpleNamespace with advisor defaults, overriding with kwargs."""
    defaults = {
        "model_params": 7e9,
        "world_size": 1,
        "gpus_per_node": None,
        "backend": "auto",
        "ci": False,
        "toml": False,
        "topology": False,
        "mode": "training",
        "prefill": None,
        "decode": None,
        "prefill_memory": None,
        "decode_memory": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------

class TestArgRegistration:
    def _make_parser(self):
        import argparse

        from torchbridge.cli.advisor import AdvisorCommand

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        AdvisorCommand.register(sub)
        return parser

    def test_mode_arg_registered(self):
        parser = self._make_parser()
        parsed = parser.parse_args(["advisor", "--model-params", "7e9", "--mode", "training"])
        assert parsed.mode == "training"

    def test_prefill_arg_registered(self):
        parser = self._make_parser()
        parsed = parser.parse_args(
            ["advisor", "--model-params", "7e9", "--mode", "disaggregated",
             "--prefill", "nvidia:hopper", "--decode", "amd:cdna3"]
        )
        assert parsed.prefill == "nvidia:hopper"

    def test_decode_arg_registered(self):
        parser = self._make_parser()
        parsed = parser.parse_args(
            ["advisor", "--model-params", "7e9", "--mode", "disaggregated",
             "--prefill", "nvidia:hopper", "--decode", "amd:cdna3"]
        )
        assert parsed.decode == "amd:cdna3"

    def test_prefill_memory_arg_registered(self):
        parser = self._make_parser()
        parsed = parser.parse_args(
            ["advisor", "--model-params", "7e9", "--mode", "disaggregated",
             "--prefill", "nvidia:hopper", "--decode", "amd:cdna3",
             "--prefill-memory", "80"]
        )
        assert parsed.prefill_memory == 80.0

    def test_decode_memory_arg_registered(self):
        parser = self._make_parser()
        parsed = parser.parse_args(
            ["advisor", "--model-params", "7e9", "--mode", "disaggregated",
             "--prefill", "nvidia:hopper", "--decode", "amd:cdna3",
             "--decode-memory", "192"]
        )
        assert parsed.decode_memory == 192.0

    def test_default_mode_is_training(self):
        parser = self._make_parser()
        parsed = parser.parse_args(["advisor", "--model-params", "7e9"])
        assert parsed.mode == "training"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestDisaggregatedGuards:
    def test_disaggregated_missing_prefill_returns_error(self):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(mode="disaggregated", prefill=None, decode="amd:cdna3")
        rc = AdvisorCommand.execute(args)
        assert rc == 1

    def test_disaggregated_missing_decode_returns_error(self):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(mode="disaggregated", prefill="nvidia:hopper", decode=None)
        rc = AdvisorCommand.execute(args)
        assert rc == 1

    def test_disaggregated_missing_both_returns_error(self):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(mode="disaggregated", prefill=None, decode=None)
        rc = AdvisorCommand.execute(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# Smoke runs
# ---------------------------------------------------------------------------

class TestSmokeRuns:
    def test_nvidia_amd_exits_zero(self, capsys):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(
            mode="disaggregated",
            prefill="nvidia:hopper",
            decode="amd:cdna3",
        )
        rc = AdvisorCommand.execute(args)
        assert rc == 0

    def test_cpu_cpu_exits_zero(self, capsys):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(mode="disaggregated", prefill="cpu", decode="cpu")
        rc = AdvisorCommand.execute(args)
        assert rc == 0

    def test_amd_nvidia_exits_zero(self, capsys):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(
            mode="disaggregated",
            prefill="amd:cdna3",
            decode="nvidia:ampere",
        )
        rc = AdvisorCommand.execute(args)
        assert rc == 0

    def test_no_arch_spec_exits_zero(self, capsys):
        from torchbridge.cli.advisor import AdvisorCommand

        # Just backend, no arch
        args = _args(mode="disaggregated", prefill="nvidia", decode="amd")
        rc = AdvisorCommand.execute(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# CI JSON
# ---------------------------------------------------------------------------

class TestCiJson:
    def _run_ci(self, **kwargs):
        import io

        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(ci=True, **kwargs)
        buf = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buf
        try:
            rc = AdvisorCommand.execute(args)
        finally:
            sys.stdout = original_stdout

        output = buf.getvalue()
        return rc, output

    def test_ci_json_schema_valid(self):
        rc, output = self._run_ci(
            mode="disaggregated",
            prefill="nvidia:hopper",
            decode="amd:cdna3",
        )
        assert rc == 0
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_ci_json_has_prefill_and_decode(self):
        rc, output = self._run_ci(
            mode="disaggregated",
            prefill="nvidia:hopper",
            decode="amd:cdna3",
        )
        parsed = json.loads(output)
        assert "prefill" in parsed
        assert "decode" in parsed
        assert "kv_transfer_format" in parsed
        assert "model_params" in parsed

    def test_ci_json_prefill_has_required_fields(self):
        rc, output = self._run_ci(
            mode="disaggregated",
            prefill="nvidia:hopper",
            decode="rocm:cdna3",
        )
        parsed = json.loads(output)
        prefill = parsed["prefill"]
        for field in ("role", "backend", "kv_dtype", "kv_cache_budget_gb",
                      "max_batch_size", "max_seq_len"):
            assert field in prefill, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Memory override
# ---------------------------------------------------------------------------

class TestMemoryOverride:
    def test_prefill_memory_override_accepted(self):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(
            mode="disaggregated",
            prefill="nvidia:hopper",
            decode="amd:cdna3",
            prefill_memory=160.0,
            ci=True,
        )
        import io
        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            rc = AdvisorCommand.execute(args)
        finally:
            sys.stdout = orig

        assert rc == 0
        parsed = json.loads(buf.getvalue())
        # 160 GB × 20% (prefill KV fraction) = 32 GB
        assert abs(parsed["prefill"]["kv_cache_budget_gb"] - 32.0) < 0.5

    def test_decode_memory_override_accepted(self):
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(
            mode="disaggregated",
            prefill="nvidia:hopper",
            decode="amd:cdna3",
            decode_memory=192.0,
            ci=True,
        )
        import io
        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            rc = AdvisorCommand.execute(args)
        finally:
            sys.stdout = orig

        assert rc == 0
        parsed = json.loads(buf.getvalue())
        # 192 GB × 80% (decode KV fraction) = 153.6 GB
        assert abs(parsed["decode"]["kv_cache_budget_gb"] - 153.6) < 1.0


# ---------------------------------------------------------------------------
# Training mode backward compat
# ---------------------------------------------------------------------------

class TestTrainingModeUnchanged:
    def test_training_mode_still_works(self, capsys):
        """Default training mode must not be affected by disaggregated changes."""
        from torchbridge.cli.advisor import AdvisorCommand

        args = _args(mode="training", model_params=7e9, world_size=8, backend="cpu")
        rc = AdvisorCommand.execute(args)
        # Training path hits recommend_parallelism → exits 0 on CPU
        assert rc == 0

    def test_default_mode_routes_to_training(self, capsys):
        from torchbridge.cli.advisor import AdvisorCommand

        # No --mode → defaults to "training"
        args = _args(model_params=7e9, world_size=4, backend="cpu")
        assert args.mode == "training"
        rc = AdvisorCommand.execute(args)
        assert rc == 0
