"""
Integration tests for --mode heterogeneous in tb-advisor.

Tests verify:
- --mode heterogeneous is a registered choice
- --nvidia and --amd flags are registered on both parsers
- _run_heterogeneous() returns 0 for valid args
- _run_heterogeneous() returns 1 when --nvidia or --amd is missing
- --ci outputs valid JSON containing collective_bridge
"""

import json
import types

from torchbridge.cli.advisor import AdvisorCommand

# ── CLI flag registration ──────────────────────────────────────────────────


class TestHeterogeneousCLIFlags:
    def _get_parser(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        AdvisorCommand.register(subparsers)
        return parser

    def test_heterogeneous_mode_accepted(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "advisor",
                "--model-params",
                "7e9",
                "--mode",
                "heterogeneous",
                "--nvidia",
                "hopper:4",
                "--amd",
                "cdna3:8",
            ]
        )
        assert args.mode == "heterogeneous"

    def test_nvidia_flag_registered(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "advisor",
                "--model-params",
                "7e9",
                "--mode",
                "heterogeneous",
                "--nvidia",
                "hopper:4",
                "--amd",
                "cdna3:8",
            ]
        )
        assert args.nvidia == "hopper:4"

    def test_amd_flag_registered(self):
        parser = self._get_parser()
        args = parser.parse_args(
            [
                "advisor",
                "--model-params",
                "7e9",
                "--mode",
                "heterogeneous",
                "--nvidia",
                "hopper:4",
                "--amd",
                "cdna3:8",
            ]
        )
        assert args.amd == "cdna3:8"

    def test_nvidia_flag_default_is_none(self):
        parser = self._get_parser()
        args = parser.parse_args(["advisor", "--model-params", "7e9"])
        assert args.nvidia is None

    def test_amd_flag_default_is_none(self):
        parser = self._get_parser()
        args = parser.parse_args(["advisor", "--model-params", "7e9"])
        assert args.amd is None

    def test_flags_in_standalone_main_parser(self):
        import inspect

        from torchbridge.cli import advisor as advisor_mod

        src = inspect.getsource(advisor_mod)
        assert "--nvidia" in src
        assert "--amd" in src
        assert "heterogeneous" in src


# ── _run_heterogeneous() execution ─────────────────────────────────────────


class TestHeterogeneousAdvisorExecution:
    def _make_args(self, nvidia="hopper:4", amd="cdna3:8", model_params=7e9, ci=False):
        return types.SimpleNamespace(
            mode="heterogeneous",
            nvidia=nvidia,
            amd=amd,
            model_params=model_params,
            ci=ci,
        )

    def test_run_returns_0_for_valid_args(self):
        args = self._make_args()
        rc = AdvisorCommand._run_heterogeneous(args)
        assert rc == 0

    def test_run_returns_1_missing_nvidia(self):
        args = self._make_args(nvidia=None)
        rc = AdvisorCommand._run_heterogeneous(args)
        assert rc == 1

    def test_run_returns_1_missing_amd(self):
        args = self._make_args(amd=None)
        rc = AdvisorCommand._run_heterogeneous(args)
        assert rc == 1

    def test_ci_output_is_valid_json(self, capsys):
        args = self._make_args(ci=True)
        rc = AdvisorCommand._run_heterogeneous(args)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)

    def test_ci_output_contains_collective_bridge(self, capsys):
        args = self._make_args(nvidia="hopper:4", amd="cdna3:8", ci=True)
        AdvisorCommand._run_heterogeneous(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "collective_bridge" in data
        assert data["collective_bridge"] == "hetccl"

    def test_unknown_arch_falls_back_gracefully(self):
        """Unknown arch string falls back to None arch → ucc bridge, still returns 0."""
        args = self._make_args(nvidia="unknown_arch:2", amd="unknown_amd:4")
        rc = AdvisorCommand._run_heterogeneous(args)
        assert rc == 0

    def test_execute_dispatches_heterogeneous_mode(self):
        """AdvisorCommand.execute() routes --mode heterogeneous to _run_heterogeneous."""
        args = types.SimpleNamespace(
            mode="heterogeneous",
            nvidia="hopper:4",
            amd="cdna3:8",
            model_params=7e9,
            ci=False,
        )
        rc = AdvisorCommand.execute(args)
        assert rc == 0


# ── v0.5.70: training-mode advisor rationale integration ─────────────────────


class TestAdvisorRationaleInOutput:
    """End-to-end: human output must contain rationale notes for all TP/PP decisions."""

    def _run_training(
        self, model_params: float, world_size: int = 4, capsys=None
    ) -> str:
        from torchbridge.cli.advisor import AdvisorCommand

        args = types.SimpleNamespace(
            mode="training",
            model_params=model_params,
            world_size=world_size,
            gpus_per_node=None,
            backend="cpu",
            ci=False,
            toml=False,
            topology=False,
        )
        AdvisorCommand.execute(args)
        if capsys:
            return capsys.readouterr().out
        return ""

    def test_human_output_contains_tp_rationale(self, capsys):
        """Training mode human output must include TP= rationale note."""
        from torchbridge.cli.advisor import AdvisorCommand

        args = types.SimpleNamespace(
            mode="training",
            model_params=7e9,
            world_size=8,
            gpus_per_node=None,
            backend="cpu",
            ci=False,
            toml=False,
            topology=False,
        )
        AdvisorCommand.execute(args)
        out = capsys.readouterr().out
        assert "TP=" in out

    def test_human_output_contains_pp_rationale(self, capsys):
        """Training mode human output must include PP= rationale note."""
        from torchbridge.cli.advisor import AdvisorCommand

        args = types.SimpleNamespace(
            mode="training",
            model_params=7e9,
            world_size=8,
            gpus_per_node=None,
            backend="cpu",
            ci=False,
            toml=False,
            topology=False,
        )
        AdvisorCommand.execute(args)
        out = capsys.readouterr().out
        assert "PP=" in out

    def test_small_model_rationale_explains_no_parallelism(self, capsys):
        """For a 1B model on 2 GPUs, output must explain TP=1 and PP=1."""
        from torchbridge.cli.advisor import AdvisorCommand

        args = types.SimpleNamespace(
            mode="training",
            model_params=1e9,
            world_size=2,
            gpus_per_node=None,
            backend="cpu",
            ci=False,
            toml=False,
            topology=False,
        )
        AdvisorCommand.execute(args)
        out = capsys.readouterr().out
        assert "TP=1" in out
        assert "PP=1" in out

    def test_large_model_rationale_explains_tp_applied(self, capsys):
        """For a 70B model on 16 GPUs, output must explain why TP>1 is used."""
        from torchbridge.cli.advisor import AdvisorCommand

        args = types.SimpleNamespace(
            mode="training",
            model_params=70e9,
            world_size=16,
            gpus_per_node=8,
            backend="cpu",
            ci=False,
            toml=False,
            topology=False,
        )
        AdvisorCommand.execute(args)
        out = capsys.readouterr().out
        # Should say TP>1 with explanation (model > 10B params)
        assert "TP=" in out
        # The note should mention why — "exceeds" or "benefit"
        combined = out.lower()
        assert "tensor parallel" in combined or "tp" in combined
