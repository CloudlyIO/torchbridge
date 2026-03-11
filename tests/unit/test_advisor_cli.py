"""Tests for the advisor CLI command."""

import json

from torchbridge.cli import main as cli_main


class TestAdvisorCLI:
    """Tests for the advisor CLI command."""

    def test_advisor_help(self):
        result = cli_main(["advisor", "--help"])
        # argparse prints help and exits with 0
        assert result == 0

    def test_advisor_basic(self):
        result = cli_main([
            "advisor",
            "--model-params", "7e9",
            "--world-size", "8",
            "--backend", "nvidia",
        ])
        assert result == 0

    def test_advisor_ci_json(self, capsys):
        result = cli_main([
            "advisor",
            "--model-params", "7e9",
            "--world-size", "4",
            "--backend", "cpu",
            "--ci",
        ])
        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "recommendation" in data
        assert data["model_params"] == int(7e9)
        assert data["world_size"] == 4

    def test_advisor_toml_output(self, capsys):
        result = cli_main([
            "advisor",
            "--model-params", "1e9",
            "--world-size", "1",
            "--backend", "cpu",
            "--toml",
        ])
        assert result == 0
        output = capsys.readouterr().out
        assert "[fsdp]" in output
        assert "[pipeline]" in output

    def test_advisor_topology(self, capsys):
        result = cli_main([
            "advisor",
            "--model-params", "1e9",
            "--topology",
            "--backend", "cpu",
        ])
        assert result == 0
        output = capsys.readouterr().out
        assert "Topology" in output or "World size" in output

    def test_advisor_topology_ci(self, capsys):
        result = cli_main([
            "advisor",
            "--model-params", "1e9",
            "--topology",
            "--ci",
            "--backend", "cpu",
        ])
        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "mesh" in data

    def test_advisor_large_model(self):
        result = cli_main([
            "advisor",
            "--model-params", "70e9",
            "--world-size", "16",
            "--gpus-per-node", "8",
            "--backend", "nvidia",
        ])
        assert result == 0

    def test_advisor_single_gpu(self):
        result = cli_main([
            "advisor",
            "--model-params", "1e9",
            "--world-size", "1",
            "--backend", "cpu",
        ])
        assert result == 0

    def test_advisor_amd_backend(self):
        result = cli_main([
            "advisor",
            "--model-params", "7e9",
            "--world-size", "4",
            "--backend", "amd",
        ])
        assert result == 0

    def test_advisor_trainium_backend(self):
        result = cli_main([
            "advisor",
            "--model-params", "7e9",
            "--world-size", "8",
            "--backend", "trainium",
        ])
        assert result == 0

    def test_advisor_tpu_backend(self):
        result = cli_main([
            "advisor",
            "--model-params", "7e9",
            "--world-size", "8",
            "--backend", "tpu",
        ])
        assert result == 0

    def test_advisor_missing_model_params(self):
        result = cli_main(["advisor"])
        assert result != 0


# ── v0.5.69: rationale in recommendation output ──────────────────────────────

class TestAdvisorRationale:
    def test_recommendation_notes_include_tp_rationale(self, capsys):
        from torchbridge.core.config import HardwareBackend
        from torchbridge.distributed.config import recommend_parallelism
        rec = recommend_parallelism(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            world_size=4,
        )
        combined = " ".join(rec.notes)
        # Should explain why TP=N was chosen (or not chosen)
        assert "TP=" in combined

    def test_recommendation_notes_include_pp_rationale(self):
        from torchbridge.core.config import HardwareBackend
        from torchbridge.distributed.config import recommend_parallelism
        rec = recommend_parallelism(
            model_params=int(7e9),
            backend=HardwareBackend.CUDA,
            world_size=4,
        )
        combined = " ".join(rec.notes)
        # Should explain PP decision
        assert "PP=" in combined

    def test_small_model_tp1_pp1_rationale_present(self):
        from torchbridge.core.config import HardwareBackend
        from torchbridge.distributed.config import recommend_parallelism
        # 1B model, 2 GPUs — should be TP=1, PP=1 with explanation
        rec = recommend_parallelism(
            model_params=int(1e9),
            backend=HardwareBackend.CUDA,
            world_size=2,
        )
        combined = " ".join(rec.notes)
        assert "TP=1" in combined
        assert "PP=1" in combined
