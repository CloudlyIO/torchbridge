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
        assert "[fsdp2]" in output
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
