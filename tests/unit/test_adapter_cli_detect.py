"""
Tests for tb-adapter detect CLI Subcommand

Tests the detect subcommand of the adapter CLI for model family detection.
"""

import json
from unittest.mock import MagicMock, patch

from torchbridge.cli.adapter import main


class _FakeAutoConfig:
    """Fake transformers AutoConfig."""

    def __init__(self, model_type: str = "llama"):
        self.model_type = model_type

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        """Return a config based on model name."""
        if "qwen" in name.lower():
            return cls(model_type="qwen3")
        if "falcon" in name.lower():
            return cls(model_type="falcon")
        if "llama" in name.lower():
            return cls(model_type="llama")
        return cls(model_type="unknown_arch")


class TestAdapterDetectCLI:
    """Tests for tb-adapter detect subcommand."""

    @patch("torchbridge.cli.adapter.sys")
    def test_detect_qwen_human(self, mock_sys, capsys):
        """detect --model qwen should show qwen family."""
        mock_sys.stderr = MagicMock()
        with patch.dict("sys.modules", {
            "transformers": MagicMock(AutoConfig=_FakeAutoConfig),
        }):
            # Re-import to pick up mock
            import importlib

            import torchbridge.cli.adapter as adapter_mod
            importlib.reload(adapter_mod)

            ret = adapter_mod.main(["detect", "--model", "Qwen/Qwen3-0.6B"])

        captured = capsys.readouterr()
        assert ret == 0
        assert "qwen" in captured.out.lower()

    @patch("torchbridge.cli.adapter.sys")
    def test_detect_qwen_ci_json(self, mock_sys, capsys):
        """detect --model qwen --ci should output valid JSON."""
        mock_sys.stderr = MagicMock()
        mock_sys.stdout = MagicMock()

        with patch.dict("sys.modules", {
            "transformers": MagicMock(AutoConfig=_FakeAutoConfig),
        }):
            import importlib

            import torchbridge.cli.adapter as adapter_mod
            importlib.reload(adapter_mod)

            # Capture json.dump output
            written = []
            mock_sys.stdout.write = lambda x: written.append(x)

            ret = adapter_mod.main(["detect", "--model", "Qwen/Qwen3-0.6B", "--ci"])

        assert ret == 0
        # json.dump writes to sys.stdout, collect what was written
        output = "".join(str(w) for w in written if isinstance(w, str))
        if output:
            data = json.loads(output)
            assert data["family"] == "qwen"
            assert "q_proj" in data["target_modules"]

    def test_detect_requires_model_arg(self):
        """detect without --model should fail."""
        import pytest

        with pytest.raises(SystemExit):
            main(["detect"])

    def test_no_subcommand_shows_help(self, capsys):
        """Running with no subcommand should show help and return 1."""
        ret = main([])
        assert ret == 1

    def test_recommend_still_works(self, capsys):
        """recommend subcommand should still work after adding detect."""
        ret = main(["recommend", "--backend", "cpu"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Adapter Method" in captured.out
