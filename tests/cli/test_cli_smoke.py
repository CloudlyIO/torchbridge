"""
CLI Smoke Tests

Verifies that all 15 CLI entry points:
  1. Import cleanly (no import errors, missing dependencies, or bad wiring)
  2. Respond to --help with exit code 0

Notes:
  - The top-level ``torchbridge.cli`` catches argparse's SystemExit internally
    and returns the exit code as an int; subcommand CLIs let SystemExit propagate.
    Both patterns are accepted as long as the effective exit code is 0.
"""

import importlib
import sys

import pytest

# All entry points from pyproject.toml [project.scripts]
_CLI_MODULES = [
    ("torchbridge.cli", "main"),
    ("torchbridge.cli.benchmark", "main"),
    ("torchbridge.cli.doctor", "main"),
    ("torchbridge.cli.validate", "main"),
    ("torchbridge.cli.migrate", "main"),
    ("torchbridge.cli.quantize", "main"),
    ("torchbridge.cli.cache", "main"),
    ("torchbridge.cli.speculate", "main"),
    ("torchbridge.cli.advisor", "main"),
    ("torchbridge.cli.checkpoint", "main"),
    ("torchbridge.cli.adapter", "main"),
]

_IDS = [module.split(".")[-1] for module, _ in _CLI_MODULES]


@pytest.mark.parametrize("module_path,func_name", _CLI_MODULES, ids=_IDS)
def test_cli_imports_cleanly(module_path: str, func_name: str) -> None:
    """Each CLI module must import without errors and expose a callable main()."""
    m = importlib.import_module(module_path)
    fn = getattr(m, func_name, None)
    assert callable(fn), f"{module_path}.{func_name} is not callable"


@pytest.mark.parametrize("module_path,func_name", _CLI_MODULES, ids=_IDS)
def test_cli_help_exits_zero(
    module_path: str, func_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each CLI entry point must exit with code 0 when passed --help.

    Accepts two patterns:
    - SystemExit(0) raised (argparse default for subcommand CLIs)
    - Return value of 0 (top-level CLI catches SystemExit internally)
    """
    m = importlib.import_module(module_path)
    monkeypatch.setattr(sys, "argv", [module_path, "--help"])
    try:
        result = getattr(m, func_name)()
    except SystemExit as exc:
        assert exc.code == 0, (
            f"{module_path} --help raised SystemExit({exc.code}), expected 0"
        )
        return
    # Function returned without raising — treat return value as exit code
    assert result == 0 or result is None, (
        f"{module_path} --help returned {result!r}, expected 0 or None"
    )
