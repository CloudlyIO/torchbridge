"""
Tests for the tb-migrate CLI command.
"""

import argparse
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from torchbridge.cli.migrate import (
    MIGRATION_PATTERNS,
    MigrateCommand,
    MigrationReport,
    MigrationSuggestion,
    main,
)


class TestMigrationSuggestion:
    """Test MigrationSuggestion dataclass."""

    def test_suggestion_creation(self):
        """Test creating a MigrationSuggestion."""
        suggestion = MigrationSuggestion(
            file="test.py",
            line=10,
            pattern_name="cuda_is_available",
            matched_text="torch.cuda.is_available()",
            suggestion="Use torchbridge.backends.detect_best_backend()",
            context="if torch.cuda.is_available():",
        )
        assert suggestion.file == "test.py"
        assert suggestion.line == 10
        assert suggestion.pattern_name == "cuda_is_available"
        assert suggestion.matched_text == "torch.cuda.is_available()"
        assert suggestion.suggestion == "Use torchbridge.backends.detect_best_backend()"
        assert suggestion.context == "if torch.cuda.is_available():"


class TestMigrationReport:
    """Test MigrationReport dataclass."""

    def test_report_creation(self):
        """Test creating a MigrationReport."""
        suggestions = [
            MigrationSuggestion("a.py", 1, "p1", "m1", "s1", "c1"),
            MigrationSuggestion("a.py", 5, "p2", "m2", "s2", "c2"),
            MigrationSuggestion("b.py", 3, "p1", "m3", "s3", "c3"),
        ]
        report = MigrationReport(
            path="/some/path",
            suggestions=suggestions,
            files_scanned=5,
            files_with_suggestions=2,
        )
        assert report.path == "/some/path"
        assert report.files_scanned == 5
        assert report.files_with_suggestions == 2
        assert len(report.suggestions) == 3

    def test_total_suggestions_property(self):
        """Test the total_suggestions computed property."""
        suggestions = [
            MigrationSuggestion("a.py", 1, "p1", "m1", "s1", "c1"),
            MigrationSuggestion("b.py", 2, "p2", "m2", "s2", "c2"),
        ]
        report = MigrationReport(
            path="/path",
            suggestions=suggestions,
            files_scanned=3,
            files_with_suggestions=2,
        )
        assert report.total_suggestions == 2

    def test_total_suggestions_empty(self):
        """Test total_suggestions with no suggestions."""
        report = MigrationReport(
            path="/path",
            suggestions=[],
            files_scanned=3,
            files_with_suggestions=0,
        )
        assert report.total_suggestions == 0


class TestMigrationPatterns:
    """Test the MIGRATION_PATTERNS list."""

    def test_patterns_is_list(self):
        """Test that MIGRATION_PATTERNS is a non-empty list."""
        assert isinstance(MIGRATION_PATTERNS, list)
        assert len(MIGRATION_PATTERNS) > 0

    def test_patterns_structure(self):
        """Test that each pattern is a (name, regex, suggestion) tuple."""
        for pattern in MIGRATION_PATTERNS:
            assert len(pattern) == 3, f"Pattern should have 3 elements: {pattern}"
            name, regex, suggestion = pattern
            assert isinstance(name, str)
            assert isinstance(suggestion, str)


class TestPatternDetection:
    """Test CUDA pattern detection for each pattern type."""

    def test_detect_cuda_is_available(self, tmp_path):
        """Test detection of torch.cuda.is_available()."""
        f = tmp_path / "test.py"
        f.write_text("if torch.cuda.is_available():\n    pass\n")
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 1
        assert any("is_available" in s.pattern_name for s in suggestions)

    def test_detect_dot_cuda(self, tmp_path):
        """Test detection of .cuda() calls."""
        f = tmp_path / "test.py"
        f.write_text("model = model.cuda()\n")
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 1
        assert any(".cuda()" in s.pattern_name for s in suggestions)

    def test_detect_cuda_device(self, tmp_path):
        """Test detection of torch.device('cuda...')."""
        f = tmp_path / "test.py"
        f.write_text('device = torch.device("cuda:0")\n')
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 1

    def test_detect_nccl_hardcoded(self, tmp_path):
        """Test detection of hardcoded 'nccl'."""
        f = tmp_path / "test.py"
        f.write_text('dist.init_process_group(backend="nccl")\n')
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 1

    def test_detect_cuda_amp(self, tmp_path):
        """Test detection of torch.cuda.amp."""
        f = tmp_path / "test.py"
        f.write_text("from torch.cuda.amp import autocast\n")
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 1

    def test_detect_cuda_synchronize(self, tmp_path):
        """Test detection of torch.cuda.synchronize."""
        f = tmp_path / "test.py"
        f.write_text("torch.cuda.synchronize()\n")
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 1

    def test_detect_memory_allocated(self, tmp_path):
        """Test detection of torch.cuda.memory_allocated."""
        f = tmp_path / "test.py"
        f.write_text("mem = torch.cuda.memory_allocated()\n")
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 1

    def test_multiple_patterns_in_one_file(self, tmp_path):
        """Test detecting multiple patterns in a single file."""
        f = tmp_path / "multi.py"
        f.write_text(
            "if torch.cuda.is_available():\n"
            "    model = model.cuda()\n"
            "    torch.cuda.synchronize()\n"
        )
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) >= 3


class TestCleanFile:
    """Test that clean files produce no suggestions."""

    def test_clean_file_no_suggestions(self, tmp_path):
        """Test that a file without CUDA patterns has no suggestions."""
        f = tmp_path / "clean.py"
        f.write_text(
            "import torch\n"
            "from torchbridge.backends import detect_best_backend\n"
            "backend = detect_best_backend()\n"
            "model = model.to(backend.device)\n"
        )
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) == 0

    def test_comment_lines_skipped(self, tmp_path):
        """Test that commented-out CUDA code is skipped."""
        f = tmp_path / "commented.py"
        f.write_text("# model = model.cuda()\n# torch.cuda.synchronize()\n")
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) == 0

    def test_empty_file(self, tmp_path):
        """Test scanning an empty file."""
        f = tmp_path / "empty.py"
        f.write_text("")
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) == 0

    def test_non_python_content(self, tmp_path):
        """Test scanning a file with no CUDA references."""
        f = tmp_path / "plain.py"
        f.write_text(
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "result = add(1, 2)\n"
            "print(result)\n"
        )
        suggestions = MigrateCommand._scan_file(str(f), verbose=False)
        assert len(suggestions) == 0


class TestOutputFormats:
    """Test output format options."""

    def test_json_output(self, tmp_path):
        """Test JSON output format."""
        f = tmp_path / "test.py"
        f.write_text("model = model.cuda()\n")

        args = argparse.Namespace(
            path=str(f), output=None, format='json',
            ci=False, verbose=False, exclude=[],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            MigrateCommand.execute(args)
        output = buf.getvalue()
        data = json.loads(output)
        assert "suggestions" in data
        assert data["files_scanned"] == 1

    def test_ci_mode_exit_code(self, tmp_path):
        """Test CI mode returns exit code 1 when suggestions found."""
        f = tmp_path / "test.py"
        f.write_text("model = model.cuda()\n")

        args = argparse.Namespace(
            path=str(f), output=None, format='markdown',
            ci=True, verbose=False, exclude=[],
        )
        result = MigrateCommand.execute(args)
        assert result == 1

    def test_ci_mode_clean_exit(self, tmp_path):
        """Test CI mode returns 0 for clean files."""
        f = tmp_path / "clean.py"
        f.write_text("import torch\nmodel = torch.nn.Linear(10, 5)\n")

        args = argparse.Namespace(
            path=str(f), output=None, format='markdown',
            ci=False, verbose=False, exclude=[],
        )
        result = MigrateCommand.execute(args)
        assert result == 0

    def test_json_output_structure(self, tmp_path):
        """Test JSON output contains all expected fields."""
        f = tmp_path / "test.py"
        f.write_text("torch.cuda.synchronize()\n")

        args = argparse.Namespace(
            path=str(f), output=None, format='json',
            ci=False, verbose=False, exclude=[],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            MigrateCommand.execute(args)
        data = json.loads(buf.getvalue())
        assert "path" in data
        assert "files_scanned" in data
        assert "files_with_suggestions" in data
        assert "total_suggestions" in data
        assert "suggestions" in data

    def test_output_to_file(self, tmp_path):
        """Test writing output to a file."""
        src = tmp_path / "test.py"
        src.write_text("model = model.cuda()\n")
        out = tmp_path / "report.json"

        args = argparse.Namespace(
            path=str(src), output=str(out), format='json',
            ci=False, verbose=False, exclude=[],
        )
        MigrateCommand.execute(args)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["files_scanned"] == 1


class TestDirectoryScanning:
    """Test directory scanning."""

    def test_scan_directory(self, tmp_path):
        """Test scanning a directory of files."""
        (tmp_path / "a.py").write_text("model.cuda()\n")
        (tmp_path / "b.py").write_text("import os\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("torch.cuda.synchronize()\n")

        args = argparse.Namespace(
            path=str(tmp_path), output=None, format='json',
            ci=False, verbose=False, exclude=[],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            MigrateCommand.execute(args)
        data = json.loads(buf.getvalue())
        assert data["files_scanned"] == 3
        assert data["total_suggestions"] >= 2

    def test_scan_directory_with_exclude(self, tmp_path):
        """Test scanning a directory with excluded paths."""
        (tmp_path / "a.py").write_text("model.cuda()\n")
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "lib.py").write_text("model.cuda()\n")

        args = argparse.Namespace(
            path=str(tmp_path), output=None, format='json',
            ci=False, verbose=False, exclude=["venv"],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            MigrateCommand.execute(args)
        data = json.loads(buf.getvalue())
        assert data["files_scanned"] == 1

    def test_scan_nonexistent_path(self):
        """Test scanning a nonexistent path."""
        args = argparse.Namespace(
            path="/nonexistent/path", output=None, format='json',
            ci=False, verbose=False, exclude=[],
        )
        result = MigrateCommand.execute(args)
        assert result == 1

    def test_scan_only_python_files(self, tmp_path):
        """Test that only .py files are scanned."""
        (tmp_path / "code.py").write_text("model.cuda()\n")
        (tmp_path / "readme.md").write_text("model.cuda()\n")
        (tmp_path / "data.txt").write_text("model.cuda()\n")

        args = argparse.Namespace(
            path=str(tmp_path), output=None, format='json',
            ci=False, verbose=False, exclude=[],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            MigrateCommand.execute(args)
        data = json.loads(buf.getvalue())
        assert data["files_scanned"] == 1


class TestRegisterSubparser:
    """Test subparser registration."""

    def test_register_adds_subparser(self):
        """Test that register adds the migrate subparser."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        MigrateCommand.register(subparsers)

        args = parser.parse_args(['migrate', '/some/path'])
        assert args.path == '/some/path'

    def test_register_accepts_flags(self):
        """Test that the registered subparser accepts expected flags."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        MigrateCommand.register(subparsers)

        args = parser.parse_args([
            'migrate', '/some/path',
            '--format', 'json',
            '--ci',
            '--verbose',
        ])
        assert args.format == 'json'
        assert args.ci is True
        assert args.verbose is True


class TestEntryPoint:
    """Test standalone entry point."""

    def test_main_entry_point(self, tmp_path):
        """Test standalone main() function."""
        f = tmp_path / "test.py"
        f.write_text("import torch\n")

        with patch('sys.argv', ['tb-migrate', str(f)]):
            result = main()
            assert result == 0

    def test_main_with_suggestions(self, tmp_path):
        """Test main() returns non-zero when suggestions found in CI mode."""
        f = tmp_path / "test.py"
        f.write_text("model = model.cuda()\n")

        with patch('sys.argv', ['tb-migrate', str(f), '--ci']):
            result = main()
            assert result == 1

    def test_main_no_args(self):
        """Test main() with no arguments exits non-zero."""
        import pytest
        with patch('sys.argv', ['tb-migrate']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
