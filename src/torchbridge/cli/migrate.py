"""
Migration command for TorchBridge CLI.

Scans Python files for CUDA-specific patterns and suggests
TorchBridge hardware-agnostic replacements.
"""

import argparse
import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MigrationSuggestion:
    """A single migration suggestion."""
    file: str
    line: int
    pattern_name: str
    matched_text: str
    suggestion: str
    context: str = ""


@dataclass
class MigrationReport:
    """Full migration report."""
    path: str
    suggestions: list[MigrationSuggestion] = field(default_factory=list)
    files_scanned: int = 0
    files_with_suggestions: int = 0

    @property
    def total_suggestions(self) -> int:
        return len(self.suggestions)


# Pattern definitions: (name, regex, suggestion text)
MIGRATION_PATTERNS = [
    (
        "torch.cuda.is_available()",
        r"torch\.cuda\.is_available\(\)",
        "Replace with detect_hardware() or detect_best_backend() from torchbridge.backends",
    ),
    (
        ".cuda()",
        r"\.cuda\(\)",
        "Replace with .to(backend.device) for backend-agnostic device placement",
    ),
    (
        'torch.device("cuda")',
        r'torch\.device\(["\']cuda',
        "Replace with backend.device from TorchBridge backend detection",
    ),
    (
        '"nccl" hardcoded',
        r"""[\"']nccl[\"']""",
        "Let TorchBridge auto-detect the distributed backend (nccl/gloo/mpi)",
    ),
    (
        "torch.cuda.amp",
        r"torch\.cuda\.amp",
        "Replace with torch.amp and TorchBridge precision config for cross-backend AMP",
    ),
    (
        "torch.cuda.synchronize",
        r"torch\.cuda\.synchronize",
        "Replace with torch.accelerator.synchronize() for backend-agnostic sync",
    ),
    (
        "torch.cuda.memory_allocated",
        r"torch\.cuda\.memory_allocated",
        "Use TorchBridge backend-agnostic memory API for cross-backend memory tracking",
    ),
]

# Directories to skip by default when scanning
DEFAULT_SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".tox",
    ".eggs",
    "dist",
    "build",
}


class MigrateCommand:
    """Migration scanner command implementation."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the migrate command with argument parser."""
        parser = subparsers.add_parser(
            'migrate',
            help='Scan for CUDA-specific patterns and suggest TorchBridge replacements',
            description='Scans Python files for CUDA-specific code patterns and suggests '
                        'TorchBridge hardware-agnostic replacements.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Detected Patterns:
  torch.cuda.is_available()   - Hardware detection calls
  .cuda()                     - Device placement calls
  torch.device("cuda")        - Hardcoded CUDA device creation
  "nccl" hardcoded            - Hardcoded distributed backend
  torch.cuda.amp              - CUDA-specific AMP usage
  torch.cuda.synchronize      - CUDA-specific synchronization
  torch.cuda.memory_allocated - CUDA-specific memory tracking

Examples:
  tb-migrate .                           # Scan current directory
  tb-migrate src/model.py                # Scan a single file
  tb-migrate src/ --ci                   # CI mode (exit 1 if suggestions)
  tb-migrate . --format json -o report   # Save JSON report
  tb-migrate . --exclude "tests/*"       # Exclude test files
            """
        )

        parser.add_argument(
            'path',
            type=str,
            help='File or directory to scan for CUDA patterns'
        )

        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Save migration report to file'
        )

        parser.add_argument(
            '--format',
            choices=['json', 'markdown'],
            default='markdown',
            help='Output format (default: markdown)'
        )

        parser.add_argument(
            '--ci',
            action='store_true',
            help='CI mode: exit 1 if migration suggestions found'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show context lines around matches'
        )

        parser.add_argument(
            '--exclude',
            nargs='*',
            default=[],
            help='Additional glob patterns to exclude (e.g. "tests/*" "legacy/*.py")'
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the migrate command."""
        target_path = getattr(args, 'path', '.')
        ci_mode = getattr(args, 'ci', False)
        verbose = getattr(args, 'verbose', False)
        output_path = getattr(args, 'output', None)
        fmt = getattr(args, 'format', 'markdown')
        exclude_patterns = getattr(args, 'exclude', []) or []

        target = Path(target_path)
        if not target.exists():
            if not ci_mode:
                print(f"Error: path does not exist: {target_path}")
            return 1

        # Collect Python files to scan
        py_files = MigrateCommand._collect_files(target, exclude_patterns)

        if not ci_mode and verbose:
            print(f"Scanning {len(py_files)} Python file(s) in: {target_path}")

        # Build migration report
        report = MigrationReport(path=target_path)
        report.files_scanned = len(py_files)
        files_with_hits = set()

        for filepath in py_files:
            suggestions = MigrateCommand._scan_file(filepath, verbose)
            if suggestions:
                files_with_hits.add(filepath)
                report.suggestions.extend(suggestions)

        report.files_with_suggestions = len(files_with_hits)

        # Save report to file if requested
        if output_path:
            MigrateCommand._save_report(report, output_path, fmt, verbose)

        # Handle CI mode
        if ci_mode:
            if verbose:
                MigrateCommand._display_report(report, verbose)
            if report.total_suggestions > 0:
                return 1
            return 0

        # Display results to terminal
        if fmt == 'json':
            print(MigrateCommand._format_json(report))
        else:
            MigrateCommand._display_report(report, verbose)

        return 0

    @staticmethod
    def _collect_files(target: Path, exclude_patterns: list[str]) -> list[str]:
        """Collect Python files from a file or directory path."""
        py_files = []

        if target.is_file():
            if target.suffix == '.py':
                py_files.append(str(target.resolve()))
            return py_files

        for dirpath, dirnames, filenames in os.walk(str(target)):
            # Remove default skip directories from traversal in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in DEFAULT_SKIP_DIRS
            ]

            for filename in filenames:
                if not filename.endswith('.py'):
                    continue

                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, str(target))

                # Check user-supplied exclude patterns against the relative path
                excluded = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(rel_path, pattern):
                        excluded = True
                        break
                if excluded:
                    continue

                py_files.append(str(Path(filepath).resolve()))

        py_files.sort()
        return py_files

    @staticmethod
    def _scan_file(filepath: str, verbose: bool) -> list[MigrationSuggestion]:
        """Scan a single file for CUDA patterns."""
        suggestions = []

        try:
            with open(filepath, encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except OSError:
            return suggestions

        for line_idx, line in enumerate(lines):
            line_num = line_idx + 1
            stripped = line.lstrip()

            # Skip comment lines
            if stripped.startswith('#'):
                continue

            for pattern_name, pattern_regex, suggestion_text in MIGRATION_PATTERNS:
                matches = re.findall(pattern_regex, line)
                if not matches:
                    continue

                matched_text = matches[0]

                context = ""
                if verbose:
                    context_lines = []
                    start = max(0, line_idx - 2)
                    end = min(len(lines), line_idx + 3)
                    for ctx_idx in range(start, end):
                        prefix = ">>" if ctx_idx == line_idx else "  "
                        ctx_line = lines[ctx_idx].rstrip('\n')
                        context_lines.append(f"  {prefix} {ctx_idx + 1}: {ctx_line}")
                    context = "\n".join(context_lines)

                suggestions.append(MigrationSuggestion(
                    file=filepath,
                    line=line_num,
                    pattern_name=pattern_name,
                    matched_text=matched_text,
                    suggestion=suggestion_text,
                    context=context,
                ))

        return suggestions

    @staticmethod
    def _format_markdown(report: MigrationReport, verbose: bool) -> str:
        """Format report as markdown."""
        parts = []
        parts.append("# TorchBridge Migration Report")
        parts.append("")
        parts.append(f"**Path:** `{report.path}`")
        parts.append(f"**Files scanned:** {report.files_scanned}")
        parts.append(f"**Files with suggestions:** {report.files_with_suggestions}")
        parts.append(f"**Total suggestions:** {report.total_suggestions}")
        parts.append("")

        if report.total_suggestions == 0:
            parts.append("No CUDA-specific patterns found. Code is already hardware-agnostic.")
            return "\n".join(parts)

        # Group suggestions by file
        by_file: dict[str, list[MigrationSuggestion]] = {}
        for s in report.suggestions:
            by_file.setdefault(s.file, []).append(s)

        for filepath, file_suggestions in sorted(by_file.items()):
            parts.append(f"## `{filepath}`")
            parts.append("")
            for s in file_suggestions:
                parts.append(f"- **Line {s.line}** -- `{s.pattern_name}`")
                parts.append(f"  - Matched: `{s.matched_text}`")
                parts.append(f"  - Suggestion: {s.suggestion}")
                if verbose and s.context:
                    parts.append("  - Context:")
                    parts.append("    ```")
                    parts.append(s.context)
                    parts.append("    ```")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _format_json(report: MigrationReport) -> str:
        """Format report as JSON."""
        data = {
            "path": report.path,
            "files_scanned": report.files_scanned,
            "files_with_suggestions": report.files_with_suggestions,
            "total_suggestions": report.total_suggestions,
            "suggestions": [
                {
                    "file": s.file,
                    "line": s.line,
                    "pattern_name": s.pattern_name,
                    "matched_text": s.matched_text,
                    "suggestion": s.suggestion,
                    "context": s.context,
                }
                for s in report.suggestions
            ],
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def _save_report(
        report: MigrationReport,
        output_path: str,
        fmt: str,
        verbose: bool,
    ) -> None:
        """Save the migration report to a file."""
        if verbose:
            print(f"Saving report to: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if fmt == 'json':
            content = MigrateCommand._format_json(report)
        else:
            content = MigrateCommand._format_markdown(report, verbose)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.write("\n")

        if verbose:
            print(f"Report saved in {fmt} format")

    @staticmethod
    def _display_report(report: MigrationReport, verbose: bool) -> None:
        """Display report to stdout in human-readable format."""
        print("")
        print("TorchBridge Migration Scanner")
        print("=" * 50)
        print(f"Path:                   {report.path}")
        print(f"Files scanned:          {report.files_scanned}")
        print(f"Files with suggestions: {report.files_with_suggestions}")
        print(f"Total suggestions:      {report.total_suggestions}")
        print("")

        if report.total_suggestions == 0:
            print("No CUDA-specific patterns found. Code is already hardware-agnostic.")
            return

        # Group suggestions by file
        by_file: dict[str, list[MigrationSuggestion]] = {}
        for s in report.suggestions:
            by_file.setdefault(s.file, []).append(s)

        for filepath, file_suggestions in sorted(by_file.items()):
            print("-" * 50)
            print(f"File: {filepath}")
            print(f"  {len(file_suggestions)} suggestion(s)")
            print("")

            for s in file_suggestions:
                print(f"  Line {s.line}: [{s.pattern_name}]")
                print(f"    Matched:    {s.matched_text}")
                print(f"    Suggestion: {s.suggestion}")
                if verbose and s.context:
                    print("    Context:")
                    print(s.context)
                print("")

        print("=" * 50)
        print(
            f"Summary: {report.total_suggestions} suggestion(s) "
            f"across {report.files_with_suggestions} file(s)"
        )


def main():
    """Standalone entry point for tb-migrate."""
    parser = argparse.ArgumentParser(
        prog='tb-migrate',
        description='Scan for CUDA-specific patterns and suggest TorchBridge replacements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'path',
        type=str,
        help='File or directory to scan for CUDA patterns'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save migration report to file'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='markdown',
        help='Output format (default: markdown)'
    )

    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: exit 1 if migration suggestions found'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show context lines around matches'
    )

    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='Additional glob patterns to exclude (e.g. "tests/*" "legacy/*.py")'
    )

    args = parser.parse_args()
    return MigrateCommand.execute(args)


if __name__ == '__main__':
    sys.exit(main())
