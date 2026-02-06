#!/usr/bin/env python3
"""
Documentation Version Synchronization Script

This script ensures all documentation files have consistent version references
that match the version defined in src/torchbridge/__init__.py.

Usage:
    python scripts/sync_doc_versions.py          # Check for mismatches
    python scripts/sync_doc_versions.py --fix    # Auto-fix mismatches

Integrates with pre-commit hooks to prevent version drift.
"""

import re
import sys
from pathlib import Path

# Root of the repository
REPO_ROOT = Path(__file__).parent.parent

# Source of truth for version
VERSION_FILE = REPO_ROOT / "src" / "torchbridge" / "__init__.py"

# Documentation files to sync
DOC_PATTERNS = [
    "*.md",
    "docs/**/*.md",
    "benchmarks/**/*.md",
    "demos/**/*.md",
    "tests/**/*.md",
]

# Files to exclude from version sync
EXCLUDE_FILES = [
    "CHANGELOG.md",  # Contains historical versions
    "docs/unified_roadmap.md",  # Contains historical milestones
    "docs/immediate_tasks.md",  # Contains historical achievements
]

# Patterns to match version references in docs
VERSION_PATTERNS = [
    # Header patterns: "# Title (v0.3.3)" or "# Title (v0.2.4)"
    (r"^(#+ .+)\(v\d+\.\d+\.\d+\)", r"\1(v{version})"),
    # Bold version: "**TorchBridge v0.3.3**"
    (r"\*\*TorchBridge v\d+\.\d+\.\d+\*\*", "**TorchBridge v{version}**"),
    # Plain version in text: "TorchBridge v0.3.3" (not in code blocks)
    (r"TorchBridge v\d+\.\d+\.\d+(?!\*)", "TorchBridge v{version}"),
    # Version output comments: "# Should output: 0.3.3"
    (r"# Should output: \d+\.\d+\.\d+", "# Should output: {version}"),
    # Last Updated version: "**Version**: 0.3.3"
    (r"\*\*Version\*\*: \d+\.\d+\.\d+", "**Version**: {version}"),
]


def get_current_version() -> str:
    """Read the current version from __init__.py."""
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {VERSION_FILE}")
    return match.group(1)


def get_doc_files() -> list[Path]:
    """Get all documentation files to check."""
    files = []
    for pattern in DOC_PATTERNS:
        files.extend(REPO_ROOT.glob(pattern))

    # Filter out excluded files and non-tracked files
    exclude_paths = [REPO_ROOT / f for f in EXCLUDE_FILES]
    files = [f for f in files if f not in exclude_paths]
    files = [f for f in files if ".archive" not in str(f)]
    files = [f for f in files if "local/" not in str(f)]

    return sorted(set(files))


def check_file_versions(file_path: Path, current_version: str) -> list[tuple[int, str, str]]:
    """
    Check a file for outdated version references.

    Returns list of (line_number, old_text, new_text) tuples.
    """
    mismatches = []
    content = file_path.read_text()
    lines = content.split("\n")

    # Skip files that are mostly code (like triton_kernels_README.md)
    if "```" in content and content.count("```") > 10:
        return []

    for line_num, line in enumerate(lines, 1):
        # Skip lines inside code blocks
        # (Simple heuristic - skip lines that look like code)
        if line.strip().startswith("```"):
            continue

        for pattern, replacement in VERSION_PATTERNS:
            match = re.search(pattern, line, re.MULTILINE)
            if match:
                old_text = match.group(0)
                new_text = replacement.format(version=current_version)

                # Check if it's actually outdated
                if old_text != new_text and current_version not in old_text:
                    mismatches.append((line_num, old_text, new_text))

    return mismatches


def fix_file_versions(file_path: Path, current_version: str) -> int:
    """
    Fix outdated version references in a file.

    Returns number of fixes made.
    """
    content = file_path.read_text()
    original_content = content

    for pattern, replacement in VERSION_PATTERNS:
        new_text = replacement.format(version=current_version)
        content = re.sub(pattern, new_text, content, flags=re.MULTILINE)

    if content != original_content:
        file_path.write_text(content)
        # Count approximate number of changes
        return len(re.findall(r"v\d+\.\d+\.\d+", original_content)) - len(re.findall(f"v{current_version}", original_content))

    return 0


def main():
    """Main entry point."""
    fix_mode = "--fix" in sys.argv

    try:
        current_version = get_current_version()
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Current version: {current_version}")
    print(f"Mode: {'FIX' if fix_mode else 'CHECK'}")
    print("-" * 60)

    doc_files = get_doc_files()
    print(f"Checking {len(doc_files)} documentation files...")
    print()

    total_mismatches = 0
    files_with_issues = []

    for file_path in doc_files:
        rel_path = file_path.relative_to(REPO_ROOT)

        if fix_mode:
            fixes = fix_file_versions(file_path, current_version)
            if fixes > 0:
                print(f"  FIXED: {rel_path} ({fixes} changes)")
                total_mismatches += fixes
        else:
            mismatches = check_file_versions(file_path, current_version)
            if mismatches:
                files_with_issues.append((rel_path, mismatches))
                total_mismatches += len(mismatches)

    print()
    print("=" * 60)

    if fix_mode:
        if total_mismatches > 0:
            print("FIXED: Updated version references in documentation")
            print(f"       Total changes: {total_mismatches}")
            return 0
        else:
            print("OK: All documentation versions are current")
            return 0
    else:
        if files_with_issues:
            print(f"FAILED: Found {total_mismatches} outdated version references")
            print()
            for rel_path, mismatches in files_with_issues:
                print(f"  {rel_path}:")
                for line_num, old_text, new_text in mismatches[:3]:  # Show first 3
                    print(f"    Line {line_num}: '{old_text}' -> '{new_text}'")
                if len(mismatches) > 3:
                    print(f"    ... and {len(mismatches) - 3} more")
            print()
            print("Run with --fix to auto-correct these issues:")
            print("  python scripts/sync_doc_versions.py --fix")
            return 1
        else:
            print("OK: All documentation versions match current version")
            return 0


if __name__ == "__main__":
    sys.exit(main())
