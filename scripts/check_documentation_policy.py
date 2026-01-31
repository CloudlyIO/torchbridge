#!/usr/bin/env python3
"""
Documentation Policy Enforcement Pre-Commit Hook

This hook enforces strict documentation file policies:
1. NO new .md files in root directory (only approved essential ones)
2. ALL validation/test reports MUST go to /local folder
3. ANY new .md files in docs/ require explicit user approval
4. Prefer modifying existing files over creating new ones

Approved root .md files (MUST remain constant):
- README.md
- CHANGELOG.md
- CONTRIBUTING.md

Version: 0.3.4
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# Approved root-level .md files (these are the ONLY allowed ones)
APPROVED_ROOT_MD_FILES = {
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
}

# Patterns for files that MUST go to /local
REPORT_PATTERNS = [
    "_REPORT.md",
    "_SUMMARY.md",
    "_ANALYSIS.md",
    "_VALIDATION.md",
    "TESTING_",
    "POST_",
    "DOCUMENTATION_SYNC",
    "CLEANUP_SUMMARY",
]


def get_staged_files() -> List[str]:
    """Get list of staged files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
        capture_output=True,
        text=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def check_root_md_files(staged_files: List[str]) -> Tuple[bool, List[str]]:
    """Check for unauthorized root .md files."""
    violations = []

    for file in staged_files:
        # Check if it's a root-level .md file
        if file.endswith(".md") and "/" not in file:
            if file not in APPROVED_ROOT_MD_FILES:
                violations.append(file)

    return len(violations) == 0, violations


def check_report_files(staged_files: List[str]) -> Tuple[bool, List[str]]:
    """Check for report files that should be in /local."""
    violations = []

    for file in staged_files:
        # Check if it matches report patterns
        for pattern in REPORT_PATTERNS:
            if pattern in file and not file.startswith("local/"):
                violations.append(file)
                break

    return len(violations) == 0, violations


def check_docs_md_files(staged_files: List[str]) -> Tuple[bool, List[str]]:
    """Check for new .md files in docs/ directory."""
    new_docs = []

    for file in staged_files:
        if file.startswith("docs/") and file.endswith(".md"):
            new_docs.append(file)

    return len(new_docs) == 0, new_docs


def main():
    """Run all documentation policy checks."""
    staged_files = get_staged_files()

    if not staged_files:
        sys.exit(0)

    has_violations = False

    # Check 1: Root .md files
    root_ok, root_violations = check_root_md_files(staged_files)
    if not root_ok:
        print("❌ BLOCKED: Unauthorized root .md files detected!")
        print("\n🚫 The following root .md files are NOT approved:")
        for file in root_violations:
            print(f"   - {file}")
        print("\n📋 ONLY these root .md files are allowed:")
        for file in sorted(APPROVED_ROOT_MD_FILES):
            print(f"   ✅ {file}")
        print("\n💡 Action required:")
        print("   1. Move these files to local/reports/ or local/planning/")
        print("   2. Update .gitignore if needed")
        print("   3. Use `git rm --cached <file>` to unstage")
        has_violations = True

    # Check 2: Report files
    report_ok, report_violations = check_report_files(staged_files)
    if not report_ok:
        print("\n❌ BLOCKED: Validation/test reports outside /local folder!")
        print("\n🚫 The following files should be in /local:")
        for file in report_violations:
            print(f"   - {file}")
        print("\n💡 Action required:")
        print("   1. Move these files to local/reports/")
        print("   2. Commit will be blocked until files are in correct location")
        has_violations = True

    # Check 3: New docs/ .md files
    docs_ok, new_docs = check_docs_md_files(staged_files)
    if not docs_ok:
        print("\n⚠️  WARNING: New .md files detected in docs/")
        print("\n📝 The following new documentation files require explicit approval:")
        for file in new_docs:
            print(f"   - {file}")
        print("\n❓ Before adding new documentation:")
        print("   1. Can this content be added to an EXISTING file?")
        print("   2. Is this documentation truly necessary?")
        print("   3. Have you asked for explicit approval?")
        print("\n💡 Recommended action:")
        print("   - Review existing docs and update them instead")
        print("   - If new file is essential, get explicit approval first")
        print("   - Use `git rm --cached <file>` to unstage until approved")
        has_violations = True

    if has_violations:
        print("\n" + "="*70)
        print("🛑 COMMIT BLOCKED - Documentation policy violations detected")
        print("="*70)
        print("\nRefer to local/DOCUMENTATION_GUIDELINES.md for complete policies.")
        sys.exit(1)

    print("✅ Documentation policy check passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
