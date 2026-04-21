#!/usr/bin/env python3
"""
check_workflow_paths.py — Validate that all explicit file/directory paths
referenced in .github/workflows/*.yml actually exist in the repo.

Catches cases like a test directory being deleted while a workflow still
points to it (which produces a silent exit-code-4 in pytest).

Patterns checked:
  - pytest <path>       (pytest invocation targets)
  - python <path>       (python script invocations)
  - paths: - '<glob>'  (workflow path-filter globs — checks the base dir)

Exits 0 if everything exists, 1 if anything is missing.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# ── Patterns that reference concrete paths in workflow YAML ───────────────
# Matches: pytest tests/foo/ or pytest tests/foo tests/bar
PYTEST_TARGET = re.compile(r"pytest\s+((?:tests/\S+\s*)+)")
# Matches: python scripts/ci/foo.py or python -m ... (skip -m patterns)
PYTHON_SCRIPT = re.compile(r"python(?:3)?\s+(?!-)(scripts/\S+\.py)")
# Matches path-filter globs like: - 'tests/benchmark/**'
PATHS_FILTER = re.compile(r"-\s+'(tests/[^'*]+?)(?:/\*\*)?'")


def check_workflows() -> list[tuple[str, str]]:
    """Return list of (workflow_file, missing_path) pairs."""
    missing: list[tuple[str, str]] = []

    for yml in sorted(WORKFLOWS_DIR.glob("*.yml")):
        content = yml.read_text()
        workflow_name = yml.name

        # Collect all candidate paths
        candidates: list[str] = []

        for match in PYTEST_TARGET.finditer(content):
            # May be multiple space-separated targets
            for part in match.group(1).split():
                # Strip trailing slash
                candidates.append(part.rstrip("/"))

        for match in PYTHON_SCRIPT.finditer(content):
            candidates.append(match.group(1))

        for match in PATHS_FILTER.finditer(content):
            # Only check the directory prefix, not the glob pattern itself
            candidates.append(match.group(1))

        for path_str in candidates:
            full_path = REPO_ROOT / path_str
            if not full_path.exists():
                missing.append((workflow_name, path_str))

    return missing


def main() -> int:
    missing = check_workflows()

    if not missing:
        print("check_workflow_paths: all paths exist ✓")
        return 0

    print("check_workflow_paths: MISSING PATHS DETECTED\n")
    for workflow, path in missing:
        print(f"  {workflow}: '{path}' does not exist")
    print(
        f"\n{len(missing)} path(s) missing — fix the workflow(s) or restore the path(s)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
