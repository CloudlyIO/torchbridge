#!/usr/bin/env python3
"""Pre-release readiness check for TorchBridge PyPI publish.

Run: python scripts/ci/check_release_readiness.py
Exits 0 if ready, 1 with details if not.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
ERRORS = []


def check_version_consistency():
    """All version declarations must match."""
    toml = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version = "([^"]+)"', toml, re.MULTILINE)
    toml_ver = m.group(1) if m else "MISSING"

    changelog = (ROOT / "CHANGELOG.md").read_text()
    if f"## [{toml_ver}]" not in changelog:
        ERRORS.append(f"CHANGELOG.md missing entry for [{toml_ver}]")

    print(f"Version: {toml_ver} {'OK' if not ERRORS else 'MISMATCH'}")
    return toml_ver


def check_pybind11_not_in_runtime():
    """pybind11 must not be in runtime dependencies."""
    toml = (ROOT / "pyproject.toml").read_text()
    deps_section = re.search(
        r"^\[project\]\n.*?^dependencies\s*=\s*\[(.*?)\]",
        toml,
        re.DOTALL | re.MULTILINE,
    )
    if deps_section and "pybind11" in deps_section.group(1):
        ERRORS.append("pybind11 found in [project] runtime dependencies — move to build-system only")
        print("pybind11 runtime check: FAIL")
    else:
        print("pybind11 runtime check: OK")


def check_ruff():
    result = subprocess.run(
        ["ruff", "check", "src/"], capture_output=True, text=True, cwd=ROOT
    )
    if result.returncode != 0:
        ERRORS.append(f"ruff violations:\n{result.stdout}")
        print("ruff: FAIL")
    else:
        print("ruff: OK")


def check_tests_collect():
    result = subprocess.run(
        ["pytest", "--co", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    combined = result.stdout + result.stderr
    m = re.search(r"(\d+) test", combined)
    if m:
        count = int(m.group(1))
        print(f"Tests collected: {count}")
        if count < 2000:
            ERRORS.append(f"Unexpectedly few tests: {count} (expected >= 2000)")
    else:
        ERRORS.append("Could not collect test count")
        print("Tests collected: UNKNOWN")


def main():
    check_version_consistency()
    check_pybind11_not_in_runtime()
    check_ruff()
    check_tests_collect()

    if ERRORS:
        print("\nNOT READY FOR RELEASE:")
        for e in ERRORS:
            print(f"  * {e}")
        sys.exit(1)
    else:
        print("\nRelease readiness: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
