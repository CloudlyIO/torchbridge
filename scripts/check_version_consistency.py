#!/usr/bin/env python3
"""
Check version consistency across TorchBridge project files.
Ensures version numbers match in all key locations.
"""

import re
import sys
from pathlib import Path

# Critical version files (MUST match for any release)
CRITICAL_VERSION_FILES = {
    'pyproject.toml': r'version\s*=\s*"(\d+\.\d+\.\d+)"',
    'src/torchbridge/__init__.py': r'__version__\s*=\s*"(\d+\.\d+\.\d+)"',
    'src/torchbridge/cli/__init__.py': r"version='%\(prog\)s\s+(\d+\.\d+\.\d+)'",
    'CHANGELOG.md': r'## \[(\d+\.\d+\.\d+)\]',
}

# Secondary version files (should match, but not blocking)
SECONDARY_VERSION_FILES = {
    'src/torchbridge/backends/nvidia/__init__.py': r'__version__\s*=\s*"(\d+\.\d+\.\d+)"',
    'src/torchbridge/backends/tpu/__init__.py': r'__version__\s*=\s*"(\d+\.\d+\.\d+)"',
    'src/torchbridge/backends/amd/__init__.py': r'__version__\s*=\s*"(\d+\.\d+\.\d+)"',
    'src/torchbridge/backends/intel/__init__.py': r'__version__\s*=\s*"(\d+\.\d+\.\d+)"',
}

def extract_version(file_path: Path, pattern: str) -> str:
    """Extract version from file using regex pattern."""
    try:
        content = file_path.read_text()
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"⚠️  Warning: Could not read {file_path}: {e}")
    return None

def main():
    """Check version consistency across all files."""
    project_root = Path(__file__).parent.parent
    critical_versions = {}
    secondary_versions = {}
    exit_code = 0

    print("🔍 Checking version consistency...\n")

    # Check critical files (blocking)
    print("CRITICAL FILES (must match):")
    for file_rel_path, pattern in CRITICAL_VERSION_FILES.items():
        file_path = project_root / file_rel_path
        version = extract_version(file_path, pattern)
        if version:
            critical_versions[file_rel_path] = version
            print(f"   ✓ {file_rel_path}: v{version}")
        else:
            print(f"   ✗ {file_rel_path}: NOT FOUND")
            exit_code = 1

    # Check critical version consistency
    unique_critical = set(critical_versions.values())
    if len(unique_critical) > 1:
        print("\n❌ CRITICAL: Version mismatch in critical files!")
        print(f"   Found versions: {sorted(unique_critical)}")
        for version in sorted(unique_critical):
            files = [f for f, v in critical_versions.items() if v == version]
            print(f"   v{version}: {', '.join(files)}")
        exit_code = 1
    elif len(unique_critical) == 1:
        canonical_version = list(unique_critical)[0]
        print(f"\n✅ Critical files consistent: v{canonical_version}")
    else:
        print("\n❌ No versions found in critical files!")
        exit_code = 1
        canonical_version = None

    # Check secondary files (warnings only)
    print("\nSECONDARY FILES (should match):")
    for file_rel_path, pattern in SECONDARY_VERSION_FILES.items():
        file_path = project_root / file_rel_path
        version = extract_version(file_path, pattern)
        if version:
            secondary_versions[file_rel_path] = version
            status = "✓" if (canonical_version and version == canonical_version) else "⚠"
            print(f"   {status} {file_rel_path}: v{version}")
        else:
            print(f"   - {file_rel_path}: not found (optional)")

    # Summary
    if exit_code == 0:
        print(f"\n✅ Version check passed: v{canonical_version}")
    else:
        print("\n❌ Version check failed! Fix critical files before commit.")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
