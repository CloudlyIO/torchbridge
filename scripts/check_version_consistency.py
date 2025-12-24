#!/usr/bin/env python3
"""
Check version consistency across KernelPyTorch project files.
Ensures version numbers match in all key locations.
"""

import re
import sys
from pathlib import Path

# Files to check for version consistency
VERSION_FILES = {
    'CHANGELOG.md': r'\[(\d+\.\d+\.\d+)\]',
    'docs/unified_roadmap.md': r'v(\d+\.\d+\.\d+)',
    'docs/immediate_tasks.md': r'v(\d+\.\d+\.\d+)',
    'src/kernel_pytorch/backends/nvidia/__init__.py': r'__version__\s*=\s*"(\d+\.\d+\.\d+)"',
    'src/kernel_pytorch/backends/tpu/__init__.py': r'__version__\s*=\s*"(\d+\.\d+\.\d+)"',
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
    versions = {}

    print("🔍 Checking version consistency...")

    for file_rel_path, pattern in VERSION_FILES.items():
        file_path = project_root / file_rel_path
        version = extract_version(file_path, pattern)
        if version:
            versions[file_rel_path] = version
            print(f"   {file_rel_path}: v{version}")

    # Check if all versions match
    unique_versions = set(versions.values())

    if len(unique_versions) == 0:
        print("❌ Error: No versions found in any files!")
        return 1
    elif len(unique_versions) == 1:
        current_version = list(unique_versions)[0]
        print(f"\n✅ All versions consistent: v{current_version}")
        return 0
    else:
        print(f"\n❌ Error: Version mismatch detected!")
        print(f"   Found versions: {sorted(unique_versions)}")
        print(f"\n   Files by version:")
        for version in sorted(unique_versions):
            files = [f for f, v in versions.items() if v == version]
            print(f"   v{version}:")
            for f in files:
                print(f"      - {f}")
        print(f"\n💡 Please update all version references to match.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
