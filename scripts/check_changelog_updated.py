#!/usr/bin/env python3
"""
Check if CHANGELOG.md has been updated when code changes.
Prevents commits without changelog updates.
"""

import subprocess
import sys
from datetime import datetime


def get_staged_files():
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n') if result.stdout else []
    except subprocess.CalledProcessError:
        return []

def main():
    """Check if CHANGELOG.md updated when code changed."""
    staged_files = get_staged_files()

    if not staged_files or staged_files == ['']:
        # No staged files, skip check
        return 0

    # Check if any code files are staged
    code_files = [f for f in staged_files if f.startswith(('src/', 'tests/', 'benchmarks/', 'demos/'))]

    if not code_files:
        # No code changes, skip check
        return 0

    # Check if CHANGELOG.md is staged
    if 'CHANGELOG.md' in staged_files:
        print("✅ CHANGELOG.md updated")
        return 0

    print("❌ Error: Code changes detected but CHANGELOG.md not updated!")
    print("\n   Changed code files:")
    for f in code_files[:5]:  # Show first 5
        print(f"      - {f}")
    if len(code_files) > 5:
        print(f"      ... and {len(code_files) - 5} more")

    print("\n💡 Please update CHANGELOG.md with:")
    print(f"   - Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("   - Version: (increment as needed)")
    print("   - Description of changes")
    print("\n   Then stage CHANGELOG.md: git add CHANGELOG.md")

    return 1

if __name__ == "__main__":
    sys.exit(main())
