#!/usr/bin/env python3
"""
Check if documentation updated when backend code changes.
Ensures docs stay in sync with implementation.
"""

import subprocess
import sys

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
    """Check if docs updated when backend changed."""
    staged_files = get_staged_files()

    if not staged_files or staged_files == ['']:
        return 0

    # Check if backend code changed
    backend_files = [f for f in staged_files if 'backends/' in f and f.endswith('.py')]

    if not backend_files:
        # No backend changes
        return 0

    # Check if any doc files are staged
    doc_files = [f for f in staged_files if f.startswith('docs/') or f == 'README.md']

    if doc_files:
        print("✅ Documentation updated")
        return 0

    print("⚠️  Warning: Backend code changed but no documentation updated")
    print(f"\n   Changed backend files:")
    for f in backend_files[:3]:
        print(f"      - {f}")
    if len(backend_files) > 3:
        print(f"      ... and {len(backend_files) - 3} more")

    print(f"\n💡 Consider updating:")
    print(f"   - docs/unified_roadmap.md")
    print(f"   - docs/immediate_tasks.md")
    print(f"   - README.md (if API changed)")

    # Warning only, don't fail
    return 0

if __name__ == "__main__":
    sys.exit(main())
