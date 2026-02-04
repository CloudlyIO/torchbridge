"""
Automated Dead Code Cleanup Script

Systematically removes identified dead code including:
- Legacy/duplicate files
- Debug print statements
- Unused imports
- TODO/FIXME comments that are outdated
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Set, Dict


class DeadCodeCleaner:
    """Automates removal of identified dead code"""

    def __init__(self, src_path: str = "src", dry_run: bool = True):
        self.src_path = Path(src_path)
        self.dry_run = dry_run
        self.changes_made = 0

    def cleanup_all(self):
        """Run all cleanup operations"""
        print(f"🧹 Starting dead code cleanup (dry_run={self.dry_run})...")

        # 1. Remove obvious legacy files
        self._remove_legacy_files()

        # 2. Clean debug prints
        self._clean_debug_prints()

        # 3. Clean simple unused imports
        self._clean_unused_imports()

        # 4. Remove empty pass statements
        self._clean_redundant_pass()

        print(f"\n✅ Cleanup complete! Made {self.changes_made} changes.")

    def _remove_legacy_files(self):
        """Remove clearly legacy/duplicate files"""
        legacy_files = [
            "src/torchbridge/distributed_scale/hardware_adaptation_original.py",
            "src/torchbridge/utils/compiler_assistant_legacy.py"
        ]

        print(f"\n🗑️  Removing legacy files...")
        for file_path in legacy_files:
            if Path(file_path).exists():
                if self.dry_run:
                    print(f"  [DRY RUN] Would remove: {file_path}")
                else:
                    os.remove(file_path)
                    print(f"  ✅ Removed: {file_path}")
                self.changes_made += 1

    def _clean_debug_prints(self):
        """Remove debug print statements"""
        print(f"\n🔧 Cleaning debug prints...")

        debug_patterns = [
            r'^\s*print\(f?".*FlexAttention.*\).*$',
            r'^\s*print\(f?".*Input shape.*\).*$',
            r'^\s*print\(f?".*output shape.*\).*$',
            r'^\s*print\(f?".*Benchmark Results.*\).*$',
            r'^\s*print\(f?".*pattern.*ms.*\).*$',
            r'^\s*print\(f?".*Available.*\).*$',
            r'^\s*print\(".*Starting.*"\).*$',
            r'^\s*print\(".*Complete.*"\).*$'
        ]

        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                removed_count = 0

                for line in lines:
                    should_remove = False
                    for pattern in debug_patterns:
                        if re.match(pattern, line):
                            should_remove = True
                            removed_count += 1
                            break

                    if not should_remove:
                        new_lines.append(line)

                if removed_count > 0:
                    if self.dry_run:
                        print(f"  [DRY RUN] {py_file.name}: would remove {removed_count} debug prints")
                    else:
                        with open(py_file, 'w') as f:
                            f.writelines(new_lines)
                        print(f"  ✅ {py_file.name}: removed {removed_count} debug prints")
                    self.changes_made += removed_count

            except Exception as e:
                print(f"  ⚠️  Error processing {py_file}: {e}")

    def _clean_unused_imports(self):
        """Remove obviously unused imports"""
        print(f"\n📦 Cleaning unused imports...")

        # Focus on commonly unused imports
        common_unused = {
            'matplotlib.pyplot', 'plt', 'numpy', 'np', 'pandas', 'pd',
            'time', 'os', 'sys', 'json', 'pickle', 'warnings'
        }

        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')

                # Simple heuristic: if import is not used in rest of file
                new_lines = []
                removed_imports = []

                for line in lines:
                    stripped = line.strip()
                    should_remove = False

                    # Check for standalone imports that aren't used
                    if stripped.startswith('import ') and not stripped.startswith('import torch'):
                        import_name = stripped.replace('import ', '').split('.')[0]
                        rest_of_file = '\n'.join(lines[lines.index(line)+1:])

                        if import_name in common_unused and import_name not in rest_of_file:
                            should_remove = True
                            removed_imports.append(import_name)

                    if not should_remove:
                        new_lines.append(line)

                if removed_imports:
                    if self.dry_run:
                        print(f"  [DRY RUN] {py_file.name}: would remove imports: {removed_imports}")
                    else:
                        with open(py_file, 'w') as f:
                            f.write('\n'.join(new_lines))
                        print(f"  ✅ {py_file.name}: removed imports: {removed_imports}")
                    self.changes_made += len(removed_imports)

            except Exception as e:
                print(f"  ⚠️  Error processing {py_file}: {e}")

    def _clean_redundant_pass(self):
        """Remove redundant pass statements"""
        print(f"\n✂️  Cleaning redundant pass statements...")

        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                removed_count = 0

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Remove standalone pass statements that aren't needed
                    if stripped == "pass":
                        # Check if this is really redundant
                        # (Simple heuristic: if it's followed by non-indented code)
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line and not next_line.startswith(' ') and not next_line.startswith('\t'):
                                removed_count += 1
                                continue  # Skip this pass

                    new_lines.append(line)

                if removed_count > 0:
                    if self.dry_run:
                        print(f"  [DRY RUN] {py_file.name}: would remove {removed_count} redundant pass statements")
                    else:
                        with open(py_file, 'w') as f:
                            f.writelines(new_lines)
                        print(f"  ✅ {py_file.name}: removed {removed_count} redundant pass statements")
                    self.changes_made += removed_count

            except Exception as e:
                print(f"  ⚠️  Error processing {py_file}: {e}")


def run_cleanup(dry_run=True):
    """Run the cleanup process"""
    print(f"🧹 DEAD CODE CLEANUP")
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'LIVE (making changes)'}")
    print("=" * 50)

    cleaner = DeadCodeCleaner(dry_run=dry_run)
    cleaner.cleanup_all()

    if dry_run:
        print(f"\n💡 To apply these changes, run with dry_run=False")
    else:
        print(f"\n🎉 Cleanup complete! Repository cleaned.")


if __name__ == "__main__":
    # Run in live mode for actual cleanup
    run_cleanup(dry_run=False)