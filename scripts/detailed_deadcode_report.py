"""
Detailed Dead Code Analysis Report

Creates a comprehensive report of dead code issues with specific recommendations
for removal and cleanup.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DeadCodeIssue:
    file_path: str
    issue_type: str
    issue_description: str
    line_number: int
    code_snippet: str
    recommendation: str


class DetailedDeadCodeAnalyzer:
    """Provides detailed analysis of dead code with specific recommendations"""

    def __init__(self, src_path: str = "src"):
        self.src_path = Path(src_path)
        self.issues: List[DeadCodeIssue] = []

    def analyze(self) -> Dict[str, List[DeadCodeIssue]]:
        """Run detailed analysis"""
        print("🔍 Running detailed dead code analysis...")

        # 1. Find legacy/redundant files
        self._find_legacy_files()

        # 2. Find obvious dead code patterns
        self._find_dead_code_patterns()

        # 3. Find files with excessive unused imports
        self._find_excessive_unused_imports()

        # 4. Find stub/placeholder implementations
        self._find_stub_implementations()

        # 5. Find duplicate implementations
        self._find_duplicate_implementations()

        return self._categorize_issues()

    def _find_legacy_files(self):
        """Find files that are clearly legacy or redundant"""
        legacy_patterns = [
            "*legacy*", "*original*", "*old*", "*deprecated*",
            "*backup*", "*temp*", "*test_*", "*_test*"
        ]

        for pattern in legacy_patterns:
            for file_path in self.src_path.rglob(pattern):
                if file_path.suffix == ".py" and "__pycache__" not in str(file_path):
                    # Check if file is actually used
                    if self._is_file_unused(file_path):
                        self.issues.append(DeadCodeIssue(
                            file_path=str(file_path),
                            issue_type="legacy_file",
                            issue_description=f"Legacy file that appears unused",
                            line_number=1,
                            code_snippet=f"File: {file_path.name}",
                            recommendation=f"Consider removing if not needed"
                        ))

    def _find_dead_code_patterns(self):
        """Find common dead code patterns"""
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')

                tree = ast.parse(content)

                # Find stub functions (only raise NotImplementedError)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if self._is_stub_function(node):
                            self.issues.append(DeadCodeIssue(
                                file_path=str(py_file),
                                issue_type="stub_function",
                                issue_description=f"Stub function '{node.name}' only raises NotImplementedError",
                                line_number=node.lineno,
                                code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                                recommendation="Remove if not planned for implementation"
                            ))

                # Find debug/print statements left in code
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if (stripped.startswith('print(') or
                        stripped.startswith('# TODO') or
                        stripped.startswith('# FIXME') or
                        stripped.startswith('# XXX')):
                        self.issues.append(DeadCodeIssue(
                            file_path=str(py_file),
                            issue_type="debug_code",
                            issue_description=f"Debug/TODO code: {stripped[:50]}...",
                            line_number=i,
                            code_snippet=stripped,
                            recommendation="Remove debug prints or complete TODOs"
                        ))

            except Exception as e:
                continue

    def _find_excessive_unused_imports(self):
        """Find files with many unused imports"""
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                tree = ast.parse(content)
                imports = []

                # Collect all imports
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        else:
                            for alias in node.names:
                                imports.append(alias.name)

                # Simple heuristic: if file has many imports, check usage
                if len(imports) > 10:
                    unused_count = 0
                    for imp in imports:
                        if imp not in content.replace(f"import {imp}", ""):
                            unused_count += 1

                    if unused_count > 5:
                        self.issues.append(DeadCodeIssue(
                            file_path=str(py_file),
                            issue_type="excessive_imports",
                            issue_description=f"File has {unused_count} potentially unused imports out of {len(imports)}",
                            line_number=1,
                            code_snippet=f"Total imports: {len(imports)}, potentially unused: {unused_count}",
                            recommendation="Review and remove unused imports"
                        ))

            except Exception:
                continue

    def _find_stub_implementations(self):
        """Find functions that are just stubs"""
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for functions with only pass, return None, or raise NotImplementedError
                        if len(node.body) == 1:
                            body_node = node.body[0]
                            if (isinstance(body_node, ast.Pass) or
                                (isinstance(body_node, ast.Return) and body_node.value is None) or
                                (isinstance(body_node, ast.Raise) and
                                 isinstance(body_node.exc, ast.Call) and
                                 hasattr(body_node.exc.func, 'id') and
                                 body_node.exc.func.id == 'NotImplementedError')):

                                self.issues.append(DeadCodeIssue(
                                    file_path=str(py_file),
                                    issue_type="stub_implementation",
                                    issue_description=f"Function '{node.name}' is a stub implementation",
                                    line_number=node.lineno,
                                    code_snippet=f"def {node.name}(...): # stub",
                                    recommendation="Implement or remove if not needed"
                                ))

            except Exception:
                continue

    def _find_duplicate_implementations(self):
        """Find potentially duplicate function implementations"""
        function_bodies = defaultdict(list)

        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create a simple signature for comparison
                        body_str = ast.dump(node.body) if node.body else ""
                        if len(body_str) > 50:  # Only consider substantial functions
                            function_bodies[body_str].append((str(py_file), node.name, node.lineno))

            except Exception:
                continue

        # Find duplicates
        for body, locations in function_bodies.items():
            if len(locations) > 1:
                for file_path, func_name, line_no in locations[1:]:  # Skip first occurrence
                    self.issues.append(DeadCodeIssue(
                        file_path=file_path,
                        issue_type="duplicate_implementation",
                        issue_description=f"Function '{func_name}' appears to be duplicate of implementation in {locations[0][0]}",
                        line_number=line_no,
                        code_snippet=f"def {func_name}(...)",
                        recommendation="Review for duplication and consolidate if possible"
                    ))

    def _is_file_unused(self, file_path: Path) -> bool:
        """Check if a file is likely unused"""
        file_name = file_path.stem

        # Check if file is imported anywhere
        for py_file in self.src_path.rglob("*.py"):
            if py_file == file_path:
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                if file_name in content:
                    return False

            except Exception:
                continue

        return True

    def _is_stub_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is just a stub"""
        if len(node.body) == 1:
            body_node = node.body[0]
            if (isinstance(body_node, ast.Raise) and
                isinstance(body_node.exc, ast.Call) and
                hasattr(body_node.exc.func, 'id') and
                body_node.exc.func.id == 'NotImplementedError'):
                return True
        return False

    def _categorize_issues(self) -> Dict[str, List[DeadCodeIssue]]:
        """Categorize issues by type"""
        categorized = defaultdict(list)

        for issue in self.issues:
            categorized[issue.issue_type].append(issue)

        return categorized

    def generate_cleanup_plan(self) -> List[str]:
        """Generate specific cleanup actions"""
        categorized = self._categorize_issues()
        actions = []

        # High priority: Remove legacy files
        if "legacy_file" in categorized:
            for issue in categorized["legacy_file"]:
                actions.append(f"🗑️  REMOVE: {issue.file_path} (legacy file)")

        # High priority: Remove stub implementations
        if "stub_implementation" in categorized:
            for issue in categorized["stub_implementation"][:5]:  # Top 5
                actions.append(f"✂️  REMOVE: {Path(issue.file_path).name}:{issue.line_number} - {issue.issue_description}")

        # Medium priority: Clean up imports
        if "excessive_imports" in categorized:
            for issue in categorized["excessive_imports"][:3]:  # Top 3
                actions.append(f"🧹 CLEAN: {Path(issue.file_path).name} - remove unused imports")

        # Medium priority: Remove debug code
        if "debug_code" in categorized:
            debug_count = len(categorized["debug_code"])
            actions.append(f"🔧 CLEAN: Remove {debug_count} debug/TODO statements")

        return actions


def create_detailed_report():
    """Create detailed dead code report"""
    analyzer = DetailedDeadCodeAnalyzer()
    categorized_issues = analyzer.analyze()

    print(f"\n📋 DETAILED DEAD CODE ANALYSIS")
    print("=" * 50)

    total_issues = sum(len(issues) for issues in categorized_issues.values())
    print(f"Total issues found: {total_issues}")

    for issue_type, issues in categorized_issues.items():
        print(f"\n🔍 {issue_type.replace('_', ' ').title()}: {len(issues)} issues")

        # Show top 3 issues of each type
        for issue in issues[:3]:
            file_name = Path(issue.file_path).name
            print(f"  📁 {file_name}:{issue.line_number} - {issue.issue_description}")
            print(f"     💡 {issue.recommendation}")

        if len(issues) > 3:
            print(f"     ... and {len(issues) - 3} more")

    print(f"\n🎯 CLEANUP PLAN")
    print("=" * 30)

    cleanup_actions = analyzer.generate_cleanup_plan()
    for action in cleanup_actions:
        print(f"  {action}")

    return categorized_issues, cleanup_actions


if __name__ == "__main__":
    create_detailed_report()