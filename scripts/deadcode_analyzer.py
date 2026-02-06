"""
Dead Code Analyzer

Analyzes the codebase for:
- Unused imports
- Dead/unreachable code
- Redundant implementations
- Deprecated features ready for removal
- Ineffective code patterns
"""

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DeadCodeResult:
    """Result of dead code analysis"""
    file_path: str
    unused_imports: list[str]
    unused_functions: list[str]
    unused_classes: list[str]
    unreachable_code: list[str]
    redundant_code: list[str]
    deprecated_usage: list[str]


class DeadCodeAnalyzer:
    """Analyzes codebase for dead and ineffective code"""

    def __init__(self, src_path: str = "src"):
        self.src_path = Path(src_path)
        self.results: list[DeadCodeResult] = []
        self.global_usage = defaultdict(set)
        self.defined_names = defaultdict(set)

    def analyze_codebase(self) -> dict[str, Any]:
        """Perform comprehensive dead code analysis"""
        print("🔍 Starting dead code analysis...")

        # First pass: collect all definitions and usages
        self._collect_definitions_and_usage()

        # Second pass: analyze each file for dead code
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            result = self._analyze_file(py_file)
            if result:
                self.results.append(result)

        return self._generate_summary()

    def _collect_definitions_and_usage(self):
        """Collect all function/class definitions and their usage"""
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file) as f:
                    content = f.read()

                tree = ast.parse(content)

                # Collect definitions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.defined_names[str(py_file)].add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        self.defined_names[str(py_file)].add(node.name)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        self.global_usage[node.id].add(str(py_file))

            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def _analyze_file(self, file_path: Path) -> DeadCodeResult:
        """Analyze a single file for dead code"""
        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)

            return DeadCodeResult(
                file_path=str(file_path),
                unused_imports=self._find_unused_imports(tree, content),
                unused_functions=self._find_unused_functions(tree, file_path),
                unused_classes=self._find_unused_classes(tree, file_path),
                unreachable_code=self._find_unreachable_code(tree),
                redundant_code=self._find_redundant_code(tree, content),
                deprecated_usage=self._find_deprecated_usage(tree, content)
            )

        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return None

    def _find_unused_imports(self, tree: ast.AST, content: str) -> list[str]:
        """Find imports that are not used in the file"""
        unused_imports = []
        imported_names = set()
        used_names = set()

        # Collect imported names
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_names.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_names.add(name)

        # Collect used names (simple heuristic)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle module.attribute usage
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Also check string-based usage (for dynamic imports, etc.)
        lines = content.split('\n')
        for imported_name in imported_names:
            # Check if name appears in comments, strings, or other contexts
            name_in_strings = any(imported_name in line for line in lines
                                if not line.strip().startswith('import')
                                and not line.strip().startswith('from'))
            if name_in_strings:
                used_names.add(imported_name)

        # Find unused imports
        for imported_name in imported_names:
            if imported_name not in used_names and imported_name != "*":
                unused_imports.append(imported_name)

        return unused_imports

    def _find_unused_functions(self, tree: ast.AST, file_path: Path) -> list[str]:
        """Find functions that are never called"""
        unused_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name

                # Skip special methods and private methods (might be used dynamically)
                if func_name.startswith('__') or func_name.startswith('_'):
                    continue

                # Check if function is used globally
                if func_name not in self.global_usage or len(self.global_usage[func_name]) == 1:
                    # Only defined in this file and not used elsewhere
                    unused_functions.append(func_name)

        return unused_functions

    def _find_unused_classes(self, tree: ast.AST, file_path: Path) -> list[str]:
        """Find classes that are never instantiated"""
        unused_classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name

                # Skip private classes
                if class_name.startswith('_'):
                    continue

                # Check if class is used globally
                if class_name not in self.global_usage or len(self.global_usage[class_name]) == 1:
                    unused_classes.append(class_name)

        return unused_classes

    def _find_unreachable_code(self, tree: ast.AST) -> list[str]:
        """Find unreachable code patterns"""
        unreachable = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for if False: or if 0: patterns
                if isinstance(node.test, ast.Constant):
                    if node.test.value is False or node.test.value == 0:
                        unreachable.append(f"Unreachable if block at line {node.lineno}")

            elif isinstance(node, ast.FunctionDef):
                # Check for functions that only raise NotImplementedError
                if (len(node.body) == 1 and
                    isinstance(node.body[0], ast.Raise) and
                    isinstance(node.body[0].exc, ast.Call) and
                    isinstance(node.body[0].exc.func, ast.Name) and
                    node.body[0].exc.func.id == "NotImplementedError"):
                    unreachable.append(f"Stub function '{node.name}' at line {node.lineno}")

        return unreachable

    def _find_redundant_code(self, tree: ast.AST, content: str) -> list[str]:
        """Find redundant code patterns"""
        redundant = []

        # Check for redundant pass statements
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "pass":
                # Check if this pass is actually needed
                if i < len(lines):
                    next_line = lines[i].strip() if i < len(lines) else ""
                    if next_line and not next_line.startswith((' ', '\t')):
                        redundant.append(f"Redundant pass statement at line {i}")

        # Check for empty except blocks
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    redundant.append(f"Empty except block at line {node.lineno}")

        return redundant

    def _find_deprecated_usage(self, tree: ast.AST, content: str) -> list[str]:
        """Find usage of deprecated features"""
        deprecated = []

        # Check for deprecated module imports
        deprecated_modules = [
            "hardware_adaptation",
            "communication_optimization",
            "orchestration",
            "compiler_optimization_assistant"
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    for dep_mod in deprecated_modules:
                        if dep_mod in node.module:
                            deprecated.append(f"Deprecated import from {node.module} at line {node.lineno}")

        return deprecated

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary of dead code analysis"""
        summary = {
            "total_files_analyzed": len(self.results),
            "files_with_issues": len([r for r in self.results if any([
                r.unused_imports, r.unused_functions, r.unused_classes,
                r.unreachable_code, r.redundant_code, r.deprecated_usage
            ])]),
            "total_unused_imports": sum(len(r.unused_imports) for r in self.results),
            "total_unused_functions": sum(len(r.unused_functions) for r in self.results),
            "total_unused_classes": sum(len(r.unused_classes) for r in self.results),
            "total_unreachable_code": sum(len(r.unreachable_code) for r in self.results),
            "total_redundant_code": sum(len(r.redundant_code) for r in self.results),
            "total_deprecated_usage": sum(len(r.deprecated_usage) for r in self.results),
            "detailed_results": self.results
        }

        return summary

    def generate_cleanup_recommendations(self) -> list[str]:
        """Generate specific cleanup recommendations"""
        recommendations = []

        # High-priority cleanups
        total_unused_imports = sum(len(r.unused_imports) for r in self.results)
        if total_unused_imports > 0:
            recommendations.append(f"🧹 Remove {total_unused_imports} unused imports")

        total_unreachable = sum(len(r.unreachable_code) for r in self.results)
        if total_unreachable > 0:
            recommendations.append(f"🚫 Remove {total_unreachable} unreachable code blocks")

        total_redundant = sum(len(r.redundant_code) for r in self.results)
        if total_redundant > 0:
            recommendations.append(f"✂️ Clean up {total_redundant} redundant code patterns")

        # Medium-priority cleanups
        total_unused_functions = sum(len(r.unused_functions) for r in self.results)
        if total_unused_functions > 0:
            recommendations.append(f"🔍 Review {total_unused_functions} potentially unused functions")

        total_unused_classes = sum(len(r.unused_classes) for r in self.results)
        if total_unused_classes > 0:
            recommendations.append(f"📦 Review {total_unused_classes} potentially unused classes")

        # Deprecation cleanup
        total_deprecated = sum(len(r.deprecated_usage) for r in self.results)
        if total_deprecated > 0:
            recommendations.append(f"⚠️ Update {total_deprecated} deprecated imports")

        return recommendations


def analyze_dead_code(src_path: str = "src") -> None:
    """Main function to analyze dead code"""
    analyzer = DeadCodeAnalyzer(src_path)
    results = analyzer.analyze_codebase()

    print("\n📊 Dead Code Analysis Summary:")
    print(f"Files analyzed: {results['total_files_analyzed']}")
    print(f"Files with issues: {results['files_with_issues']}")
    print(f"Unused imports: {results['total_unused_imports']}")
    print(f"Unused functions: {results['total_unused_functions']}")
    print(f"Unused classes: {results['total_unused_classes']}")
    print(f"Unreachable code: {results['total_unreachable_code']}")
    print(f"Redundant code: {results['total_redundant_code']}")
    print(f"Deprecated usage: {results['total_deprecated_usage']}")

    print("\n🎯 Cleanup Recommendations:")
    recommendations = analyzer.generate_cleanup_recommendations()
    for rec in recommendations:
        print(f"  {rec}")

    # Show detailed results for files with issues
    if results['files_with_issues'] > 0:
        print("\n📋 Detailed Issues (Top 10 files):")
        issue_files = [r for r in results['detailed_results'] if any([
            r.unused_imports, r.unused_functions, r.unused_classes,
            r.unreachable_code, r.redundant_code, r.deprecated_usage
        ])]

        for result in issue_files[:10]:
            file_name = Path(result.file_path).name
            issues = []
            if result.unused_imports:
                issues.append(f"{len(result.unused_imports)} unused imports")
            if result.unused_functions:
                issues.append(f"{len(result.unused_functions)} unused functions")
            if result.unreachable_code:
                issues.append(f"{len(result.unreachable_code)} unreachable blocks")
            if result.deprecated_usage:
                issues.append(f"{len(result.deprecated_usage)} deprecated usage")

            if issues:
                print(f"  📁 {file_name}: {', '.join(issues)}")

    return results


if __name__ == "__main__":
    print("🔍 Starting dead code analysis...")
    results = analyze_dead_code()
    print("\n✅ Dead code analysis complete!")
