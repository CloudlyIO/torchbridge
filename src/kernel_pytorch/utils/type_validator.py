"""
Type Validation and Coverage Analysis (2025)

Provides utilities for validating type hints across the codebase and ensuring
comprehensive type coverage for public APIs.
"""

import inspect
import ast
from typing import (
    Dict, List, Set, Tuple, Optional, Union, Any, Callable,
    get_type_hints, get_origin, get_args
)
from dataclasses import dataclass
from pathlib import Path
import importlib


@dataclass
class TypeCoverageResult:
    """Results from type coverage analysis."""
    module_name: str
    total_functions: int
    typed_functions: int
    total_classes: int
    typed_classes: int
    coverage_percentage: float
    missing_types: List[str]
    issues: List[str]


class TypeValidator:
    """
    Validates and analyzes type hint coverage across modules.

    Ensures public APIs have comprehensive type annotations for better
    IDE support and code quality.
    """

    def __init__(self):
        self.results: Dict[str, TypeCoverageResult] = {}

    def analyze_module_coverage(self, module_name: str) -> TypeCoverageResult:
        """
        Analyze type hint coverage for a specific module.

        Args:
            module_name: Full module name to analyze

        Returns:
            Type coverage analysis results
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            return TypeCoverageResult(
                module_name=module_name,
                total_functions=0,
                typed_functions=0,
                total_classes=0,
                typed_classes=0,
                coverage_percentage=0.0,
                missing_types=[],
                issues=[f"Import error: {e}"]
            )

        total_functions = 0
        typed_functions = 0
        total_classes = 0
        typed_classes = 0
        missing_types = []
        issues = []

        # Analyze functions and methods
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue  # Skip private members

            if inspect.isfunction(obj):
                total_functions += 1
                if self._has_complete_type_hints(obj):
                    typed_functions += 1
                else:
                    missing_types.append(f"Function: {name}")

            elif inspect.isclass(obj):
                total_classes += 1
                class_typed = True

                # Check class methods
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if method_name.startswith('_') and method_name != '__init__':
                        continue  # Skip private methods except __init__

                    if not self._has_complete_type_hints(method):
                        class_typed = False
                        missing_types.append(f"Class {name}.{method_name}")

                if class_typed:
                    typed_classes += 1

        # Calculate coverage
        total_items = total_functions + total_classes
        typed_items = typed_functions + typed_classes
        coverage = (typed_items / total_items * 100) if total_items > 0 else 100.0

        result = TypeCoverageResult(
            module_name=module_name,
            total_functions=total_functions,
            typed_functions=typed_functions,
            total_classes=total_classes,
            typed_classes=typed_classes,
            coverage_percentage=coverage,
            missing_types=missing_types,
            issues=issues
        )

        self.results[module_name] = result
        return result

    def _has_complete_type_hints(self, func: Callable) -> bool:
        """
        Check if a function has complete type hints.

        Args:
            func: Function to check

        Returns:
            True if function has complete type annotations
        """
        try:
            # Get function signature
            sig = inspect.signature(func)

            # Skip if no parameters (except self/cls)
            params = [p for name, p in sig.parameters.items()
                     if name not in ('self', 'cls')]

            # Check parameter annotations
            for param in params:
                if param.annotation == inspect.Parameter.empty:
                    return False

            # Check return annotation
            if sig.return_annotation == inspect.Signature.empty:
                return False

            return True

        except (ValueError, TypeError):
            return False

    def analyze_public_apis(self, package_name: str) -> Dict[str, TypeCoverageResult]:
        """
        Analyze type coverage for all public APIs in a package.

        Args:
            package_name: Package to analyze (e.g., 'kernel_pytorch')

        Returns:
            Dictionary of module names to their coverage results
        """
        # Key public API modules to analyze
        key_modules = [
            f"{package_name}",
            f"{package_name}.components",
            f"{package_name}.utils.validation_framework",
            f"{package_name}.utils.compiler_assistant",
            f"{package_name}.distributed_scale",
            f"{package_name}.testing_framework",
        ]

        results = {}
        for module_name in key_modules:
            result = self.analyze_module_coverage(module_name)
            results[module_name] = result

        return results

    def generate_coverage_report(self) -> str:
        """Generate a comprehensive type coverage report."""
        if not self.results:
            return "No type coverage analysis results available."

        report = "# Type Coverage Analysis Report\n\n"

        # Summary statistics
        total_modules = len(self.results)
        total_functions = sum(r.total_functions for r in self.results.values())
        total_typed_functions = sum(r.typed_functions for r in self.results.values())
        total_classes = sum(r.total_classes for r in self.results.values())
        total_typed_classes = sum(r.typed_classes for r in self.results.values())

        overall_coverage = 0.0
        if total_functions + total_classes > 0:
            overall_coverage = ((total_typed_functions + total_typed_classes) /
                              (total_functions + total_classes) * 100)

        report += f"**Overall Coverage**: {overall_coverage:.1f}%\n"
        report += f"**Modules Analyzed**: {total_modules}\n"
        report += f"**Functions**: {total_typed_functions}/{total_functions} typed\n"
        report += f"**Classes**: {total_typed_classes}/{total_classes} typed\n\n"

        # Per-module breakdown
        report += "## Module Coverage Details\n\n"
        report += "| Module | Coverage | Functions | Classes | Issues |\n"
        report += "|--------|----------|-----------|---------|--------|\n"

        for module_name, result in self.results.items():
            issues_count = len(result.issues)
            missing_count = len(result.missing_types)

            report += (
                f"| {module_name} | {result.coverage_percentage:.1f}% | "
                f"{result.typed_functions}/{result.total_functions} | "
                f"{result.typed_classes}/{result.total_classes} | "
                f"{issues_count + missing_count} |\n"
            )

        # Missing type hints
        report += "\n## Missing Type Hints\n\n"
        for module_name, result in self.results.items():
            if result.missing_types:
                report += f"### {module_name}\n"
                for missing in result.missing_types[:10]:  # Limit to first 10
                    report += f"- {missing}\n"
                if len(result.missing_types) > 10:
                    report += f"- ... and {len(result.missing_types) - 10} more\n"
                report += "\n"

        return report

    def identify_priority_improvements(self) -> List[Tuple[str, str]]:
        """
        Identify priority areas for type hint improvements.

        Returns:
            List of (module_name, reason) tuples sorted by priority
        """
        improvements = []

        for module_name, result in self.results.items():
            if result.coverage_percentage < 80:
                reason = f"Low coverage: {result.coverage_percentage:.1f}%"
                improvements.append((module_name, reason))

            if result.issues:
                reason = f"Has issues: {len(result.issues)} problems"
                improvements.append((module_name, reason))

        # Sort by coverage percentage (lowest first)
        improvements.sort(key=lambda x: self.results[x[0]].coverage_percentage)
        return improvements


def enhance_type_hints_for_module(module_name: str) -> List[str]:
    """
    Suggest type hint enhancements for a specific module.

    Args:
        module_name: Module to enhance

    Returns:
        List of suggested improvements
    """
    validator = TypeValidator()
    result = validator.analyze_module_coverage(module_name)

    suggestions = []

    if result.coverage_percentage < 100:
        suggestions.append(f"Add type hints to {len(result.missing_types)} missing items")

    if result.issues:
        suggestions.extend(result.issues)

    return suggestions


def validate_critical_apis() -> Dict[str, Any]:
    """
    Validate type coverage for critical public APIs.

    Returns:
        Validation results with pass/fail status
    """
    validator = TypeValidator()

    # Critical modules that must have high type coverage
    critical_modules = [
        'kernel_pytorch.utils.validation_framework',
        'kernel_pytorch.utils.compiler_assistant',
        'kernel_pytorch.distributed_scale.hardware_discovery',
        'kernel_pytorch.testing_framework.unified_validator'
    ]

    results = {}
    all_passed = True

    for module_name in critical_modules:
        result = validator.analyze_module_coverage(module_name)
        passed = result.coverage_percentage >= 90  # 90% threshold for critical APIs

        results[module_name] = {
            'coverage': result.coverage_percentage,
            'passed': passed,
            'missing_types': len(result.missing_types),
            'issues': len(result.issues)
        }

        if not passed:
            all_passed = False

    return {
        'all_passed': all_passed,
        'modules': results,
        'summary': {
            'total_modules': len(critical_modules),
            'passed_modules': sum(1 for r in results.values() if r['passed']),
            'average_coverage': sum(r['coverage'] for r in results.values()) / len(results)
        }
    }


if __name__ == "__main__":
    print("🔍 Analyzing type hint coverage...")

    validator = TypeValidator()
    results = validator.analyze_public_apis('kernel_pytorch')

    print(f"\n📊 Analysis Complete:")
    print(f"  Modules analyzed: {len(results)}")

    # Print summary
    for module_name, result in results.items():
        status = "✅" if result.coverage_percentage >= 80 else "⚠️" if result.coverage_percentage >= 60 else "❌"
        print(f"  {status} {module_name}: {result.coverage_percentage:.1f}% coverage")

    print("\n" + validator.generate_coverage_report())