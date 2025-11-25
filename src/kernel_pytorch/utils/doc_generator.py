"""
Documentation Auto-Generation System (2025)

Automatically generates comprehensive API documentation from the clean package
structure created in Phase 2 refactoring, leveraging type hints and docstrings.
"""

import inspect
import importlib
import ast
from typing import Dict, List, Set, Tuple, Optional, Any, Type
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime


@dataclass
class APIDocItem:
    """Documentation item for an API component."""
    name: str
    type: str  # 'function', 'class', 'method', 'module'
    signature: str
    docstring: str
    module_path: str
    parameters: List[Dict[str, Any]]
    return_type: str
    examples: List[str]
    deprecation_info: Optional[str] = None


class DocumentationGenerator:
    """
    Generates comprehensive API documentation from package structure.

    Leverages the clean module organization from Phase 2 refactoring to create
    organized, navigable documentation with examples and migration guides.
    """

    def __init__(self, package_name: str = "kernel_pytorch"):
        self.package_name = package_name
        self.doc_items: List[APIDocItem] = []
        self.module_hierarchy: Dict[str, List[str]] = {}

    def generate_full_documentation(self) -> Dict[str, Any]:
        """
        Generate complete documentation for the package.

        Returns:
            Comprehensive documentation structure with API reference,
            examples, and migration guides.
        """
        # Discover and analyze modules
        modules = self._discover_modules()

        # Generate documentation for each module
        for module_name in modules:
            self._process_module(module_name)

        # Organize documentation
        return self._organize_documentation()

    def _discover_modules(self) -> List[str]:
        """Discover all modules in the package."""
        try:
            package = importlib.import_module(self.package_name)
            package_path = Path(package.__file__).parent

            modules = []
            for py_file in package_path.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                # Convert path to module name
                rel_path = py_file.relative_to(package_path.parent)
                module_name = str(rel_path.with_suffix("")).replace("/", ".")

                if not module_name.endswith("_test"):
                    modules.append(module_name)

            return sorted(modules)
        except ImportError:
            return []

    def _process_module(self, module_name: str) -> None:
        """Process a single module to extract documentation."""
        try:
            module = importlib.import_module(module_name)

            # Extract module-level documentation
            module_doc = APIDocItem(
                name=module_name,
                type="module",
                signature="",
                docstring=inspect.getdoc(module) or "",
                module_path=module_name,
                parameters=[],
                return_type="",
                examples=[]
            )
            self.doc_items.append(module_doc)

            # Process classes and functions
            for name, obj in inspect.getmembers(module):
                if name.startswith("_"):
                    continue

                if inspect.isclass(obj) and obj.__module__ == module_name:
                    self._process_class(obj, module_name)
                elif inspect.isfunction(obj) and obj.__module__ == module_name:
                    self._process_function(obj, module_name)

        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")

    def _process_class(self, cls: Type, module_name: str) -> None:
        """Process a class to extract documentation."""
        class_doc = self._extract_api_doc_item(cls, "class", module_name)
        self.doc_items.append(class_doc)

        # Process class methods
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_") or name == "__init__":
                method_doc = self._extract_api_doc_item(method, "method", module_name, parent_class=cls.__name__)
                self.doc_items.append(method_doc)

    def _process_function(self, func: Any, module_name: str) -> None:
        """Process a function to extract documentation."""
        func_doc = self._extract_api_doc_item(func, "function", module_name)
        self.doc_items.append(func_doc)

    def _extract_api_doc_item(
        self,
        obj: Any,
        obj_type: str,
        module_name: str,
        parent_class: Optional[str] = None
    ) -> APIDocItem:
        """Extract documentation for an API object."""
        name = obj.__name__
        if parent_class:
            name = f"{parent_class}.{name}"

        # Get signature
        try:
            signature = str(inspect.signature(obj))
        except (ValueError, TypeError):
            signature = ""

        # Extract parameters and return type
        parameters, return_type = self._extract_type_info(obj)

        # Check for deprecation
        deprecation_info = self._check_deprecation(obj, module_name)

        # Extract examples from docstring
        examples = self._extract_examples(inspect.getdoc(obj) or "")

        return APIDocItem(
            name=name,
            type=obj_type,
            signature=signature,
            docstring=inspect.getdoc(obj) or "",
            module_path=module_name,
            parameters=parameters,
            return_type=return_type,
            examples=examples,
            deprecation_info=deprecation_info
        )

    def _extract_type_info(self, obj: Any) -> Tuple[List[Dict[str, Any]], str]:
        """Extract parameter and return type information."""
        try:
            sig = inspect.signature(obj)
            parameters = []

            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                    "required": param.default == inspect.Parameter.empty
                }
                parameters.append(param_info)

            return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "Any"

            return parameters, return_type

        except (ValueError, TypeError):
            return [], "Any"

    def _check_deprecation(self, obj: Any, module_name: str) -> Optional[str]:
        """Check if an object or its module is deprecated."""
        # Check for deprecated modules from Phase 2 refactoring
        deprecated_modules = [
            "hardware_adaptation",
            "communication_optimization",
            "orchestration",
            "compiler_optimization_assistant"
        ]

        for deprecated in deprecated_modules:
            if deprecated in module_name:
                return f"Module {deprecated} is deprecated. See REFACTORING_GUIDE.md for migration."

        return None

    def _extract_examples(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []
        lines = docstring.split('\n')
        in_example = False
        current_example = []

        for line in lines:
            if 'Example:' in line or '```python' in line:
                in_example = True
                continue
            elif '```' in line and in_example:
                if current_example:
                    examples.append('\n'.join(current_example))
                    current_example = []
                in_example = False
            elif in_example:
                current_example.append(line)

        if current_example:
            examples.append('\n'.join(current_example))

        return examples

    def _organize_documentation(self) -> Dict[str, Any]:
        """Organize documentation into structured format."""
        organized = {
            "package_info": {
                "name": self.package_name,
                "generated_at": datetime.now().isoformat(),
                "total_items": len(self.doc_items)
            },
            "modules": {},
            "api_reference": {},
            "deprecated_apis": [],
            "migration_guide": self._generate_migration_guide()
        }

        # Group by module
        for item in self.doc_items:
            module_name = item.module_path
            if module_name not in organized["modules"]:
                organized["modules"][module_name] = {
                    "classes": [],
                    "functions": [],
                    "methods": []
                }

            if item.type == "module":
                organized["modules"][module_name]["description"] = item.docstring
            elif item.type == "class":
                organized["modules"][module_name]["classes"].append(item)
            elif item.type == "function":
                organized["modules"][module_name]["functions"].append(item)
            elif item.type == "method":
                organized["modules"][module_name]["methods"].append(item)

            # Add to API reference
            organized["api_reference"][item.name] = {
                "signature": item.signature,
                "docstring": item.docstring,
                "module": item.module_path,
                "parameters": item.parameters,
                "return_type": item.return_type,
                "examples": item.examples
            }

            # Track deprecated APIs
            if item.deprecation_info:
                organized["deprecated_apis"].append({
                    "name": item.name,
                    "module": item.module_path,
                    "deprecation_info": item.deprecation_info
                })

        return organized

    def _generate_migration_guide(self) -> Dict[str, str]:
        """Generate migration guide for deprecated APIs."""
        from .deprecation_manager import DeprecationManager

        guide = {}
        for module_name in DeprecationManager.DEPRECATION_SCHEDULE:
            guide[module_name] = DeprecationManager.create_migration_guide_entry(module_name)

        return guide

    def generate_markdown_docs(self, output_dir: Path) -> None:
        """Generate markdown documentation files."""
        docs = self.generate_full_documentation()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Generate main API reference
        self._write_api_reference_md(docs, output_dir / "API_REFERENCE.md")

        # Generate module-specific docs
        for module_name, module_info in docs["modules"].items():
            if module_name.startswith(self.package_name):
                filename = module_name.replace(".", "_") + ".md"
                self._write_module_md(module_name, module_info, output_dir / filename)

        # Generate migration guide
        self._write_migration_guide_md(docs["migration_guide"], output_dir / "MIGRATION_GUIDE.md")

        print(f"📚 Documentation generated in {output_dir}")

    def _write_api_reference_md(self, docs: Dict[str, Any], output_file: Path) -> None:
        """Write main API reference markdown file."""
        content = f"""# {docs['package_info']['name']} API Reference

Generated on: {docs['package_info']['generated_at']}
Total API items: {docs['package_info']['total_items']}

## Quick Navigation

"""

        # Add navigation links
        for module_name in sorted(docs["modules"].keys()):
            if module_name.startswith(self.package_name):
                clean_name = module_name.replace(self.package_name + ".", "")
                content += f"- [{clean_name}]({module_name.replace('.', '_')}.md)\n"

        content += "\n## Deprecated APIs\n\n"
        for dep_api in docs["deprecated_apis"]:
            content += f"- **{dep_api['name']}** ({dep_api['module']}): {dep_api['deprecation_info']}\n"

        content += "\n## Package Structure\n\n"
        content += "The package is organized into focused modules following the Phase 2 refactoring:\n\n"

        for module_name, module_info in docs["modules"].items():
            if module_name.startswith(self.package_name) and "description" in module_info:
                clean_name = module_name.replace(self.package_name + ".", "")
                content += f"### {clean_name}\n\n"
                content += f"{module_info.get('description', 'No description available.')}\n\n"

        output_file.write_text(content)

    def _write_module_md(self, module_name: str, module_info: Dict[str, Any], output_file: Path) -> None:
        """Write module-specific markdown documentation."""
        clean_name = module_name.replace(self.package_name + ".", "")

        content = f"""# {clean_name}

{module_info.get('description', 'No description available.')}

"""

        # Classes
        if module_info.get('classes'):
            content += "## Classes\n\n"
            for cls in module_info['classes']:
                content += f"### {cls.name}\n\n"
                content += f"```python\n{cls.signature}\n```\n\n"
                content += f"{cls.docstring}\n\n"

                if cls.examples:
                    content += "**Examples:**\n\n"
                    for example in cls.examples:
                        content += f"```python\n{example}\n```\n\n"

        # Functions
        if module_info.get('functions'):
            content += "## Functions\n\n"
            for func in module_info['functions']:
                content += f"### {func.name}\n\n"
                content += f"```python\n{func.signature}\n```\n\n"
                content += f"{func.docstring}\n\n"

                if func.examples:
                    content += "**Examples:**\n\n"
                    for example in func.examples:
                        content += f"```python\n{example}\n```\n\n"

        output_file.write_text(content)

    def _write_migration_guide_md(self, migration_guide: Dict[str, str], output_file: Path) -> None:
        """Write migration guide markdown file."""
        content = "# Phase 2 Migration Guide\n\n"
        content += "This guide helps migrate from deprecated modules to the new refactored structure.\n\n"

        for module_name, guide_text in migration_guide.items():
            content += guide_text + "\n\n"

        output_file.write_text(content)


def generate_package_documentation(package_name: str = "kernel_pytorch", output_dir: str = "docs/api") -> None:
    """
    Generate comprehensive documentation for the package.

    Args:
        package_name: Name of package to document
        output_dir: Output directory for documentation files
    """
    generator = DocumentationGenerator(package_name)

    # Generate JSON API reference
    docs = generator.generate_full_documentation()

    # Write JSON documentation
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "api_reference.json", "w") as f:
        json.dump(docs, f, indent=2, default=str)

    # Generate markdown documentation
    generator.generate_markdown_docs(output_path)

    print(f"✅ Documentation generated successfully in {output_path}")


if __name__ == "__main__":
    print("📚 Generating package documentation...")
    generate_package_documentation()
    print("✅ Documentation generation complete!")