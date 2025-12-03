"""
Enhanced Deprecation Management System (2025)

Provides centralized deprecation warnings with clear migration paths, timelines,
and automated tracking for the Phase 2 refactoring transition period.
"""

import warnings
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os


class DeprecationManager:
    """
    Centralized deprecation warning system with enhanced migration guidance.

    Provides consistent deprecation messages across the codebase with:
    - Clear migration paths
    - Specific removal timelines
    - Documentation references
    - Usage tracking capabilities
    """

    # Centralized deprecation schedule
    DEPRECATION_SCHEDULE = {
        "hardware_adaptation": {
            "removal_version": "2.0.0",
            "removal_date": "June 2026",
            "new_imports": {
                "HardwareTopologyManager": "kernel_pytorch.distributed_scale.hardware_discovery",
                "DeviceMeshOptimizer": "kernel_pytorch.distributed_scale.hardware_adapter",
                "ThermalAwareScheduler": "kernel_pytorch.distributed_scale.thermal_power_management",
                "PowerEfficiencyOptimizer": "kernel_pytorch.distributed_scale.thermal_power_management"
            }
        },
        "communication_optimization": {
            "removal_version": "2.0.0",
            "removal_date": "June 2026",
            "new_imports": {
                "AdvancedCollectiveOps": "kernel_pytorch.distributed_scale.communication_primitives",
                "NetworkTopologyOptimizer": "kernel_pytorch.distributed_scale.network_optimization",
                "BandwidthAwareScheduler": "kernel_pytorch.distributed_scale.network_optimization",
                "CommunicationProfiler": "kernel_pytorch.distributed_scale.communication_profiling"
            }
        },
        "orchestration": {
            "removal_version": "2.0.0",
            "removal_date": "June 2026",
            "new_imports": {
                "KubernetesDistributedOrchestrator": "kernel_pytorch.distributed_scale.cluster_management",
                "SLURMClusterManager": "kernel_pytorch.distributed_scale.cluster_management",
                "AutoScalingManager": "kernel_pytorch.distributed_scale.scaling_fault_tolerance",
                "FaultToleranceManager": "kernel_pytorch.distributed_scale.scaling_fault_tolerance"
            }
        },
        "compiler_optimization_assistant": {
            "removal_version": "2.0.0",
            "removal_date": "June 2026",
            "new_imports": {
                "CompilerOptimizationAssistant": "kernel_pytorch.utils.compiler_assistant",
                "ModelArchitectureAnalyzer": "kernel_pytorch.utils.model_analyzer",
                "OptimizationRecommendationEngine": "kernel_pytorch.utils.optimization_recommendations"
            }
        }
    }

    @classmethod
    def warn_deprecated_module(
        cls,
        module_name: str,
        imported_class: Optional[str] = None,
        stacklevel: int = 3
    ) -> None:
        """
        Issue enhanced deprecation warning for deprecated module imports.

        Args:
            module_name: Name of the deprecated module (e.g., 'hardware_adaptation')
            imported_class: Specific class being imported (optional)
            stacklevel: Stack level for warning location
        """
        if module_name not in cls.DEPRECATION_SCHEDULE:
            # Fallback for unknown modules
            warnings.warn(
                f"Module {module_name} has been refactored. Check documentation for new import paths.",
                FutureWarning,
                stacklevel=stacklevel
            )
            return

        schedule = cls.DEPRECATION_SCHEDULE[module_name]
        removal_version = schedule["removal_version"]
        removal_date = schedule["removal_date"]

        if imported_class and imported_class in schedule["new_imports"]:
            new_import_path = schedule["new_imports"][imported_class]
            message = (
                f"Importing {imported_class} from {module_name} is deprecated and will be removed "
                f"in v{removal_version} ({removal_date}). "
                f"\n\nMigration: Use 'from {new_import_path} import {imported_class}' instead."
                f"\n\nSee REFACTORING_GUIDE.md for complete migration instructions."
            )
        else:
            new_imports = schedule["new_imports"]
            migration_examples = "\n".join([
                f"  from {path} import {cls_name}"
                for cls_name, path in list(new_imports.items())[:3]
            ])

            message = (
                f"Module {module_name} has been refactored and will be removed in v{removal_version} ({removal_date})."
                f"\n\nMigration examples:\n{migration_examples}"
                f"\n\nSee REFACTORING_GUIDE.md for complete details."
            )

        warnings.warn(message, FutureWarning, stacklevel=stacklevel)

    @classmethod
    def create_migration_guide_entry(cls, module_name: str) -> str:
        """Generate migration guide entry for a deprecated module."""
        if module_name not in cls.DEPRECATION_SCHEDULE:
            return f"# {module_name}: Migration guide not available"

        schedule = cls.DEPRECATION_SCHEDULE[module_name]
        new_imports = schedule["new_imports"]

        guide = f"""
### {module_name.replace('_', ' ').title()} Migration

**Removal Timeline**: v{schedule['removal_version']} ({schedule['removal_date']})

**Before (Deprecated)**:
```python
from kernel_pytorch.{module_name.replace('_', '.')} import ClassName
```

**After (Recommended)**:
```python
# New focused imports
"""
        for cls_name, new_path in new_imports.items():
            guide += f"from {new_path} import {cls_name}\n"

        guide += "```"
        return guide

    @classmethod
    def check_deprecation_status(cls) -> Dict[str, Any]:
        """
        Check status of all deprecations for monitoring purposes.

        Returns:
            Dictionary with deprecation status information
        """
        status = {
            "total_deprecated_modules": len(cls.DEPRECATION_SCHEDULE),
            "next_removal_date": "June 2026",
            "modules": {}
        }

        for module_name, schedule in cls.DEPRECATION_SCHEDULE.items():
            status["modules"][module_name] = {
                "removal_version": schedule["removal_version"],
                "removal_date": schedule["removal_date"],
                "migration_paths_count": len(schedule["new_imports"])
            }

        return status


def warn_deprecated_import(
    old_module: str,
    class_name: Optional[str] = None,
    stacklevel: int = 3
) -> None:
    """
    Convenience function for issuing deprecation warnings.

    Args:
        old_module: Name of deprecated module
        class_name: Specific class being imported
        stacklevel: Stack level for warning location
    """
    DeprecationManager.warn_deprecated_module(old_module, class_name, stacklevel)


# Migration utility functions
def generate_full_migration_guide() -> str:
    """Generate complete migration guide for all deprecated modules."""
    guide = """
# Phase 2 Refactoring Migration Guide

This guide helps you migrate from deprecated module imports to the new focused module structure.

## Overview

The Phase 2 refactoring split large monolithic files into focused modules while maintaining
100% backward compatibility through deprecation warnings.

"""

    for module_name in DeprecationManager.DEPRECATION_SCHEDULE.keys():
        guide += DeprecationManager.create_migration_guide_entry(module_name)
        guide += "\n\n"

    guide += """
## Migration Timeline

- **Phase 1 (Current - Next 6 months)**: Deprecated imports work with warnings
- **Phase 2 (6-12 months)**: Warnings become more prominent
- **Phase 3 (12+ months)**: Deprecated imports removed in v2.0.0

## Getting Help

For migration assistance:
1. Check REFACTORING_GUIDE.md for detailed examples
2. Use the new focused imports for better maintainability
3. Update your imports when convenient - no rush required
"""

    return guide