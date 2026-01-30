"""
Infrastructure Management for KernelPyTorch.

This module provides unified infrastructure management:
- Testing and validation infrastructure
- CI/CD pipeline management
- Deprecation tracking and lifecycle management

Consolidates functionality from:
- TestEnvironmentManager, CIPipelineManager
- DeprecationManager, and other infrastructure managers

Version: 0.3.11
"""

import warnings
from typing import Any

from .base import BaseManager, ManagerState, ManagerType


class InfrastructureManager(BaseManager):
    """
    Unified infrastructure management.

    Consolidates:
    - TestEnvironmentManager
    - CIPipelineManager
    - DeprecationManager
    - And other infrastructure managers
    """

    def _get_manager_type(self) -> ManagerType:
        return ManagerType.INFRASTRUCTURE

    def _initialize(self) -> None:
        """Initialize infrastructure management."""
        self.validation_config = self.config.validation

        # Initialize testing infrastructure
        self.testing_enabled = self.validation_config.enabled

        # Initialize lifecycle management
        self.deprecation_tracking: dict[str, dict[str, Any]] = {}

        # Track validation results
        self.validation_results: list[dict[str, Any]] = []

        self.context.state = ManagerState.READY
        self._initialized = True

    def optimize(self, target: Any, **kwargs) -> Any:
        """Apply infrastructure optimizations."""
        if not self._initialized:
            raise RuntimeError("InfrastructureManager not initialized")

        # Apply validation optimizations
        if self.testing_enabled:
            self._validate_target(target, **kwargs)

        # Apply deprecation management
        self._check_deprecations(target, **kwargs)

        return target

    def _validate_target(self, target: Any, **kwargs) -> None:
        """Validate target for infrastructure requirements."""
        validation_result = {
            'target_type': type(target).__name__,
            'valid': True,
            'warnings': []
        }

        # Check for common issues
        if hasattr(target, 'parameters'):
            param_count = sum(p.numel() for p in target.parameters())
            if param_count == 0:
                validation_result['warnings'].append('Model has no parameters')

        self.validation_results.append(validation_result)

    def _check_deprecations(self, target: Any, **kwargs) -> None:
        """Check for deprecated usage patterns."""
        target_type = type(target).__name__

        if target_type in self.deprecation_tracking:
            deprecation_info = self.deprecation_tracking[target_type]
            warnings.warn(
                f"{target_type} is deprecated: {deprecation_info.get('message', 'No details')}",
                DeprecationWarning,
            stacklevel=2,
            )

    def register_deprecation(
        self,
        component_name: str,
        message: str,
        removal_version: str | None = None,
        replacement: str | None = None
    ) -> None:
        """Register a deprecation for a component."""
        self.deprecation_tracking[component_name] = {
            'message': message,
            'removal_version': removal_version,
            'replacement': replacement
        }

    def get_deprecations(self) -> dict[str, dict[str, Any]]:
        """Get all registered deprecations."""
        return self.deprecation_tracking.copy()

    def get_validation_results(self) -> list[dict[str, Any]]:
        """Get validation results from last validation pass."""
        return self.validation_results.copy()

    def clear_validation_results(self) -> None:
        """Clear accumulated validation results."""
        self.validation_results.clear()
