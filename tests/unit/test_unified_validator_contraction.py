"""
Regression tests for Contraction VIII — unified_validator.py dead-code removal.

Guards against re-introduction of:
- validate_custom_kernels() and its sub-methods (import deleted modules)
- validate_precision_allocation() and its stub sub-methods
- Stub sub-methods from validate_configuration()

Also verifies that the kept public API still works correctly.
"""

import pathlib


class TestDeletedMethods:
    # Public methods
    def test_validate_custom_kernels_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "validate_custom_kernels")

    def test_validate_precision_allocation_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "validate_precision_allocation")

    # Stub private methods from validate_configuration()
    def test_stub_validate_attention_config_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_attention_config")

    def test_stub_validate_hardware_config_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_hardware_config")

    def test_stub_validate_distributed_config_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_distributed_config")

    # Stub private methods from validate_precision_allocation()
    def test_stub_validate_precision_formats_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_precision_formats")

    def test_stub_validate_entropy_thresholds_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_entropy_thresholds")

    def test_stub_validate_memory_budget_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_memory_budget")

    # Private helpers from validate_custom_kernels()
    def test_validate_cuda_available_private_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_cuda_available")

    def test_validate_fused_activation_kernels_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_fused_activation_kernels")

    def test_validate_fp8_kernels_does_not_exist(self):
        from torchbridge.validation.unified_validator import UnifiedValidator
        assert not hasattr(UnifiedValidator, "_validate_fp8_kernels")


class TestDeadImportsAbsent:
    """Verify no dead imports to deleted modules remain in source."""

    _src = pathlib.Path(
        "src/torchbridge/validation/unified_validator.py"
    ).read_text()

    def test_no_kernel_registry_import(self):
        """core.kernel_registry was deleted in v0.5.55."""
        assert "kernel_registry" not in self._src

    def test_no_custom_kernels_import(self):
        """hardware.gpu.custom_kernels was deleted in v0.5.57."""
        assert "custom_kernels" not in self._src


class TestKeptAPIStillWorks:
    def test_validate_configuration_passes_on_default_config(self):
        from torchbridge.core.config import TorchBridgeConfig
        from torchbridge.validation.unified_validator import UnifiedValidator

        validator = UnifiedValidator()
        summary = validator.validate_configuration(TorchBridgeConfig())
        assert summary.total_tests > 0
        assert summary.failed == 0

    def test_validate_model_passes_on_simple_linear(self):
        import torch.nn as nn

        from torchbridge.validation.unified_validator import UnifiedValidator

        model = nn.Linear(32, 16)
        validator = UnifiedValidator()
        summary = validator.validate_model(model, (4, 32))
        assert summary.total_tests > 0

    def test_validate_hardware_compatibility_runs_on_cpu(self):
        import torch

        from torchbridge.validation.unified_validator import UnifiedValidator

        validator = UnifiedValidator()
        summary = validator.validate_hardware_compatibility(torch.device("cpu"))
        assert summary.total_tests > 0
        assert summary.failed == 0

    def test_convenience_functions_still_importable(self):
        from torchbridge.validation.unified_validator import (
            validate_configuration,
            validate_hardware,
            validate_model,
        )
        assert callable(validate_configuration)
        assert callable(validate_hardware)
        assert callable(validate_model)

    def test_convenience_validate_custom_kernels_does_not_exist(self):
        """Module-level convenience function must also be gone."""
        import torchbridge.validation.unified_validator as m
        assert not hasattr(m, "validate_custom_kernels")
