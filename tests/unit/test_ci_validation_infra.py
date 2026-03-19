"""
CI validation infrastructure existence tests.

Verifies that the GPU validation workflow and standalone script exist with the
right properties. These tests FAIL until the files are created.
"""

import os
import stat
import subprocess
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_WORKFLOW = os.path.join(_REPO_ROOT, ".github", "workflows", "gpu-validation.yml")
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "validation", "run_gpu_validation.py")


class TestCIValidationInfra:
    """GPU validation CI infrastructure must exist and be correctly configured."""

    def test_workflow_file_exists(self):
        assert os.path.isfile(_WORKFLOW), (
            f"GPU validation workflow not found at {_WORKFLOW}"
        )

    def test_workflow_has_workflow_dispatch(self):
        with open(_WORKFLOW) as f:
            content = f.read()
        assert "workflow_dispatch" in content, (
            "gpu-validation.yml must have a workflow_dispatch trigger for manual runs"
        )

    def test_validation_script_exists(self):
        assert os.path.isfile(_SCRIPT), (
            f"Validation script not found at {_SCRIPT}"
        )

    def test_validation_script_is_executable(self):
        mode = os.stat(_SCRIPT).st_mode
        assert mode & stat.S_IXUSR, (
            f"{_SCRIPT} is not executable — run: chmod +x {_SCRIPT}"
        )

    def test_validation_script_imports_cleanly(self):
        """python3 run_gpu_validation.py --help must exit 0."""
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"run_gpu_validation.py --help exited {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
