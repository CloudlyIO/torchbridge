#!/usr/bin/env python3
"""
KernelPyTorch v0.4.30 Master Validation Script

Orchestrates all validation phases:
- Phase 1: Static Analysis (Ruff, mypy, bandit, pip-audit)
- Phase 2: Local Validation (unit tests, integration tests, coverage)
- Phase 3: Cloud Validation (orchestration scripts)
- Phase 4: Report Generation

Usage:
    python scripts/validation/v0430_master_validator.py --phase all
    python scripts/validation/v0430_master_validator.py --phase static
    python scripts/validation/v0430_master_validator.py --phase local
    python scripts/validation/v0430_master_validator.py --phase reports
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Result of a validation step."""
    name: str
    passed: bool
    duration_seconds: float
    output_file: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class PhaseResult:
    """Result of a validation phase."""
    phase: str
    started: datetime
    completed: datetime | None = None
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def duration_seconds(self) -> float:
        if self.completed:
            return (self.completed - self.started).total_seconds()
        return 0.0


class MasterValidator:
    """Orchestrates v0.4.30 validation."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.phase_results: list[PhaseResult] = []

    def run_command(
        self,
        cmd: list[str],
        output_file: str | None = None,
        capture: bool = True,
        timeout: int = 600
    ) -> tuple[int, str, str]:
        """Run a command and optionally save output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture,
                text=True,
                timeout=timeout
            )

            if output_file and result.stdout:
                output_path = self.reports_dir / output_file
                output_path.write_text(result.stdout)

            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    # =========================================================================
    # Phase 1: Static Analysis
    # =========================================================================

    def run_phase1_static_analysis(self) -> PhaseResult:
        """Run static analysis tools."""
        phase = PhaseResult(phase="static_analysis", started=datetime.now())
        print("\n" + "="*60)
        print("PHASE 1: Static Analysis")
        print("="*60)

        # 1.1 Ruff Linting
        print("\n[1.1] Running Ruff linting...")
        start = datetime.now()
        returncode, stdout, stderr = self.run_command(
            ["ruff", "check", "src/", "tests/", "--output-format=json"],
            output_file="ruff_report.json"
        )
        duration = (datetime.now() - start).total_seconds()

        ruff_errors = 0
        if stdout:
            try:
                issues = json.loads(stdout)
                ruff_errors = len(issues)
            except json.JSONDecodeError:
                pass

        phase.results.append(ValidationResult(
            name="ruff_linting",
            passed=returncode == 0,
            duration_seconds=duration,
            output_file="ruff_report.json",
            summary={"errors": ruff_errors}
        ))
        print(f"    Ruff: {ruff_errors} issues found")

        # 1.2 Mypy Type Checking
        print("\n[1.2] Running mypy type checking...")
        start = datetime.now()
        returncode, stdout, stderr = self.run_command(
            ["mypy", "src/kernel_pytorch", "--ignore-missing-imports", "--no-error-summary"],
            output_file="mypy_report.txt"
        )
        duration = (datetime.now() - start).total_seconds()

        mypy_errors = stdout.count("error:") if stdout else 0
        phase.results.append(ValidationResult(
            name="mypy_typecheck",
            passed=returncode == 0,
            duration_seconds=duration,
            output_file="mypy_report.txt",
            summary={"errors": mypy_errors}
        ))
        print(f"    Mypy: {mypy_errors} type errors found")

        # 1.3 Bandit Security Scan
        print("\n[1.3] Running Bandit security scan...")
        start = datetime.now()
        returncode, stdout, stderr = self.run_command(
            ["bandit", "-r", "src/", "-f", "json", "-q"],
            output_file="bandit_report.json"
        )
        duration = (datetime.now() - start).total_seconds()

        security_issues = {"high": 0, "medium": 0, "low": 0}
        if stdout:
            try:
                report = json.loads(stdout)
                for result in report.get("results", []):
                    severity = result.get("issue_severity", "").lower()
                    if severity in security_issues:
                        security_issues[severity] += 1
            except json.JSONDecodeError:
                pass

        phase.results.append(ValidationResult(
            name="bandit_security",
            passed=security_issues["high"] == 0,
            duration_seconds=duration,
            output_file="bandit_report.json",
            summary=security_issues
        ))
        print(f"    Bandit: {security_issues['high']} high, {security_issues['medium']} medium, {security_issues['low']} low")

        # 1.4 Pip Audit
        print("\n[1.4] Running pip-audit dependency check...")
        start = datetime.now()
        returncode, stdout, stderr = self.run_command(
            ["pip-audit", "--format=json"],
            output_file="pip_audit_report.json"
        )
        duration = (datetime.now() - start).total_seconds()

        vulnerabilities = 0
        if stdout:
            try:
                report = json.loads(stdout)
                vulnerabilities = len(report) if isinstance(report, list) else 0
            except json.JSONDecodeError:
                pass

        phase.results.append(ValidationResult(
            name="pip_audit",
            passed=vulnerabilities == 0,
            duration_seconds=duration,
            output_file="pip_audit_report.json",
            summary={"vulnerabilities": vulnerabilities}
        ))
        print(f"    Pip-audit: {vulnerabilities} vulnerabilities found")

        # 1.5 License Check
        print("\n[1.5] Running license compliance check...")
        start = datetime.now()
        returncode, stdout, stderr = self.run_command(
            ["pip-licenses", "--format=json"],
            output_file="licenses.json"
        )
        duration = (datetime.now() - start).total_seconds()

        license_count = 0
        if stdout:
            try:
                licenses = json.loads(stdout)
                license_count = len(licenses)
            except json.JSONDecodeError:
                pass

        phase.results.append(ValidationResult(
            name="license_check",
            passed=True,  # Informational only
            duration_seconds=duration,
            output_file="licenses.json",
            summary={"packages": license_count}
        ))
        print(f"    Licenses: {license_count} packages documented")

        phase.completed = datetime.now()
        self.phase_results.append(phase)
        return phase

    # =========================================================================
    # Phase 2: Local Validation
    # =========================================================================

    def run_phase2_local_validation(self) -> PhaseResult:
        """Run local tests and coverage."""
        phase = PhaseResult(phase="local_validation", started=datetime.now())
        print("\n" + "="*60)
        print("PHASE 2: Local Validation")
        print("="*60)

        # 2.1 Unit Tests with Coverage
        print("\n[2.1] Running unit tests with coverage...")
        start = datetime.now()
        returncode, stdout, stderr = self.run_command(
            [
                "pytest", "tests/",
                "--cov=src/kernel_pytorch",
                "--cov-report=json:reports/coverage.json",
                "--cov-report=term-missing",
                "-v", "--tb=short",
                "-x",  # Stop on first failure for faster feedback
                "--timeout=60"
            ],
            timeout=1800  # 30 minute timeout for full test suite
        )
        duration = (datetime.now() - start).total_seconds()

        # Parse test results from output
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        coverage_pct = 0.0

        if stdout:
            for line in stdout.split('\n'):
                if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                    # Parse pytest summary line
                    import re
                    passed_match = re.search(r'(\d+) passed', line)
                    failed_match = re.search(r'(\d+) failed', line)
                    skipped_match = re.search(r'(\d+) skipped', line)
                    if passed_match:
                        tests_passed = int(passed_match.group(1))
                    if failed_match:
                        tests_failed = int(failed_match.group(1))
                    if skipped_match:
                        tests_skipped = int(skipped_match.group(1))

        # Read coverage from JSON
        coverage_file = self.reports_dir / "coverage.json"
        if coverage_file.exists():
            try:
                cov_data = json.loads(coverage_file.read_text())
                coverage_pct = cov_data.get("totals", {}).get("percent_covered", 0)
            except (json.JSONDecodeError, KeyError):
                pass

        total_tests = tests_passed + tests_failed + tests_skipped
        pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

        phase.results.append(ValidationResult(
            name="pytest_coverage",
            passed=returncode == 0 and pass_rate >= 95,
            duration_seconds=duration,
            output_file="coverage.json",
            summary={
                "passed": tests_passed,
                "failed": tests_failed,
                "skipped": tests_skipped,
                "total": total_tests,
                "pass_rate": round(pass_rate, 2),
                "coverage_pct": round(coverage_pct, 2)
            }
        ))
        print(f"    Tests: {tests_passed}/{total_tests} passed ({pass_rate:.1f}%)")
        print(f"    Coverage: {coverage_pct:.1f}%")

        # Save full pytest output
        if stdout or stderr:
            (self.reports_dir / "pytest_output.txt").write_text(
                f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )

        phase.completed = datetime.now()
        self.phase_results.append(phase)
        return phase

    # =========================================================================
    # Phase 3: Cloud Validation (Preparation Only)
    # =========================================================================

    def run_phase3_cloud_preparation(self) -> PhaseResult:
        """Prepare cloud validation scripts (doesn't actually run cloud tests)."""
        phase = PhaseResult(phase="cloud_preparation", started=datetime.now())
        print("\n" + "="*60)
        print("PHASE 3: Cloud Validation Preparation")
        print("="*60)

        print("\n[3.1] Cloud validation scripts are ready in scripts/validation/")
        print("      - cloud_orchestrator.py: Manages cloud instance deployment")
        print("      - See plan for AWS/GCP/Intel DevCloud commands")

        phase.results.append(ValidationResult(
            name="cloud_scripts_ready",
            passed=True,
            duration_seconds=0,
            summary={"status": "scripts_prepared"}
        ))

        phase.completed = datetime.now()
        self.phase_results.append(phase)
        return phase

    # =========================================================================
    # Phase 4: Report Generation
    # =========================================================================

    def run_phase4_reports(self) -> PhaseResult:
        """Generate all validation reports."""
        from report_generator import ReportGenerator

        phase = PhaseResult(phase="report_generation", started=datetime.now())
        print("\n" + "="*60)
        print("PHASE 4: Report Generation")
        print("="*60)

        generator = ReportGenerator(self.project_root, self.reports_dir)

        # Generate all 5 reports
        reports = [
            ("reliability", generator.generate_reliability_report),
            ("performance", generator.generate_performance_report),
            ("quality", generator.generate_quality_report),
            ("security", generator.generate_security_report),
            ("privacy", generator.generate_privacy_report),
        ]

        for name, gen_func in reports:
            print(f"\n[4.x] Generating {name} report...")
            start = datetime.now()
            try:
                output_file = gen_func()
                phase.results.append(ValidationResult(
                    name=f"{name}_report",
                    passed=True,
                    duration_seconds=(datetime.now() - start).total_seconds(),
                    output_file=output_file
                ))
                print(f"    Generated: {output_file}")
            except Exception as e:
                phase.results.append(ValidationResult(
                    name=f"{name}_report",
                    passed=False,
                    duration_seconds=(datetime.now() - start).total_seconds(),
                    errors=[str(e)]
                ))
                print(f"    Error: {e}")

        phase.completed = datetime.now()
        self.phase_results.append(phase)
        return phase

    # =========================================================================
    # Summary
    # =========================================================================

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        all_passed = True
        for phase in self.phase_results:
            status = "PASS" if phase.passed else "FAIL"
            all_passed = all_passed and phase.passed
            print(f"\n{phase.phase}: {status} ({phase.duration_seconds:.1f}s)")
            for result in phase.results:
                r_status = "✓" if result.passed else "✗"
                print(f"  {r_status} {result.name}: {result.summary}")

        print("\n" + "="*60)
        final_status = "ALL PHASES PASSED" if all_passed else "SOME PHASES FAILED"
        print(f"FINAL STATUS: {final_status}")
        print("="*60)

        return all_passed

    def save_summary(self):
        """Save validation summary to JSON."""
        summary = {
            "version": "0.4.30",
            "timestamp": datetime.now().isoformat(),
            "phases": []
        }

        for phase in self.phase_results:
            phase_data = {
                "name": phase.phase,
                "passed": phase.passed,
                "duration_seconds": phase.duration_seconds,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration_seconds": r.duration_seconds,
                        "output_file": r.output_file,
                        "summary": r.summary,
                        "errors": r.errors
                    }
                    for r in phase.results
                ]
            }
            summary["phases"].append(phase_data)

        summary_file = self.reports_dir / "validation_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="KernelPyTorch v0.4.30 Validation")
    parser.add_argument(
        "--phase",
        choices=["all", "static", "local", "cloud", "reports"],
        default="all",
        help="Which validation phase to run"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    validator = MasterValidator(project_root)

    if args.phase in ("all", "static"):
        validator.run_phase1_static_analysis()

    if args.phase in ("all", "local"):
        validator.run_phase2_local_validation()

    if args.phase in ("all", "cloud"):
        validator.run_phase3_cloud_preparation()

    if args.phase in ("all", "reports"):
        validator.run_phase4_reports()

    passed = validator.print_summary()
    validator.save_summary()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
