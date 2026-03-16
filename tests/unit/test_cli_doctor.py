"""
Unit tests for tb-doctor CLI fixes (v0.5.75).

Covers:
- --fix help text accurately describes behavior (does not imply system modification)
"""

import argparse


def _get_fix_help_text() -> str:
    """Extract the --fix flag help string from the doctor parser."""
    from torchbridge.cli.doctor import DoctorCommand

    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd")
    DoctorCommand.register(sub)

    # Walk the subparser actions to find --fix
    doctor_parser = None
    for action in root._subparsers._group_actions:
        for choice_name, choice_parser in action.choices.items():
            if choice_name == "doctor":
                doctor_parser = choice_parser
                break

    assert doctor_parser is not None, "doctor subparser not found"

    for action in doctor_parser._actions:
        if "--fix" in getattr(action, "option_strings", []):
            return action.help or ""

    raise AssertionError("--fix action not found in doctor parser")


class TestFixFlagHelpText:
    def test_fix_flag_help_does_not_imply_system_modification(self):
        """--fix help must NOT claim the system is modified (e.g., 'fix detected issues')."""
        help_text = _get_fix_help_text().lower()
        # These phrases imply actual modification — must not appear
        misleading_phrases = ["attempt to fix", "fix detected issues"]
        for phrase in misleading_phrases:
            assert phrase not in help_text, (
                f"--fix help text is misleading: contains '{phrase}'. "
                f"Full help: '{_get_fix_help_text()}'"
            )

    def test_fix_flag_help_mentions_no_system_modification(self):
        """--fix help must clarify it does not modify the system."""
        help_text = _get_fix_help_text().lower()
        assert "does not modify" in help_text, (
            f"--fix help must state 'does not modify'. Full help: '{_get_fix_help_text()}'"
        )

    def test_fix_flag_help_mentions_remediation_or_steps(self):
        """--fix help must mention remediation steps (the actual behavior)."""
        help_text = _get_fix_help_text().lower()
        keywords = ["remediation", "steps", "instructions", "print"]
        assert any(kw in help_text for kw in keywords), (
            f"--fix help must mention remediation/steps. Full help: '{_get_fix_help_text()}'"
        )
