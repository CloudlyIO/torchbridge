# TorchBridge Templates

Ready-to-use templates for integrating TorchBridge into your CI/CD pipeline.

## GitHub Actions

### `torchbridge-validate.yml`

A GitHub Actions workflow that runs TorchBridge validation on every push and PR.

**Setup:**

1. Copy the workflow file into your project:

```bash
mkdir -p .github/workflows
cp templates/github-actions/torchbridge-validate.yml .github/workflows/
```

2. Ensure your project has TorchBridge as a dependency:

```
# requirements.txt or pyproject.toml
torchbridge>=0.5.0
```

3. Push to GitHub. The workflow will run automatically.

**What it does:**

- Runs `tb-doctor --ci` to check system compatibility
- Runs `tb-validate --ci --level quick` for quick validation
- Runs your test suite with pytest
- (Optional) Runs GPU validation on self-hosted runners

**Customization:**

- Change `--level quick` to `--level standard` or `--level full` for more thorough checks
- Add `--output report.json` to save validation reports as artifacts
- Adjust the Python version matrix to match your project requirements

## Using TorchBridge CLI in CI

The TorchBridge CLI tools use structured exit codes for CI integration:

| Exit Code | Meaning |
|-----------|---------|
| 0 | All checks passed |
| 1 | Failures detected |
| 2 | Warnings only (no failures) |

Use the `--ci` flag on `tb-doctor` and `tb-validate` for JSON output suitable for parsing in CI scripts.
