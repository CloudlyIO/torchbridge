# Git Hooks for TorchBridge

This directory contains custom git hooks to maintain code quality and commit message standards.

## Available Hooks

### `commit-msg`

**Purpose**: Automatically removes Claude Code footers from commit messages

**What it removes**:
- Lines containing `🤖 Generated with [Claude Code]`
- Lines containing `Co-Authored-By: Claude Opus` or `Co-Authored-By: Claude Sonnet`
- Lines containing `Generated with [Claude Code]` (without emoji)
- Trailing blank lines after removal

**Installation**:

```bash
# Manual installation
cp .githooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg

# Or use the install script (recommended)
./scripts/install_hooks.sh
```

**Testing**:

```bash
# Create a test commit message
cat > /tmp/test_msg.txt << 'EOF'
feat: Test commit

Some changes here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF

# Run the hook
.git/hooks/commit-msg /tmp/test_msg.txt

# Verify - should not contain footer
cat /tmp/test_msg.txt
```

**Expected Output**:
```
feat: Test commit

Some changes here.
```

## Hook Management

### Install All Hooks

```bash
./scripts/install_hooks.sh
```

### Reinstall After Updates

```bash
# If hooks are updated in .githooks/, reinstall them
./scripts/install_hooks.sh --force
```

### Bypass Hooks (Not Recommended)

```bash
# Skip pre-commit hooks
git commit --no-verify

# This will bypass:
# - commit-msg hook (footer removal)
# - pre-commit hook (version checks)
```

## Hook Development

When creating new hooks:

1. Add the hook script to `.githooks/`
2. Make it executable: `chmod +x .githooks/hook-name`
3. Update `scripts/install_hooks.sh` to include the new hook
4. Document it in this README
5. Test thoroughly before committing

## Version

- **commit-msg**: v1.0 (2025-12-26)

## Notes

- Hooks in `.githooks/` are version controlled
- Hooks in `.git/hooks/` are local and not version controlled
- Always use the install script to ensure hooks are up to date
- Pre-commit hooks from `.pre-commit-config.yaml` run before these custom hooks
