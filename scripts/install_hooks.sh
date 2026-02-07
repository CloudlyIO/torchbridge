#!/bin/bash
#
# Install Git Hooks for TorchBridge
#
# This script installs custom git hooks from .githooks/ to .git/hooks/
#
# Usage:
#   ./scripts/install_hooks.sh [--force]
#
# Options:
#   --force    Overwrite existing hooks without prompting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GITHOOKS_DIR="$REPO_ROOT/.githooks"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

FORCE=false
if [[ "$1" == "--force" ]]; then
    FORCE=true
fi

echo "=================================================="
echo "  Installing TorchBridge Git Hooks"
echo "=================================================="
echo ""

# Check if .githooks directory exists
if [[ ! -d "$GITHOOKS_DIR" ]]; then
    echo "❌ Error: .githooks directory not found"
    exit 1
fi

# Check if .git/hooks directory exists
if [[ ! -d "$HOOKS_DIR" ]]; then
    echo "❌ Error: .git/hooks directory not found"
    echo "   Are you in a git repository?"
    exit 1
fi

# Install hooks
INSTALLED=0
SKIPPED=0

for hook_file in "$GITHOOKS_DIR"/*; do
    # Skip README and non-executable files
    if [[ "$(basename "$hook_file")" == "README.md" ]]; then
        continue
    fi

    hook_name=$(basename "$hook_file")
    dest="$HOOKS_DIR/$hook_name"

    # Check if hook already exists
    if [[ -f "$dest" ]] && [[ "$FORCE" != true ]]; then
        echo "⚠️  $hook_name already exists"
        read -p "   Overwrite? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "   Skipped $hook_name"
            ((SKIPPED++))
            continue
        fi
    fi

    # Copy and make executable
    cp "$hook_file" "$dest"
    chmod +x "$dest"
    echo "✅ Installed: $hook_name"
    ((INSTALLED++))
done

echo ""
echo "=================================================="
echo "  Installation Complete"
echo "=================================================="
echo "  Installed: $INSTALLED hook(s)"
if [[ $SKIPPED -gt 0 ]]; then
    echo "  Skipped:   $SKIPPED hook(s)"
fi
echo ""
echo "Installed hooks:"
for hook in "$HOOKS_DIR"/*; do
    if [[ -x "$hook" ]] && [[ "$(basename "$hook")" != *.sample ]]; then
        echo "  - $(basename "$hook")"
    fi
done
echo ""
echo "To test the commit-msg hook:"
echo "  1. Create a test commit with Claude footer"
echo "  2. The hook will automatically remove it"
echo ""
echo "To bypass hooks (not recommended):"
echo "  git commit --no-verify"
echo ""
