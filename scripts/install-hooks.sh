#!/bin/bash
# Install git hooks from scripts/hooks to .git/hooks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

HOOKS_DIR="scripts/hooks"
GIT_HOOKS_DIR=".git/hooks"

echo "Installing git hooks..."

# Copy and make executable each hook
for hook in "$HOOKS_DIR"/*; do
    hook_name=$(basename "$hook")
    echo "  Installing $hook_name..."
    cp "$hook" "$GIT_HOOKS_DIR/$hook_name"
    chmod +x "$GIT_HOOKS_DIR/$hook_name"
done

echo "Git hooks installed successfully!"
echo ""
echo "Installed hooks:"
echo "  - pre-commit: Run linter and mypy on staged files before commit"
echo "  - pre-push: Run linter and mypy on all files before push"
echo ""
echo "To skip hooks temporarily (not recommended):"
echo "  git commit --no-verify"
echo "  git push --no-verify"
