#!/bin/bash
# Run ruff linter (without auto-fix)
# Usage: ./scripts/lint.sh [files...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ $# -eq 0 ]; then
    echo "Linting all Python files..."
    ruff check --output-format github .
else
    echo "Linting specified files..."
    ruff check --output-format github "$@"
fi
