#!/bin/bash
# Format Python code with ruff
# Usage: ./scripts/format.sh [files...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ $# -eq 0 ]; then
    echo "Formatting all Python files..."
    ruff format --output-format github .
    ruff check --output-format github --fix .
else
    echo "Formatting specified files..."
    ruff format --output-format github "$@"
    ruff check --output-format github --fix "$@"
fi

echo "Formatting complete!"
