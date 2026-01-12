#!/bin/bash
# Run pre-commit checks manually
# This mimics what the pre-commit and pre-push hooks do

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Run linter
echo "=================================="
echo "Running linter on all Python files..."
echo "=================================="
if ! ./scripts/lint.sh; then
    echo ""
    echo "=================================="
    echo "Lint failed!"
    echo "Run ./scripts/format.sh to auto-fix"
    echo "=================================="
    exit 1
fi
echo "Lint passed!"

# Run mypy
echo ""
echo "=================================="
echo "Running mypy type check..."
echo "=================================="
if ! python3 ./scripts/mypy.py; then
    echo ""
    echo "=================================="
    echo "Mypy failed!"
    echo "=================================="
    exit 1
fi
echo "Mypy passed!"

echo ""
echo "=================================="
echo "All checks passed!"
echo "=================================="
