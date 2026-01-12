.PHONY: format lint mypy mypy-all check clean help

# Default target
.DEFAULT_GOAL := help

## format: Format code with ruff (auto-fix)
format:
	@./scripts/format.sh

## lint: Run ruff linter (check only)
lint:
	@./scripts/lint.sh

## mypy: Run mypy type checking
mypy:
	@python scripts/mypy.py

## mypy-all: Run mypy for all Python versions
mypy-all:
	@echo "Running mypy for Python 3.10..."
	@python scripts/mypy.py 3.10 || true
	@echo "Running mypy for Python 3.11..."
	@python scripts/mypy.py 3.11 || true
	@echo "Running mypy for Python 3.12..."
	@python scripts/mypy.py 3.12 || true

## check: Run all checks (lint + mypy)
check: lint mypy

## clean: Clean cache directories
clean:
	@echo "Cleaning cache directories..."
	@rm -rf .mypy_cache .ruff_cache .pytest_cache __pycache__
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "Clean complete!"

## help: Show this help message
help:
	@echo "Available targets:"
	@grep -E "^## " Makefile | sed 's/## /  /'
