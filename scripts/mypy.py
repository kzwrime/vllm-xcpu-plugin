#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Run mypy type checking on Python files.

This script runs mypy on the vllm_xcpu_plugin package.
Adapted from vllm-stable/tools/pre_commit/mypy.py

Usage:
    python scripts/mypy.py [python_version] [files...]

Args:
    python_version: Optional Python version to use (e.g., "3.10").
        Defaults to local Python version.
    files: Optional list of files/directories to check.
        Defaults to the entire package.
"""

import subprocess
import sys

# Directories to check
FILES = [
    "vllm_xcpu_plugin",
]

# Exclude patterns
EXCLUDE = [
    "tests",
]


def run_mypy(
    targets: list[str],
    python_version: str | None = None,
    follow_imports: str | None = "silent",
) -> int:
    """
    Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.10") or None to use
            the default mypy version.
        follow_imports: Value for the --follow-imports option.

    Returns:
        The return code from mypy.
    """
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    print(f"$ {' '.join(args)} {' '.join(targets)}")
    return subprocess.run(args + targets, check=False).returncode


def main():
    # Get Python version (optional)
    python_version = None
    targets = []
    args = sys.argv[1:]

    if args and args[0].startswith("3."):
        python_version = args.pop(0)
    elif args and args[0] == "local":
        args.pop(0)
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Get targets
    targets = args if args else FILES

    return run_mypy(targets, python_version)


if __name__ == "__main__":
    sys.exit(main())
