# SPDX-License-Identifier: Apache-2.0
"""Source text of upstream symbols: current files, reviewed snapshots, diffs.

Symbols are located by AST so the reviewed snapshot (``git show
<local_commit>:<path>``) and the current file are fingerprinted the same way
as the runtime check (``fake_triton.runtime._source_fingerprint``).
"""

from __future__ import annotations

import ast
import difflib
import functools
import hashlib
import importlib.util
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

MODULE_SCOPE = "<module>"


def ast_fingerprint(source: str) -> str:
    normalized = ast.dump(
        ast.parse(textwrap.dedent(source)),
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class SymbolSource:
    qualname: str
    lineno: int
    text: str
    # Whole definition, as the runtime fingerprints the object.
    fingerprint: str
    # What this symbol alone contributes: a class without its methods, so a
    # changed method is reported once, under the method.
    own_fingerprint: str


def symbols(text: str) -> dict[str, SymbolSource]:
    """Top-level functions/classes/assignments, class methods and attributes,
    plus ``<module>`` for the remaining top-level statements (imports)."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    out: dict[str, SymbolSource] = {}

    def segment(node: ast.stmt) -> tuple[int, str]:
        decorators = getattr(node, "decorator_list", [])
        start = min([d.lineno for d in decorators] + [node.lineno])
        end = node.end_lineno
        assert end is not None
        return start, textwrap.dedent("".join(lines[start - 1 : end]))

    def add(qualname: str, node: ast.stmt, own: str | None = None) -> None:
        start, text = segment(node)
        fingerprint = ast_fingerprint(text)
        out[qualname] = SymbolSource(
            qualname, start, text, fingerprint, own or fingerprint
        )

    def assigned_names(node: ast.stmt) -> list[str]:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            return []
        return [t.id for t in targets if isinstance(t, ast.Name)]

    def visit(body: list[ast.stmt], prefix: str) -> list[str]:
        rest = []
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(prefix + node.name, node)
            elif isinstance(node, ast.ClassDef):
                qualname = prefix + node.name
                own = visit(node.body, qualname + ".")
                header = ast.unparse(
                    ast.ClassDef(
                        name=node.name,
                        bases=node.bases,
                        keywords=node.keywords,
                        body=[ast.parse(s).body[0] for s in own] or [ast.Pass()],
                        decorator_list=node.decorator_list,
                        type_params=getattr(node, "type_params", []),
                    )
                )
                add(qualname, node, ast_fingerprint(header))
            elif names := assigned_names(node):
                for name in names:
                    add(prefix + name, node)
                if prefix:
                    # Class attributes also belong to the class itself.
                    rest.append(segment(node)[1])
            else:
                rest.append(segment(node)[1])
        return rest

    module_rest = visit(tree.body, "")
    module_text = "".join(module_rest)
    out[MODULE_SCOPE] = SymbolSource(
        MODULE_SCOPE, 1, module_text, ast_fingerprint(module_text),
        ast_fingerprint(module_text),
    )  # fmt: skip
    return out


def module_path(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def vllm_repo_path(relative: str) -> Path | None:
    """A path inside the vLLM checkout that provides the ``vllm`` package."""
    package = module_path("vllm")
    return package.parents[1] / relative if package else None


@functools.cache
def file_symbols(path: Path) -> dict[str, SymbolSource]:
    return symbols(path.read_text())


def location(target: str) -> str | None:
    """``path:line`` of ``module:qualname`` without importing the module."""
    module, _, qualname = target.partition(":")
    path = module_path(module)
    if path is None:
        return None
    symbol = file_symbols(path).get(qualname)
    return f"{path}:{symbol.lineno}" if symbol else str(path)


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


@functools.cache
def _repo_root(directory: Path) -> Path | None:
    root = _git(directory, "rev-parse", "--show-toplevel")
    return Path(root.strip()) if root else None


@functools.cache
def reviewed_text(path: Path, commit: str) -> str | None:
    """The file as of ``commit`` in the repository that contains it."""
    root = _repo_root(path.parent)
    if root is None:
        return None
    return _git(root, "show", f"{commit}:{path.relative_to(root).as_posix()}")


def reviewed_symbols(path: Path, commit: str) -> dict[str, SymbolSource] | None:
    text = reviewed_text(path, commit)
    if text is None:
        return None
    try:
        return symbols(text)
    except SyntaxError:
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def unified_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=old_name,
            tofile=new_name,
        )
    )
