# SPDX-License-Identifier: Apache-2.0
"""Patch-site lint: every way a category's code changes upstream behavior
must be declared by a ``Patch`` of its manifest.

Discovered sites:

* ``rebind``: ``rebind(owner, "name", value)`` calls;
* ``raw-rebind``: attribute assignment / ``setattr`` on an imported object
  (rejected: use ``rebind`` so the original is pinned and must exist);
* ``override``: a method of a plugin class that overrides one defined in a
  vLLM class (judged on the bases written in the source, not ones injected
  by decorators such as ``support_torch_compile``);
* ``class-swap``: ``x.__class__ = ...`` (its enclosing function is the site);
* ``private-use``: a read of an upstream ``_private`` callable, through
  ``self`` (defined in a vLLM base class) or an imported upstream name. The
  downstream then relies on an internal helper, which must be pinned.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import Manifest, _source


@dataclass(frozen=True)
class Site:
    kind: str
    module: str
    lineno: int
    # Plugin code: ``module:qualname`` (override, class-swap).
    downstream: str | None = None
    # Upstream path: rebound attribute or overridden method.
    upstream: str | None = None

    def __str__(self) -> str:
        where = f"{self.module}:{self.lineno}"
        parts = [p for p in (self.downstream, self.upstream) if p]
        return f"{self.kind} at {where}: {' -> '.join(parts)}"


def _modules(packages: tuple[str, ...]) -> list[str]:
    names = []
    for package in packages:
        names.append(package)
        module = importlib.import_module(package)
        if hasattr(module, "__path__"):
            names.extend(
                info.name
                for info in pkgutil.walk_packages(module.__path__, f"{package}.")
            )
    return names


def _aliases(tree: ast.Module, module: str, is_package: bool) -> dict[str, str]:
    package = module if is_package else module.rpartition(".")[0]
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    aliases[head] = head
        elif isinstance(node, ast.ImportFrom):
            base = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package
            ) if node.level else node.module or ""  # fmt: skip
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{base}.{alias.name}"
    return aliases


def _dotted(node: ast.expr) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


@functools.cache
def _split_target(dotted: str) -> str:
    """``a.b.C.d`` -> ``a.b:C.d`` at the longest importable module prefix."""
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        module = ".".join(parts[:end])
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        return f"{module}:{'.'.join(parts[end:])}" if end < len(parts) else module
    return dotted


def _external(dotted: str, own_prefix: str) -> bool:
    return not dotted.startswith(own_prefix)


class _Visitor(ast.NodeVisitor):
    def __init__(self, module: str, aliases: dict[str, str], own: str) -> None:
        self.module = module
        self.aliases = aliases
        self.own = own
        self.scope: list[str] = []
        self.sites: list[Site] = []

    def _resolve(self, node: ast.expr) -> str | None:
        parts = _dotted(node)
        if not parts or parts[0] not in self.aliases:
            return None
        return ".".join([self.aliases[parts[0]], *parts[1:]])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and _is_private(node.attr)
            and isinstance(node.value, ast.Name)
            and node.value.id != "self"
        ):
            owner = self._resolve(node.value)
            if owner and _external(owner, self.own) and owner.startswith("vllm."):
                self.sites.append(
                    Site(
                        "private-use",
                        self.module,
                        node.lineno,
                        downstream=self._enclosing(),
                        upstream=_split_target(f"{owner}.{node.attr}"),
                    )
                )
        self.generic_visit(node)

    def _enclosing(self) -> str:
        return f"{self.module}:{'.'.join(self.scope)}"

    def _assign(self, target: ast.expr, lineno: int) -> None:
        if not isinstance(target, ast.Attribute):
            return
        if target.attr == "__class__":
            site = Site("class-swap", self.module, lineno, self._enclosing())
            self.sites.append(site)
            return
        owner = self._resolve(target.value)
        if owner and _external(owner, self.own):
            self.sites.append(
                Site(
                    "raw-rebind",
                    self.module,
                    lineno,
                    upstream=_split_target(f"{owner}.{target.attr}"),
                )
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._assign(target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._assign(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._assign(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = _dotted(node.func)
        name = func[-1] if func else None
        if (
            name in ("rebind", "setattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            owner = self._resolve(node.args[0])
            if owner and _external(owner, self.own):
                self.sites.append(
                    Site(
                        "rebind" if name == "rebind" else "raw-rebind",
                        self.module,
                        node.lineno,
                        upstream=_split_target(f"{owner}.{node.args[1].value}"),
                    )
                )
        self.generic_visit(node)


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _callable_member(value: Any) -> bool:
    return inspect.isfunction(value) or isinstance(
        value, (staticmethod, classmethod, property)
    )


def _self_private_sites(module_name: str, cls: type, node: ast.ClassDef) -> list[Site]:
    """``self._helper`` reads that resolve to a callable of a vLLM base."""
    sites = []
    for method in node.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(method):
            if not (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.ctx, ast.Load)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
                and _is_private(sub.attr)
            ):
                continue
            owner = next((b for b in cls.__mro__ if sub.attr in vars(b)), None)
            if (
                owner is None
                or not owner.__module__.startswith("vllm.")
                or not _callable_member(vars(owner)[sub.attr])
            ):
                continue
            sites.append(
                Site(
                    "private-use",
                    module_name,
                    sub.lineno,
                    downstream=f"{module_name}:{node.name}.{method.name}",
                    upstream=f"{owner.__module__}:{owner.__qualname__}.{sub.attr}",
                )
            )
    return sites


def _override_sites(module_name: str, tree: ast.Module) -> list[Site]:
    module = importlib.import_module(module_name)
    sites = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        cls = getattr(module, node.name, None)
        if not inspect.isclass(cls):
            continue
        sites.extend(_self_private_sites(module_name, cls, node))
        bases: list[type] = []
        for base_node in node.bases:
            try:
                base = eval(ast.unparse(base_node), vars(module))  # noqa: S307
            except Exception:
                continue
            if inspect.isclass(base):
                bases.extend(b for b in base.__mro__ if b not in bases)
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            owner = next((b for b in bases if item.name in vars(b)), None)
            if owner is None or not owner.__module__.startswith("vllm."):
                continue
            sites.append(
                Site(
                    "override",
                    module_name,
                    item.lineno,
                    downstream=f"{module_name}:{node.name}.{item.name}",
                    upstream=f"{owner.__module__}:{owner.__qualname__}.{item.name}",
                )
            )
    return sites


def discover(manifest: Manifest) -> list[Site]:
    own = "vllm_xcpu_plugin."
    sites = []
    for name in _modules(manifest.packages):
        if name == _manifest_module(manifest):
            continue
        path = _source.module_path(name)
        if path is None:
            continue
        tree = ast.parse(path.read_text())
        visitor = _Visitor(name, _aliases(tree, name, path.name == "__init__.py"), own)
        visitor.visit(tree)
        sites.extend(visitor.sites)
        sites.extend(_override_sites(name, tree))
    return sites


def _manifest_module(manifest: Manifest) -> str:
    from . import MANIFESTS

    return MANIFESTS[manifest.category]


def _resolves(target: str) -> bool:
    module_name, _, qualname = target.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = inspect.getattr_static(obj, part)
    except (ImportError, AttributeError):
        return False
    return True


def _workspace_root(
    anchor: Path | None = None, cwd: Path | None = None
) -> Path | None:
    """Workspace root holding vllm/, vllm-xcpu-plugin/, torch_xcpu/, docs/.

    Editable installs sit at ``<workspace>/vllm-xcpu-plugin/vllm_xcpu_plugin/...``.
    Non-editable installs sit under ``site-packages/``, so fall back to walking
    up from ``cwd``. Returns ``None`` outside a workspace checkout.
    """
    file_path = (anchor or Path(__file__)).resolve()
    candidates: list[Path] = []
    if len(file_path.parents) > 3:
        candidates.append(file_path.parents[3])
    cwd_path = (cwd or Path.cwd()).resolve()
    candidates.append(cwd_path)
    candidates.extend(cwd_path.parents)
    for candidate in candidates:
        if (candidate / "vllm-xcpu-plugin" / "vllm_xcpu_plugin").is_dir():
            return candidate
    return None


def _reference_problem(entry: str, workspace: Path | None) -> str | None:
    """``path[::symbol] [note]`` entries must name an existing file/symbol."""
    token = entry.split(" ", 1)[0]
    path_part, _, symbol = token.partition("::")
    if "/" not in path_part or workspace is None:
        return None
    matches = sorted(workspace.glob(path_part))
    if not matches:
        return f"{path_part} does not exist under {workspace}"
    if symbol and not any(
        re.search(
            rf"\b(def|class)\s+{re.escape(symbol)}\b|^{re.escape(symbol)}\s*=",
            m.read_text(),
            re.M,
        )
        for m in matches
        if m.is_file()
    ):
        return f"{symbol} is not defined in {path_part}"
    return None


def _ref_origins(manifest: Manifest) -> set[str]:
    from . import origin, resolve

    origins = set()
    for patch in manifest.patches:
        for ref in patch.upstream:
            try:
                obj_origin = origin(resolve(ref.target))
            except (ImportError, AttributeError):
                continue
            if obj_origin:
                origins.add(obj_origin)
    return origins


def lint(manifest: Manifest) -> list[str]:
    """Problems that would let a patch drift unnoticed; empty when clean."""
    problems = []
    ids = [p.id for p in manifest.patches]
    for duplicate in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate patch id {duplicate}")

    workspace = _workspace_root()
    pins: dict[str, tuple[str | None, str | None, str | None]] = {}
    for patch in manifest.patches:
        if not patch.contract.strip():
            problems.append(f"{patch.id}: empty contract")
        if not patch.downstream:
            problems.append(f"{patch.id}: no downstream site")
        for target in patch.downstream:
            if not _resolves(target):
                problems.append(f"{patch.id}: downstream {target} does not resolve")
        for entry in (*patch.impl, *patch.tests):
            if problem := _reference_problem(entry, workspace):
                problems.append(f"{patch.id}: {problem}")
        for ref in patch.upstream:
            pin = (ref.source_hash, ref.signature_hash, ref.expect)
            if pins.setdefault(ref.target, pin) != pin:
                problems.append(
                    f"{patch.id}: {ref.target} is pinned differently by another "
                    "patch; re-pin every reference after the same review"
                )

    files = {w.path for w in manifest.watched_files}
    for patch in manifest.patches:
        for path in patch.upstream_files:
            if path not in files:
                problems.append(f"{patch.id}: upstream file {path} is not pinned")
    for watched_file in manifest.watched_files:
        resolved = _source.vllm_repo_path(watched_file.path)
        if resolved is not None and not resolved.is_file():
            problems.append(f"watched file {watched_file.path} does not exist")

    downstream = {t: p for p in manifest.patches for t in p.downstream}
    refs = {r.target for p in manifest.patches for r in p.upstream}
    pinned = refs | _ref_origins(manifest)
    watched = {w.module for w in manifest.watched}
    for site in discover(manifest):
        if site.kind == "raw-rebind":
            problems.append(
                f"{site}: use upstream_contracts.rebind() so the replaced "
                "attribute must exist and its original stays pinned"
            )
        elif site.kind == "rebind":
            if site.upstream not in refs:
                problems.append(f"{site}: no patch pins the rebound attribute")
        elif site.kind == "override":
            assert site.downstream and site.upstream
            override = downstream.get(site.downstream)
            if override is None:
                problems.append(f"{site}: no patch declares this override")
            elif site.upstream.partition(":")[0] in watched and site.upstream not in {
                r.target for r in override.upstream
            }:
                problems.append(
                    f"{site}: overrides a method of a watched module; "
                    f"{override.id} must pin {site.upstream}"
                )
        elif site.kind == "class-swap" and site.downstream not in downstream:
            problems.append(f"{site}: no patch declares the enclosing function")
        elif site.kind == "private-use" and site.upstream not in pinned:
            problems.append(
                f"{site}: relies on an upstream private helper; pin it in the "
                "patch that owns this code"
            )
    return problems


def _uses_vllm(manifest: Manifest) -> bool:
    if manifest.watched_files:
        return True
    if any(
        w.module == "vllm" or w.module.startswith("vllm.") for w in manifest.watched
    ):
        return True
    return any(
        ref.target.startswith(("vllm:", "vllm."))
        for patch in manifest.patches
        for ref in patch.upstream
    )


def provenance(manifest: Manifest) -> list[str]:
    """Each pin must match its ``KernelSource.local_commit`` snapshot, so the
    reviewed source can be recovered for the next diff."""
    from vllm_xcpu_plugin.fake_triton.provenance import KernelSource

    from . import _reviewed_symbol, origin, resolve

    if _uses_vllm(manifest) and _source.vllm_git_root() is None:
        return [
            "vllm is not installed from a git checkout; provenance requires "
            "a vllm git checkout (editable install)"
        ]

    problems = []
    for patch in manifest.patches:
        if not isinstance(patch.source, KernelSource):
            problems.append(f"{patch.id}: {patch.source} has no reviewed snapshot")
            continue
        commit = patch.source.local_commit[:12]
        for ref in patch.upstream:
            if ref.source_hash is None:
                continue
            obj_origin = origin(resolve(ref.target))
            if obj_origin is None:
                continue
            reviewed = _reviewed_symbol(obj_origin, patch.source)
            if reviewed is None:
                problems.append(f"{patch.id}: {obj_origin} is not in {commit}")
            elif reviewed.fingerprint != ref.source_hash:
                problems.append(
                    f"{patch.id}: {obj_origin} at {commit} is not the pinned "
                    "source; commit the reviewed vLLM state and point "
                    "local_commit at it"
                )
    for watched_file in manifest.watched_files:
        path = _source.vllm_repo_path(watched_file.path)
        if not isinstance(watched_file.source, KernelSource) or path is None:
            problems.append(f"{watched_file.path}: no reviewed snapshot")
            continue
        commit = watched_file.source.local_commit
        text = _source.reviewed_text(path, commit)
        if text is None or _source.sha256_text(text) != watched_file.sha256:
            problems.append(
                f"{watched_file.path}: the file at {commit[:12]} is not the "
                "pinned SHA256"
            )
    for watched in manifest.watched:
        if not isinstance(watched.source, KernelSource):
            problems.append(f"{watched.module}: {watched.source} has no snapshot")
            continue
        path = _source.module_path(watched.module)
        commit = watched.source.local_commit
        text = _source.reviewed_text(path, commit) if path else None
        if text is None or _source.sha256_text(text) != watched.sha256:
            problems.append(
                f"{watched.module}: the file at "
                f"{watched.source.local_commit[:12]} is not the pinned SHA256"
            )
    return problems
