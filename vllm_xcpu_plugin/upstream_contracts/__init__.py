# SPDX-License-Identifier: Apache-2.0
"""Upstream contracts: which vLLM code each XCPU patch relies on, and why.

A category manifest (``MANIFESTS``) declares

* ``Patch``: one group of downstream patch sites (OOT subclasses, overrides,
  rebinds, adapters), every upstream symbol it mirrors or relies on
  (``UpstreamRef``, pinned by source and signature fingerprint, or by
  identity), the reviewed ``KernelSource`` and the semantic ``contract`` the
  downstream keeps, plus the implementations and tests that encode it;
* ``WatchedModule``: upstream files the category mirrors as a whole (SHA256).
  A change anywhere in them is broken down to the changed symbols and mapped
  back to the patches that reference them.
* ``WatchedFile``: a non-Python upstream source (a CUDA kernel an XCPU op
  mirrors), pinned by SHA256 and diffed as a whole; patches name it in
  ``upstream_files``.

``verify(category)`` runs before the patches install and raises a single
``UpstreamDriftError`` with every drift, its downstream sites, contract and
reviewed-vs-current diff. Fingerprints are pinned by hand after a semantic
review (see ``docs/UPSTREAM_CONTRACTS.md``); nothing here refreshes them.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from vllm_xcpu_plugin.fake_triton.provenance import KernelSource, LegacyKernelSource
from vllm_xcpu_plugin.fake_triton.runtime import (
    KernelVersionError,
    _signature_fingerprint,
    _source_fingerprint,
)

from . import _source

# Category -> module defining ``MANIFEST``.
MANIFESTS: dict[str, str] = {
    "deepseek_v41": "vllm_xcpu_plugin.models.deepseek_v41.upstream",
}

DRIFT_MARKER = "[XCPU-UPSTREAM-DRIFT]"


@dataclass(frozen=True, kw_only=True)
class UpstreamRef:
    """An upstream object, named by the path the patch reaches it through
    (``module:Qual.name``), e.g. the consumer-module name a rebind replaces.

    ``source_hash``/``signature_hash`` pin the object's definition;
    ``expect`` pins that the path still resolves to ``module:qualname``.
    """

    target: str
    source_hash: str | None = None
    signature_hash: str | None = None
    expect: str | None = None

    def __post_init__(self) -> None:
        if ":" not in self.target:
            raise ValueError(f"{self.target}: target must be 'module:qualname'")
        if not (self.source_hash or self.signature_hash or self.expect):
            raise ValueError(f"{self.target}: pins nothing")


@dataclass(frozen=True, kw_only=True)
class Patch:
    id: str
    # ``module:qualname`` of the XCPU code that keeps the contract.
    downstream: tuple[str, ...]
    upstream: tuple[UpstreamRef, ...]
    source: KernelSource | LegacyKernelSource
    # The upstream semantics the downstream mirrors or assumes: what to
    # re-check in the diff.
    contract: str
    # Implementations outside this package (torch_xcpu ops, references).
    impl: tuple[str, ...] = ()
    # Tests and E2E checks that cover the contract.
    tests: tuple[str, ...] = ()
    # Non-Python upstream sources the semantics come from (WatchedFile.path).
    upstream_files: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class WatchedModule:
    module: str
    sha256: str
    source: KernelSource | LegacyKernelSource


@dataclass(frozen=True, kw_only=True)
class WatchedFile:
    # Relative to the vLLM repository root, e.g. ``csrc/.../kernel.cu``.
    path: str
    sha256: str
    source: KernelSource | LegacyKernelSource


@dataclass(frozen=True, kw_only=True)
class Manifest:
    category: str
    # Downstream packages the patch-site lint scans.
    packages: tuple[str, ...]
    patches: tuple[Patch, ...]
    watched: tuple[WatchedModule, ...] = ()
    watched_files: tuple[WatchedFile, ...] = ()
    # Category-wide E2E checks to rerun after any drift fix.
    checks: tuple[str, ...] = ()


@dataclass
class Drift:
    # missing | identity | source | signature | module | file
    kind: str
    target: str
    detail: str
    # ``module:qualname`` where the changed upstream object is defined.
    origin: str | None = None
    patches: list[Patch] = field(default_factory=list)
    # Patches bound to an enclosing class or module scope.
    related: list[Patch] = field(default_factory=list)
    location: str | None = None
    reviewed: str | None = None
    diff: str | None = None


class UpstreamDriftError(KernelVersionError):
    """Upstream code an XCPU patch relies on changed; see ``drifts``."""

    def __init__(self, category: str, drifts: list[Drift], message: str) -> None:
        super().__init__(message)
        self.category = category
        self.drifts = drifts


def load_manifest(category: str) -> Manifest:
    manifest = importlib.import_module(MANIFESTS[category]).MANIFEST
    assert manifest.category == category
    return manifest


# (id(owner), name) -> the upstream attribute a ``rebind`` replaced.
_ORIGINALS: dict[tuple[int, str], Any] = {}


def rebind(owner: Any, name: str, value: Any) -> None:
    """``setattr(owner, name, value)`` for an upstream attribute that must
    already exist; the original stays resolvable for the contract check."""
    try:
        original = inspect.getattr_static(owner, name)
    except AttributeError:
        owner_name = getattr(owner, "__name__", repr(owner))
        raise KernelVersionError(
            f"{DRIFT_MARKER} {owner_name}.{name} no longer exists upstream; the "
            "XCPU rebind would silently add a new attribute instead of replacing "
            "it. Run `python -m vllm_xcpu_plugin.upstream_contracts check` for "
            "the affected contracts."
        ) from None
    _ORIGINALS.setdefault((id(owner), name), original)
    setattr(owner, name, value)


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    if isinstance(obj, property):
        return obj.fget
    return obj


def resolve(target: str) -> Any:
    """The upstream object at ``module:qualname``, seen through rebinds."""
    module_name, _, qualname = target.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        key = (id(obj), part)
        if key in _ORIGINALS:
            obj = _ORIGINALS[key]
        else:
            obj = inspect.getattr_static(obj, part)
        obj = _unwrap(obj)
    return obj


def _definition(obj: Any) -> Any:
    # Triton (and Fake Triton) JIT functions keep the Python function in fn.
    return getattr(obj, "fn", obj)


def fingerprint(obj: Any) -> tuple[str, str | None]:
    """(source hash, signature hash) as the manifest pins them."""
    fn = _definition(obj)
    if inspect.isfunction(fn) or inspect.isclass(fn) or inspect.ismethod(fn):
        try:
            signature = _signature_fingerprint(inspect.signature(fn))
        except (TypeError, ValueError):
            signature = None
        return _source_fingerprint(fn), signature
    return hashlib.sha256(repr(obj).encode()).hexdigest(), None


def origin(obj: Any) -> str | None:
    """``module:qualname`` where ``obj`` is defined."""
    fn = inspect.unwrap(_definition(obj))
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module is None or qualname is None or "<locals>" in qualname:
        return None
    return f"{module}:{qualname}"


def _reviewed_symbol(
    origin_target: str, source: KernelSource | LegacyKernelSource
) -> _source.SymbolSource | None:
    if not isinstance(source, KernelSource):
        return None
    module, _, qualname = origin_target.partition(":")
    path = _source.module_path(module)
    if path is None:
        return None
    reviewed = _source.reviewed_symbols(path, source.local_commit)
    return None if reviewed is None else reviewed.get(qualname)


def _symbol_diff(
    origin_target: str, patches: list[Patch], source_hash: str | None
) -> tuple[str | None, str | None]:
    """(provenance, diff) of the reviewed snapshot against the current file.

    Prefers a patch whose ``local_commit`` holds exactly the pinned source;
    ``source_hash=None`` (constants, pinned by value) skips that check.
    """
    module, _, qualname = origin_target.partition(":")
    path = _source.module_path(module)
    current = _source.file_symbols(path).get(qualname) if path else None
    candidates = []
    for patch in patches:
        reviewed = _reviewed_symbol(origin_target, patch.source)
        if reviewed is not None:
            exact = source_hash is None or reviewed.fingerprint == source_hash
            candidates.append((exact, patch, reviewed))
    if not candidates:
        sources = sorted({str(p.source) for p in patches})
        return "; ".join(sources) or None, None
    matches, patch, reviewed = max(candidates, key=lambda c: c[0])
    provenance = str(patch.source)
    if not matches:
        provenance += (
            " (WARNING: the symbol at local_commit does not match the pinned "
            "fingerprint; the reviewed state may be uncommitted)"
        )
    if current is None:
        return provenance, None
    assert isinstance(patch.source, KernelSource)
    diff = _source.unified_diff(
        reviewed.text,
        current.text,
        f"{origin_target} @ {patch.source.local_commit[:12]} (reviewed)",
        f"{origin_target} (current)",
    )
    return provenance, diff or None


def check(category: str) -> list[Drift]:
    """Every drift of ``category``; empty when all contracts still hold."""
    manifest = load_manifest(category)
    drifts: dict[str, Drift] = {}
    # Origin of each reference -> patches relying on it.
    bound: dict[str, list[Patch]] = defaultdict(list)

    refs: dict[str, tuple[UpstreamRef, list[Patch]]] = {}
    for patch in manifest.patches:
        for ref in patch.upstream:
            refs.setdefault(ref.target, (ref, []))[1].append(patch)

    for target, (ref, patches) in refs.items():
        try:
            obj = resolve(target)
        except (ImportError, AttributeError) as exc:
            drifts[target] = Drift(
                "missing",
                target,
                f"no longer resolves ({type(exc).__name__}: {exc}); it was moved, "
                "renamed or removed upstream",
                patches=patches,
                reviewed="; ".join(sorted({str(p.source) for p in patches})),
            )
            continue
        obj_origin = origin(obj)
        # Constants have no origin; their target names the definition.
        bound[obj_origin or target].extend(patches)
        problems = []
        if ref.expect is not None and obj_origin != ref.expect:
            problems.append(("identity", f"resolves to {obj_origin}, not {ref.expect}"))
        source_hash, signature_hash = fingerprint(obj)
        if ref.source_hash is not None and source_hash != ref.source_hash:
            problems.append((
                "source",
                f"source {ref.source_hash[:12]} -> {source_hash[:12]}",
            ))
        if ref.signature_hash is not None and signature_hash != ref.signature_hash:
            actual = (signature_hash or "-")[:12]
            problems.append((
                "signature",
                f"signature {ref.signature_hash[:12]} -> {actual}",
            ))
        if not problems:
            continue
        kind = next(
            (k for k in ("identity", "signature", "source") if k in dict(problems)),
            problems[0][0],
        )
        definition = obj_origin or target
        drift = Drift(
            kind,
            target,
            "; ".join(detail for _, detail in problems),
            origin=definition,
            patches=patches,
            location=_source.location(definition),
        )
        drift.reviewed, drift.diff = _symbol_diff(
            definition, patches, ref.source_hash if obj_origin else None
        )
        drifts[target] = drift

    reported_origins = {d.origin for d in drifts.values() if d.origin}
    for watched in manifest.watched:
        for drift in _watched_drifts(watched, bound, reported_origins):
            drifts.setdefault(drift.target, drift)
    for watched_file in manifest.watched_files:
        if file_drift := _watched_file_drift(watched_file, manifest.patches):
            drifts.setdefault(file_drift.target, file_drift)
    return list(drifts.values())


def _watched_file_drift(
    watched: WatchedFile, patches: tuple[Patch, ...]
) -> Drift | None:
    path = _source.vllm_repo_path(watched.path)
    users = [p for p in patches if watched.path in p.upstream_files]
    target = f"vllm:{watched.path}"
    if path is None or not path.is_file():
        return Drift(
            "missing",
            target,
            "watched upstream source no longer exists (moved or removed)",
            patches=users,
            reviewed=str(watched.source),
        )
    current = path.read_text()
    if _source.sha256_text(current) == watched.sha256:
        return None
    diff = None
    detail = "watched upstream source changed"
    if isinstance(watched.source, KernelSource):
        reviewed = _source.reviewed_text(path, watched.source.local_commit)
        if reviewed is not None and _source.sha256_text(reviewed) == watched.sha256:
            diff = _source.unified_diff(
                reviewed,
                current,
                f"{watched.path} @ {watched.source.local_commit[:12]} (reviewed)",
                f"{watched.path} (current)",
            )
        else:
            detail += "; the reviewed snapshot is unavailable, diff by hand"
    return Drift(
        "file",
        target,
        detail,
        origin=target,
        patches=users,
        location=str(path),
        reviewed=str(watched.source),
        diff=diff,
    )


def _watched_drifts(
    watched: WatchedModule,
    bound: dict[str, list[Patch]],
    reported: set[str],
) -> list[Drift]:
    path = _source.module_path(watched.module)
    if path is None:
        return [
            Drift(
                "missing",
                f"{watched.module}:{_source.MODULE_SCOPE}",
                "watched upstream module no longer exists",
                reviewed=str(watched.source),
            )
        ]
    current_text = path.read_text()
    if _source.sha256_text(current_text) == watched.sha256:
        return []
    reviewed_text = None
    if isinstance(watched.source, KernelSource):
        reviewed_text = _source.reviewed_text(path, watched.source.local_commit)
    if reviewed_text is None or _source.sha256_text(reviewed_text) != watched.sha256:
        return [
            Drift(
                "module",
                f"{watched.module}:{_source.MODULE_SCOPE}",
                "watched file changed; the reviewed snapshot is unavailable "
                "(local_commit missing, not reachable, or not the pinned "
                "SHA256), so the change cannot be broken down by symbol: diff "
                "the file by hand",
                location=str(path),
                reviewed=str(watched.source),
            )
        ]
    assert isinstance(watched.source, KernelSource)
    old = _source.symbols(reviewed_text)
    new = _source.symbols(current_text)
    drifts = []
    for qualname in sorted(old.keys() | new.keys()):
        before, after = old.get(qualname), new.get(qualname)
        if before and after and before.own_fingerprint == after.own_fingerprint:
            continue
        target = f"{watched.module}:{qualname}"
        if target in reported:
            continue
        what = "added" if before is None else "removed" if after is None else "changed"
        patches = list(dict.fromkeys(bound.get(target, [])))
        # A class body or the module scope (imports) may change what the
        # members bound below it see.
        scope = (
            f"{watched.module}:" if qualname == _source.MODULE_SCOPE else f"{target}."
        )
        related = list(
            dict.fromkeys(
                patch
                for origin_target, ps in bound.items()
                if origin_target.startswith(scope)
                for patch in ps
                if patch not in patches
            )
        )
        detail = f"{what} in watched module"
        if not patches:
            detail += (
                "; no patch references it: decide whether an XCPU patch "
                "implicitly depends on it (then add an UpstreamRef) or not"
            )
        drifts.append(
            Drift(
                "module",
                target,
                detail,
                origin=target,
                patches=patches,
                related=related,
                location=f"{path}:{after.lineno}" if after else str(path),
                reviewed=str(watched.source),
                diff=_source.unified_diff(
                    before.text if before else "",
                    after.text if after else "",
                    f"{target} @ {watched.source.local_commit[:12]} (reviewed)",
                    f"{target} (current)",
                ),
            )
        )
    return drifts


@functools.cache
def _verified(category: str) -> None:
    drifts = check(category)
    if drifts:
        from .report import format_text

        raise UpstreamDriftError(
            category, drifts, format_text(category, drifts, max_diff_lines=60)
        )


def verify(category: str) -> None:
    """Raise ``UpstreamDriftError`` unless every contract of ``category``
    holds. Call before installing the category's patches."""
    _verified(category)
