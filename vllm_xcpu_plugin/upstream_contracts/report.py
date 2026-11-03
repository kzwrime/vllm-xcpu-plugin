# SPDX-License-Identifier: Apache-2.0
"""Drift reports: the text an agent or reviewer acts on, and JSON."""

from __future__ import annotations

import textwrap
from typing import Any

from . import DRIFT_MARKER, Drift, Patch, _source, load_manifest

_KIND_TITLES = {
    "missing": "MISSING",
    "identity": "REBOUND ELSEWHERE",
    "signature": "INTERFACE CHANGED",
    "source": "SOURCE CHANGED",
    "module": "WATCHED MODULE CHANGED",
    "file": "WATCHED SOURCE CHANGED",
}


def _indent(text: str, prefix: str) -> str:
    return textwrap.indent(text.rstrip("\n"), prefix)


def _truncate(diff: str, max_lines: int | None) -> str:
    lines = diff.splitlines()
    if max_lines is None or len(lines) <= max_lines:
        return diff
    hidden = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"... ({hidden} more lines in the report)"])


def _patch_lines(patch: Patch, prefix: str) -> list[str]:
    lines = [f"{prefix}patch {patch.id}"]
    for target in patch.downstream:
        lines.append(f"{prefix}  downstream: {target}")
        if location := _source.location(target):
            lines.append(f"{prefix}              {location}")
    for impl in patch.impl:
        lines.append(f"{prefix}  impl:       {impl}")
    for path in patch.upstream_files:
        lines.append(f"{prefix}  mirrors:    vllm/{path}")
    lines.append(f"{prefix}  contract:")
    lines.append(_indent(textwrap.fill(patch.contract, 76), f"{prefix}    "))
    for test in patch.tests:
        lines.append(f"{prefix}  test:       {test}")
    return lines


def format_text(
    category: str, drifts: list[Drift], *, max_diff_lines: int | None = None
) -> str:
    manifest = load_manifest(category)
    unbound = sum(1 for d in drifts if not d.patches)
    out = [
        f"{DRIFT_MARKER} {category}: {len(drifts)} upstream change(s) reach "
        f"XCPU patches ({len(drifts) - unbound} bound, {unbound} unbound).",
        "This is a semantic-review stop, not an install problem: the XCPU code "
        "below mirrors or relies on upstream code that changed since its review.",
        "Workflow: vllm-xcpu-plugin/docs/UPSTREAM_CONTRACTS.md. Full report "
        f"with diffs: python -m vllm_xcpu_plugin.upstream_contracts report {category}",
        "For each item: read the diff against its contract, update the "
        "downstream/impl/tests where the semantics moved, then re-pin that ref "
        f"(hashes + KernelSource) in {_manifest_module(category)}.",
    ]
    for index, drift in enumerate(drifts, 1):
        out.append("")
        out.append(f"[{index}] {_KIND_TITLES[drift.kind]}: {drift.target}")
        out.append(f"    detail:   {drift.detail}")
        if drift.origin and drift.origin != drift.target:
            out.append(f"    defined:  {drift.origin}")
        if drift.location:
            out.append(f"    upstream: {drift.location}")
        if drift.reviewed:
            out.append(f"    reviewed: {drift.reviewed}")
        for patch in drift.patches:
            out.extend(_patch_lines(patch, "    "))
        if drift.related:
            out.append(
                "    related (bound inside this scope): "
                + ", ".join(p.id for p in drift.related)
            )
        if drift.diff:
            out.append("    diff (reviewed -> current):")
            out.append(_indent(_truncate(drift.diff, max_diff_lines), "      "))
    if manifest.checks:
        out.append("")
        out.append("After the fixes, rerun:")
        out.extend(f"  {check}" for check in manifest.checks)
    return "\n".join(out)


def _manifest_module(category: str) -> str:
    from . import MANIFESTS

    return MANIFESTS[category]


def to_json(category: str, drifts: list[Drift]) -> dict[str, Any]:
    def patch(p: Patch) -> dict[str, Any]:
        return {
            "id": p.id,
            "downstream": [
                {"target": t, "location": _source.location(t)} for t in p.downstream
            ],
            "impl": list(p.impl),
            "upstream_files": list(p.upstream_files),
            "contract": p.contract,
            "tests": list(p.tests),
            "source": str(p.source),
        }

    return {
        "category": category,
        "manifest": _manifest_module(category),
        "drifts": [
            {
                "kind": d.kind,
                "target": d.target,
                "origin": d.origin,
                "detail": d.detail,
                "location": d.location,
                "reviewed": d.reviewed,
                "patches": [patch(p) for p in d.patches],
                "related": [p.id for p in d.related],
                "diff": d.diff,
            }
            for d in drifts
        ],
        "checks": list(load_manifest(category).checks),
    }
