# SPDX-License-Identifier: Apache-2.0
"""``python -m vllm_xcpu_plugin.upstream_contracts <command>``.

check [CATEGORY ...] [--json]   drift summary (exit 1 on drift)
report CATEGORY                 drift with full diffs
fingerprint TARGET ...          hashes and location of module:qualname
lint [CATEGORY ...]             undeclared patch sites, broken references
provenance [CATEGORY ...]       pins vs their local_commit snapshots
sites CATEGORY                  discovered patch sites and their patches
"""

from __future__ import annotations

import argparse
import json
import sys

from . import MANIFESTS, _source, check, fingerprint, load_manifest, origin, resolve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_xcpu_plugin.upstream_contracts"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "lint", "provenance"):
        p = sub.add_parser(name)
        p.add_argument("categories", nargs="*", default=sorted(MANIFESTS))
        if name == "check":
            p.add_argument("--json", action="store_true")
    sub.add_parser("report").add_argument("category")
    sub.add_parser("sites").add_argument("category")
    sub.add_parser("fingerprint").add_argument("targets", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "fingerprint":
        for target in args.targets:
            obj = resolve(target)
            source_hash, signature_hash = fingerprint(obj)
            obj_origin = origin(obj)
            print(target)
            print(f"  source_hash={source_hash!r},")
            print(f"  signature_hash={signature_hash!r},")
            if obj_origin and obj_origin != target:
                print(f"  defined at {obj_origin}")
            print(f"  {_source.location(obj_origin or target)}")
        return 0

    if args.command in ("check", "report"):
        from .report import format_text, to_json

        categories = [args.category] if args.command == "report" else args.categories
        status = 0
        reports = []
        for category in categories:
            drifts = check(category)
            status |= bool(drifts)
            if getattr(args, "json", False):
                reports.append(to_json(category, drifts))
            elif drifts:
                limit = None if args.command == "report" else 30
                print(format_text(category, drifts, max_diff_lines=limit))
            else:
                print(f"{category}: all upstream contracts hold")
        if reports:
            json.dump(reports, sys.stdout, indent=2)
            print()
        return status

    from .lint import discover, lint, provenance

    if args.command == "sites":
        manifest = load_manifest(args.category)
        owners = {t: p.id for p in manifest.patches for t in p.downstream}
        refs = {r.target: p.id for p in manifest.patches for r in p.upstream}
        for site in discover(manifest):
            key = site.downstream if site.kind != "rebind" else site.upstream
            patch = owners.get(key or "") or refs.get(site.upstream or "") or "-"
            print(f"{site}  [{patch}]")
        return 0

    status = 0
    for category in args.categories:
        run = lint if args.command == "lint" else provenance
        problems = run(load_manifest(category))
        status |= bool(problems)
        for problem in problems:
            print(f"{category}: {problem}")
        if not problems:
            print(f"{category}: {args.command} clean")
    return status


if __name__ == "__main__":
    sys.exit(main())
