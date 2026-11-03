---
name: upstream-drift
description: Handle vLLM upstream drift against XCPU plugin patches. Use whenever output, a log or a test shows "[XCPU-UPSTREAM-DRIFT]", UpstreamDriftError, KernelVersionError, "upstream operator compatibility mismatch" or "upstream source file changed"; when upgrading or rebasing the vLLM fork; and when adding or changing anything in vllm_xcpu_plugin that overrides a vLLM method, rebinds a vLLM attribute, swaps __class__, registers an OOT layer or a Fake Triton adapter.
---

# Upstream drift (vllm-xcpu-plugin)

XCPU patches mirror or rely on upstream vLLM semantics. Each category manifest
(`vllm_xcpu_plugin/models/<category>/upstream.py`, registry in
`vllm_xcpu_plugin.upstream_contracts.MANIFESTS`) lists every patch: its
downstream sites, the upstream symbols it relies on (fingerprint pins), the
semantic `contract`, the `impl` (torch_xcpu ops) and `tests`. Full guide:
`vllm-xcpu-plugin/docs/UPSTREAM_CONTRACTS.md`.

A drift error is a semantic-review stop. Never "fix" it by refreshing hashes.

## When a drift is reported

Run from `vllm-xcpu-plugin/` (at the dev-kit root `./vllm` shadows the vllm
package), not by restarting servers:

1. Collect every drift at once:
   `python -m vllm_xcpu_plugin.upstream_contracts check --json`
   (or `report <category>` for full text diffs).
2. For each drift, read in this order: the diff, the `contract`, the
   `downstream` locations, the `impl` sources and their torch_xcpu reference
   (`references.py` next to the tests). Decide:
   - semantics unchanged for the contract -> only re-pin that ref;
   - semantics changed -> change the downstream patch, the torch_xcpu op and
     its reference, add test cases for the new behavior (edge cases first:
     empty batches, padding rows, SP shards, dead/pad ids, spec decode),
     update the contract text;
   - `MISSING` -> locate the moved/renamed symbol (`git -C vllm log -S`, grep),
     update `target` and the downstream;
   - unbound (`no patch references it`) -> decide whether any patch implicitly
     depends on it; if so add an `UpstreamRef` with a contract sentence,
     otherwise only update that watched module's `sha256`.
3. Re-pin per reference, after review only: commit the reviewed vLLM state,
   point `KernelSource.local_commit` (and `main_commit` or `backport_commit`)
   at it, and copy the values printed by
   `python -m vllm_xcpu_plugin.upstream_contracts fingerprint <module:qualname>`.
4. Verify: `... upstream_contracts lint`, `... upstream_contracts provenance`,
   `pytest tests/upstream_contracts`, the tests listed in the
   patch, then the E2E docs in the manifest `checks`.
5. Report to the user which drifts changed semantics, what was changed, and
   which were re-pinned as unchanged, with the reasoning for each.

## When writing or changing a patch

- Rebind upstream attributes with `upstream_contracts.rebind(owner, name, value)`,
  never by assignment.
- Every override of a vLLM method, rebind, `__class__` swap and Fake Triton
  adapter must belong to a `Patch` of the category manifest; `lint` fails
  otherwise. Also add refs for upstream behavior relied on without overriding
  (call sites, output shapes, inherited helpers).
- Write the contract as the upstream facts the downstream depends on, plus what
  to re-check in a future diff.
