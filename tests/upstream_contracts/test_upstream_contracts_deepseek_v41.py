# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 manifest covers every patch site and still holds."""

from vllm_xcpu_plugin.upstream_contracts import check, load_manifest
from vllm_xcpu_plugin.upstream_contracts.lint import lint, provenance


def test_every_patch_site_is_declared():
    assert lint(load_manifest("deepseek_v41")) == []


def test_contracts_hold():
    assert check("deepseek_v41") == []


def test_pins_match_their_reviewed_snapshots():
    assert provenance(load_manifest("deepseek_v41")) == []


def test_adapters_take_their_pins_from_the_manifest():
    from vllm_xcpu_plugin.models.deepseek_v41.triton_adapters import _ADAPTERS

    manifest = load_manifest("deepseek_v41")
    pinned = {ref.target for patch in manifest.patches for ref in patch.upstream}
    assert {f"{module}:{qualname}" for module, qualname, _ in _ADAPTERS} <= pinned
