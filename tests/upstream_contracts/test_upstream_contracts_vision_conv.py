# SPDX-License-Identifier: Apache-2.0
"""The vision_conv manifest covers every patch site and still holds."""

from vllm_xcpu_plugin.upstream_contracts import check, load_manifest
from vllm_xcpu_plugin.upstream_contracts.lint import lint, provenance


def test_every_patch_site_is_declared():
    assert lint(load_manifest("vision_conv")) == []


def test_contracts_hold():
    assert check("vision_conv") == []


def test_pins_match_their_reviewed_snapshots():
    assert provenance(load_manifest("vision_conv")) == []
