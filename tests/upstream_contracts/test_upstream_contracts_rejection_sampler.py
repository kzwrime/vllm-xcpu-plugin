# SPDX-License-Identifier: Apache-2.0
"""The rejection_sampler manifest covers every patch site and still holds."""

from vllm_xcpu_plugin.upstream_contracts import check, load_manifest
from vllm_xcpu_plugin.upstream_contracts.lint import lint, provenance


def test_every_patch_site_is_declared():
    assert lint(load_manifest("rejection_sampler")) == []


def test_contracts_hold():
    assert check("rejection_sampler") == []


def test_pins_match_their_reviewed_snapshots():
    assert provenance(load_manifest("rejection_sampler")) == []
