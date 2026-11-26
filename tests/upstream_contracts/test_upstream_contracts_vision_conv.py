# SPDX-License-Identifier: Apache-2.0
"""The vision_conv manifest covers every patch site and still holds."""

import pytest

from vllm_xcpu_plugin.upstream_contracts import _source, check, load_manifest
from vllm_xcpu_plugin.upstream_contracts.lint import lint, provenance


def test_every_patch_site_is_declared():
    assert lint(load_manifest("vision_conv")) == []


def test_contracts_hold():
    assert check("vision_conv") == []


def test_pins_match_their_reviewed_snapshots():
    if _source.vllm_git_root() is None:
        pytest.skip("provenance check requires a vllm git checkout (editable install)")
    assert provenance(load_manifest("vision_conv")) == []
