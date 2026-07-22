# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from pathlib import Path

import pytest

from vllm_xcpu_plugin import upstream_compatibility as compatibility
from vllm_xcpu_plugin.fake_triton.runtime import (
    KernelVersionError,
    _signature_fingerprint,
    _source_fingerprint,
)


def _reference_kernel(value: int = 1) -> int:
    return value + 1


def test_non_registry_operator_version_is_per_target(monkeypatch):
    target = compatibility.UpstreamOperator(
        category="test",
        module=__name__,
        name="_reference_kernel",
        source_version="v0.19.0",
        expected_source_hash=_source_fingerprint(_reference_kernel),
        expected_signature_hash=_signature_fingerprint(
            inspect.signature(_reference_kernel)
        ),
        replacement="torch_xcpu.ops.reference",
    )
    monkeypatch.setattr(compatibility, "UPSTREAM_OPERATORS", (target,))

    assert compatibility.verify_upstream_compatibility(("test",)) == (target,)


def test_non_registry_source_drift_reports_manual_update(monkeypatch):
    target = compatibility.UpstreamOperator(
        category="test",
        module=__name__,
        name="_reference_kernel",
        source_version="v0.23.0",
        expected_source_hash="outdated-source-hash",
        expected_signature_hash=_signature_fingerprint(
            inspect.signature(_reference_kernel)
        ),
        replacement="torch_xcpu.ops.reference",
    )
    monkeypatch.setattr(compatibility, "UPSTREAM_OPERATORS", (target,))

    with pytest.raises(KernelVersionError) as exc_info:
        compatibility.verify_upstream_compatibility(("test",))

    message = str(exc_info.value)
    assert "outdated-source-hash" in message
    assert "v0.23.0" in message
    assert "torch_xcpu.ops.reference" in message
    assert "Manually review" in message
    assert "source_version" in message


def test_manifest_has_explicit_mixed_source_versions():
    versions = {
        category: {
            target.source_version
            for target in compatibility.UPSTREAM_OPERATORS
            if target.category == category
        }
        for category in {"attention", "grouped_topk", "topk_topp"}
    }
    assert versions == {
        "attention": {"v0.24.0", "v0.25.0"},
        "grouped_topk": {"v0.19.0"},
        "topk_topp": {"v0.24.0"},
    }


def test_gdn_manifest_stops_at_the_composite_replacement_boundary():
    gdn_names = {
        target.name
        for target in compatibility.UPSTREAM_OPERATORS
        if target.category == "gdn"
    }
    assert gdn_names == {
        "ChunkGatedDeltaRule.forward_native",
        "chunk_gated_delta_rule",
        "fused_gdn_gating_kernel",
        "_fused_post_conv_kernel",
        "fused_sigmoid_gating_delta_rule_update_kernel",
        "fused_recurrent_gated_delta_rule_packed_decode_kernel",
    }


def test_fake_triton_versions_are_literals_not_baseline_defaults():
    source_path = (
        Path(__file__).parents[2]
        / "vllm_xcpu_plugin"
        / "fake_triton"
        / "vllm_kernels.py"
    )
    tree = ast.parse(source_path.read_text())
    manifest = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_KERNELS"
    )
    assert isinstance(manifest.value, ast.Tuple)
    assert len(manifest.value.elts) == 39
    assert all(
        isinstance(record, ast.Tuple)
        and isinstance(record.elts[4], ast.Constant)
        and isinstance(record.elts[4].value, str)
        for record in manifest.value.elts
    )
    baseline_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "_MANIFEST_BASELINE_VERSION"
        and isinstance(node.ctx, ast.Load)
    ]
    assert baseline_reads == []
