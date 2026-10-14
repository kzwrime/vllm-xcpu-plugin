# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
from dataclasses import replace
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


def test_source_guard_accepts_only_listed_signature_variants(monkeypatch):
    actual_signature = _signature_fingerprint(inspect.signature(_reference_kernel))
    target = compatibility.UpstreamOperator(
        category="test",
        module=__name__,
        name="_reference_kernel",
        source_version="v0.25.1",
        expected_source_hash=_source_fingerprint(_reference_kernel),
        expected_signature_hash="other-environment-signature",
        replacement="torch_xcpu.ops.reference",
        alternate_signature_hashes=(actual_signature,),
    )
    monkeypatch.setattr(compatibility, "UPSTREAM_OPERATORS", (target,))
    assert compatibility.verify_upstream_compatibility(("test",)) == (target,)

    monkeypatch.setattr(
        compatibility,
        "UPSTREAM_OPERATORS",
        (replace(target, alternate_signature_hashes=()),),
    )
    with pytest.raises(KernelVersionError, match="signature hash"):
        compatibility.verify_upstream_compatibility(("test",))


def test_source_file_guard_catches_helper_changes(monkeypatch):
    monkeypatch.setattr(compatibility, "UPSTREAM_OPERATORS", ())
    target = compatibility.UpstreamSourceFile(
        category="test", module=__name__, expected_sha256="stale-file-hash"
    )
    monkeypatch.setattr(compatibility, "UPSTREAM_SOURCE_FILES", (target,))
    with pytest.raises(KernelVersionError, match="source file changed"):
        compatibility.verify_upstream_compatibility(("test",))

    correct_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(
        compatibility,
        "UPSTREAM_SOURCE_FILES",
        (replace(target, expected_sha256=correct_hash),),
    )
    assert compatibility.verify_upstream_compatibility(("test",)) == ()


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
        for category in {
            "attention",
            "grouped_topk",
            "topk_softmax",
            "topk_topp",
        }
    }
    assert versions == {
        "attention": {
            "v0.24.0",
            "v0.25.0",
            "v0.31 dense causal attention; local=dd24619b0d47831e; "
            "upstream=07ea9350baf84e33,1117140edbaf3d1e",
        },
        "grouped_topk": {"v0.19.0"},
        "topk_softmax": {"v0.25.0"},
        "topk_topp": {
            "v0.24.0",
            "v0.31 XCPU full filter; local=dd24619b0d47831e; "
            "upstream=bf95f58d10891293,aed894c190aa3db7,1ee7f78e806b5e9b",
        },
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
