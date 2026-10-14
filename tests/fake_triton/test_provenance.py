# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_xcpu_plugin.fake_triton.provenance import KernelSource, LegacyKernelSource


@pytest.mark.parametrize("origin", ["main", "backport"])
def test_source_description(origin):
    source = KernelSource(
        tag="v0.31.0-release-branch",
        local_commit="a" * 40,
        **{f"{origin}_commit": "b" * 40},
    )
    assert (
        str(source) == f"v0.31.0-release-branch; local={'a' * 40}; {origin}={'b' * 40}"
    )
    assert str(LegacyKernelSource("v0.24.0")) == "legacy-unverified: v0.24.0"


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"main_commit": "b" * 40, "backport_commit": "c" * 40},
        {"main_commit": "HEAD"},
    ],
)
def test_source_requires_one_commit_origin(fields):
    with pytest.raises(ValueError):
        KernelSource(tag="v0.31.0", local_commit="a" * 40, **fields)
