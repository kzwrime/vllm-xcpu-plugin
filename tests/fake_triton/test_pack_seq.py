# SPDX-License-Identifier: Apache-2.0

"""PackSeq/UnpackSeq kernels must reject with actionable guidance.

The XCPU sparse indexer replaces the upstream path that launches these
triton kernels; the only live dispatch is the JIT warmup registered by
``SparseAttnIndexer.__init__`` when ``enable_jit_warmup`` stays at its
upstream default. The adapters therefore raise with the exact flag to
set instead of emulating dead code paths.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_pack_unpack_seq_reject_with_guidance():
    repo = Path(__file__).parents[2]
    code = """
import torch
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
from vllm_xcpu_plugin.fake_triton.runtime import (
    InvalidLaunchError, get_registry,
)
register_vllm_kernels()

# The registration must exist so the provenance hash guard stays active
# against upstream changes to these kernels.
counts = get_registry().launch_counts()
assert "vllm.v1.attention.ops.common.PackSeqTritonKernel.kernel" in counts
assert "vllm.v1.attention.ops.common.UnpackSeqTritonKernel.kernel" in counts

# A real-tensor launch is an unsupported configuration and must fail with
# the flag guidance, not emulate a code path XCPU never exercises.
x = torch.arange(20, dtype=torch.float32, device="mcpu").reshape(10, 2)
lengths = torch.tensor([3, 0, 5, 2], dtype=torch.int32, device="mcpu")
try:
    pack_seq_triton(x, lengths)
except InvalidLaunchError as exc:
    assert "enable_jit_warmup" in str(exc), str(exc)
else:
    raise AssertionError("pack_seq_triton must reject on XCPU")

packed = torch.zeros(4, 5, 2, dtype=torch.float32, device="mcpu")
out = torch.zeros(10, 2, dtype=torch.float32, device="mcpu")
try:
    unpack_seq_triton(packed, lengths)
except InvalidLaunchError as exc:
    assert "enable_jit_warmup" in str(exc), str(exc)
else:
    raise AssertionError("unpack_seq_triton must reject on XCPU")

# JIT warmup dispatches with metadata-only stand-ins on a (1,) grid: the
# same actionable rejection, so the user learns the flag at startup.
class _StandIn:
    def __init__(self, dtype):
        self.dtype = dtype
        self.shape = (1,)
    def stride(self, dim=None):
        return (1,)

from vllm.v1.attention.ops.common import (
    PackSeqTritonKernel, UnpackSeqTritonKernel,
)
for kernel, kwargs in (
    (PackSeqTritonKernel.kernel, dict(
        x_ptr=_StandIn(torch.float32), out_ptr=_StandIn(torch.float32),
        lengths_ptr=_StandIn(torch.int32), N=1, D=1, Lmax=1,
        PAD_VALUE=float("-inf"), PAD_IS_UINT8=False, BLOCK_T=64, BLOCK_D=64,
    )),
    (UnpackSeqTritonKernel.kernel, dict(
        packed_ptr=_StandIn(torch.int32), out_ptr=_StandIn(torch.int32),
        lengths_ptr=_StandIn(torch.int32), B=1, Lmax=1, D=1,
        BLOCK_T=64, BLOCK_D=64,
    )),
):
    try:
        get_registry().dispatch(kernel, (1,), (), kwargs)
    except InvalidLaunchError as exc:
        assert "enable_jit_warmup" in str(exc), str(exc)
    else:
        raise AssertionError(f"{kernel.qualname} warmup must reject")
print("rejection guidance OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        timeout=120,
        env={
            **os.environ,
            "PYTHONPATH": str(repo),
            "VLLM_PLUGINS": "xcpu_platform_plugin",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    pytest.main([__file__])
