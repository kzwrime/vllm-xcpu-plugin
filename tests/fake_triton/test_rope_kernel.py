# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_rope_kernel_dispatches_current_v2_launch_abi():
    repo = Path(__file__).parents[2]
    torch_mcpu_repo = repo.parent / "torch_mcpu"
    code = """
import json
import torch

from vllm.v1.worker.gpu.mm.rope import _prepare_rope_positions_kernel
from vllm_xcpu_plugin.fake_triton.runtime import get_registry
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

register_vllm_kernels()
positions = torch.full((3, 6), -1, dtype=torch.int64, device="mcpu")
prefill_positions = torch.tensor(
    [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
    dtype=torch.int32,
    device="mcpu",
)
_prepare_rope_positions_kernel[(1,)](
    positions,
    positions.stride(0),
    prefill_positions,
    3 * prefill_positions.stride(0),
    prefill_positions.stride(0),
    torch.tensor([9], dtype=torch.int32, device="mcpu"),
    torch.tensor([0], dtype=torch.int32, device="mcpu"),
    torch.tensor([0, 3], dtype=torch.int32, device="mcpu"),
    torch.tensor([4], dtype=torch.int32, device="mcpu"),
    torch.tensor([1], dtype=torch.int32, device="mcpu"),
    BLOCK_SIZE=1024,
    NUM_DIMS=3,
)
torch.mcpu.synchronize()
print(json.dumps({
    "positions": positions.cpu().tolist(),
    "launches": get_registry().launch_counts()[
        "vllm.v1.worker.gpu.mm.rope._prepare_rope_positions_kernel"
    ],
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(torch_mcpu_repo), str(repo)))
    env["VLLM_PLUGINS"] = "xcpu_platform_plugin"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "positions": [
            [11, 12, 13, -1, -1, -1],
            [21, 22, 23, -1, -1, -1],
            [31, 32, 33, -1, -1, -1],
        ],
        "launches": 1,
    }
