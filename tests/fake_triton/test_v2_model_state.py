# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_scatter_num_accepted_kernel_dispatches_with_current_vllm_abi():
    repo = Path(__file__).parents[2]
    code = """
import json
import torch

from vllm.v1.worker.gpu.model_states.mamba_hybrid import _scatter_num_accepted_kernel
from vllm_xcpu_plugin.fake_triton.runtime import InvalidLaunchError, get_registry
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

register_vllm_kernels()
idx_mapping = torch.tensor([2, -1, 0, 3], dtype=torch.int32, device="mcpu")
num_sampled = torch.tensor([0, 5, -3, 6], dtype=torch.int32, device="mcpu")
num_accepted = torch.full((4,), 77, dtype=torch.int32, device="mcpu")
_scatter_num_accepted_kernel[(4,)](idx_mapping, num_sampled, num_accepted)
bad_grid_rejected = False
try:
    _scatter_num_accepted_kernel[(3,)](idx_mapping, num_sampled, num_accepted)
except InvalidLaunchError:
    bad_grid_rejected = True
torch.mcpu.synchronize()
registry = get_registry()
print(json.dumps({
    "num_accepted": num_accepted.cpu().tolist(),
    "launches": registry.launch_counts()[
        "vllm.v1.worker.gpu.model_states.mamba_hybrid._scatter_num_accepted_kernel"
    ],
    "bad_grid_rejected": bad_grid_rejected,
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
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
        "num_accepted": [1, 77, 1, 6],
        "launches": 1,
        "bad_grid_rejected": True,
    }


def test_preprocess_mamba_align_dispatches_with_current_vllm_abi():
    repo = Path(__file__).parents[2]
    code = """
import json
import torch

from vllm.v1.worker.mamba_utils import preprocess_mamba_align_fused_kernel
from vllm_xcpu_plugin.fake_triton.runtime import InvalidLaunchError, get_registry
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

register_vllm_kernels()
idx_mapping = torch.tensor([2, 0, 3], dtype=torch.int32, device="mcpu")
state_idx = torch.tensor([0, 88, -1, 1], dtype=torch.int32, device="mcpu")
num_computed = torch.tensor([4, 0, 0, 7], dtype=torch.int32, device="mcpu")
query_start = torch.tensor([0, 1, 5, 7], dtype=torch.int32, device="mcpu")
num_accepted = torch.tensor([3, 77, 2, 4], dtype=torch.int32, device="mcpu")
src_col = torch.full((4,), 99, dtype=torch.int32, device="mcpu")
src_off = torch.full((4,), 99, dtype=torch.int32, device="mcpu")
preprocess_mamba_align_fused_kernel[(1,)](
    idx_mapping,
    state_idx,
    num_computed,
    query_start,
    num_accepted,
    src_col,
    src_off,
    3,
    BLOCK_SIZE=256,
    MAMBA_BLOCK_SIZE=4,
)
bad_block_rejected = False
try:
    preprocess_mamba_align_fused_kernel[(1,)](
        idx_mapping,
        state_idx,
        num_computed,
        query_start,
        num_accepted,
        src_col,
        src_off,
        3,
        BLOCK_SIZE=128,
        MAMBA_BLOCK_SIZE=4,
    )
except InvalidLaunchError:
    bad_block_rejected = True
torch.mcpu.synchronize()
registry = get_registry()
print(json.dumps({
    "state_idx": state_idx.cpu().tolist(),
    "num_accepted": num_accepted.cpu().tolist(),
    "src_col": src_col.cpu().tolist(),
    "src_off": src_off.cpu().tolist(),
    "launches": registry.launch_counts()[
        "vllm.v1.worker.mamba_utils.preprocess_mamba_align_fused_kernel"
    ],
    "bad_block_rejected": bad_block_rejected,
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
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
        "state_idx": [1, 88, 0, 2],
        "num_accepted": [1, 77, 2, 1],
        "src_col": [0, 99, -1, 1],
        "src_off": [2, 99, 1, 3],
        "launches": 1,
        "bad_block_rejected": True,
    }
