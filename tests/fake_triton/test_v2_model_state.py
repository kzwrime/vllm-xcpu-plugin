# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_scatter_num_accepted_kernel_dispatches_with_current_vllm_abi():
    repo = Path(__file__).parents[2]
    torch_mcpu_repo = repo.parent / "torch_mcpu"
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
        "num_accepted": [1, 77, 1, 6],
        "launches": 1,
        "bad_grid_rejected": True,
    }
