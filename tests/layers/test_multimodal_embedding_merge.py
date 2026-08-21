# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_multimodal_embedding_merge_accepts_cpu_mask():
    repo = Path(__file__).parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["VLLM_PLUGINS"] = "xcpu_platform_plugin"
    code = r'''
import json

import torch

from vllm.model_executor.models.utils import _merge_multimodal_embeddings


inputs = torch.arange(20, dtype=torch.float32).reshape(5, 4).to("mcpu")
multimodal = [
    torch.tensor(
        [[101.0, 102.0, 103.0, 104.0],
         [201.0, 202.0, 203.0, 204.0],
         [301.0, 302.0, 303.0, 304.0]],
        device="mcpu",
    )
]
mask = torch.tensor([True, False, True, True, False], device="cpu")
expected = inputs.cpu()
expected[mask] = multimodal[0].cpu()

merged = _merge_multimodal_embeddings(inputs, multimodal, mask)
assert merged.data_ptr() == inputs.data_ptr()
torch.mcpu.synchronize()
torch.testing.assert_close(merged.cpu(), expected, rtol=0, atol=0)

print(json.dumps({
    "input_device": str(inputs.device),
    "mask_device": str(mask.device),
    "shape": list(merged.shape),
    "dtype": str(merged.dtype),
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary == {
        "input_device": "mcpu:0",
        "mask_device": "cpu",
        "shape": [5, 4],
        "dtype": "torch.float32",
    }
