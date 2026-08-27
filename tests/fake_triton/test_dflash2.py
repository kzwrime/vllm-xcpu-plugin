# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_dflash2_kernels_dispatch_to_mcpu_ops():
    repo = Path(__file__).parents[2]
    code = """
import json
import torch

from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    _cache_draft_logits_kernel,
    _selector_walk_kernel,
)
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

register_vllm_kernels()
scores = torch.tensor(
    [[[[1.0, 3.0], [4.0, 2.0]], [[8.0, 5.0], [6.0, 9.0]]]],
    device="mcpu",
)
candidates = torch.tensor([[[10, 11], [20, 21]]], dtype=torch.int64, device="mcpu")
sample_pos = torch.tensor([3, 4], dtype=torch.int64, device="mcpu")
req_state = torch.tensor([0, 0], dtype=torch.int32, device="mcpu")
temperature = torch.zeros(1, device="mcpu")
seeds = torch.zeros(1, dtype=torch.int64, device="mcpu")
tokens = torch.full((1, 2), -1, dtype=torch.int64, device="mcpu")
realized = torch.full((1, 2, 2), -99.0, device="mcpu")
_selector_walk_kernel[(1,)](
    scores,
    candidates,
    sample_pos,
    req_state,
    temperature,
    seeds,
    tokens,
    realized,
    num_steps=2,
    top_k=2,
    BLOCK_K=2,
    SAMPLE_PROBABILISTIC=False,
    USE_FP64=False,
    num_warps=1,
)

draft_logits = torch.full((1, 2, 32), -float("inf"), device="mcpu")
cached = torch.zeros((1, 2, 2), dtype=torch.int64, device="mcpu")
_cache_draft_logits_kernel[(2,)](
    draft_logits,
    cached,
    candidates,
    realized,
    req_state,
    draft_logits.stride(0),
    draft_logits.stride(1),
    num_steps=2,
    top_k=2,
    BLOCK_K=2,
    num_warps=1,
)
torch.mcpu.synchronize()
print(json.dumps({
    "tokens": tokens.cpu().tolist(),
    "realized": realized.cpu().tolist(),
    "cached": cached.cpu().tolist(),
    "cached_scores": [
        draft_logits.cpu()[0, 0, 10:12].tolist(),
        draft_logits.cpu()[0, 1, 20:22].tolist(),
    ],
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
        "tokens": [[11, 21]],
        "realized": [[[1.0, 3.0], [6.0, 9.0]]],
        "cached": [[[10, 11], [20, 21]]],
        "cached_scores": [[1.0, 3.0], [6.0, 9.0]],
    }
