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


def test_probabilistic_selector_matches_dense_candidate_sampling():
    """Candidate walk uses token-ID noise, temperature, and the draft salt."""
    repo = Path(__file__).parents[2]
    code = """
import json
import torch
import torch_xcpu
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import _selector_walk_kernel
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()
torch.manual_seed(19)
rows, steps, top_k, vocab = 12, 3, 4, 128
scores = torch.randn(rows, steps, top_k, top_k)
candidates = torch.tensor(
    [[[7, 29, 80, 3], [9, 33, 101, 12], [77, 42, 6, 119]]]
).expand(rows, -1, -1).contiguous()
positions = torch.arange(rows * steps, dtype=torch.int64) + 10
states = torch.arange(rows, dtype=torch.int32).repeat_interleave(steps)
states[-steps:] = -1
seeds = torch.arange(rows, dtype=torch.int64) + 30
results = []
for use_fp64 in (False, True):
    temperatures = torch.full((rows,), 0.7)
    temperatures[0] = 0.0
    expected = torch.full((rows, steps), -1, dtype=torch.int64)
    expected_scores = torch.full((rows, steps, top_k), -99.0)
    for row in range(rows - 1):
        previous = 0
        for step in range(steps):
            row_scores = scores[row, step, previous]
            ids = candidates[row, step]
            dense = torch.full((1, vocab), -float('inf'), device='mcpu')
            dense[0, ids.to('mcpu')] = row_scores.to('mcpu')
            sampled = torch.zeros(1, dtype=torch.int64, device='mcpu')
            torch_xcpu.ops.gumbel_sample(
                dense, sampled, torch.tensor([row], dtype=torch.int32, device='mcpu'),
                temperatures.to('mcpu'), seeds.to('mcpu'),
                (positions[row * steps + step:row * steps + step + 1] - 1).to('mcpu'),
                apply_temperature=True, use_fp64=use_fp64, is_drafting=True,
            )
            token = sampled.cpu().item()
            expected[row, step] = token
            expected_scores[row, step] = row_scores
            previous = ids.tolist().index(token)
    tokens = torch.full((rows, steps), -1, dtype=torch.int64, device='mcpu')
    realized = torch.full((rows, steps, top_k), -99.0, device='mcpu')
    _selector_walk_kernel[(rows,)](
        scores.to('mcpu'), candidates.to('mcpu'), positions.to('mcpu'),
        states.to('mcpu'), temperatures.to('mcpu'), seeds.to('mcpu'), tokens, realized,
        num_steps=steps, top_k=top_k, BLOCK_K=4, SAMPLE_PROBABILISTIC=True,
        USE_FP64=use_fp64, num_warps=1,
    )
    assert torch.equal(tokens.cpu(), expected)
    assert torch.equal(realized.cpu(), expected_scores)
    results.append(use_fp64)
print(json.dumps({'passed': results}))
"""
    env = dict(os.environ, PYTHONPATH=str(repo), VLLM_PLUGINS="xcpu_platform_plugin")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "passed": [False, True]
    }
