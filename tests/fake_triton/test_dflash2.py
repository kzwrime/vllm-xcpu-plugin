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
rows, steps, top_k, vocab = 12, 3, 16, 128
scores = torch.randn(rows, steps, top_k, top_k)
candidates = torch.randperm(vocab)[:steps * top_k].view(1, steps, top_k)
candidates = candidates.expand(rows, -1, -1).contiguous()
positions = torch.arange(rows * steps, dtype=torch.int64) + 10
states = torch.arange(rows, dtype=torch.int32).repeat_interleave(steps)
states[-steps:] = -1
seeds = torch.arange(rows, dtype=torch.int64) + 30
results = []
# The dense reference explicitly rejects FP32 sampling in the v0.31 port.
# Compare the supported FP64 path; do not claim FP32 parity.
for use_fp64 in (True,):
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
        num_steps=steps, top_k=top_k, BLOCK_K=16, SAMPLE_PROBABILISTIC=True,
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
        "passed": [True]
    }


def test_draft_gather_masks_prefill_placeholders_after_state_reuse():
    """Prefill placeholders must be rejected, including across a Triton tile."""
    repo = Path(__file__).parents[2]
    code = """
import torch
import torch_xcpu
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _gather_draft_sampled_kernel,
)
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()
torch.manual_seed(71)
rows = 1025
ids = torch.arange(rows, dtype=torch.int32) + 10
positions = torch.arange(rows, dtype=torch.int64) + 100
states = torch.arange(rows, dtype=torch.int32) % 4
local = torch.arange(rows, dtype=torch.int32) % 6
prefill = torch.tensor([100, 500, 900, 1200], dtype=torch.int32)
for index_dtype in (torch.int32, torch.int64):
    indices = torch.randperm(rows).to(index_dtype)
    inputs = [t.to('mcpu')
              for t in (ids, positions, indices, states, local, prefill)]
    draft = torch.full((rows + 1,), -77, dtype=torch.int32, device='mcpu')
    pos = torch.full((rows + 1,), -77, dtype=torch.int64, device='mcpu')
    for lengths in (prefill, prefill.flip(0)):
        inputs[-1].copy_(lengths)
        _gather_draft_sampled_kernel[(2,)](
            draft, pos, *inputs, rows, BLOCK_SIZE=1024,
        )
        expected_pos = positions[indices.long()]
        expected = ids[indices.long()].clone()
        placeholder = (local > 0) & (expected_pos - local < lengths[states.long()])
        expected[placeholder] = -1
        torch.testing.assert_close(draft.cpu()[:rows], expected)
        torch.testing.assert_close(pos.cpu()[:rows], expected_pos)
        assert draft.cpu()[-1].item() == -77 and pos.cpu()[-1].item() == -77
print('passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(repo), VLLM_PLUGINS="xcpu_platform_plugin"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("passed")


def test_standard_rejection_ignores_stale_proposal_for_prefill_placeholders():
    """An invalid draft must sample the target even when the slot has old logits."""
    repo = Path(__file__).parents[2]
    code = """
import torch
import torch_xcpu
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()
def t(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device='mcpu')
target = torch.full((6, 32), -1000., device='mcpu')
for row, token in enumerate([2, 3, 4, 5, 6, 7]):
    target[row, token] = 0.
draft = torch.full((4, 2, 32), -1000., device='mcpu')
draft[1, 0, 2] = draft[1, 1, 3] = 0.
draft[3, 0, 5] = draft[3, 1, 6] = 0.
for temp in (0., 0.7):
    for probabilities in (None, draft):
        sampled, counts = rejection_sample(
            target_logits=target, draft_logits=probabilities,
            draft_sampled=t([99, 2, 3, 99, -1, -1]),
            cu_num_logits=t([0, 3, 6]), pos=t([10, 11, 12, 20, 21, 22], torch.int64),
            idx_mapping=t([1, 3]), expanded_idx_mapping=t([1, 1, 1, 3, 3, 3]),
            expanded_local_pos=t([0, 1, 2, 0, 1, 2]),
            temperature=t([0., temp, 0., temp], torch.float32),
            seed=t([0, 71, 0, 97], torch.int64), num_speculative_steps=2,
            use_fp64=True,
        )
        assert counts.cpu().tolist() == [3, 1], counts.cpu()
        assert sampled.cpu()[0].tolist() == [2, 3, 4], sampled.cpu()
        assert sampled.cpu()[1, 0].item() == 5, sampled.cpu()
print('passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(repo), VLLM_PLUGINS="xcpu_platform_plugin"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("passed")


def test_prepare_dflash_inputs_abi_matches_upstream():
    """Catch reordered metadata tensors before a positional call corrupts KV state."""
    import ast
    import inspect

    import torch
    import torch_xcpu
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator

    from vllm_xcpu_plugin.dflash_patch import _xcpu_prepare_dflash_inputs

    tree = ast.parse(Path(speculator.__file__).read_text())
    definitions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    wrapper = definitions["prepare_dflash_inputs"]
    assert list(inspect.signature(_xcpu_prepare_dflash_inputs).parameters) == [
        arg.arg for arg in wrapper.args.args
    ]
    kernel = definitions["_prepare_dflash_inputs_kernel"]
    # BLOCK_SIZE tiles the Triton launch; num_reqs supplies its grid on XCPU.
    expected = [arg.arg.removesuffix("_ptr").lower() for arg in kernel.args.args[:-1]]
    expected.append("num_reqs")
    schema = torch.ops.torch_xcpu.prepare_dflash_inputs.default._schema
    assert [arg.name for arg in schema.arguments] == expected
    assert (
        list(inspect.signature(torch_xcpu.ops.prepare_dflash_inputs).parameters)
        == expected
    )
