# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_v031_kernels_dispatch():
    repo = Path(__file__).parents[2]
    code = """
import numpy as np
import torch
from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel as slots_kernel
from vllm.v1.worker.gpu.input_batch import prepare_prefill_inputs
from vllm.v1.worker.gpu.structured_outputs import (
    _apply_grammar_bitmask_kernel as grammar_kernel, _build_grammar_mapping,
)
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()

def tensor(data, dtype=torch.int32):
    return torch.tensor(data, dtype=dtype, device="mcpu")

# Sharded, replicated and disabled groups; dummy request and tail padding.
table = tensor([[10, 11, 12, 13]])
slots = tensor([[999] * 12] * 3, torch.int64)
slots_kernel[(3, 3)](
    12, tensor([0, -1]), tensor([0, 8, 10]),
    tensor(list(range(8)) + [0, 1000], torch.int64),
    tensor([table.data_ptr()] * 3, torch.uint64), tensor([4] * 3, torch.int64),
    tensor([4] * 3), tensor([2] * 3), tensor([True, True, False], torch.bool),
    tensor([True, False, True], torch.bool), slots, 12, 1,
    CP_SIZE=2, CP_INTERLEAVE=1, PAD_ID=-1, TRITON_BLOCK_SIZE=1024,
)
assert slots.cpu().tolist() == [
    [-1, 20, -1, 21, -1, 22, -1, 23] + [-1] * 4,
    list(range(20, 28)) + [-1] * 4, [-1] * 12,
]

# Partial/final prefill writes lookahead tokens or zero; decode stays untouched.
inputs = tensor([-99] * 5)
next_tokens = tensor([[-99] * 5] * 3)[:, :3]
prepare_prefill_inputs(
    inputs, next_tokens, tensor([2, 0, 1]), tensor([0, 1, 3, 5]),
    tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]]),
    tensor([4, 3, 2]), tensor([1, 1, 2]),
)
assert inputs.cpu().tolist() == [-99, 11, 12, 21, 22]
assert next_tokens.cpu().tolist() == [[13, 0, -99], [0, 0, -99], [0, 0, -99]]

# Scheduled grammar positions may be trimmed by compacted device offsets.
mapping = _build_grammar_mapping(
    ["a", "b", "c"], ["c", "a"], np.array([0, 1, 1, 2]),
    np.array([2, 0, 1]), 1, 4,
)
assert mapping == [8, 9, 0, 1, 2]
logits = torch.arange(70, dtype=torch.float32, device="mcpu").reshape(2, 35)
grammar_kernel[(5, 1)](
    logits, 35, tensor(mapping), tensor([0, 1, 1, 2]),
    tensor([[-3, 5], [0, 0], [-1, -1], [0, 0], [0, 0]]), 2, 35,
    MASK_STRIDE=4, BLOCK_SIZE=8192,
)
expected = torch.arange(70, dtype=torch.float32).reshape(2, 35)
expected[1, 1] = expected[1, 33] = -float("inf")
torch.testing.assert_close(logits.cpu(), expected)
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


@pytest.mark.parametrize("omp_threads", [1, 2, 4])
def test_topk_logprobs_preserve_large_common_offsets(omp_threads):
    """Subtract the row maximum before log(sumexp), including padded rows."""
    repo = Path(__file__).parents[2]
    code = """
import torch
from vllm.v1.worker.gpu.sample.logprob import (
    compute_token_logprobs, _topk_log_softmax_kernel,
)
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()

for dtype in (torch.float32, torch.float16, torch.bfloat16):
 for vocab in (2, 1025, 8193):
  for offset in ((0., 1000., 1e4) if dtype == torch.float16 else (0., 1000., 1e8)):
    storage = torch.full((2, vocab + 3), offset, dtype=dtype)
    storage[1, :vocab - 1] += torch.linspace(-64, 64, vocab - 1).to(dtype)
    storage[1, vocab - 1] = -float('inf')
    storage[:, vocab:] = 123
    logits = storage.to('mcpu')[:, :vocab]
    ids = torch.full((2, 1708), -999, dtype=torch.int64, device='mcpu')
    ids[:, :1701] = torch.arange(1701, device='mcpu') % vocab
    ids = ids[:, :1701]
    actual = compute_token_logprobs(logits, ids).cpu()
    expected = storage[:, :vocab].float().log_softmax(-1).gather(1, ids.cpu())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logits.cpu(), storage[:, :vocab], rtol=0, atol=0)
    if vocab == 1025 and offset == 0.:
        for block_size in (64, 2048, 4096):
            out = torch.empty((2, 1701), device='mcpu')
            _topk_log_softmax_kernel[(2,)](
                out, logits, logits.stride(0), ids, ids.stride(0), 1701, vocab,
                BLOCK_SIZE=block_size, TOPK_BLOCK_SIZE=1024)
            torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)

# The native default remains usable by existing callers.
logits = torch.full((1, 2), 1e8, device='mcpu')
ids = torch.zeros((1, 1), dtype=torch.int64, device='mcpu')
out = torch.empty((1, 1), device='mcpu')
torch.ops.torch_xcpu.topk_log_softmax(out, logits, ids, 1, 2)
torch.testing.assert_close(out.cpu(), logits.cpu().log_softmax(-1)[:, :1])

# Empty batches and one-row vocabularies exercise both sides of row parallelism.
for rows in (0, 1, 7):
    logits = (torch.arange(rows * 2057).reshape(rows, 2057) % 127).float().to('mcpu')
    ids = (torch.arange(rows * 1305).reshape(rows, 1305) % 2057).to('mcpu')
    expected = logits.cpu().log_softmax(-1).gather(1, ids.cpu())
    actual = compute_token_logprobs(logits, ids).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    out = torch.empty((rows, 1305), device='mcpu')
    for topk_block in (1, 64, 1024, 2048):
        torch.ops.torch_xcpu.topk_log_softmax(
            out, logits, ids, 1305, 2057, 1024, topk_block)
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)

out = torch.empty((1, 0), device='mcpu')
torch.ops.torch_xcpu.topk_log_softmax(
    out, torch.zeros((1, 2), device='mcpu'),
    torch.empty((1, 0), dtype=torch.int64, device='mcpu'), 0, 2)
assert out.numel() == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        timeout=120,
        env={
            **os.environ,
            "PYTHONPATH": str(repo),
            "VLLM_PLUGINS": "xcpu_platform_plugin",
            "OMP_NUM_THREADS": str(omp_threads),
            "OMP_DYNAMIC": "FALSE",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_zero_kv_blocks_clears_only_launched_chunks():
    """A partial chunk grid must leave the unlaunched payload and padding intact."""
    repo = Path(__file__).parents[2]
    code = """
import torch
import torch_xcpu
from vllm.v1.worker.utils import _zero_kv_blocks_kernel
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()
def t(data, dtype=torch.int64): return torch.tensor(data, dtype=dtype, device='mcpu')

for block_size in (4, 8):
 for chunks in (0, 1, 2, 3, 5):
    segments = [torch.full((4 * stride + 4,), 99, dtype=torch.int32, device='mcpu')
                for stride in (13, 21)]
    addresses = t([s.data_ptr() + 8 for s in segments], torch.uint64)
    _zero_kv_blocks_kernel[(2, 2, chunks)](
        addresses, t([13, 21]), t([5, 17]), t([1, 3]), BLOCK_SIZE=block_size)
    for s, stride, size in zip(segments, (13, 21), (5, 17)):
        expected = torch.full(s.shape, 99, dtype=torch.int32)
        for block in (1, 3):
            start = 2 + block * stride
            expected[start:start + min(size, chunks * block_size)] = 0
        torch.testing.assert_close(s.cpu(), expected, rtol=0, atol=0)

# Existing direct callers retain full-payload zeroing without a coverage cap.
storage = t([99] * 8, torch.int32)
torch.ops.torch_xcpu.zero_kv_blocks(
    t([storage.data_ptr()], torch.uint64), t([8]), t([8]), t([0]))
torch.testing.assert_close(storage.cpu(), torch.zeros(8, dtype=torch.int32))
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


def test_bad_words_masks_only_launched_words():
    """The word grid bounds filtering even when request metadata lists more words."""
    repo = Path(__file__).parents[2]
    code = """
import torch
import torch_xcpu
from vllm.v1.worker.gpu.sample.bad_words import _bad_words_kernel
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()
def t(data): return torch.tensor(data, dtype=torch.int32, device='mcpu')

mapping, counts = t([1, 0, 1]), t([3, 2])
words, offsets = t([[1, 2, 3], [3, 4, 4]]), t([[0, 1, 2, 3]] * 2)
for dtype in (torch.float32, torch.float16, torch.bfloat16):
 for grid in ((3,), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)):
    storage = torch.full((3, 8), 123., dtype=dtype, device='mcpu')
    logits = storage[:, :6]
    _bad_words_kernel[grid](
        logits, 8, mapping, words, 3, offsets, 4, counts,
        t([[0], [0]]), 1, t([0, 0]), t([0, 0]), t([0, 0, 0]), t([0, 0, 0]))
    expected = torch.full((3, 8), 123., dtype=dtype)
    word_limit = grid[1] if len(grid) > 1 else 1
    for row, req in enumerate((1, 0, 1)):
        for word in range(min((3, 2)[req], word_limit)):
            expected[row, ((1, 2, 3), (3, 4, 4))[req][word]] = -float('inf')
    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)

# Default native API still processes every word, including duplicate endings.
logits = torch.zeros((3, 6), device='mcpu')
torch.ops.torch_xcpu.bad_words(logits, mapping, words, offsets, counts,
    t([[0], [0]]), t([0, 0]), t([0, 0]), t([0, 0, 0]), t([0, 0, 0]))
expected = torch.zeros((3, 6))
expected[0, 3:5] = expected[2, 3:5] = -float('inf')
expected[1, 1:4] = -float('inf')
torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
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


def test_v031_gdn_cache_layouts_and_state_migration():
    """Clear only page payloads; migrate SD/DS conv and padded SSM states."""
    repo = Path(__file__).parents[2]
    code = """
import torch
import torch_xcpu
from vllm.v1.worker.utils import _zero_kv_blocks_kernel
from vllm.v1.worker.mamba_utils import (
    precopy_mamba_align_fused_kernel, postprocess_mamba_fused_kernel,
)
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()
def t(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device="mcpu")

segments = [torch.full((4 * stride + 4,), 99, dtype=torch.int32, device="mcpu")
            for stride in (13, 21)]
_zero_kv_blocks_kernel[(2, 2, 3)](
    t([s.data_ptr() + 8 for s in segments], torch.uint64),
    t([13, 21], torch.int64), t([5, 17], torch.int64), t([1, 3], torch.int64),
    BLOCK_SIZE=8,
)
for s, stride, size in zip(segments, (13, 21), (5, 17)):
    expected = torch.full(s.shape, 99, dtype=torch.int32, device="cpu")
    for block in (1, 3):
        expected[2 + block * stride:2 + block * stride + size] = 0
    torch.testing.assert_close(s.cpu(), expected)

for ds in (False, True):
 for mapped in (False, True):
  for tiles in (1, 16):
    conv_storage = torch.arange(5 * 23, dtype=torch.float32, device="mcpu")
    conv = conv_storage.as_strided((5, 3, 5) if ds else (5, 5, 3),
                                  (23, 7, 1) if ds else (23, 3, 1))
    ssm_storage = torch.arange(5 * 11, dtype=torch.float32, device="mcpu")
    ssm = ssm_storage.as_strided((5, 7), (11, 1))
    conv_before, ssm_before = conv_storage.cpu(), ssm_storage.cpu()
    table = t([[1, 2, 3, 4]])
    layout = (
        t([table.data_ptr()], torch.int64), 4,
        t([conv.data_ptr(), ssm.data_ptr()], torch.int64),
        t([23 * 4, 11 * 4], torch.int64), t([4, 4]),
        t([3, 7], torch.int64), t([5, 0]), t([0, 0]),
        t([3 if ds else 0, 0]), t([28 if ds else 0, 0], torch.int64),
    )
    state_idx = t([1, 1, 1] if mapped else [1])
    src = t([0, 0, 0] if mapped else [0])
    bias = t([2, 2, 2] if mapped else [2])
    mapping = t([2]) if mapped else None
    precopy_mamba_align_fused_kernel[(1, 2, tiles)](
        state_idx, src, bias, *layout, mapping, 1,
        COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=ds,
        HAS_IDX_MAPPING=mapped, TEMPORAL_TILES=tiles,
    )
    expected = conv_before.clone()
    expected_view = expected.as_strided(conv.shape, conv.stride())
    source = conv_before.as_strided(conv.shape, conv.stride())[1]
    if ds:
        expected_view[2, :, :3] = source[:, 2:]
    else:
        expected_view[2, :3] = source[2:]
    torch.testing.assert_close(conv_storage.cpu(), expected)
    expected_ssm = ssm_before.clone()
    expected_ssm[22:29] = ssm_before[33:40]
    torch.testing.assert_close(ssm_storage.cpu(), expected_ssm)

    # Postprocess performs a same-block left shift at an accepted boundary.
    conv_storage.copy_(conv_before.to("mcpu"))
    ssm_storage.copy_(ssm_before.to("mcpu"))
    accepted = t([3])
    postprocess_mamba_fused_kernel[(1, 2, tiles)](
        accepted, t([0]), None, t([4]), None, *layout, t([99]), t([0]), 1,
        block_size=4, COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=ds,
        HAS_IDX_MAPPING=True, PRECOMPUTED_NEW_COMPUTED=True, TEMPORAL_TILES=tiles,
    )
    expected = conv_before.clone()
    expected_view = expected.as_strided(conv.shape, conv.stride())
    if ds:
        expected_view[1, :, :3] = source[:, 2:]
    else:
        expected_view[1, :3] = source[2:]
    torch.testing.assert_close(conv_storage.cpu(), expected)
    expected_ssm = ssm_before.clone()
    expected_ssm[11:18] = ssm_before[33:40]
    torch.testing.assert_close(ssm_storage.cpu(), expected_ssm)
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


@pytest.mark.parametrize("omp_threads", [1, 2, 4])
def test_v031_logit_bias_restore_and_strides(omp_threads):
    """Match bias/masking order, per-request restore and expanded decode rows."""
    repo = Path(__file__).parents[2]
    code = r"""
import torch
from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()

for vocab in (35, 8193, 248320):
 for dtype in (torch.float32, torch.float16, torch.bfloat16):
  for pos_dtype in (torch.int32, torch.int64):
   for check in (False, True):
     rows, reqs = 10, 9
     mapping = torch.tensor([2, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)
     pos = torch.tensor([0, 0, 0, 9, 9, 0, 0, 0, 0, 0], dtype=pos_dtype)
     lengths = torch.full((reqs,), 10, dtype=torch.int32)
     restore = torch.ones(reqs, dtype=torch.int32)
     restore[1] = 0
     na = torch.zeros(reqs, dtype=torch.int32)
     na[0] = 2
     nb = torch.ones(reqs, dtype=torch.int32)
     ns = torch.full((reqs,), 2, dtype=torch.int32)
     ns[7] = 0
     # Independent table row strides, with out-of-vocabulary padding canaries.
     allowed = torch.full((reqs, 7), -999, dtype=torch.int32)
     allowed[:, :2] = torch.tensor([3, 4])
     ids = torch.full((reqs, 9), -999, dtype=torch.int32)
     ids[:, 0] = 3
     bias = torch.full((reqs, 11), -999.0)
     bias[:, 0] = torch.arange(reqs) + 0.5
     stops = torch.full((reqs, 6), -999, dtype=torch.int32)
     stops[:, :2] = torch.tensor([3, 4])
     original = torch.full((rows, vocab + 8), 123.0, dtype=dtype)
     original[:, :vocab] = -float('inf')
     original[:, 3] = 1
     original[:, 4] = 2
     original[0, 6] = 3  # legal non-stop: keep stop mask
     original[1, 7] = 9  # allowed IDs filter must run first
     original[5, 3:5] = torch.tensor([float('inf'), float('nan')])
     original[6, 6] = float('nan')  # Triton maximum ignores NaN
     original[7, :vocab] = -float('inf')  # no finite stop to restore
     original[9, vocab - 1] = float('inf')  # +inf is not an all-masked row
     expected = original.clone()
     for i, req in enumerate(mapping.tolist()):
         row = expected[i, :vocab]
         if na[req] > 0:
             keep = allowed[req, :int(na[req])].long()
             saved = row[keep].clone()
             row.fill_(-float('inf'))
             row[keep] = saved
         token_ids = ids[req, :int(nb[req])].long()
         row[token_ids] = (row[token_ids].float() + bias[req, :int(nb[req])]).to(dtype)
         if ns[req] > 0 and pos[i] + 1 < lengths[req]:
             stop = stops[req, :int(ns[req])].long()
             saved = row[stop].clone()
             row[stop] = -float('inf')
             row_max = torch.fmax(row, torch.full_like(row, -float('inf'))).amax()
             if check and restore[req] and row_max == -float('inf'):
                 finite = torch.isfinite(saved)
                 row[stop[finite]] = saved[finite]
     def t(x): return x.to('mcpu')
     # Include single-row and empty launches, in addition to parallel rows.
     for selection in (slice(None), slice(1, 2), slice(0, 0)):
         storage = original[selection].to('mcpu')
         # Slice after transfer so padding and independent strides survive.
         apply_logit_bias(storage[:, :vocab], t(mapping[selection]), t(pos[selection]),
                          t(na), t(allowed)[:, :2], t(nb), t(ids)[:, :2],
                          t(bias)[:, :2], t(lengths), t(ns), t(restore),
                          t(stops)[:, :2], check_all_masked_rows=check)
         torch.testing.assert_close(storage.cpu(), expected[selection],
                                    rtol=0, atol=0, equal_nan=True)
print('36 vocab/dtype/position/restore combinations, multi/single/empty rows passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        timeout=120,
        env={
            **os.environ,
            "PYTHONPATH": str(repo),
            "VLLM_PLUGINS": "xcpu_platform_plugin",
            "OMP_NUM_THREADS": str(omp_threads),
            "OMP_DYNAMIC": "FALSE",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("omp_threads", [1, 2, 4])
def test_v031_bad_words_draft_boundary_and_logprob_row_stride(omp_threads):
    repo = Path(__file__).parents[2]
    code = r"""
import torch
from vllm.v1.worker.gpu.sample.bad_words import apply_bad_words
from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels
register_vllm_kernels()

def t(x, dtype=torch.int32): return torch.tensor(x, dtype=dtype, device='mcpu')
# Duplicate words and different words ending at the same token must not race.
word_lists = [[[2, 3], [3, 4], [10], [3], [2, 3]],
              [[4, 5], [5, 6], [4, 5, 7], [6, 8], [11], [6],
               [5, 6], [1, 2, 3, 4, 5, 6, 7]], []]
words = torch.full((3, 32), -999, dtype=torch.int32)
offsets = torch.full((3, 12), -999, dtype=torch.int32)
for req, req_words in enumerate(word_lists):
    end = 0
    offsets[req, 0] = 0
    for j, word in enumerate(req_words):
        words[req, end:end + len(word)] = torch.tensor(word)
        end += len(word)
        offsets[req, j + 1] = end
mapping, local = [1, 1, 1, 0, 0, 2], [0, 1, 2, 0, 1, 0]
inputs = [4, 5, 6, 2, 3, 8]
committed = [[1, 2], [4], []]
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    expected = torch.zeros((6, 40), dtype=dtype)
    expected[:, 35:] = 123
    for row, (req, pos) in enumerate(zip(mapping, local)):
        first = row - pos
        context = committed[req] + inputs[first + 1:first + 1 + pos]
        for word in word_lists[req]:
            prefix = word[:-1]
            matches = len(context) >= len(prefix) and context[-len(prefix):] == prefix
            if not prefix or matches:
                expected[row, word[-1]] = -float('inf')
    for selection in (slice(None), slice(0, 1), slice(0, 0)):
        storage = torch.zeros((len(mapping[selection]), 40), dtype=dtype, device='mcpu')
        storage[:, 35:] = 123
        apply_bad_words(storage[:, :35], t(mapping[selection]),
            words.to('mcpu')[:, :24],
            offsets.to('mcpu')[:, :9], t([len(ws) for ws in word_lists]),
            t([[9, 1, 2, 99, 99], [8, 4, 99, 99, 99], [8, 99, 99, 99, 99]])[:, :3],
            t([1, 1, 1]), t([3, 2, 1]), t(inputs), t(local[selection]), 8)
        torch.testing.assert_close(storage.cpu(), expected[selection], rtol=0, atol=0)

    logits = (torch.arange(90, dtype=torch.float32).reshape(2, 45) / 17).to(dtype)
    ids = t([[3, 7, 34, 0, -1, -1, -1], [30, 1, 4, 9, -1, -1, -1]], torch.int64)[:, :4]
    actual = compute_token_logprobs(logits.to('mcpu')[:, :35], ids)
    expected = logits[:, :35].float().log_softmax(-1).gather(1, ids.cpu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
print('bad words and logprobs: 3 dtypes passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        timeout=120,
        env={
            **os.environ,
            "PYTHONPATH": str(repo),
            "VLLM_PLUGINS": "xcpu_platform_plugin",
            "OMP_NUM_THREADS": str(omp_threads),
            "OMP_DYNAMIC": "FALSE",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
