# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_xcpu_plugin.attn_backend import xcpu_sparse_mla_attention
from vllm_xcpu_plugin.layers.sparse_attn_indexer import (
    xcpu_sparse_indexer_topk,
)


def _make_paged_cache(
    logical_values: torch.Tensor,
    block_size: int,
    block_table: torch.Tensor,
    device: str,
) -> torch.Tensor:
    num_blocks = int(block_table.max().item()) + 1
    cache = torch.zeros(
        num_blocks,
        block_size,
        logical_values.shape[-1],
        dtype=logical_values.dtype,
    )
    for logical_idx in range(logical_values.shape[0]):
        block = int(block_table[logical_idx // block_size])
        cache[block, logical_idx % block_size] = logical_values[logical_idx]
    return cache.to(device)


@pytest.mark.parametrize("seq_len", [2049, 4096, 4101])
def test_sparse_indexer_arbitrary_long_context(seq_len: int) -> None:
    torch.manual_seed(7)
    device = "mcpu"
    block_size = 64
    query_len = 1
    num_heads = 2
    head_dim = 4
    topk = 4
    num_logical_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(num_logical_blocks - 1, -1, -1, dtype=torch.int32)

    logical_k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    cache = _make_paged_cache(logical_k, block_size, block_table, device)
    q = torch.randn(query_len, num_heads, head_dim).to(torch.float8_e4m3fn)
    weights = torch.randn(query_len, num_heads, dtype=torch.float32)
    output = torch.empty(query_len, topk, dtype=torch.int32, device=device)

    actual = xcpu_sparse_indexer_topk(
        q.to(device),
        weights.to(device),
        cache,
        block_table.unsqueeze(0).to(device),
        torch.tensor([0, query_len], dtype=torch.int32),
        torch.tensor([seq_len], dtype=torch.int32),
        block_size,
        topk,
        output,
    ).cpu()

    score = torch.matmul(q.to(torch.bfloat16), logical_k.T)
    logits = (torch.clamp(score, min=0).float() * weights.unsqueeze(-1)).sum(dim=1)
    expected = torch.topk(logits, topk, dim=-1).indices.to(torch.int32)
    assert set(actual[0].tolist()) == set(expected[0].tolist())


def test_sparse_indexer_prefill_is_causal() -> None:
    torch.manual_seed(11)
    device = "mcpu"
    seq_len = 2051
    query_len = 3
    block_size = 64
    topk = 8
    num_heads = 2
    head_dim = 4
    blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.randperm(blocks, dtype=torch.int32)
    logical_k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    cache = _make_paged_cache(logical_k, block_size, block_table, device)
    q = torch.randn(query_len, num_heads, head_dim).to(torch.float8_e4m3fn)
    weights = torch.randn(query_len, num_heads)
    output = torch.empty(query_len, topk, dtype=torch.int32, device=device)

    actual = xcpu_sparse_indexer_topk(
        q.to(device),
        weights.to(device),
        cache,
        block_table.unsqueeze(0).to(device),
        torch.tensor([0, query_len], dtype=torch.int32),
        torch.tensor([seq_len], dtype=torch.int32),
        block_size,
        topk,
        output,
    ).cpu()

    context_len = seq_len - query_len
    for query_idx in range(query_len):
        assert torch.all(actual[query_idx] < context_len + query_idx + 1)


def test_sparse_indexer_short_context_pads_with_minus_one() -> None:
    device = "mcpu"
    output = torch.empty(2, 8, dtype=torch.int32, device=device)
    actual = xcpu_sparse_indexer_topk(
        torch.zeros(2, 1, 2, dtype=torch.float8_e4m3fn, device=device),
        torch.ones(2, 1, device=device),
        torch.zeros(1, 4, 2, dtype=torch.bfloat16, device=device),
        torch.tensor([[0]], dtype=torch.int32, device=device),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        4,
        8,
        output,
    ).cpu()

    expected_first = torch.tensor([0, 1, -1, -1, -1, -1, -1, -1], dtype=torch.int32)
    expected_second = torch.tensor([0, 1, 2, -1, -1, -1, -1, -1], dtype=torch.int32)
    torch.testing.assert_close(actual[0], expected_first)
    torch.testing.assert_close(actual[1], expected_second)


def test_sparse_mla_consumes_only_selected_tokens() -> None:
    torch.manual_seed(13)
    device = "mcpu"
    block_size = 4
    seq_len = 9
    q = torch.randn(1, 2, 6, dtype=torch.bfloat16)
    logical_kv = torch.randn(seq_len, 6, dtype=torch.bfloat16)
    block_table = torch.tensor([2, 0, 1], dtype=torch.int32)
    cache = _make_paged_cache(logical_kv, block_size, block_table, device)
    selected = torch.tensor([[1, 5, 8, -1]], dtype=torch.int32)
    output = torch.empty(1, 2, 4, dtype=torch.bfloat16, device=device)

    actual = xcpu_sparse_mla_attention(
        q.to(device),
        cache,
        selected.to(device),
        block_table.unsqueeze(0).to(device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        0.5,
        4,
        output,
    ).cpu()

    selected_kv = logical_kv[[1, 5, 8]]
    scores = torch.matmul(q[0], selected_kv.T).float() * 0.5
    expected = torch.matmul(
        torch.softmax(scores, dim=-1).to(q.dtype), selected_kv[:, :4]
    )
    torch.testing.assert_close(actual, expected.unsqueeze(0), atol=2e-2, rtol=2e-2)

    changed = logical_kv.clone()
    changed[0] += 100
    changed_cache = _make_paged_cache(changed, block_size, block_table, device)
    unchanged = xcpu_sparse_mla_attention(
        q.to(device),
        changed_cache,
        selected.to(device),
        block_table.unsqueeze(0).to(device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        0.5,
        4,
        torch.empty_like(output),
    ).cpu()
    torch.testing.assert_close(actual, unchanged)
