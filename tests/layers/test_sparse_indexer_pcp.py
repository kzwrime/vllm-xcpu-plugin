# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers import sparse_attn_indexer as upstream
from vllm.model_executor.layers.attention import pcp
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.worker import workspace

from vllm_xcpu_plugin.layers import sparse_attn_indexer as xcpu


@pytest.mark.parametrize("history", [0, 2048])
def test_pcp_gathers_k_before_cache_write(monkeypatch, history):
    import torch_xcpu

    torch.manual_seed(7)
    device = "mcpu"
    seq_len = history + 7
    all_k = torch.randn(seq_len, 128, dtype=torch.bfloat16, device=device)
    k = all_k[history:]
    order = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], dtype=torch.int64, device=device)
    padded = torch.cat((k, torch.full_like(k[:1], 100)))
    gathered = padded[order]
    slots = torch.tensor(
        [64, 65, 70, -1, 66, 67, 68, 69], dtype=torch.int64, device=device
    )
    slots = torch.where(slots >= 0, slots + history, slots)
    calls = []

    def gather(value, dim):
        assert dim == 0
        assert torch.equal(value.cpu(), gathered[:4].cpu())
        calls.append(True)
        return gathered

    group = SimpleNamespace(world_size=2, all_gather=gather)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    monkeypatch.setattr(upstream, "get_pcp_group", lambda: group)
    cache = torch.full(
        ((seq_len + 63) // 64 + 1, 64, 132), 0xA5, dtype=torch.uint8, device=device
    )
    if history:
        torch_xcpu.ops.indexer_k_quant_and_cache(
            all_k[:history],
            cache,
            torch.arange(64, 64 + history, dtype=torch.int64, device=device),
            128,
            "float32",
        )
    expected = cache.clone()
    torch_xcpu.ops.indexer_k_quant_and_cache(
        k,
        expected,
        torch.arange(64 + history, 71 + history, dtype=torch.int64, device=device),
        128,
        "float32",
    )
    output = torch.full((4, 2048), -9, dtype=torch.int32, device=device)
    config = VllmConfig(parallel_config=ParallelConfig(prefill_context_parallel_size=2))
    with set_current_vllm_config(config):
        layer = xcpu.XcpuSparseAttnIndexer(
            SimpleNamespace(prefix="indexer", kv_cache=cache),
            128,
            "float32",
            2048,
            128,
            seq_len + 64,
            seq_len + 64,
            output,
        )
    lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
    chunk = SimpleNamespace(
        token_start=0,
        token_end=4,
        skip_kv_gather=False,
        local_total_seq_lens=seq_len,
        max_local_total_seq_lens=seq_len,
        local_cu_seq_lens=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        block_table=torch.arange(
            1, cache.shape[0], dtype=torch.int32, device=device
        ).unsqueeze(0),
        cu_seqlen_ks=torch.zeros(4, dtype=torch.int32, device=device),
        cu_seqlen_ke=torch.tensor(
            [history + 1, history + 2, seq_len, 0], dtype=torch.int32, device=device
        ),
    )
    metadata = DeepseekV32IndexerMetadata(
        lengths,
        seq_len,
        slots,
        0,
        0,
        1,
        4,
        prefill=DeepseekV32IndexerPrefillMetadata([chunk]),
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"indexer": metadata},
        slot_mapping={"indexer": slots},
    )
    monkeypatch.setattr(upstream, "get_forward_context", lambda: context)
    workspace.init_workspace_manager(torch.device(device))
    q = torch.randn(4, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.ones(4, 32, device=device)
    try:
        layer.forward_oot(
            torch.empty(4, 1, device=device),
            q,
            gathered[:4],
            weights,
        )
        actual = output.cpu().clone()
        layer.use_pcp = False
        layer.skip_k_cache_insert = True
        layer.k_cache.kv_cache = expected
        output.fill_(-9)
        layer.forward_oot(torch.empty(4, 1, device=device), q, gathered[:4], weights)
        assert torch.equal(actual, output.cpu())
    finally:
        workspace.reset_workspace_manager()
    assert calls == [True]
    assert torch.equal(cache.cpu(), expected.cpu())
    for row, length in enumerate((history + 1, history + 2, seq_len, 0)):
        values = output[row].cpu()
        valid = values[values >= 0]
        assert len(set(valid.tolist())) == min(length, 2048)
        assert torch.all(valid < length)
        assert (values == -1).sum().item() == 2048 - min(length, 2048)
