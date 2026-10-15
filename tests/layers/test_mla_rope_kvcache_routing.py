# SPDX-License-Identifier: Apache-2.0
"""Exercise the v0.31 OOT MLA path against separate RoPE/cache/projections."""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch_mcpu  # noqa: F401
import torch_xcpu

from vllm_xcpu_plugin.attention.mla import XcpuMLAAttention
from vllm_xcpu_plugin.attention.sparse_mla import (
    XcpuMLASparseBackend,
    _xcpu_do_kv_cache_update,
    _xcpu_forward_mqa,
)


def make_layer(cache_dtype, tokens):
    layer = XcpuMLAAttention.__new__(XcpuMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads, layer.kv_lora_rank = 8, 512
    layer.qk_nope_head_dim, layer.qk_rope_head_dim, layer.v_head_dim = 192, 64, 256
    layer.kv_b_proj = torch.nn.Module()
    layer.kv_b_proj.weight = torch.nn.Parameter(
        torch.randn(8 * (192 + 256), 512, dtype=torch.bfloat16).to("mcpu") / 16,
        requires_grad=False,
    )
    layer.layer_name = "test.attn"
    layer.kv_cache_dtype = cache_dtype
    layer.kv_cache = torch.zeros(
        (1, 64, 656 if cache_dtype == "fp8_ds_mla" else 576),
        dtype=torch.uint8 if cache_dtype == "fp8_ds_mla" else torch.bfloat16,
        device="mcpu",
    )
    layer._k_scale = torch.ones(1, device="mcpu")
    layer.use_direct_call = True
    layer.use_pcp = False
    layer.pcp_shard_decode_requests = False
    layer.hisparse_cache = None
    topk = torch.full((tokens, 2048), -1, dtype=torch.int32)
    for row in range(tokens):
        topk[row, : row + 1] = torch.arange(row + 1)
    layer.impl = SimpleNamespace(
        is_sparse=True,
        process_weights_after_loading=lambda dtype: None,
        kv_cache_dtype=cache_dtype,
        kv_lora_rank=512,
        topk_indices_buffer=topk.to("mcpu"),
        scale=256**-0.5,
    )
    layer.impl.forward_mqa = MethodType(_xcpu_forward_mqa, layer.impl)
    layer.impl.do_kv_cache_update = MethodType(_xcpu_do_kv_cache_update, layer.impl)
    layer.process_weights_after_loading(torch.bfloat16)
    assert layer.kv_b_proj.weight.numel() == 0
    return layer


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_ds_mla"])
@pytest.mark.parametrize("tokens", [1, 7])
def test_fused_sparse_mla_matches_separate_cache_and_projection(
    monkeypatch, cache_dtype, tokens
):
    import vllm.model_executor.layers.attention.mla_attention as upstream

    import vllm_xcpu_plugin.attention.mla as plugin

    torch_xcpu.initialize_runtime()
    torch.manual_seed(18)
    layer = make_layer(cache_dtype, tokens)
    q = torch.randn(tokens, 8, 256, dtype=torch.bfloat16).to("mcpu")
    kv = torch.randn(tokens, 512, dtype=torch.bfloat16).to("mcpu")
    k = torch.randn(tokens, 1, 64, dtype=torch.bfloat16).to("mcpu")
    positions = torch.arange(tokens, dtype=torch.int64, device="mcpu")
    cache = torch.randn(16, 64, dtype=torch.bfloat16).to("mcpu")
    starts_cpu = torch.tensor([0, tokens], dtype=torch.int32)
    starts = starts_cpu.to("mcpu")
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        num_decode_tokens=0,
        query_start_loc=starts,
        max_query_len=tokens,
        max_seq_len=tokens,
        seq_lens=torch.tensor([tokens], dtype=torch.int32, device="mcpu"),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device="mcpu"),
    )
    if cache_dtype == "fp8_ds_mla":
        metadata.xcpu_backend = torch_xcpu.ops.SparseMlaFp8(8)
        metadata.xcpu_schedule = metadata.xcpu_backend.build_metadata(
            starts,
            tokens,
            query_start_loc_cpu=starts_cpu,
            has_prefill=True,
            kv_slots_hint=tokens,
        )
    context = SimpleNamespace(
        attn_metadata={layer.layer_name: metadata},
        slot_mapping={layer.layer_name: positions},
    )
    monkeypatch.setattr(plugin, "get_forward_context", lambda: context)
    monkeypatch.setattr(upstream, "get_forward_context", lambda: context)

    class Rope:
        cos_sin_cache = cache
        is_neox_style = False

        def __call__(self, pos, q_pe, k_pe):
            torch_xcpu.ops.rotary_embedding(pos, q_pe, k_pe, 64, cache, False)
            return q_pe, k_pe

    with torch.inference_mode():
        layer._fuse_rope_cache = False
        reference = layer.forward_with_rope(
            q.clone(), kv, k.clone(), positions, Rope(), None
        ).cpu()
        expected_cache = layer.kv_cache.cpu().clone()
        layer.kv_cache.zero_()
        layer._fuse_rope_cache = True
        actual = layer.forward_with_rope(
            q.clone(), kv, k.clone(), positions, Rope(), None
        ).cpu()
        actual_cache = layer.kv_cache.cpu()
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_fp8_cache_format_is_scoped_to_xcpu_backend():
    from vllm.v1.attention.backend import AttentionBackend

    assert XcpuMLASparseBackend.canonicalize_kv_cache_dtype("fp8") == "fp8_ds_mla"
    assert AttentionBackend.canonicalize_kv_cache_dtype("fp8") == "fp8"
    assert XcpuMLASparseBackend.canonicalize_kv_cache_dtype("auto") == "auto"
