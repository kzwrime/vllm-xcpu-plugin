# SPDX-License-Identifier: Apache-2.0
"""Unit tests for XCPU Dense MLA backend, unfused fallback, and wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch_mcpu  # noqa: F401
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backend import CommonAttentionMetadata

import vllm_xcpu_plugin.attention.mla  # noqa: F401
from vllm_xcpu_plugin.attention.dense_mla import (
    XcpuMLADenseBackend,
    XcpuMLADenseMetadataBuilder,
    _validate_dense_mla_config,
)


class _TupleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str) -> None:
        super().__init__()
        w_cpu = (
            torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cpu")
            * 0.02
        )
        self.weight = nn.Parameter(w_cpu.to(device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(x, self.weight), None


def _make_mock_config(
    *,
    dcp: int = 1,
    pcp: int = 1,
    ubatches: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    spec_method: str | None = None,
) -> Any:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            num_ubatches=ubatches,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=(
            None if spec_method is None else SimpleNamespace(method=spec_method)
        ),
        model_config=SimpleNamespace(dtype=dtype),
    )


def test_dense_mla_config_validation_rejects_invalid_modes() -> None:
    _validate_dense_mla_config(_make_mock_config())
    with pytest.raises(NotImplementedError, match="DCP=1"):
        _validate_dense_mla_config(_make_mock_config(dcp=2))
    with pytest.raises(NotImplementedError, match="PCP=1"):
        _validate_dense_mla_config(_make_mock_config(pcp=2))
    with pytest.raises(NotImplementedError, match="ubatching"):
        _validate_dense_mla_config(_make_mock_config(ubatches=2))
    with pytest.raises(NotImplementedError, match="BF16"):
        _validate_dense_mla_config(_make_mock_config(dtype=torch.float16))
    with pytest.raises(NotImplementedError, match="DFlash"):
        _validate_dense_mla_config(_make_mock_config(spec_method="mtp"))


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("q_lora_rank", [1536, None])
@torch.inference_mode()
def test_dense_mla_fused_vs_unfused_and_q_lora_modes(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_dtype: str,
    q_lora_rank: int | None,
) -> None:
    torch.manual_seed(0)
    device = "mcpu"
    hidden_size = 256
    num_heads = 8
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 128
    kv_lora_rank = 512
    block_size = 64
    num_blocks = 4

    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        cache_config=CacheConfig(block_size=block_size, cache_dtype=kv_cache_dtype),
    )
    vllm_config.model_config = SimpleNamespace(  # type: ignore[assignment]
        dtype=torch.bfloat16,
        rswa_window=None,
    )

    def _build_wrapper(prefix: str, fused_mla: bool) -> MultiHeadLatentAttentionWrapper:
        monkeypatch.setenv("VLLM_XCPU_FUSED_MLA", "1" if fused_mla else "0")
        rope = get_rope(
            qk_rope_head_dim,
            max_position=4096,
            is_neox_style=False,
            rope_parameters={
                "rope_type": "deepseek_yarn",
                "rope_theta": 50000.0,
                "factor": 32.0,
                "beta_fast": 1.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
            },
        ).to(device=device, dtype=torch.bfloat16)

        if q_lora_rank is not None:
            fused_qkv_a = _TupleLinear(
                hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim, device
            )
            q_a_ln = RMSNorm(q_lora_rank, eps=1e-6).to(
                device=device, dtype=torch.bfloat16
            )
            q_b = _TupleLinear(q_lora_rank, num_heads * qk_head_dim, device)
            kv_a = None
            q_proj = None
        else:
            fused_qkv_a = None
            q_a_ln = None
            q_b = None
            kv_a = _TupleLinear(hidden_size, kv_lora_rank + qk_rope_head_dim, device)
            q_proj = _TupleLinear(hidden_size, num_heads * qk_head_dim, device)

        kv_a_ln = RMSNorm(kv_lora_rank, eps=1e-6).to(
            device=device, dtype=torch.bfloat16
        )
        kv_b = _TupleLinear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), device
        )
        o_proj = _TupleLinear(num_heads * v_head_dim, hidden_size, device)

        modules = MLAModules(
            kv_a_layernorm=kv_a_ln,
            kv_b_proj=kv_b,
            rotary_emb=rope,
            o_proj=o_proj,
            fused_qkv_a_proj=fused_qkv_a,
            kv_a_proj_with_mqa=kv_a,
            q_a_layernorm=q_a_ln,
            q_b_proj=q_b,
            q_proj=q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        wrapper = MultiHeadLatentAttentionWrapper(
            hidden_size=hidden_size,
            num_heads=num_heads,
            scale=qk_head_dim**-0.5,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=modules,
            cache_config=CacheConfig(block_size=block_size, cache_dtype=kv_cache_dtype),
            prefix=prefix,
        )
        wrapper.mla_attn.process_weights_after_loading(torch.bfloat16)
        return wrapper

    old_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            torch.manual_seed(42)
            w_fused = _build_wrapper("layer_fused", fused_mla=True)
            torch.manual_seed(42)
            w_unfused = _build_wrapper("layer_unfused", fused_mla=False)

            assert w_fused.mla_attn.attn_backend is XcpuMLADenseBackend

            def _alloc_cache() -> torch.Tensor:
                if kv_cache_dtype == "fp8":
                    return torch.zeros(
                        (num_blocks, 1, block_size, 656),
                        dtype=torch.uint8,
                        device=device,
                    )
                return torch.zeros(
                    (num_blocks, 1, block_size, 576),
                    dtype=torch.bfloat16,
                    device=device,
                )

            w_fused.mla_attn.bind_kv_cache(_alloc_cache())
            w_unfused.mla_attn.bind_kv_cache(_alloc_cache())

            # Mixed batch: 1 decode req (1 token) + 1 prefill req (4 tokens)
            builder = XcpuMLADenseMetadataBuilder(
                w_fused.mla_attn.get_kv_cache_spec(vllm_config),
                ["layer_fused.attn", "layer_unfused.attn"],
                vllm_config,
                torch.device(device),
            )

            qsl_cpu = torch.tensor([0, 1, 5], dtype=torch.int32)
            common = CommonAttentionMetadata(
                query_start_loc=qsl_cpu.to(device),
                query_start_loc_cpu=qsl_cpu,
                seq_lens=torch.tensor([1, 4], dtype=torch.int32, device=device),
                num_reqs=2,
                num_actual_tokens=5,
                max_query_len=4,
                max_seq_len=4,
                block_table_tensor=torch.tensor(
                    [[0, 1], [2, 3]], dtype=torch.int32, device=device
                ),
                slot_mapping=torch.tensor(
                    [0, 128, 129, 130, 131], dtype=torch.int64, device=device
                ),
            )
            meta = builder.build(0, common)
            assert meta.num_decode_tokens == 1

            hidden = torch.randn((5, hidden_size), dtype=torch.bfloat16, device=device)
            positions = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int64, device=device)
            attn_meta_map = {"layer_fused.attn": meta, "layer_unfused.attn": meta}
            slot_map = {
                "layer_fused.attn": common.slot_mapping,
                "layer_unfused.attn": common.slot_mapping,
            }
            with set_forward_context(attn_meta_map, vllm_config, slot_mapping=slot_map):
                out_fused = w_fused(positions, hidden)
                out_unfused = w_unfused(positions, hidden)

            torch.mcpu.synchronize()
            torch.testing.assert_close(
                out_fused.cpu(), out_unfused.cpu(), atol=2e-2, rtol=2e-2
            )
    finally:
        torch.set_default_dtype(old_default)
