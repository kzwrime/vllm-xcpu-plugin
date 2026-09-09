# SPDX-License-Identifier: Apache-2.0

"""Install XCPU kernels into vLLM's original sparse-indexer operator chain.

This module intentionally defines no ``SparseAttnIndexer`` subclass and does
not register an out-of-tree replacement layer.  vLLM keeps ownership of cache
insertion, prefill chunking, workspace allocation, metadata, logits lifetime,
TopK invocation and output merging.  We patch only the exact kernel functions
that ``vllm/model_executor/layers/sparse_attn_indexer.py`` already calls.
"""

from __future__ import annotations

import sys

import torch


def _require_supported_q(q: torch.Tensor, *, topk: int | None = None) -> None:
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise NotImplementedError(
            "XCPU sparse indexer supports unquantized BF16 Q with head_dim=128 only"
        )
    heads = q.shape[-2]
    if heads not in (32, 64) or (topk is not None and topk not in (512, 1024, 2048)):
        raise NotImplementedError(
            "XCPU sparse indexer supports H={32,64}, D=128 and "
            "K={512,1024,2048}; got "
            f"H={heads}, D={q.shape[-1]}, K={topk}"
        )


def _indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    import torch_xcpu

    torch_xcpu.ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    )


def _cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    import torch_xcpu

    torch_xcpu.ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )


def _fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    import torch_xcpu

    q_values, q_scale = q
    k_values, k_scale = kv
    if q_scale is not None:
        raise NotImplementedError("XCPU sparse indexer does not support FP4 Q/cache")
    _require_supported_q(q_values)
    return torch_xcpu.ops.fp8_fp4_mqa_logits(
        q_values,
        k_values,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits,
    )


def _top_k_per_row_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    import torch_xcpu

    torch_xcpu.ops.top_k_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    )


def _top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    del next_n
    import torch_xcpu

    torch_xcpu.ops.top_k_per_row_decode(
        logits,
        seq_lens,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    )


def _pack_seq_triton(x, lengths, pad_value=0, block_t=64, block_d=64):
    if pad_value != 0:
        raise NotImplementedError("XCPU indexer sequence packing requires zero padding")
    output = torch.empty(
        (lengths.numel(), int(lengths.max().item()), *x.shape[1:]),
        dtype=x.dtype,
        device=x.device,
    )
    torch.ops.torch_xcpu.sparse_indexer_pack_seq(x, lengths, output, False)
    return output


def _unpack_seq_triton(packed_tensor, lengths, block_t=64, block_d=64):
    output = torch.empty(
        (int(lengths.sum().item()), *packed_tensor.shape[2:]),
        dtype=packed_tensor.dtype,
        device=packed_tensor.device,
    )
    torch.ops.torch_xcpu.sparse_indexer_pack_seq(
        packed_tensor, lengths, output, True
    )
    return output


def _get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int, max_model_len: int
) -> torch.Tensor:
    import torch_xcpu

    return torch_xcpu.ops.get_paged_mqa_logits_metadata(
        context_lens, block_size, num_sms, max_model_len
    )


def _fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    import torch_xcpu

    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("XCPU sparse indexer does not support FP4 Q/cache")
    _require_supported_q(q_values)
    return torch_xcpu.ops.fp8_fp4_paged_mqa_logits(
        q_values,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits,
    )


def _fused_indexer_q_rope_quant_glm(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_xcpu

    return torch_xcpu.ops.fused_indexer_q_rope_quant(
        positions,
        q,
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        is_neox,
        False,
    )


def _fused_indexer_q_rope_quant_deepseek_v4(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    use_fp4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_fp4:
        raise NotImplementedError("XCPU sparse indexer does not support FP4 Q/cache")
    import torch_xcpu

    return torch_xcpu.ops.fused_indexer_q_rope_quant(
        positions,
        index_q,
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        False,
        True,
    )


def maybe_patch_vllm_sparse_attn_indexer() -> None:
    """Patch only the named kernel functions in vLLM's original chain."""

    import vllm._custom_ops as ops
    import vllm.model_executor.layers.sparse_attn_indexer as indexer_module
    import vllm.utils.deep_gemm as deep_gemm

    if getattr(indexer_module, "_xcpu_sparse_kernel_patch_installed", False):
        return

    # Exact vLLM operator names used by sparse_attn_indexer().
    ops.indexer_k_quant_and_cache = _indexer_k_quant_and_cache
    ops.cp_gather_indexer_k_quant_cache = _cp_gather_indexer_k_quant_cache
    ops.top_k_per_row_prefill = _top_k_per_row_prefill
    ops.top_k_per_row_decode = _top_k_per_row_decode

    # Exact DeepGEMM function names imported by sparse_attn_indexer.py.
    deep_gemm.fp8_fp4_mqa_logits = _fp8_fp4_mqa_logits
    deep_gemm.fp8_fp4_paged_mqa_logits = _fp8_fp4_paged_mqa_logits
    indexer_module.fp8_fp4_mqa_logits = _fp8_fp4_mqa_logits
    indexer_module.fp8_fp4_paged_mqa_logits = _fp8_fp4_paged_mqa_logits
    indexer_module.pack_seq_triton = _pack_seq_triton
    indexer_module.unpack_seq_triton = _unpack_seq_triton

    # The two model families expose the same Python function name but differ
    # in where RoPE lives. Keep that ABI distinction in tiny adapters while a
    # single XCPU operator owns BF16 RoPE and fixed head scaling (no Q scale).
    # The patched `quant`/`fp8_fp4` symbols retain upstream names only; tensors
    # carry BF16 Q all the way to the XCPU logits kernels.
    indexer_module.fused_indexer_q_rope_quant = _fused_indexer_q_rope_quant_glm
    glm_module = sys.modules.get("vllm.model_executor.models.deepseek_v2")
    if glm_module is not None:
        glm_module.__dict__["fused_indexer_q_rope_quant"] = (
            _fused_indexer_q_rope_quant_glm
        )
    try:
        import vllm.models.deepseek_v4.common.ops as deepseek_v4_ops

        deepseek_v4_ops.fused_indexer_q_rope_quant = (
            _fused_indexer_q_rope_quant_deepseek_v4
        )
    except ImportError:
        pass
    deepseek_v4_attention = sys.modules.get("vllm.models.deepseek_v4.attention")
    if deepseek_v4_attention is not None:
        deepseek_v4_attention.__dict__["fused_indexer_q_rope_quant"] = (
            _fused_indexer_q_rope_quant_deepseek_v4
        )

    indexer_module._xcpu_sparse_kernel_patch_installed = True
