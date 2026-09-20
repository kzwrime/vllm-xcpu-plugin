# SPDX-License-Identifier: Apache-2.0

"""Install XCPU kernels into vLLM's original sparse-indexer operator chain.

This module intentionally defines no ``SparseAttnIndexer`` subclass and does
not register an out-of-tree replacement layer. vLLM keeps ownership of
prefill chunking, workspace allocation, metadata, logits lifetime, TopK
invocation and output merging. This module binds the named XCPU kernels and
installs BF16 Q / fused K preprocessing for DeepSeek V3.2 and GLM DSA.
"""

from __future__ import annotations

import functools
import os
import sys

import torch
from vllm.forward_context import get_forward_context


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
    kv_cache_dtype: str,
) -> None:
    import torch_xcpu

    torch_xcpu.ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
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
    seq_lens_cpu: torch.Tensor | None = None,
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
        seq_lens_cpu,
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
    seq_lens_cpu: torch.Tensor | None = None,
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
        seq_lens_cpu,
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
    torch.ops.torch_xcpu.sparse_indexer_pack_seq(packed_tensor, lengths, output, True)
    return output


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
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_fp4:
        raise NotImplementedError("XCPU sparse indexer does not support FP4 Q/cache")
    import torch_xcpu

    return torch_xcpu.ops.fused_indexer_q_rope_quant(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        False,
        True,
    )


@torch.library.custom_op("vllm_xcpu::indexer_k_cache", mutates_args=("cache",))
def _indexer_k_cache(
    k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    cache: torch.Tensor,
    prefix: str,
    eps: float,
    is_neox: bool,
) -> None:
    # Resolve per-step metadata inside the opaque op, including after compile.
    metadata = get_forward_context().attn_metadata
    # Profiling has no slot mapping, so it must not write to the K cache.
    if metadata is None:
        return
    assert isinstance(metadata, dict)
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

    layer_metadata = metadata[prefix]
    assert isinstance(layer_metadata, DeepseekV32IndexerMetadata)
    import torch_xcpu

    torch_xcpu.ops.fused_indexer_k_norm_rope_cache(
        k,
        weight,
        bias,
        positions,
        cos_sin,
        layer_metadata.slot_mapping,
        cache,
        eps,
        is_neox,
    )


@_indexer_k_cache.register_fake
def _indexer_k_cache_fake(
    k, weight, bias, positions, cos_sin, cache, prefix, eps, is_neox
):
    return None


def _indexer_forward(self, hidden_states, qr, positions, rotary_emb):
    import torch_xcpu

    q = self.wq_b(qr)[0].view(-1, self.n_head, self.head_dim)
    kw = self.wk_weights_proj(hidden_states)[0]
    k, weights = kw[:, : self.head_dim], kw[:, self.head_dim :]
    q, weights = torch_xcpu.ops.fused_indexer_q_rope_quant(
        positions,
        q,
        rotary_emb.cos_sin_cache,
        weights,
        self.softmax_scale,
        self.n_head_scale,
        rotary_emb.is_neox_style,
        False,
    )
    if self._xcpu_fuse_k_cache:
        _indexer_k_cache(
            k,
            self.k_norm.weight,
            self.k_norm.bias,
            positions,
            rotary_emb.cos_sin_cache,
            self.k_cache.kv_cache,
            self.k_cache.prefix,
            self.k_norm.eps,
            rotary_emb.is_neox_style,
        )
    else:
        # Reference path for A/B validation; Q remains BF16 on both paths.
        k = self.k_norm(k)
        k_pe = k[:, : self.rope_dim].unsqueeze(1)
        _, k_pe = rotary_emb(positions, torch.empty_like(k_pe), k_pe)
        k = torch.cat((k_pe.reshape(-1, self.rope_dim), k[:, self.rope_dim :]), dim=-1)

    # Reuse upstream scheduling/logits/TopK with XCPU kernel bindings. Calling
    # the shared implementation explicitly avoids CustomOp's CUDA-only native
    # dispatch guard on OOT platforms.
    return self.indexer_op.forward_cuda(hidden_states, q, k, weights)


def _install_indexer_preprocessing() -> None:
    from vllm.model_executor.models.deepseek_v2 import Indexer

    if getattr(Indexer, "_xcpu_preprocessing_installed", False):
        return
    original_init = Indexer.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        parallel = self.vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise NotImplementedError(
                "XCPU DSA indexer preprocessing requires PCP=DCP=1"
            )
        if self.head_dim != 128 or self.rope_dim != 64 or self.n_head not in (32, 64):
            raise NotImplementedError(
                "XCPU DSA indexer requires H={32,64}, D=128, RoPE=64"
            )
        self._xcpu_fuse_k_cache = os.getenv("VLLM_XCPU_FUSED_INDEXER_K", "1") != "0"
        self.indexer_op.skip_k_cache_insert = self._xcpu_fuse_k_cache

    Indexer.__init__ = initialize  # type: ignore[method-assign]
    Indexer.forward = _indexer_forward  # type: ignore[method-assign]
    Indexer._xcpu_preprocessing_installed = True


def maybe_patch_vllm_sparse_attn_indexer() -> None:
    """Install sparse-indexer kernels and model preprocessing."""

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

    _install_indexer_preprocessing()
    indexer_module._xcpu_sparse_kernel_patch_installed = True  # type: ignore[attr-defined]
