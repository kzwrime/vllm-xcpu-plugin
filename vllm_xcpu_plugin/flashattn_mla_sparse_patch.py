# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import torch
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseMetadata,
)


def _xcpu_do_kv_cache_update(
    self,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> None:
    if kv_cache.numel() == 0:
        return

    import torch_xcpu

    torch_xcpu.ops.reshape_and_cache(
        kv_c_normed,  # [tokens, kv_lora_rank]
        k_pe.squeeze(1),  # [tokens, qk_rope]
        kv_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope]
        slot_mapping.flatten(),
        kv_cache_dtype=kv_cache_dtype,
    )


def _xcpu_forward_mqa(
    self,
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: FlashAttnMLASparseMetadata,
    layer: AttentionLayer,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(q, tuple):
        raise NotImplementedError(
            "FlashAttnMLASparseImpl expects split (q_nope, q_rope) input."
        )

    q_nope, q_rope = q
    q = torch.cat((q_nope, q_rope), dim=-1)
    output = torch.empty_like(q_nope)

    logical_topk = None
    if self.is_sparse:
        assert self.topk_indices_buffer is not None, (
            "XCPU sparse MLA requires the shared topk indices buffer."
        )
        logical_topk = self.topk_indices_buffer

    # MQA covers the leading decode rows (decode-first reorder). On a mixed
    # batch the varlen metadata must be narrowed to the decode requests:
    # with the full-batch cu_seqlens_q the kernel would walk the prefill
    # requests too and read q past its end.
    if attn_metadata.num_decodes == attn_metadata.num_reqs:
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        max_seqlen_q = attn_metadata.max_query_len
    else:
        n = q.size(0)
        cu_seqlens_q = attn_metadata.query_start_loc[: n + 1]
        seqused_k = attn_metadata.seq_lens[:n]
        block_table = attn_metadata.block_table[:n]
        max_seqlen_q = max(1, int(torch.diff(cu_seqlens_q).max().item()))

    import torch_xcpu

    torch_xcpu.ops.unified_attention(
        q=q,  # [tokens, q_num_heads, kv_lora_rank + qk_rope]
        kv=kv_c_and_k_pe_cache,
        out=output,  # [tokens, q_num_heads, kv_lora_rank]
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        softmax_scale=self.scale,
        window_size=0,
        block_table=block_table,
        logical_topk=logical_topk,
        kv_cache_dtype=self.kv_cache_dtype,
    )

    return output, None


def _xcpu_make_sparse_forward_mha(orig_forward_mha):
    def _xcpu_sparse_forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        # Bridge the shared top-k buffer into the prefill backend call.
        # Rows are ordered decode-first, so the prefill rows are
        # [num_decode_tokens, num_decode_tokens + q tokens); entries are
        # request-relative logical KV positions. The downstream
        # attn_varlen_diff_headdims op demotes to dense per request when the
        # packed KV prefix does not exceed topk_tokens.
        backend = None
        if self.topk_indices_buffer is not None and attn_metadata.prefill is not None:
            backend = getattr(attn_metadata.prefill, "prefill_backend", None)
            if backend is not None:
                num_decode_tokens = attn_metadata.num_decode_tokens
                backend._xcpu_prefill_logical_topk = self.topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + q.shape[0]
                ]
        try:
            return orig_forward_mha(
                self,
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
        finally:
            if backend is not None:
                backend._xcpu_prefill_logical_topk = None

    return _xcpu_sparse_forward_mha


def maybe_patch_vllm_flashattn_mla_sparse() -> None:
    from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
        FlashAttnMLASparseImpl,
    )

    sparse_any = cast(Any, FlashAttnMLASparseImpl)
    if getattr(sparse_any, "_xcpu_flashattn_mla_sparse_patched", False):
        return
    sparse_any.do_kv_cache_update = _xcpu_do_kv_cache_update
    sparse_any.forward_mqa = _xcpu_forward_mqa
    # Note: register this when prefill backend needs to run with topk_indices_buffer
    # sparse_any.forward_mha = _xcpu_make_sparse_forward_mha(sparse_any.forward_mha)
    sparse_any._xcpu_flashattn_mla_sparse_patched = True
