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


def _xcpu_sparse_mla_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    logical_topk: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    value_dim: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Dispatch experimental paged sparse MLA prefill and decode kernels."""
    query_starts = query_start_loc.cpu().tolist()
    query_lens = [end - start for start, end in zip(query_starts, query_starts[1:])]
    num_decodes = 0
    while num_decodes < len(query_lens) and query_lens[num_decodes] == 1:
        num_decodes += 1
    if any(query_len == 1 for query_len in query_lens[num_decodes:]):
        raise NotImplementedError(
            "XCPU experimental sparse MLA requires decode requests before prefills"
        )

    assert q.shape[0] == query_starts[-1]
    assert output.shape == (q.shape[0], q.shape[1], value_dim)
    logical_topk = logical_topk[: q.shape[0]]

    import torch_xcpu

    if num_decodes:
        torch_xcpu.ops.sparse_mla_decode(
            q[:num_decodes].contiguous(),
            kv_cache,
            logical_topk[:num_decodes].contiguous(),
            block_table[:num_decodes].contiguous(),
            seq_lens[:num_decodes].contiguous(),
            scale,
            output[:num_decodes].contiguous(),
        )

    num_decode_tokens = query_starts[num_decodes]
    if num_decode_tokens < q.shape[0]:
        request_ids = [
            request_idx
            for request_idx, query_len in enumerate(query_lens)
            for _ in range(query_len)
        ]
        req_id_per_token = torch.tensor(
            request_ids[num_decode_tokens:],
            dtype=torch.int32,
            device=q.device,
        )
        torch_xcpu.ops.sparse_mla_prefill(
            q[num_decode_tokens:].contiguous(),
            kv_cache,
            logical_topk[num_decode_tokens:].contiguous(),
            block_table.contiguous(),
            req_id_per_token,
            query_start_loc.contiguous(),
            seq_lens.contiguous(),
            num_decode_tokens,
            scale,
            output[num_decode_tokens:].contiguous(),
        )
    return output


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
    if self.is_sparse and attn_metadata.max_seq_len > attn_metadata.topk_tokens:
        assert self.topk_indices_buffer is not None
        use_sparse_unified_attention = True
        if use_sparse_unified_attention:
            logical_topk = self.topk_indices_buffer[: q.shape[0]].contiguous()
            import torch_xcpu

            torch_xcpu.ops.unified_attention(
                q=q,  # [tokens, q_num_heads, kv_lora_rank + qk_rope]
                kv=kv_c_and_k_pe_cache,
                out=output,  # [tokens, q_num_heads, kv_lora_rank]
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                softmax_scale=self.scale,
                window_size=0,
                block_table=attn_metadata.block_table,
                logical_topk=logical_topk,
                kv_cache_dtype=self.kv_cache_dtype,
            )
        else:
            _xcpu_sparse_mla_attention(
                q,
                kv_c_and_k_pe_cache,
                self.topk_indices_buffer,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                self.scale,
                self.kv_lora_rank,
                output,
            )
    else:
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        import torch_xcpu  # noqa: E402

        torch_xcpu.ops.unified_attention(
            q=q,  # [tokens, q_num_heads, kv_lora_rank + qk_rope]
            kv=kv_c_and_k_pe_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope]
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


def maybe_patch_vllm_flashattn_mla_sparse() -> None:
    from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
        FlashAttnMLASparseImpl,
    )

    sparse_any = cast(Any, FlashAttnMLASparseImpl)
    if getattr(sparse_any, "_xcpu_flashattn_mla_sparse_patched", False):
        return
    sparse_any.do_kv_cache_update = _xcpu_do_kv_cache_update
    sparse_any.forward_mqa = _xcpu_forward_mqa
    sparse_any._xcpu_flashattn_mla_sparse_patched = True
