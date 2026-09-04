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
