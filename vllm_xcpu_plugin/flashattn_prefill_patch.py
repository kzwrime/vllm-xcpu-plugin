# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import torch
from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend


def _xcpu_run_prefill_new_tokens(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    return_softmax_lse: bool,
    out: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    # q: [tokens, num_heads, k_nope + k_pe], = 192 + 64 = 256
    # k: [tokens, num_heads, k_nope + k_pe],
    # v: [tokens, num_heads, v_head_dim], = 256

    import torch_xcpu  # noqa: E402
    output = torch_xcpu.ops.attn_varlen_diff_headdims(
        query=q,
        key=k,
        value=v,
        cu_seqlens_q=self._prefill_metadata.query_start_loc,
        cu_seqlens_k=self._prefill_metadata.query_start_loc,
        is_causal=True,
        scale=self.scale,
        enable_gqa=False,
    )

    # Note: scaled_dot_product_attention_varlen is also usable
    
    # output = torch_xcpu.ops.scaled_dot_product_attention_varlen(
    #     query=q,
    #     key=k,
    #     value=v,
    #     cu_seqlens_q=self._prefill_metadata.query_start_loc,
    #     cu_seqlens_k=self._prefill_metadata.query_start_loc,
    #     is_causal=True,
    #     scale=self.scale,
    #     enable_gqa=False,
    # )

    return output


def _xcpu_is_available() -> bool:
    return True


def maybe_patch_vllm_flashattn_prefill() -> None:
    prefill_any = cast(Any, FlashAttnPrefillBackend)
    if getattr(prefill_any, "_xcpu_flash_attn_prefill_patched", False):
        return
    prefill_any.is_available = _xcpu_is_available
    prefill_any.run_prefill_new_tokens = _xcpu_run_prefill_new_tokens
    prefill_any._xcpu_flashattn_mla_sparse_patched = True
