# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 GatedDeltaNet attention patches for xcpu.

The lightweight Python shims keep the vLLM GDN interfaces intact while routing
supported fused ops to torch_xcpu custom operators.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

logger = init_logger(__name__)

_GDN_ATTENTION_PATCHED = False


def _xcpu_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_xcpu

    return torch_xcpu.ops.fused_gdn_gating(A_log, a, b, dt_bias, beta, threshold)


def _xcpu_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
    **_: object,
) -> torch.Tensor:
    import torch_xcpu

    return torch_xcpu.ops.causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=pad_slot_id,
        null_block_id=null_block_id,
    )


def _xcpu_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int = NULL_BLOCK_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
    **kwargs: object,
) -> torch.Tensor:
    import torch_xcpu

    if kwargs:
        raise TypeError(f"unsupported causal_conv1d_update kwargs: {sorted(kwargs)}")
    return torch_xcpu.ops.causal_conv1d_update(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
        null_block_id=null_block_id,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        validate_data=validate_data,
    )


def _xcpu_fused_post_conv_prep(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> torch.Tensor:
    import torch_xcpu

    return torch_xcpu.ops.fused_post_conv_prep(
        conv_output=conv_output,
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        apply_l2norm=apply_l2norm,
        output_g_exp=output_g_exp,
    )


def _xcpu_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    import torch_xcpu

    return torch_xcpu.ops.fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        out=out,
    )


def _xcpu_fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_xcpu

    return torch_xcpu.ops.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


# TODO: may be we should enable warmup in vllm v0.23.0
def _disable_gdn_warmup(
    self,
    mixed_qkv: torch.Tensor,
    v_dim: int = 0,
) -> None:
    del self, mixed_qkv, v_dim
    return None


def maybe_patch_gdn_attention() -> None:
    global _GDN_ATTENTION_PATCHED

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    from vllm_xcpu_plugin.upstream_compatibility import (
        verify_upstream_compatibility,
    )

    if _GDN_ATTENTION_PATCHED:
        return
    verify_upstream_compatibility(("conv", "gdn"))

    _ = (
        gdn.fused_gdn_gating,
        gdn.fused_post_conv_prep,
        gdn.fused_sigmoid_gating_delta_rule_update,
        gdn.fused_recurrent_gated_delta_rule_packed_decode,
        gdn.causal_conv1d_fn,
        gdn.causal_conv1d_update,
    )

    gdn_any = cast(Any, gdn)
    gdn_any.fused_gdn_gating = _xcpu_fused_gdn_gating
    gdn_any.fused_post_conv_prep = _xcpu_fused_post_conv_prep
    gdn_any.fused_sigmoid_gating_delta_rule_update = (
        _xcpu_fused_sigmoid_gating_delta_rule_update
    )
    gdn_any.fused_recurrent_gated_delta_rule_packed_decode = (
        _xcpu_fused_recurrent_gated_delta_rule_packed_decode
    )
    gdn_any.causal_conv1d_fn = _xcpu_causal_conv1d_fn
    gdn_any.causal_conv1d_update = _xcpu_causal_conv1d_update

    for cls_name in ("QwenGatedDeltaNetAttention", "GatedDeltaNetAttention"):
        gdn_attention_cls = cast(Any, getattr(gdn, cls_name, None))
        if gdn_attention_cls is not None and hasattr(
            gdn_attention_cls, "_warmup_prefill_kernels"
        ):
            gdn_attention_cls._warmup_prefill_kernels = _disable_gdn_warmup

    _GDN_ATTENTION_PATCHED = True
    logger.info("Patched GDN attention with xcpu custom ops")
