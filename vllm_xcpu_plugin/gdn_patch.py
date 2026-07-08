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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

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
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata: object | None = None,
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
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        num_computed_tokens=num_computed_tokens,
        block_size_to_align=block_size_to_align,
        metadata=metadata,
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
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    **_: object,
) -> torch.Tensor:
    import torch_xcpu

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
        pad_slot_id=pad_slot_id,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
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
    **kwargs: object,
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
        **kwargs,
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


def _disable_gdn_warmup(self, mixed_qkv: torch.Tensor) -> None:
    del self, mixed_qkv
    return None


def maybe_patch_gdn_attention() -> None:
    global _GDN_ATTENTION_PATCHED

    from vllm.model_executor.layers.mamba import gdn_linear_attn as gdn

    if _GDN_ATTENTION_PATCHED:
        return

    _ = (
        gdn.fused_gdn_gating,
        gdn.fused_sigmoid_gating_delta_rule_update,
        gdn.fused_recurrent_gated_delta_rule_packed_decode,
        gdn.causal_conv1d_fn,
        gdn.causal_conv1d_update,
        gdn.GatedDeltaNetAttention._warmup_prefill_kernels,
    )

    gdn_any = cast(Any, gdn)
    gdn_attention_cls = cast(Any, gdn.GatedDeltaNetAttention)
    gdn_any.fused_gdn_gating = _xcpu_fused_gdn_gating
    gdn_any.fused_sigmoid_gating_delta_rule_update = (
        _xcpu_fused_sigmoid_gating_delta_rule_update
    )
    gdn_any.fused_recurrent_gated_delta_rule_packed_decode = (
        _xcpu_fused_recurrent_gated_delta_rule_packed_decode
    )
    gdn_any.causal_conv1d_fn = _xcpu_causal_conv1d_fn
    gdn_any.causal_conv1d_update = _xcpu_causal_conv1d_update
    gdn_attention_cls._warmup_prefill_kernels = _disable_gdn_warmup

    _GDN_ATTENTION_PATCHED = True
    logger.info("Patched GDN attention with xcpu custom ops")
