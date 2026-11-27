# SPDX-License-Identifier: Apache-2.0
"""Adapters between the CUDA SM100 call signatures and ``torch_xcpu.ops``.

Only entries whose signature differs from the ``torch_xcpu.ops`` operator
live here; the rest are called as ``torch_xcpu.ops.deepseek_v41_*`` at the call
site (the operators' torch references and tests live in torch_xcpu,
``test/ops/custom_ops/deepseek_v41``).

CUDA path                                         XCPU entry
fused wo_a / fp8_einsum                           grouped_wo_a
mhc_pre_delayed_tilelang                          mhc_pre_delayed
mhc_post_tilelang                                 mhc_post
mhc_shifted_post_pre                              mhc_shifted_post_pre
hc_collapse_triton                                hc_collapse
fused_save_compress_norm                          fused_save_compress_norm
rope_quant_insert                                 rope_quant_insert
_custom_ops.topk_hash_softplus_sqrt               topk_hash_softplus_sqrt

``torch_xcpu`` is imported inside function bodies only: importing it
initializes the device on real platforms, and the EngineCore process imports
this plugin too.
"""

import torch


def mhc_pre_delayed(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lazy wrapper around ``torch_xcpu.ops.deepseek_v41_mhc_pre_delayed``."""
    import torch_xcpu

    return torch_xcpu.ops.deepseek_v41_mhc_pre_delayed(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        x=x,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    """Lazy wrapper around ``torch_xcpu.ops.deepseek_v4_mhc_post``."""
    import torch_xcpu

    return torch_xcpu.ops.deepseek_v4_mhc_post(x, residual, post_mix, comb_mix)


def hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Lazy wrapper around ``torch_xcpu.ops.deepseek_v41_hc_collapse``."""
    import torch_xcpu

    return torch_xcpu.ops.deepseek_v41_hc_collapse(x, pre_mix)


def grouped_wo_a(
    o: torch.Tensor, packed_weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """W8A16 grouped wo_a ``tgd,grd->tgr`` over the [T, H, 512] attention
    output (heads folded into G groups of D), flattened to [T, G * R]."""
    import torch_xcpu

    groups, rank = packed_weight.shape[:2]
    x = o.reshape(o.shape[0], groups, -1)
    out = o.new_empty((o.shape[0], groups, rank))
    if packed_weight.dtype == torch.int8:
        torch_xcpu.ops.einsum_int8_tgd_grd_to_tgr(x, packed_weight, weight_scale, out)
    else:
        torch_xcpu.ops.einsum_fp8_tgd_grd_to_tgr(x, packed_weight, weight_scale, out)
    return out.flatten(1)


def mhc_shifted_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
    *,
    stream=None,
    reduce_results: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Mega-mHC seam: post of the previous sublayer, then this one's pre with
    the carried pre-mix and the RMSNorm folded in."""
    import torch_xcpu

    assert not reduce_results, "the fused all-reduce + mHC is CUDA only"
    return torch_xcpu.ops.deepseek_v41_mhc_shifted_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        capture_aux=capture_aux,
    )


def fused_save_compress_norm(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    latent_out: torch.Tensor,
) -> None:
    """The compressor's state save + pool + RMSNorm, into ``latent_out``."""
    import torch_xcpu

    torch_xcpu.ops.deepseek_v41_compress_save_norm(
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        rms_norm_weight,
        rms_norm_eps,
        compress_ratio,
        out=latent_out,
    )


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """RoPE + record insert into the compressed cache. XCPU caches are the
    uint8 NVFP4/MXFP8 records, which carry their own scales."""
    import torch_xcpu

    assert fp8_scale is None, "per-tensor FP8 compressed caches are FlashInfer only"
    torch_xcpu.ops.deepseek_v41_compress_rope_insert(
        latent, positions, cos_sin_cache, kv_cache, slot_mapping, compress_ratio
    )


def topk_hash_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> None:
    """``_custom_ops.topk_hash_softplus_sqrt`` for the score-routed V4.1 MoE:
    image rows pick with ``bias_vl``, padding rows get expert -1, all in one
    launch. ``token_expert_indices`` is not written; no XCPU MoE reads it."""
    import torch_xcpu

    if hash_indices_table is not None:
        raise NotImplementedError("XCPU sqrtsoftplus routing has no hash layers")
    if e_score_correction_bias is None:
        raise NotImplementedError("XCPU sqrtsoftplus routing needs the gate bias")
    image_routing = bias_vl is not None and image_sentinel_lo > 0
    if image_routing:
        assert input_tokens is not None, "bias_vl routing requires input_tokens"
    torch_xcpu.ops.deepseek_v41_topk_softplus_sqrt(
        gating_output,
        topk_weights.shape[1],
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
        input_ids=(
            input_tokens.contiguous()
            if image_routing and input_tokens is not None
            else None
        ),
        bias_vl=bias_vl if image_routing else None,
        image_sentinel_lo=image_sentinel_lo if image_routing else 0,
        is_padding=is_padding,
        out=(topk_weights, topk_indices),
    )
