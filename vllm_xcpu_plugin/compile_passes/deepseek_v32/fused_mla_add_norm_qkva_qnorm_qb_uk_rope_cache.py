# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2 pattern: the MLA front half plus its input add-norm and the
fused_qkv_a FP8 linear, all fused into one operator.

Extends the phase-1 window (:mod:`_front_window`) upwards:

    h, res = fused_add_rms_norm_out_bf16(prev_out, running_residual)
    qkv_a = fp8_mm(h, fused_qkv_a_proj packed weights)   # [T, 2048+512+64]
    <phase-1 window: q/kv RMSNorm, q_b_proj, RoPE, KV-cache write, W_UK>

Replaced by ``fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache`` returning
``(hidden, residual, q_c_norm, kv_c_norm, out_q)`` through its mutated-arg
clones (the raw op returns nothing): ``hidden`` holds the normalized token
rows the qkv_a linear used to consume, ``residual`` the updated running
residual the next layer's add-norm consumes.

The pattern requires the input norm to be ``fused_add_rms_norm_out_bf16``
(the L2..LN layers). The first layer normalizes the embedding with a plain
``rms_norm_bf16`` and no residual add, so it is skipped here and left to the
phase-1 pass. The post-attention ``fused_add_rms_norm_chunk_out_bf16``
variant never appears on this def-use path.

The linear must be ``fp8_scaled_mm_intel_amx_out`` with a bias-free
[2624, 6144] packed weight and [21, 48] scales (block 128); other shapes
stay unfused. ``ag_send``/``ag_comm`` stay null/zero for now — absorbing the
SP all-gather is the follow-up phase.
"""

import logging

import torch
from torch.fx import Node

from vllm_xcpu_plugin.compile_passes.deepseek_v32 import _front_window as _fw

logger = logging.getLogger(__name__)

_AF = _fw._AF
_ATEN = _fw._ATEN
_GETITEM = _fw._GETITEM
_EMPTY = _fw._EMPTY
_SKIP = _fw._Skip

_FP8_MM = _fw._FP8_MM
_CACHE_WRITE = _fw._CACHE_WRITE
_ADD_NORM = torch.ops.torch_xcpu.fused_add_rms_norm_out_bf16.default
_FUSED = torch.ops.torch_xcpu.fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache.default

_HIDDEN = 6144
_W_QKVA_SHAPE = (_fw._QKV_WIDTH, _HIDDEN)
_W_QKVA_SCALE_SHAPE = (
    -(-_fw._QKV_WIDTH // 128),  # ceil: 2624 -> 21 scale rows
    -(-_HIDDEN // 128),  # 6144 -> 48 scale cols
)

_IDX = _fw._af_indices(_FUSED)
_IDX_ADD = _fw._af_indices(_ADD_NORM)
_IDX_MM_OUT = _fw._af_indices(_FP8_MM)["out"]
_IDX_CACHE_KV = _fw._af_indices(_CACHE_WRITE)["kv_cache"]


def _backend_supported() -> bool:
    return _fw.backend_supported(
        torch.ops.torch_xcpu.fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache_backend_supported
    )


def _replace_one(graph, cache_af: Node, fp8_backend: int) -> bool:
    m = _fw.match_window(graph, cache_af)

    # Upstream: the fused_qkv_a FP8 linear the two rms_norms read.
    qkva_af = m.qkv_base.args[0]
    if not _fw._is_af_of(qkva_af, _FP8_MM):
        raise _SKIP("qkv source is not fp8_scaled_mm_intel_amx")
    if qkva_af.kwargs.get("bias") is not None:
        raise _SKIP("fused_qkv_a linear has a bias")
    _fw._check_shape(
        "w_qkva", qkva_af.kwargs["packed_weight"], _W_QKVA_SHAPE
    )
    _fw._check_shape(
        "w_qkva_scale", qkva_af.kwargs["weight_scale"], _W_QKVA_SCALE_SHAPE
    )

    # And its input: the layer's add-norm. The first layer uses a plain
    # rms_norm (no residual add) and stays with the phase-1 pass.
    hidden_gi = _fw._view_ancestor(qkva_af.kwargs["input"])
    if not (
        _fw._is_getitem(hidden_gi) and hidden_gi.args[1] == _IDX_ADD["output"]
    ):
        raise _SKIP("fused_qkv_a input is not an add-norm output clone")
    add_norm_af = hidden_gi.args[0]
    if not _fw._is_af_of(add_norm_af, _ADD_NORM):
        raise _SKIP("input norm is not fused_add_rms_norm_out (first layer?)")
    _fw._check_shape("add-norm hidden", hidden_gi, (None, _HIDDEN))
    _fw._check_shape(
        "add-norm weight", add_norm_af.kwargs["weight"], (_HIDDEN,)
    )

    # Redirect producers: region results consumed outside the region.
    residual_gi = next(
        (
            u
            for u in add_norm_af.users
            if _fw._is_getitem(u) and u.args[1] == _IDX_ADD["residual_out"]
        ),
        None,
    )
    cache_out = next(
        (
            u
            for u in cache_af.users
            if _fw._is_getitem(u) and u.args[1] == _IDX_CACHE_KV
        ),
        None,
    )
    if cache_out is None:
        raise _SKIP("KV-cache result missing")
    redirects: dict[Node, Node] = {
        hidden_gi: None,
        m.q_norm_out: None,
        cache_out: None,
        m.cat: None,
    }
    if residual_gi is not None:
        redirects[residual_gi] = None
    region = _fw.close_region(
        graph,
        [
            add_norm_af,
            qkva_af,
            m.q_norm_af,
            m.kv_norm_af,
            m.q_b_af,
            m.rotary_af,
            m.einsum_af,
            cache_af,
            m.cat,
        ],
    )
    _fw.ensure_no_escape(region, redirects)

    # --- mutation below this line; no more _Skip ---
    tokens = m.q_c_out.args[0][0]
    with graph.inserting_before(m.cat):
        slots = graph.create_node(
            "call_function",
            _ATEN.slice.Tensor,
            (cache_af.kwargs["slot_mapping"], 0, 0, tokens),
        )
        # qkv workspace: the linear kernels write whole 16-row groups, so the
        # buffer must hold aligned_linear_rows(tokens) rows. The qkv_a linear's
        # own out buffer is exactly that ([16*ceil(T/16), 2624]).
        qkv = qkva_af.kwargs["out"]
        out_q = graph.create_node(
            "call_function",
            _EMPTY,
            ([tokens, _fw._HEADS_PER_RANK, _fw._OUT_HEAD_DIM],),
            dict(m.q_c_out.kwargs),
        )
        fused = graph.create_node(
            "call_function",
            _AF,
            (_FUSED,),
            dict(
                hidden=add_norm_af.kwargs["input"],
                residual=add_norm_af.kwargs["residual"],
                add_norm_weight=add_norm_af.kwargs["weight"],
                add_norm_eps=add_norm_af.kwargs["epsilon"],
                w_qkva=qkva_af.kwargs["packed_weight"],
                w_qkva_scale=qkva_af.kwargs["weight_scale"],
                q_a_weight=_fw._strip_alias(m.q_norm_af.kwargs["weight"]),
                kv_a_weight=_fw._strip_alias(m.kv_norm_af.kwargs["weight"]),
                q_a_eps=m.q_norm_af.kwargs["epsilon"],
                kv_a_eps=m.kv_norm_af.kwargs["epsilon"],
                w_qb=m.q_b_af.kwargs["packed_weight"],
                w_qb_scale=m.q_b_af.kwargs["weight_scale"],
                fp8_backend=fp8_backend,
                w_uk_t=m.einsum_af.kwargs["weight"],
                positions=m.rotary_af.kwargs["positions"],
                cos_sin=m.rotary_af.kwargs["cos_sin_cache"],
                slots=slots,
                kv_cache=cache_af.kwargs["kv_cache"],
                fp8_ds_mla=True,
                is_neox=m.rotary_af.kwargs["is_neox"],
                ag_send=None,
                ag_comm=0,
                qkv=qkv,
                q_c_norm=m.q_c_out,
                kv_c_norm=m.kv_c_out,
                q=m.q_ws,
                out_q=out_q,
            ),
        )
        redirects[hidden_gi] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["hidden"])
        )
        redirects[cache_out] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["kv_cache"])
        )
        redirects[m.q_norm_out] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["q_c_norm"])
        )
        if residual_gi is not None:
            redirects[residual_gi] = graph.create_node(
                "call_function", _GETITEM, (fused, _IDX["residual"])
            )
        redirects[m.cat] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["out_q"])
        )

    _fw.rebase_and_erase(graph, region, redirects)
    return True


def replace_with_fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache(graph) -> int:
    if not _backend_supported():
        logger.info(
            "deepseek_v32 pass: "
            "fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache backend "
            "unsupported on this build, graph left unfused"
        )
        return 0
    from torch_xcpu.ops_defs.fp8_linear import Fp8LinearBackend

    fp8_backend = int(Fp8LinearBackend.INTEL_AMX)
    cache_afs = [
        n
        for n in list(graph.nodes)
        if _fw._is_af_of(n, _CACHE_WRITE)
    ]
    if not cache_afs:
        return 0
    replaced = 0
    for cache_af in cache_afs:
        try:
            replaced += bool(_replace_one(graph, cache_af, fp8_backend))
        except _SKIP as reason:
            logger.info(
                "deepseek_v32 fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache "
                "pass: layer skipped (%s)",
                reason,
            )
    return replaced
