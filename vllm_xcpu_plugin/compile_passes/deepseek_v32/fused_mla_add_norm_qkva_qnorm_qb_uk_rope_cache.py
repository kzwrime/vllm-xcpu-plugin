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
stay unfused.

With sequence-parallel MoE the layer input is a ``[0:T]`` slice of the TP
all-gather (``all_gather_into_tensor_out_v2``) of the previous MoE's
per-rank shard. That gather is absorbed too: the fused operator receives
``ag_send`` (the shard) and ``ag_comm`` (the graph's communicator-wrapper
placeholder, so the MPI handle stays a runtime value) and ``hidden``
becomes the gather's landing buffer (``world_size * shard_rows`` rows).
The collective then runs on the mega-kernel's helper thread while the
weight DMAs are in flight, instead of as a separate graph node. Windows
whose add-norm input is not such a slice (e.g. TP1) fall back to the
non-absorbed form.
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
_ALL_GATHER = torch.ops.torch_xcpu.all_gather_into_tensor_out_v2.default
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
_IDX_AG_OUT = _fw._af_indices(_ALL_GATHER)["output"]


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
    _fw._check_shape("w_qkva", qkva_af.kwargs["packed_weight"], _W_QKVA_SHAPE)
    _fw._check_shape(
        "w_qkva_scale", qkva_af.kwargs["weight_scale"], _W_QKVA_SCALE_SHAPE
    )

    # And its input: the layer's add-norm. The first layer uses a plain
    # rms_norm (no residual add) and stays with the phase-1 pass.
    hidden_gi = _fw._view_ancestor(qkva_af.kwargs["input"])
    if not (_fw._is_getitem(hidden_gi) and hidden_gi.args[1] == _IDX_ADD["output"]):
        raise _SKIP("fused_qkv_a input is not an add-norm output clone")
    add_norm_af = hidden_gi.args[0]
    if not _fw._is_af_of(add_norm_af, _ADD_NORM):
        raise _SKIP("input norm is not fused_add_rms_norm_out (first layer?)")
    _fw._check_shape("add-norm hidden", hidden_gi, (None, _HIDDEN))
    _fw._check_shape("add-norm weight", add_norm_af.kwargs["weight"], (_HIDDEN,))

    # Upstream of the add-norm: with sequence-parallel MoE the layer input
    # is a [0:T] slice of the TP all-gather of the previous MoE's per-rank
    # shard. When present, absorb the gather into the fused operator
    # (ag_send shard / ag_comm wrapper placeholder / landing buffer as
    # hidden), so the collective runs on the mega-kernel's helper thread
    # instead of as a separate graph node.
    gather_af = None
    add_in = _fw._view_ancestor(add_norm_af.kwargs["input"])
    if (
        _fw._is_getitem(add_in)
        and isinstance(add_in.args[0], Node)
        and _fw._is_af_of(add_in.args[0], _ALL_GATHER)
        and add_in.args[1] == _IDX_AG_OUT
    ):
        gather_af = add_in.args[0]
        if gather_af.kwargs.get("dim") != 0:
            raise _SKIP("all-gather along a non-leading dim")
        _fw._check_shape("all-gather landing buffer", add_in, (None, _HIDDEN))
        _fw._check_shape("all-gather shard", gather_af.kwargs["input"], (None, _HIDDEN))
        _fw._check_shape(
            "all-gather comm wrapper",
            gather_af.kwargs["comm_ptr_wrapper"],
            (1,),
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
            *([gather_af] if gather_af is not None else []),
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
        if gather_af is not None:
            ag_send = gather_af.kwargs["input"]
            # The graph's communicator-wrapper placeholder: the MPI handle
            # stays a runtime value, safe across compiled-artifact reuse.
            ag_comm = gather_af.kwargs["comm_ptr_wrapper"]
            hidden = gather_af.kwargs["output"]
        else:
            ag_send = None
            ag_comm = None
            hidden = add_norm_af.kwargs["input"]
        fused = graph.create_node(
            "call_function",
            _AF,
            (_FUSED,),
            dict(
                hidden=hidden,
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
                ag_send=ag_send,
                ag_comm=ag_comm,
                qkv=qkv,
                q_c_norm=m.q_c_out,
                kv_c_norm=m.kv_c_out,
                q=m.q_ws,
                out_q=out_q,
            ),
        )
        hidden_out = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["hidden"])
        )
        if gather_af is not None:
            # hidden is now the full gather landing buffer; consumers expect
            # the [0:T] slice the unfused gather+slice produced.
            redirects[hidden_gi] = graph.create_node(
                "call_function",
                _ATEN.slice.Tensor,
                (hidden_out, 0, 0, tokens),
            )
        else:
            redirects[hidden_gi] = hidden_out
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
    cache_afs = [n for n in list(graph.nodes) if _fw._is_af_of(n, _CACHE_WRITE)]
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
