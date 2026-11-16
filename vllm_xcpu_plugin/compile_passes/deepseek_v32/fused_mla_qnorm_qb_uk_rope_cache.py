# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-1 pattern: the GLM-5.2 (DeepSeek-V3.2-style) MLA front half fused.

In the post-grad graph every MLA layer runs (all wrapped in
``auto_functionalized``; the pass runs before Inductor's reinplace pass) the
window described in ``_front_window`` and it is replaced by
``fused_mla_qnorm_qb_uk_rope_cache`` returning
``(q_c_norm, kv_c_norm, out_q)``. The indexer operators between q_b_proj and
the rope (layers with a sparse indexer) are not part of the pattern; matching
is by def-use, not by emission order.

Boundary redirects preserve the consumers outside the region: the indexer
wq_b linear keeps reading q_c_norm, the sparse MLA keeps reading out_q and
the post-write KV cache; everything else inside the region is erased.

The operator requires 16 heads per rank (TP4 sharding of the 64-head model),
[4096, 2048] q_b FP8 weights and [16, 192, 512] W_UK_T; layers whose
parameters do not match those shapes stay unfused (e.g. TP1 keeps all 64
heads per rank), mirroring the eager wrapper's support gate.
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
_STRIP_ALIAS = _fw._strip_alias

_FUSED = torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache.default
_CACHE_WRITE = _fw._CACHE_WRITE

_IDX = _fw._af_indices(_FUSED)
_IDX_CACHE_KV = _fw._af_indices(_CACHE_WRITE)["kv_cache"]


def _backend_supported() -> bool:
    return _fw.backend_supported(
        torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache_backend_supported
    )


def _replace_one(graph, cache_af: Node) -> bool:
    m = _fw.match_window(graph, cache_af)

    # The packed-weight layout the operator must read is decided by the
    # backend that packed q_b_proj -- the graph encodes it in the fp8 mm
    # overload itself.
    fp8_backend = _fw._fp8_mm_backend(m.q_b_af)
    if not torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache_backend_supported(
        fp8_backend
    ):
        raise _SKIP(f"fp8 backend {fp8_backend} unsupported by the fused operator")

    # Redirect producers: region results consumed outside the region.
    cache_out = next(
        u
        for u in cache_af.users
        if _fw._is_getitem(u) and u.args[1] == _IDX_CACHE_KV
    )
    redirects: dict[Node, Node | None] = {
        m.q_norm_out: None,
        cache_out: None,
        m.cat: None,
    }
    region = _fw.close_region(
        graph,
        [
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
    q_c_shape = m.q_c_out.args[0]
    assert isinstance(q_c_shape, (list, tuple))
    tokens = q_c_shape[0]
    with graph.inserting_before(m.cat):
        slots = graph.create_node(
            "call_function",
            _ATEN.slice.Tensor,
            (cache_af.kwargs["slot_mapping"], 0, 0, tokens),
        )
        qkv = graph.create_node(
            "call_function",
            _ATEN.as_strided.default,
            (m.qkv_base, [tokens, _fw._QKV_WIDTH], [_fw._QKV_WIDTH, 1]),
        )
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
                qkv=qkv,
                q_a_weight=_STRIP_ALIAS(m.q_norm_af.kwargs["weight"]),
                kv_a_weight=_STRIP_ALIAS(m.kv_norm_af.kwargs["weight"]),
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
                q_c_norm=m.q_c_out,
                kv_c_norm=m.kv_c_out,
                q=m.q_ws,
                out_q=out_q,
            ),
        )
        redirects[m.q_norm_out] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["q_c_norm"])
        )
        redirects[cache_out] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["kv_cache"])
        )
        redirects[m.cat] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["out_q"])
        )

    _fw.rebase_and_erase(graph, region, redirects)
    return True


def replace_with_fused_mla_qnorm_qb_uk_rope_cache(graph) -> int:
    if not _backend_supported():
        logger.info(
            "deepseek_v32 pass: fused_mla_qnorm_qb_uk_rope_cache backend "
            "unsupported on this build, graph left unfused"
        )
        return 0
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
            replaced += bool(_replace_one(graph, cache_af))
        except _SKIP as reason:
            logger.info(
                "deepseek_v32 fused_mla_qnorm_qb_uk_rope_cache pass: layer "
                "skipped (%s)",
                reason,
            )
    return replaced
