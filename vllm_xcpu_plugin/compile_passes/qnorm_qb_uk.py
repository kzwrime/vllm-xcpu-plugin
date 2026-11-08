# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-1 pattern: the GLM-5.2 MLA front half as one torch_xcpu operator.

In the post-grad graph every MLA layer runs (all wrapped in
``auto_functionalized``; the pass runs before Inductor's reinplace pass):

    qkv = fp8_mm(fused_qkv_a_proj output)          # [T, 2048 + 512 + 64]
    q_norm  = rms_norm(qkv[:, :2048])              # q_a_layernorm
    kv_norm = rms_norm(qkv[:, 2048:2560])          # kv_a_layernorm
    q    = fp8_mm(q_norm, q_b_proj)                # [T, heads*256]
    rope(q_pe, k_pe)                               # + write-back scatter dance
    reshape_and_cache_mla_fp8_ds(kv_norm, roped k_pe)
    q_nope_uk = einsum(q_nope, W_UK_T)             # [T, heads, 512]
    cat([q_nope_uk, roped q_pe])                   # attention query

and is replaced by ``fused_mla_qnorm_qb_uk_rope_cache`` returning
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
import operator

import torch
from torch.fx import Node

logger = logging.getLogger(__name__)

_ATEN = torch.ops.aten
_AF = torch.ops.higher_order.auto_functionalized
_RMS_NORM = torch.ops.torch_xcpu.rms_norm_bf16.default
_FP8_MM = torch.ops.torch_xcpu.fp8_scaled_mm_intel_amx_out.default
_ROTARY = torch.ops.torch_xcpu.rotary_embedding_bf16.default
_EINSUM = torch.ops.torch_xcpu.einsum_mhk_hkn_to_mhn.default
_CACHE_WRITE = torch.ops.torch_xcpu.reshape_and_cache_mla_fp8_ds.default
_FUSED = torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache.default

# Nodes the replacement region may contain; the walk stops at anything else
# (custom operators, placeholders, foreign auto_functionalized results).
_TRANSPARENT = {
    _ATEN.alias.default,
    _ATEN.as_strided.default,
    _ATEN.as_strided_scatter.default,
    _ATEN.cat.default,
    _ATEN.copy.default,
    _ATEN.expand.default,
    _ATEN.permute.default,
    _ATEN.reshape.default,
    _ATEN.slice.Tensor,
    _ATEN.slice_scatter.default,
    _ATEN.split_with_sizes.default,
    _ATEN.squeeze.dim,
    _ATEN.squeeze.dims,
    _ATEN.t.default,
    _ATEN.transpose.int,
    _ATEN.unsqueeze.default,
    _ATEN.view.default,
}
_GETITEM = operator.getitem
_EMPTY = _ATEN.empty.memory_format

_QKV_WIDTH = 2048 + 512 + 64
_HEADS_PER_RANK = 16
_NOPE_DIM = 192
_ROPE_DIM = 64
_KV_LORA = 512
# out_q head layout: up-projected nope (kv_lora) then roped q_pe.
_OUT_HEAD_DIM = _KV_LORA + _ROPE_DIM
_W_UK_SHAPE = (_HEADS_PER_RANK, _NOPE_DIM, _KV_LORA)
_QB_PACKED_SHAPE = (_HEADS_PER_RANK * (_NOPE_DIM + _ROPE_DIM), 2048)
_QB_SCALE_SHAPE = (_QB_PACKED_SHAPE[0] // 128, 2048 // 128)


def _af_indices(op) -> dict[str, int]:
    """auto_functionalized result index per mutated argument name.

    The af tuple is ``(op_return..., *clones_of_mutated_args_in_schema_order)``
    (torch/_higher_order_ops/auto_functionalize.py); a ``-> ()`` op still
    contributes a leading ``None`` slot, hence the offset of one.
    """
    mutated = [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    offset = max(1, len(op._schema.returns))
    return {name: offset + i for i, name in enumerate(mutated)}


_IDX = _af_indices(_FUSED)
# Result index of the mutated-output clone each pattern reads downstream.
_IDX_RMS_OUT = _af_indices(_RMS_NORM)["out"]
_IDX_MM_OUT = _af_indices(_FP8_MM)["out"]
_IDX_EINSUM_OUT = _af_indices(_EINSUM)["output"]
_IDX_CACHE_KV = _af_indices(_CACHE_WRITE)["kv_cache"]


def _backend_supported() -> bool:
    from torch_xcpu.ops_defs.fp8_linear import Fp8LinearBackend

    backend = int(Fp8LinearBackend.INTEL_AMX)
    return bool(
        torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache_backend_supported(
            backend
        )
    )


def _shape(node: Node) -> tuple | None:
    val = node.meta.get("val")
    return tuple(val.shape) if val is not None else None


def _is_af_of(node: Node, op) -> bool:
    return (
        node.op == "call_function" and node.target == _AF and node.args[0] == op
    )


def _arg_nodes(node: Node):
    for a in node.args + tuple(node.kwargs.values()):
        if isinstance(a, Node):
            yield a
        elif isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, Node):
                    yield x


def _is_getitem(node: Node) -> bool:
    return node.op == "call_function" and node.target == _GETITEM


def _is_result_view(node: Node) -> bool:
    return node.op == "call_function" and (
        node.target in _TRANSPARENT or _is_getitem(node)
    )


def _walkable(node: Node) -> bool:
    """Whether def-use walks may descend into this node's inputs."""
    if not _is_result_view(node):
        return False
    if _is_getitem(node) and isinstance(node.args[0], Node):
        # A getitem of an auto_functionalized call is the op's result
        # itself: the walk stops there.
        return node.args[0].target != _AF
    return True


def _roots_at(node: Node | None, targets: set[Node]) -> bool:
    """Whether a view-chain walk from ``node`` reaches any of ``targets``."""
    seen = set()
    work = [node]
    while work:
        n = work.pop()
        if not isinstance(n, Node) or id(n) in seen:
            continue
        seen.add(id(n))
        if n in targets:
            return True
        if _is_result_view(n):
            work.extend(_arg_nodes(n))
    return False


def _view_ancestor(node: Node) -> Node:
    """Walk a view chain down to its first non-view producer.

    The result of a foreign auto_functionalized call (e.g. the fused_qkv_a
    linear) is a producer: the walk stops at its result getitem.
    """
    while _walkable(node):
        node = node.args[0]
    return node


def _strip_alias(node: Node) -> Node:
    while node.op == "call_function" and node.target == _ATEN.alias.default:
        node = node.args[0]
    return node


class _Skip(Exception):
    pass


def _replace_one(graph, cache_af: Node, fp8_backend: int) -> bool:
    kv_norm_af = cache_af.kwargs["key"]
    if not (isinstance(kv_norm_af, Node) and _is_getitem(kv_norm_af)):
        raise _Skip("cache-write key is not an auto_functionalized result")
    kv_norm_af = kv_norm_af.args[0]
    if not _is_af_of(kv_norm_af, _RMS_NORM):
        raise _Skip("cache-write key does not come from rms_norm_bf16")

    qkv_base = _view_ancestor(kv_norm_af.kwargs["input"])
    if not _is_getitem(qkv_base):
        raise _Skip("qkv source is not a linear output")

    q_norm_afs = [
        n
        for n in graph.nodes
        if _is_af_of(n, _RMS_NORM)
        and _roots_at(n.kwargs.get("input"), {qkv_base})
    ]
    if len(q_norm_afs) != 2:
        raise _Skip(f"expected 2 qkv rms_norms, found {len(q_norm_afs)}")
    if kv_norm_af not in q_norm_afs:
        raise _Skip("kv_a rms_norm does not read the fused_qkv_a output")
    q_norm_af = next(n for n in q_norm_afs if n is not kv_norm_af)
    q_norm_out = next(
        (
            u
            for u in q_norm_af.users
            if _is_getitem(u) and u.args[1] == _IDX_RMS_OUT
        ),
        None,
    )
    if q_norm_out is None:
        raise _Skip("q_norm result missing")

    # q_b_proj: the FP8 linear reading q_norm whose result feeds the layer's
    # W_UK einsum. Shape alone cannot pick it: on TP4 the indexer wq_b linear
    # has the same [4096, 2048] packed weights as q_b_proj and reads q_norm.
    einsum_afs = [
        n
        for n in graph.nodes
        if _is_af_of(n, _EINSUM)
        and _shape(n.kwargs["weight"]) == _W_UK_SHAPE
    ]
    q_b_af = None
    q_b_out = None
    for n in graph.nodes:
        if not _is_af_of(n, _FP8_MM):
            continue
        if not _roots_at(n.kwargs.get("input"), {q_norm_out}):
            continue
        if _shape(n.kwargs["packed_weight"]) != _QB_PACKED_SHAPE or (
            _shape(n.kwargs["weight_scale"]) != _QB_SCALE_SHAPE
        ):
            continue
        cand_out = next(
            (u for u in n.users if _is_getitem(u) and u.args[1] == _IDX_MM_OUT),
            None,
        )
        if not any(
            _roots_at(e.kwargs["input"], {cand_out}) for e in einsum_afs
        ):
            continue
        if q_b_af is not None:
            raise _Skip("ambiguous q_b_proj")
        q_b_af = n
        q_b_out = cand_out
    if q_b_af is None:
        raise _Skip(
            f"no q_b_proj with FP8 weights {_QB_PACKED_SHAPE} feeding W_UK "
            f"(heads per rank != {_HEADS_PER_RANK}?)"
        )

    einsum_afs = [
        n
        for n in graph.nodes
        if _is_af_of(n, _EINSUM)
        and _shape(n.kwargs["weight"]) == _W_UK_SHAPE
        and _roots_at(n.kwargs.get("input"), {q_b_out})
    ]
    if len(einsum_afs) != 1:
        raise _Skip(f"expected 1 W_UK einsum, found {len(einsum_afs)}")
    einsum_af = einsum_afs[0]
    einsum_out = next(
        u for u in einsum_af.users if _is_getitem(u) and u.args[1] == _IDX_EINSUM_OUT
    )

    rotary_afs = [
        n
        for n in graph.nodes
        if _is_af_of(n, _ROTARY)
        and _roots_at(n.kwargs.get("query"), {q_b_out})
    ]
    if len(rotary_afs) != 1:
        raise _Skip(f"expected 1 q rope, found {len(rotary_afs)}")
    rotary_af = rotary_afs[0]

    cats = [
        u
        for u in einsum_out.users
        if u.op == "call_function" and u.target == _ATEN.cat.default
    ]
    if len(cats) != 1:
        raise _Skip(f"expected 1 attention-query cat, found {len(cats)}")
    cat = cats[0]

    # Static shape gate, mirroring the eager wrapper's support check.
    for name, node, want in (
        ("qkv", qkv_base, (None, _QKV_WIDTH)),
        ("w_uk_t", einsum_af.kwargs["weight"], _W_UK_SHAPE),
        ("attention query", cat, (None, _HEADS_PER_RANK, _OUT_HEAD_DIM)),
    ):
        got = _shape(node)
        if got is None or any(
            w is not None and g != w for g, w in zip(got, want)
        ):
            raise _Skip(f"{name} shape {got} does not fit the operator")

    # The roped k_pe reaching the cache write must come from the q rope.
    if not _roots_at(cache_af.kwargs["value"], {rotary_af}):
        raise _Skip("cache-write value does not come from the q rope")

    # Output buffers reused by the fused operator (kept, not erased).
    q_c_out = q_norm_af.kwargs["out"]
    kv_c_out = kv_norm_af.kwargs["out"]
    q_ws = q_b_af.kwargs["out"]

    seeds = [q_norm_af, kv_norm_af, q_b_af, rotary_af, einsum_af, cache_af, cat]

    # Region closure: backward into view chains / tuple gets / aten helpers,
    # forward into view chains of the region's results. Custom operators,
    # placeholders and foreign af results are region boundaries; empty
    # allocations enter as leaves so their size math (shared with the rest of
    # the graph) stays out of the region.
    region: set[Node] = set()
    work = list(seeds)
    while work:
        n = work.pop()
        if not isinstance(n, Node) or n in region:
            continue
        region.add(n)
        if n.target == _EMPTY:
            continue
        for a in _arg_nodes(n):
            if a.op != "call_function":
                continue
            if _is_getitem(a):
                base = a.args[0]
                if isinstance(base, Node) and (
                    base in region or base.target != _AF
                ):
                    work.append(a)
            elif a.target in _TRANSPARENT or a.target.namespace == "aten":
                work.append(a)
        for u in n.users:
            if _is_result_view(u):
                work.append(u)

    # Redirect producers: region results consumed outside the region.
    cache_out = next(
        u for u in cache_af.users if _is_getitem(u) and u.args[1] == _IDX_CACHE_KV
    )
    redirects: dict[Node, Node] = {q_norm_out: None, cache_out: None, cat: None}
    # Redirect reachable set: the producers plus their transparent fan-out
    # inside the region; only those may have consumers outside the region.
    reach = set(redirects)
    work = list(redirects)
    while work:
        n = work.pop()
        for u in n.users:
            if u in region and u not in reach:
                reach.add(u)
                work.append(u)
    for d in region:
        if d in reach:
            continue
        if any(u not in region for u in d.users):
            raise _Skip(f"{d.target} result escapes the region unfused")

    # --- mutation below this line; no more _Skip ---
    tokens = q_c_out.args[0][0]
    with graph.inserting_before(cat):
        slots = graph.create_node(
            "call_function",
            _ATEN.slice.Tensor,
            (cache_af.kwargs["slot_mapping"], 0, 0, tokens),
        )
        qkv = graph.create_node(
            "call_function",
            _ATEN.as_strided.default,
            (qkv_base, [tokens, _QKV_WIDTH], [_QKV_WIDTH, 1]),
        )
        out_q = graph.create_node(
            "call_function",
            _EMPTY,
            ([tokens, _HEADS_PER_RANK, _OUT_HEAD_DIM],),
            dict(q_c_out.kwargs),
        )
        fused = graph.create_node(
            "call_function",
            _AF,
            (_FUSED,),
            dict(
                qkv=qkv,
                q_a_weight=_strip_alias(q_norm_af.kwargs["weight"]),
                kv_a_weight=_strip_alias(kv_norm_af.kwargs["weight"]),
                q_a_eps=q_norm_af.kwargs["epsilon"],
                kv_a_eps=kv_norm_af.kwargs["epsilon"],
                w_qb=q_b_af.kwargs["packed_weight"],
                w_qb_scale=q_b_af.kwargs["weight_scale"],
                fp8_backend=fp8_backend,
                w_uk_t=einsum_af.kwargs["weight"],
                positions=rotary_af.kwargs["positions"],
                cos_sin=rotary_af.kwargs["cos_sin_cache"],
                slots=slots,
                kv_cache=cache_af.kwargs["kv_cache"],
                fp8_ds_mla=True,
                is_neox=rotary_af.kwargs["is_neox"],
                q_c_norm=q_c_out,
                kv_c_norm=kv_c_out,
                q=q_ws,
                out_q=out_q,
            ),
        )
        redirects[q_norm_out] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["q_c_norm"])
        )
        redirects[cache_out] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["kv_cache"])
        )
        redirects[cat] = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX["out_q"])
        )

    # Rebase redirect chains: clone the region-side view nodes that outside
    # consumers read through, on top of the fused results.
    clone_map: dict[Node, Node] = dict(redirects)
    for d in list(graph.nodes):
        if d not in region or d in clone_map:
            continue
        if not any(u not in region for u in d.users):
            continue
        with graph.inserting_before(d):
            clone_map[d] = graph.create_node(
                d.op,
                d.target,
                tuple(clone_map.get(a, a) for a in d.args),
                {k: clone_map.get(v, v) for k, v in d.kwargs.items()},
            )
    for n in list(graph.nodes):
        if n in region:
            continue
        n.args = tuple(clone_map.get(a, a) for a in n.args)
        n.kwargs = {k: clone_map.get(v, v) for k, v in n.kwargs.items()}

    # Erase the region once nothing reads it (users first).
    progress = True
    while progress:
        progress = False
        for d in reversed(list(graph.nodes)):
            if d in region and not d.users:
                graph.erase_node(d)
                region.discard(d)
                progress = True
    if region:
        raise RuntimeError(
            "qnorm_qb_uk pass could not erase "
            f"{[str(d.target) for d in region]}"
        )
    return True


def replace_with_fused_mla_qnorm_qb_uk(graph) -> int:
    if not _backend_supported():
        logger.info(
            "qnorm_qb_uk pass: fused_mla_qnorm_qb_uk_rope_cache backend "
            "unsupported on this build, graph left unfused"
        )
        return 0
    from torch_xcpu.ops_defs.fp8_linear import Fp8LinearBackend

    fp8_backend = int(Fp8LinearBackend.INTEL_AMX)
    cache_afs = [n for n in list(graph.nodes) if _is_af_of(n, _CACHE_WRITE)]
    if not cache_afs:
        return 0
    replaced = 0
    for cache_af in cache_afs:
        try:
            replaced += bool(_replace_one(graph, cache_af, fp8_backend))
        except _Skip as reason:
            logger.info("qnorm_qb_uk pass: layer skipped (%s)", reason)
    return replaced
