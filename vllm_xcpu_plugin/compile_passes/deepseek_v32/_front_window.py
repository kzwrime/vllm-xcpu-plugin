# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matcher and rewrite machinery shared by the DeepSeek-V3.2 MLA passes.

The core window every pass in this package replaces (all operators wrapped in
``auto_functionalized``; the passes run before Inductor's reinplace pass):

    qkv = fp8_mm(fused_qkv_a_proj output)          # [T, 2048 + 512 + 64]
    q_norm  = rms_norm(qkv[:, :2048])              # q_a_layernorm
    kv_norm = rms_norm(qkv[:, 2048:2560])          # kv_a_layernorm
    q    = fp8_mm(q_norm, q_b_proj)                # [T, heads*256]
    rope(q_pe, k_pe)                               # + write-back scatter dance
    reshape_and_cache_mla_fp8_ds(kv_norm, roped k_pe)
    q_nope_uk = einsum(q_nope, W_UK_T)             # [T, heads, 512]
    cat([q_nope_uk, roped q_pe])                   # attention query

Matching is by def-use from the cache write, not by emission order, so the
indexer operators a sparse-indexer layer runs between q_b_proj and the rope
are invisible to it. Passes extend the window upwards from
:func:`match_window` (e.g. the input add-norm and the fused_qkv_a linear)
and emit their fused operator; :func:`close_region`, :func:`ensure_no_escape`
and :func:`rebase_and_erase` then perform the rewrite.
"""

import logging
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeGuard

import torch
from torch.fx import Node
from torch.fx.node import map_arg

logger = logging.getLogger(__name__)

_ATEN = torch.ops.aten
_AF = torch.ops.higher_order.auto_functionalized
_RMS_NORM = torch.ops.torch_xcpu.rms_norm_bf16.default
# The linear backends each register their own fp8_scaled_mm_<suffix>_out
# overload with an identical schema; matching accepts all of them and the
# backend int is read back off the matched op (``_fp8_mm_backend``).
_FP8_MM = torch.ops.torch_xcpu.fp8_scaled_mm_intel_amx_out.default
_ROTARY = torch.ops.torch_xcpu.rotary_embedding_bf16.default
_EINSUM = torch.ops.torch_xcpu.einsum_mhk_hkn_to_mhn.default
_CACHE_WRITE = torch.ops.torch_xcpu.reshape_and_cache_mla_fp8_ds.default

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


def _fp8_mm_ops() -> dict:
    from torch_xcpu.ops_defs.fp8_linear import _BACKEND_OP_SUFFIXES

    return {
        getattr(torch.ops.torch_xcpu, f"fp8_scaled_mm_{suffix}_out").default: int(
            backend
        )
        for backend, suffix in _BACKEND_OP_SUFFIXES.items()
    }


_FP8_MM_OPS = _fp8_mm_ops()

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


def _is_fp8_mm_af(node: object) -> TypeGuard[Node]:
    """auto_functionalized call of any fp8_scaled_mm_<backend>_out overload."""
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == _AF
        and node.args[0] in _FP8_MM_OPS
    )


def _fp8_mm_backend(node: Node) -> int:
    """The Fp8LinearBackend int of a matched fp8 mm (see _is_fp8_mm_af)."""
    return _FP8_MM_OPS[node.args[0]]


def backend_supported(supported_op, backend: int | None = None) -> bool:
    if backend is not None:
        return bool(supported_op(backend))
    # Early-out before any graph is seen: any fp8 backend will do here, the
    # per-layer gate re-checks the backend the graph actually uses.
    from torch_xcpu.ops_defs.fp8_linear import Fp8LinearBackend

    return any(bool(supported_op(int(b))) for b in Fp8LinearBackend)


def _shape(node: object) -> tuple | None:
    if not isinstance(node, Node):
        return None
    val = node.meta.get("val")
    return tuple(val.shape) if val is not None else None


def _check_shape(where: str, node: object, want: tuple) -> None:
    got = _shape(node)
    if got is None or any(w is not None and g != w for g, w in zip(got, want)):
        raise _Skip(f"{where} shape {got} does not fit the operator")


def _is_af_of(node: object, op) -> TypeGuard[Node]:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == _AF
        and node.args[0] == op
    )


def _arg_nodes(node: Node):
    for a in node.args + tuple(node.kwargs.values()):
        if isinstance(a, Node):
            yield a
        elif isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, Node):
                    yield x


def _is_getitem(node: object) -> TypeGuard[Node]:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == _GETITEM
    )


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


def _roots_at(node: object, targets: set[Node]) -> bool:
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


def _view_ancestor(node: object) -> Node:
    """Walk a view chain down to its first non-view producer.

    The result of a foreign auto_functionalized call (e.g. the fused_qkv_a
    linear) is a producer: the walk stops at its result getitem.
    """
    while isinstance(node, Node) and _walkable(node):
        node = node.args[0]
    if not isinstance(node, Node):
        raise _Skip("expected FX Node in view chain")
    return node


def _strip_alias(node: object) -> Node:
    while (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == _ATEN.alias.default
    ):
        node = node.args[0]
    assert isinstance(node, Node)
    return node


class _Skip(Exception):
    pass


@dataclass
class WindowMatch:
    """The core window's nodes, as matched around one cache write."""

    cache_af: Node
    kv_norm_af: Node
    q_norm_af: Node
    q_norm_out: Node  # getitem of q_norm's result clone, read by q_b_proj
    qkv_base: Node  # getitem of the fused_qkv_a linear's result
    q_b_af: Node
    einsum_af: Node
    einsum_out: Node
    rotary_af: Node
    cat: Node  # the attention-query cat
    q_c_out: Node  # q_norm's out buffer, reused by the fused operator
    kv_c_out: Node  # kv_norm's out buffer
    q_ws: Node  # q_b_proj's out buffer


def match_window(graph, cache_af: Node) -> WindowMatch:
    """Match the core window rooted at ``cache_af`` (raises _Skip)."""
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
    idx_rms_out = _af_indices(_RMS_NORM)["out"]
    q_norm_out = next(
        (
            u
            for u in q_norm_af.users
            if _is_getitem(u) and u.args[1] == idx_rms_out
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
        if _is_af_of(n, _EINSUM) and _shape(n.kwargs["weight"]) == _W_UK_SHAPE
    ]
    q_b_af = None
    q_b_out = None
    for n in graph.nodes:
        if not _is_fp8_mm_af(n):
            continue
        if not _roots_at(n.kwargs.get("input"), {q_norm_out}):
            continue
        if _shape(n.kwargs["packed_weight"]) != _QB_PACKED_SHAPE or (
            _shape(n.kwargs["weight_scale"]) != _QB_SCALE_SHAPE
        ):
            continue
        idx_mm_out = _af_indices(_FP8_MM)["out"]
        cand_out = next(
            (
                u
                for u in n.users
                if _is_getitem(u) and u.args[1] == idx_mm_out
            ),
            None,
        )
        if cand_out is None:
            continue
        if not any(_roots_at(e.kwargs["input"], {cand_out}) for e in einsum_afs):
            continue
        if q_b_af is not None:
            raise _Skip("ambiguous q_b_proj")
        q_b_af = n
        q_b_out = cand_out
    if q_b_af is None or q_b_out is None:
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
    idx_einsum_out = _af_indices(_EINSUM)["output"]
    einsum_out = next(
        u
        for u in einsum_af.users
        if _is_getitem(u) and u.args[1] == idx_einsum_out
    )

    rotary_afs = [
        n
        for n in graph.nodes
        if _is_af_of(n, _ROTARY) and _roots_at(n.kwargs.get("query"), {q_b_out})
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
    _check_shape("qkv", qkv_base, (None, _QKV_WIDTH))
    _check_shape("w_uk_t", einsum_af.kwargs["weight"], _W_UK_SHAPE)
    _check_shape("attention query", cat, (None, _HEADS_PER_RANK, _OUT_HEAD_DIM))

    # The roped k_pe reaching the cache write must come from the q rope.
    if not _roots_at(cache_af.kwargs["value"], {rotary_af}):
        raise _Skip("cache-write value does not come from the q rope")

    q_c_out = q_norm_af.kwargs["out"]
    kv_c_out = kv_norm_af.kwargs["out"]
    q_ws = q_b_af.kwargs["out"]
    if not (
        isinstance(q_c_out, Node)
        and isinstance(kv_c_out, Node)
        and isinstance(q_ws, Node)
    ):
        raise _Skip("norm/q_b out buffer is not a Node")

    return WindowMatch(
        cache_af=cache_af,
        kv_norm_af=kv_norm_af,
        q_norm_af=q_norm_af,
        q_norm_out=q_norm_out,
        qkv_base=qkv_base,
        q_b_af=q_b_af,
        einsum_af=einsum_af,
        einsum_out=einsum_out,
        rotary_af=rotary_af,
        cat=cat,
        q_c_out=q_c_out,
        kv_c_out=kv_c_out,
        q_ws=q_ws,
    )


def close_region(graph, seeds: list[Node]) -> set[Node]:
    """Region closure: backward into view chains / tuple gets / aten helpers,
    forward into view chains of the region's results. Custom operators,
    placeholders and foreign af results are region boundaries; empty
    allocations enter as leaves so their size math (shared with the rest of
    the graph) stays out of the region."""
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
    return region


def ensure_no_escape(
    region: set[Node], redirects: Mapping[Node, Node | None]
) -> None:
    """Redirect reachable set: the producers plus their transparent fan-out
    inside the region; only those may have consumers outside the region."""
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


def rebase_and_erase(
    graph, region: set[Node], redirects: Mapping[Node, Node | None]
) -> None:
    """Rebase redirect chains and erase the region (users first).

    Clone the region-side view nodes that outside consumers read through, on
    top of the fused results, repoint every outside consumer, then erase the
    region once nothing reads it."""
    clone_map: dict[Node, Node] = {}
    for orig, repl in redirects.items():
        assert repl is not None
        if "val" in orig.meta and "val" not in repl.meta:
            repl.meta["val"] = orig.meta["val"]
        clone_map[orig] = repl
    # Each clone is an outside-region consumer of its args: a region member
    # it reads may itself now need cloning, so iterate to a fixpoint (the
    # chain exits the region at foreign af results / placeholders).
    changed = True
    while changed:
        changed = False
        for d in list(graph.nodes):
            if d not in region or d in clone_map:
                continue
            if not any(u not in region for u in d.users):
                continue
            with graph.inserting_before(d):
                cloned = graph.create_node(
                    d.op,
                    d.target,
                    tuple(clone_map.get(a, a) for a in d.args),
                    {k: clone_map.get(v, v) for k, v in d.kwargs.items()},
                )
                if "val" in d.meta:
                    cloned.meta["val"] = d.meta["val"]
                clone_map[d] = cloned
            changed = True
    for n in list(graph.nodes):
        if n in region:
            continue
        n.args = map_arg(n.args, lambda a: clone_map.get(a, a))
        n.kwargs = map_arg(n.kwargs, lambda v: clone_map.get(v, v))

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
            "deepseek_v32 pass could not erase "
            f"{[str(d.target) for d in region]}"
        )
