# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-grad pattern pass: MoE front add-RMSNorm + FP32 router gate GEMM.

Every routed-MoE layer runs two back-to-back operators in front of its router
selection when the BF16 linear takeover serves ``GateLinear``::

    hidden_norm, residual_out = fused_add_rms_norm_out_bf16(
        hidden, residual, norm_weight, eps
    )
    router_logits = bf16_linear_<backend>_out(
        hidden_norm, packed_router_weight, bias=None
    )  # FP32 out [aligned_rows(T), N]

In the Inductor post-grad graph (pre-reinplace, ``auto_functionalized`` form)
this pass fuses each eligible pair into a single call to
``torch.ops.torch_xcpu.fused_add_norm_router_gemm.default``:

* ``hidden_norm`` and ``residual_out`` are produced through the fused
  operator's mutated ``hidden`` / ``residual`` clones, so downstream consumers
  (routed experts, shared experts, next layer's add-norm, or live ``input_sum``
  captures such as EAGLE3) keep exact SSA dataflow and Inductor's reinplace pass
  mutates in place when dead or clones automatically when live.
* The gate linear's private ``[aligned_linear_rows(T), N]`` FP32 output buffer
  is hoisted (with any pure symbolic-shape math) and reused as ``router_logits``;
  downstream consumers keep their existing ``[0:T, N]`` logical view.
* The operator itself dispatches per call between the instantiated mega-kernel
  segment (supported shape, ``T <= 32``, PE budget) and its internal per-op
  fallback (larger prefill batches, other shapes, ``A0``), so one compiled
  graph serves every batch size without graph breaks or token-count branches.

Safety gates (mismatches stay unfused):
* Only ``fused_add_rms_norm_out_bf16`` matches -- plain ``rms_norm_bf16``,
  ``gemma_rms_norm_bf16``, FP32 norms, and ``fused_add_rms_norm_chunk_out_bf16``
  never match.
* Only shape- and layout-preserving 2D BF16 views/aliases are traversed between
  ``hidden_norm`` and the gate input -- sequence-parallel ``slice`` / ``pad``
  between norm and gate cannot match.
* Only a unique bias-free ``bf16_linear_<backend>_out`` with FP32 output on a
  backend passing ``fused_add_norm_router_gemm_backend_supported`` and valid
  packed weight alignment matches.
"""

import logging
import operator
from typing import Any

import torch
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.node import map_arg

logger = logging.getLogger(__name__)

_ATEN = torch.ops.aten
_AF = torch.ops.higher_order.auto_functionalized
_AF_V2 = torch.ops.higher_order.auto_functionalized_v2
_AF_HOPS = (_AF, _AF_V2)
_GETITEM = operator.getitem
_EMPTY = _ATEN.empty.memory_format
_EMPTY_ALLOC_OPS = {_EMPTY, _ATEN.empty_like.default}

_ADD_NORM = torch.ops.torch_xcpu.fused_add_rms_norm_out_bf16.default
_FUSED = torch.ops.torch_xcpu.fused_add_norm_router_gemm.default

# Only shape-preserving 2D views/aliases may sit between add-norm output and
# the router gate input. Row-changing ops (e.g. aten.slice for SP chunking) are
# deliberately excluded so pre-chunk norms never fuse with post-chunk gates.
_SAME_2D_VIEW_OPS = {
    _ATEN.alias.default,
    _ATEN.detach.default,
    _ATEN.reshape.default,
    _ATEN.view.default,
    _ATEN._unsafe_view.default,
}

# Pure tensor view ops allowed when hoisting weight/buffer chains before the
# add-norm anchor.
_PURE_VIEW_OPS = _SAME_2D_VIEW_OPS | {
    _ATEN.as_strided.default,
    _ATEN.slice.Tensor,
}


class _Skip(Exception):
    """Pattern matched the anchor but failed a static gate; leave unfused."""


def _af_indices(op) -> dict[str, int]:
    """auto_functionalized result index per mutated argument name."""
    mutated = [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    offset = max(1, len(op._schema.returns))
    return {name: offset + i for i, name in enumerate(mutated)}


def _bf16_linear_ops() -> dict:
    from torch_xcpu.ops_defs.bf16_linear import _BACKEND_OP_SUFFIXES

    return {
        getattr(torch.ops.torch_xcpu, f"bf16_linear_{suffix}_out").default: int(
            backend
        )
        for backend, suffix in _BACKEND_OP_SUFFIXES.items()
    }


_BF16_LINEAR_OPS = _bf16_linear_ops()
_IDX_ADD = _af_indices(_ADD_NORM)
_IDX_FUSED = _af_indices(_FUSED)
_IDX_LINEAR_OUT = _af_indices(next(iter(_BF16_LINEAR_OPS)))["out"]


def _any_backend_supported() -> bool:
    from torch_xcpu.ops_defs.bf16_linear import Bf16LinearBackend

    gate = torch.ops.torch_xcpu.fused_add_norm_router_gemm_backend_supported
    return any(bool(gate(int(b))) for b in Bf16LinearBackend)


def _is_af_of(node: Node, op) -> bool:
    return (
        node.op == "call_function"
        and node.target in _AF_HOPS
        and bool(node.args)
        and node.args[0] == op
    )


def _is_bf16_linear_af(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target in _AF_HOPS
        and bool(node.args)
        and node.args[0] in _BF16_LINEAR_OPS
    )


def _mutable_arg_info(
    af_node: Node, arg_name: str, v1_indices: dict[str, int]
) -> tuple[Node, int] | None:
    """Return ``(backing_buffer_node, getitem_index)`` for a mutable arg on an
    ``auto_functionalized`` (V1) or ``auto_functionalized_v2`` (V2) HOP call.

    In V2, rejects view-backed or aliased bases so we only match standalone
    freshly allocated output buffers (matching V1's ``empty`` / ``empty_like``).
    """
    if af_node.target == _AF:
        buf = af_node.kwargs.get(arg_name)
        idx = v1_indices.get(arg_name)
        if isinstance(buf, Node) and idx is not None:
            return buf, idx
        return None

    if af_node.target == _AF_V2:
        # Reject subviews (slice / as_strided). Allow ``_<arg>_alias: True``
        # (AliasViewInfo), which is an exact 1:1 full-tensor alias emitted e.g.
        # under ``torch.inference_mode()``.
        for suffix in ("_storage_offset", "_slice_dim"):
            if f"_{arg_name}{suffix}" in af_node.kwargs:
                return None
        alias_key = f"_{arg_name}_alias"
        if alias_key in af_node.kwargs and af_node.kwargs[alias_key] is not True:
            return None
        base_idx = af_node.kwargs.get(f"_{arg_name}_base_index")
        all_bases = af_node.kwargs.get("_all_bases")
        if not (
            isinstance(base_idx, int)
            and isinstance(all_bases, (list, tuple))
            and 0 <= base_idx < len(all_bases)
            and isinstance(all_bases[base_idx], Node)
        ):
            return None
        # Ensure no two mutable arguments of this op share the same base slot.
        base_uses = sum(
            1
            for k, v in af_node.kwargs.items()
            if k.endswith("_base_index") and v == base_idx
        )
        if base_uses != 1:
            return None
        op: Any = af_node.args[0]
        offset = max(1, len(op._schema.returns))
        return all_bases[base_idx], offset + base_idx

    return None


def _is_getitem(node: Node) -> bool:
    return node.op == "call_function" and node.target == _GETITEM


def _strip_alias(node: object) -> Any:
    while (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target in (_ATEN.alias.default, _ATEN.detach.default)
    ):
        node = node.args[0]
    return node


def _node_tensor_val(graph: torch.fx.Graph, node: object) -> torch.Tensor | None:
    if not isinstance(node, Node):
        return None
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val
    if (
        node.op == "get_attr"
        and isinstance(node.target, str)
        and graph.owning_module is not None
    ):
        attr = getattr(graph.owning_module, node.target, None)
        if isinstance(attr, torch.Tensor):
            return attr
    return None


def _known_eq(a, b) -> bool:
    return bool(statically_known_true(a == b))


def _known_ge(a, b) -> bool:
    return bool(statically_known_true(a >= b))


def _same_2d_shape(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dim() == 2
        and b.dim() == 2
        and _known_eq(a.shape[0], b.shape[0])
        and _known_eq(a.shape[1], b.shape[1])
    )


def _is_contiguous_2d_rows(t: torch.Tensor, cols: int) -> bool:
    return (
        t.dim() == 2
        and _known_eq(t.shape[1], cols)
        and _known_eq(t.stride(1), 1)
        and (
            _known_eq(t.stride(0), cols)
            or (_known_eq(t.shape[0], 1) and _known_ge(t.stride(0), cols))
        )
    )


def _same_2d_ancestor(
    node: object, targets: set[Node]
) -> tuple[Node, list[Node]] | None:
    """Walk shape-preserving contiguous 2D BF16 views from ``node`` to ``targets``."""
    if not isinstance(node, Node):
        return None
    chain: list[Node] = []
    cur = node
    seen: set[Node] = set()
    while cur not in targets:
        if (
            cur in seen
            or cur.op != "call_function"
            or cur.target not in _SAME_2D_VIEW_OPS
            or not cur.args
            or not isinstance(cur.args[0], Node)
        ):
            return None
        parent = cur.args[0]
        cur_val = cur.meta.get("val")
        parent_val = parent.meta.get("val")
        if not (
            isinstance(cur_val, torch.Tensor)
            and isinstance(parent_val, torch.Tensor)
            and cur_val.dtype == torch.bfloat16
            and parent_val.dtype == torch.bfloat16
            and _same_2d_shape(cur_val, parent_val)
            and _known_eq(cur_val.stride(1), 1)
            and _known_eq(parent_val.stride(1), 1)
        ):
            return None
        seen.add(cur)
        chain.append(cur)
        cur = parent
    return cur, chain


def _direct_node_deps(node: Node) -> list[Node]:
    deps: list[Node] = []

    def visit(x: Node) -> Node:
        deps.append(x)
        return x

    map_arg((node.args, node.kwargs), visit)
    return deps


def _is_pure_scalar_node(node: Node) -> bool:
    if node.op != "call_function" or node.target in _AF_HOPS:
        return False
    val = node.meta.get("val")
    return isinstance(
        val,
        (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool),
    )


def _hoist_before(
    anchor: Node,
    roots: list[Node],
    allowed_alloc: Node,
    node_pos: dict[Node, int],
) -> None:
    """Hoist ``roots`` and any pure post-anchor dependencies before ``anchor``."""
    anchor_pos = node_pos[anchor]
    to_hoist: set[Node] = set()
    stack = [r for r in roots if isinstance(r, Node) and node_pos[r] >= anchor_pos]
    while stack:
        n = stack.pop()
        if n in to_hoist:
            continue
        if n is anchor:
            raise _Skip("cyclic dependency on add_norm")
        movable = (
            n is allowed_alloc
            or n.op == "get_attr"
            or _is_pure_scalar_node(n)
            or (n.op == "call_function" and n.target in _PURE_VIEW_OPS)
        )
        if not movable:
            raise _Skip(f"cannot hoist dependency {n.target} before add_norm")
        to_hoist.add(n)
        for dep in _direct_node_deps(n):
            if node_pos[dep] >= anchor_pos and dep not in to_hoist:
                stack.append(dep)

    for n in sorted(to_hoist, key=lambda x: node_pos[x]):
        anchor.prepend(n)


def _packed_layout_ok(bf16_backend: int, n_router: int, hidden_dim: int) -> bool:
    from torch_xcpu.ops_defs.bf16_linear import Bf16LinearBackend

    if n_router <= 0 or hidden_dim <= 0:
        return False
    if bf16_backend == int(Bf16LinearBackend.INTEL_AMX):
        return n_router % 32 == 0 and hidden_dim % 32 == 0
    if bf16_backend in (
        int(Bf16LinearBackend.ACC),
        int(Bf16LinearBackend.PORTABLE),
    ):
        return n_router % 32 == 0
    return False


def _try_fuse_add_norm(
    graph: torch.fx.Graph,
    add_norm_af: Node,
    fp32_bf16_linears: list[Node],
    node_pos: dict[Node, int],
) -> bool:
    norm_out_info = _mutable_arg_info(add_norm_af, "output", _IDX_ADD)
    res_out_info = _mutable_arg_info(add_norm_af, "residual_out", _IDX_ADD)
    if norm_out_info is None or res_out_info is None:
        raise _Skip("add-norm mutable output buffers not standalone")
    old_norm_out_buf, norm_out_gi_idx = norm_out_info
    old_res_out_buf, res_out_gi_idx = res_out_info

    hidden_gis: list[Node] = []
    residual_gis: list[Node] = []
    for u in add_norm_af.users:
        if _is_getitem(u) and u.args[1] == norm_out_gi_idx:
            hidden_gis.append(u)
        elif _is_getitem(u) and u.args[1] == res_out_gi_idx:
            residual_gis.append(u)
        else:
            raise _Skip("unexpected user on fused_add_rms_norm_out_bf16")
    if not hidden_gis:
        return False

    hidden_gi_set = set(hidden_gis)
    matched_gates: list[tuple[Node, list[Node]]] = []
    for lin_af in fp32_bf16_linears:
        if lin_af.graph is not graph:
            continue
        anc = _same_2d_ancestor(lin_af.kwargs.get("input"), hidden_gi_set)
        if anc is not None:
            matched_gates.append((lin_af, anc[1]))

    if not matched_gates:
        # Normal for attention input_layernorm, dense FFN layers, or SP-chunked
        # MoE layers where a slice sits between add-norm and router gate.
        return False
    if len(matched_gates) > 1:
        raise _Skip(
            f"ambiguous: {len(matched_gates)} fp32 bf16_linear ops read one add-norm"
        )

    gate_af, gate_view_chain = matched_gates[0]
    bf16_backend = _BF16_LINEAR_OPS[gate_af.args[0]]
    if not bool(
        torch.ops.torch_xcpu.fused_add_norm_router_gemm_backend_supported(
            bf16_backend
        )
    ):
        raise _Skip(f"bf16 backend {bf16_backend} unsupported by fused operator")

    gate_out_info = _mutable_arg_info(
        gate_af, "out", {"out": _IDX_LINEAR_OUT}
    )
    if gate_out_info is None:
        raise _Skip("router bf16_linear out buffer not standalone")
    gate_out_buf, gate_out_gi_idx = gate_out_info

    gate_out_gis: list[Node] = []
    for u in gate_af.users:
        if _is_getitem(u) and u.args[1] == gate_out_gi_idx:
            gate_out_gis.append(u)
        else:
            raise _Skip("unexpected user on router bf16_linear")
    if not gate_out_gis:
        return False

    router_weight = _strip_alias(gate_af.kwargs["packed_weight"])
    rw_val = _node_tensor_val(graph, router_weight)
    if not (
        isinstance(rw_val, torch.Tensor)
        and rw_val.dim() == 2
        and isinstance(rw_val.shape[0], int)
        and isinstance(rw_val.shape[1], int)
        and rw_val.dtype == torch.bfloat16
        and rw_val.is_contiguous()
        and rw_val.device.type in ("mcpu", "privateuseone")
    ):
        raise _Skip("router packed_weight metadata invalid")
    n_router, hidden_dim = int(rw_val.shape[0]), int(rw_val.shape[1])
    if not _packed_layout_ok(bf16_backend, n_router, hidden_dim):
        raise _Skip(
            f"packed layout unsupported for backend={bf16_backend}, "
            f"N={n_router}, H={hidden_dim}"
        )

    hidden_in = add_norm_af.kwargs["input"]
    residual_in = add_norm_af.kwargs["residual"]
    if _strip_alias(hidden_in) is _strip_alias(residual_in):
        raise _Skip("hidden and residual alias the same tensor")
    add_norm_weight = _strip_alias(add_norm_af.kwargs["weight"])
    add_norm_eps = add_norm_af.kwargs["epsilon"]

    hidden_val = _node_tensor_val(graph, hidden_in)
    residual_val = _node_tensor_val(graph, residual_in)
    hidden_gi_val = _node_tensor_val(graph, hidden_gis[0])
    norm_w_val = _node_tensor_val(graph, add_norm_weight)
    if not (
        isinstance(hidden_val, torch.Tensor)
        and isinstance(residual_val, torch.Tensor)
        and isinstance(hidden_gi_val, torch.Tensor)
        and isinstance(norm_w_val, torch.Tensor)
    ):
        raise _Skip("add-norm FakeTensor metadata missing")

    if isinstance(hidden_val.shape[0], int) and hidden_val.shape[0] <= 0:
        raise _Skip("non-positive static token count")
    if _known_eq(hidden_val.shape[0], 0):
        raise _Skip("zero token count")

    if not (
        hidden_val.dtype == torch.bfloat16
        and residual_val.dtype == torch.bfloat16
        and hidden_val.device == rw_val.device
        and residual_val.device == rw_val.device
        and _same_2d_shape(hidden_val, residual_val)
        and _same_2d_shape(hidden_val, hidden_gi_val)
        and _is_contiguous_2d_rows(hidden_val, hidden_dim)
        and _is_contiguous_2d_rows(residual_val, hidden_dim)
    ):
        raise _Skip("hidden/residual shape or layout incompatible")

    if not (
        norm_w_val.dim() == 1
        and _known_eq(norm_w_val.shape[0], hidden_dim)
        and norm_w_val.dtype == torch.bfloat16
        and norm_w_val.is_contiguous()
        and norm_w_val.device == rw_val.device
    ):
        raise _Skip("add-norm weight incompatible")

    gate_out_val = gate_out_buf.meta.get("val")
    if not (
        isinstance(gate_out_buf, Node)
        and gate_out_buf.op == "call_function"
        and gate_out_buf.target in _EMPTY_ALLOC_OPS
        and len(gate_out_buf.users) == 1
        and isinstance(gate_out_val, torch.Tensor)
        and gate_out_val.dim() == 2
        and gate_out_val.dtype == torch.float32
        and gate_out_val.device == rw_val.device
        and _known_ge(gate_out_val.shape[0], hidden_val.shape[0])
        and _known_eq(gate_out_val.shape[1], n_router)
        and _known_eq(gate_out_val.stride(1), 1)
        and _known_eq(gate_out_val.stride(0), n_router)
    ):
        raise _Skip("router_logits out buffer incompatible")

    raw_router_weight = gate_af.kwargs["packed_weight"]
    raw_norm_weight = add_norm_af.kwargs["weight"]
    _hoist_before(
        add_norm_af,
        [gate_out_buf, router_weight, add_norm_weight],
        allowed_alloc=gate_out_buf,
        node_pos=node_pos,
    )

    with graph.inserting_before(add_norm_af):
        fused = graph.create_node(
            "call_function",
            _AF,
            (_FUSED,),
            dict(
                hidden=hidden_in,
                residual=residual_in,
                add_norm_weight=add_norm_weight,
                add_norm_eps=add_norm_eps,
                router_weight=router_weight,
                bf16_backend=bf16_backend,
                router_logits=gate_out_buf,
            ),
        )
        fused.meta["val"] = (None, hidden_gi_val, residual_val, gate_out_val)
        new_hidden = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX_FUSED["hidden"])
        )
        new_hidden.meta["val"] = hidden_gi_val
        new_residual = None
        if residual_gis:
            new_residual = graph.create_node(
                "call_function", _GETITEM, (fused, _IDX_FUSED["residual"])
            )
            if "val" in residual_gis[0].meta:
                new_residual.meta["val"] = residual_gis[0].meta["val"]
        new_logits = graph.create_node(
            "call_function", _GETITEM, (fused, _IDX_FUSED["router_logits"])
        )
        if "val" in gate_out_gis[0].meta:
            new_logits.meta["val"] = gate_out_gis[0].meta["val"]

    for gi in gate_out_gis:
        gi.replace_all_uses_with(new_logits)
        graph.erase_node(gi)
    graph.erase_node(gate_af)
    for view_node in gate_view_chain:
        if not view_node.users:
            graph.erase_node(view_node)

    for gi in hidden_gis:
        gi.replace_all_uses_with(new_hidden)
        graph.erase_node(gi)
    if new_residual is not None:
        for gi in residual_gis:
            gi.replace_all_uses_with(new_residual)
            graph.erase_node(gi)
    graph.erase_node(add_norm_af)
    for buf in (old_norm_out_buf, old_res_out_buf):
        if (
            isinstance(buf, Node)
            and buf.op == "call_function"
            and buf.target in _EMPTY_ALLOC_OPS
            and not buf.users
        ):
            graph.erase_node(buf)
    for w_node in (raw_router_weight, raw_norm_weight):
        cur = w_node
        while (
            isinstance(cur, Node)
            and cur.op == "call_function"
            and cur.target in (_ATEN.alias.default, _ATEN.detach.default)
            and not cur.users
        ):
            nxt = cur.args[0]
            graph.erase_node(cur)
            cur = nxt

    return True


def _is_fp32_bf16_linear_af(node: Node) -> bool:
    if not _is_bf16_linear_af(node) or node.kwargs.get("bias") is not None:
        return False
    out_info = _mutable_arg_info(node, "out", {"out": _IDX_LINEAR_OUT})
    if out_info is None:
        return False
    out_val = out_info[0].meta.get("val")
    return isinstance(out_val, torch.Tensor) and out_val.dtype == torch.float32


def replace_with_fused_add_norm_router_gemm(graph: torch.fx.Graph) -> int:
    if not _any_backend_supported():
        logger.info(
            "fused_add_norm_router_gemm pass: no supported BF16 backend on "
            "this build, graph left unfused"
        )
        return 0

    add_norm_afs = [n for n in graph.nodes if _is_af_of(n, _ADD_NORM)]
    if not add_norm_afs:
        return 0

    fp32_bf16_linears = [
        n for n in graph.nodes if _is_fp32_bf16_linear_af(n)
    ]
    if not fp32_bf16_linears:
        return 0

    replaced = 0
    for add_norm_af in add_norm_afs:
        if add_norm_af.graph is not graph:
            continue
        node_pos = {n: i for i, n in enumerate(graph.nodes)}
        try:
            replaced += bool(
                _try_fuse_add_norm(
                    graph, add_norm_af, fp32_bf16_linears, node_pos
                )
            )
        except _Skip as reason:
            logger.info(
                "fused_add_norm_router_gemm pass: layer skipped (%s)", reason
            )
    return replaced
