# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2 pattern: the MLA front half plus its input add-norm and the
fused_qkv_a FP8 linear, fused into one operator -- one of two independent
operators, picked per layer.

Extends the phase-1 window (:mod:`_front_window`) upwards:

    h, res = fused_add_rms_norm_out_bf16(prev_out, running_residual)
    qkv_a = fp8_mm(h, fused_qkv_a_proj packed weights)   # [T, 2048+512+64]
    <phase-1 window: q/kv RMSNorm, q_b_proj, RoPE, KV-cache write, W_UK>

Layers carrying the model's sparse indexer (GLM-5.2's ``full`` layers) add
two projections off intermediates this window computes anyway:

    indexer_q  = fp8_mm(q_c_norm, indexer.wq_b_proj packed weights)  # [T, 4096]
    indexer_kw = bf16_linear(hidden_norm, indexer.wk_weights_proj
                             packed weight)                          # [T, 160]

and are rewritten to ``fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_
cache``: an operator of their own whose five indexer arguments (``w_wq`` /
``w_wq_scale`` / ``w_wk`` plus the ``indexer_q`` / ``indexer_kw``
workspaces) are required -- it has no indexer-free form -- and which
returns ``(hidden, residual, q_c_norm, out_q, indexer_q, indexer_kw)``.
Layers without a sparse indexer (GLM-5.2's ``shared`` layers -- 57 of its
78, keyed by ``indexer_types``) have neither node and are rewritten to
``fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache``, which takes no indexer
arguments at all and returns ``(hidden, residual, q_c_norm, kv_c_norm,
out_q)``. The two operators are unrelated at the interface level --
neither is the other's optional-argument form -- each has its own
``*_backend_supported`` gate, and this pass picks per layer. Seeing
exactly one of the two indexer nodes is a shape/model mismatch and skips
the window.

Both return through their mutated-arg clones (the raw ops return nothing):
``hidden`` holds the normalized token rows the qkv_a linear used to
consume, ``residual`` the updated running residual the next layer's
add-norm consumes.

``wq_b`` is an fp8 mm with q_b_proj's packed shapes reading the same q_norm
output (told apart from q_b_proj by identity: q_b is the one feeding the
W_UK einsum); ``wk`` is the BF16 linear takeover's
``bf16_linear_<backend>_out`` call on the add-norm output, and the wk
operator receives its ``packed_weight`` -- the [160, 6144] checkpoint bf16
prepacked by the layer's ``Bf16LinearParams`` backend at
``process_weights_after_loading``, after which the raw checkpoint weight no
longer exists. The graph's aligned out buffers of the two absorbed GEMMs
are reused as the wk operator's ``indexer_q`` / ``indexer_kw`` workspaces
(only rows ``[0:T]`` are defined, which is all the consumers read, before
and after the rewrite).

The pattern requires the input norm to be ``fused_add_rms_norm_out_bf16``
(the L2..LN layers). The first layer normalizes the embedding with a plain
``rms_norm_bf16`` and no residual add, so it is skipped here and left to
the phase-1 pass. The post-attention ``fused_add_rms_norm_chunk_out_bf16``
variant never appears on this def-use path.

The FP8 backend is not assumed: it is read off the matched
``fp8_scaled_mm_<backend>_out`` overloads, fused_qkv_a / q_b_proj / wq_b
must share one (the wk operator reads every packed weight of this window
with a single instantiation), and the picked operator's
``*_backend_supported`` gate decides per layer. The BF16 backend of ``wk``
is read off the matched ``bf16_linear_<backend>_out`` overload the same
way; the fp8 x bf16 pair must pass the wk operator's two-argument
``*_backend_supported`` gate, whose pairing is by platform family
(intel_amx x intel_amx, portable-family x portable, acc x acc -- ``onednn``
has no pair and stays unfused). Shapes must be the bias-free
[2624, 6144] fused_qkv_a with [21, 48] scales (block 128); other shapes
stay unfused.

With sequence-parallel MoE the layer input is a ``[0:T]`` slice of the TP
all-gather (``all_gather_into_tensor_out_v2``) of the previous MoE's
per-rank shard. That gather is absorbed too (both operators take
``ag_send`` / ``ag_comm``): the fused operator receives ``ag_send`` (the
shard) and ``ag_comm`` (the graph's communicator-wrapper placeholder, so
the MPI handle stays a runtime value) and ``hidden`` becomes the gather's
landing buffer (``world_size * shard_rows`` rows). The collective then runs
on the mega-kernel's helper thread while the weight DMAs are in flight,
instead of as a separate graph node. Windows whose add-norm input is not
such a slice (e.g. TP1) fall back to the non-absorbed form.
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
_FUSED_WKWQ = (
    torch.ops.torch_xcpu.fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache.default
)

_HIDDEN = 6144
_W_QKVA_SHAPE = (_fw._QKV_WIDTH, _HIDDEN)
_W_QKVA_SCALE_SHAPE = (
    -(-_fw._QKV_WIDTH // 128),  # ceil: 2624 -> 21 scale rows
    -(-_HIDDEN // 128),  # 6144 -> 48 scale cols
)
# indexer.wq_b_proj packs like q_b_proj ([4096, 2048], scales [32, 16]) --
# the same contract the wk operator enforces statically.
_W_WQ_SHAPE = _fw._QB_PACKED_SHAPE
_W_WQ_SCALE_SHAPE = _fw._QB_SCALE_SHAPE
# indexer.wk_weights_proj: the BF16 linear takeover packed the [160, 6144]
# checkpoint bf16 into the layer's Bf16LinearParams backend at load time;
# this is the shape of that packed buffer, viewed as [N, K].
_W_WK_SHAPE = (160, _HIDDEN)


def _bf16_linear_ops() -> dict:
    from torch_xcpu.ops_defs.bf16_linear import _BACKEND_OP_SUFFIXES

    return {
        getattr(torch.ops.torch_xcpu, f"bf16_linear_{suffix}_out").default: int(
            backend
        )
        for backend, suffix in _BACKEND_OP_SUFFIXES.items()
    }


# The BF16 linear takeover executes the indexer wk projection as a
# bf16_linear_<backend>_out call (Bf16Linear.run -> self._linear_op), one
# overload per backend. Matching accepts all of them; the backend int is
# read back off the matched op, the same way _front_window reads the fp8
# backend off the fp8_scaled_mm_<backend>_out overloads.
_BF16_LINEAR_OPS = _bf16_linear_ops()


def _is_bf16_linear_af(node: Node) -> bool:
    """auto_functionalized call of any bf16_linear_<backend>_out overload."""
    return (
        node.op == "call_function"
        and node.target == _fw._AF
        and node.args[0] in _BF16_LINEAR_OPS
    )


_IDX = _fw._af_indices(_FUSED)
_IDX_WKWQ = _fw._af_indices(_FUSED_WKWQ)
_IDX_ADD = _fw._af_indices(_ADD_NORM)
# af result index of the ``out`` clone every single-out linear op in the
# window has (fp8_scaled_mm_<b>_out, bf16_linear_<b>_out): all are -> ()
# ops whose only mutated arg is ``out``.
_IDX_LINEAR_OUT = _fw._af_indices(_FP8_MM)["out"]
_IDX_CACHE_KV = _fw._af_indices(_CACHE_WRITE)["kv_cache"]
_IDX_AG_OUT = _fw._af_indices(_ALL_GATHER)["output"]


def _shared_backend_supported(backend: int) -> bool:
    """The shared-layer operator's gate for one fp8 backend."""
    gate = (
        torch.ops.torch_xcpu
        .fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache_backend_supported
    )
    return _fw.backend_supported(gate, backend)


def _wkwq_backend_supported(fp8_backend: int, bf16_backend: int) -> bool:
    """The sparse-indexer operator's gate for one (fp8, bf16) backend pair."""
    gate = (
        torch.ops.torch_xcpu
        .fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache_backend_supported
    )
    return bool(gate(int(fp8_backend), int(bf16_backend)))


def _backend_supported() -> bool:
    """Graph-level early-out: either operator's gate passing is enough to
    enter the per-layer matching; every window re-checks the operator it
    picks with the fp8 x bf16 backends the graph actually uses. The shared
    operator's gate takes one fp8 backend, the sparse-indexer one a family
    -paired (fp8, bf16) pair."""
    from torch_xcpu.ops_defs.bf16_linear import Bf16LinearBackend
    from torch_xcpu.ops_defs.fp8_linear import Fp8LinearBackend

    shared_gate = (
        torch.ops.torch_xcpu
        .fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache_backend_supported
    )
    wkwq_gate = (
        torch.ops.torch_xcpu
        .fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache_backend_supported
    )
    for fp8 in Fp8LinearBackend:
        if bool(shared_gate(int(fp8))):
            return True
        for bf16 in Bf16LinearBackend:
            if bool(wkwq_gate(int(fp8), int(bf16))):
                return True
    return False


def _replace_one(graph, cache_af: Node) -> bool:
    m = _fw.match_window(graph, cache_af)

    # Upstream: the fused_qkv_a FP8 linear the two rms_norms read. Its fp8
    # mm overload names the backend; the picked operator reads every packed
    # weight of this window with that one backend's layout.
    qkva_af = m.qkv_base.args[0]
    if not _fw._is_fp8_mm_af(qkva_af):
        raise _SKIP("qkv source is not a fp8_scaled_mm")
    fp8_backend = _fw._fp8_mm_backend(qkva_af)
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

    # The layer's two sparse-indexer projections. wq_b: an fp8 mm with q_b's
    # packed shapes reading the same q_norm output (q_b_proj itself is
    # excluded by identity -- it is the one feeding the W_UK einsum). wk: a
    # bf16_linear_<backend>_out call of the BF16 linear takeover on the
    # add-norm output; its packed_weight is the [160, 6144] checkpoint bf16
    # after the backend's prepack. A layer with the sparse indexer has both
    # nodes and routes to fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_
    # cache; a shared layer has neither and routes to
    # fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache.
    wq_b_af = None
    for n in graph.nodes:
        if n is m.q_b_af or not _fw._is_fp8_mm_af(n):
            continue
        if not _fw._roots_at(n.kwargs.get("input"), {m.q_norm_out}):
            continue
        if (
            _fw._shape(n.kwargs["packed_weight"]) != _W_WQ_SHAPE
            or _fw._shape(n.kwargs["weight_scale"]) != _W_WQ_SCALE_SHAPE
        ):
            continue
        if wq_b_af is not None:
            raise _SKIP("ambiguous indexer wq_b linear")
        wq_b_af = n
    wq_b_out = None
    if wq_b_af is not None:
        if wq_b_af.kwargs.get("bias") is not None:
            raise _SKIP("indexer wq_b linear has a bias")
        if wq_b_af.args[0] is not qkva_af.args[0]:
            raise _SKIP("indexer wq_b backend differs from fused_qkv_a")
        wq_b_out = next(
            (
                u
                for u in wq_b_af.users
                if _fw._is_getitem(u) and u.args[1] == _IDX_LINEAR_OUT
            ),
            None,
        )
        if wq_b_out is None:
            raise _SKIP("indexer wq_b result missing")

    wk_af = None
    for n in graph.nodes:
        if not _is_bf16_linear_af(n):
            continue
        if n.kwargs.get("bias") is not None:
            continue  # not the takeover's wk linear (wk_weights_proj has none)
        if _fw._shape(n.kwargs["packed_weight"]) != _W_WK_SHAPE:
            continue
        if not _fw._roots_at(n.kwargs.get("input"), {hidden_gi}):
            continue
        if wk_af is not None:
            raise _SKIP("ambiguous indexer wk linear")
        wk_af = n
    w_wk = None
    bf16_backend = None
    wk_out = None
    if wk_af is not None:
        bf16_backend = _BF16_LINEAR_OPS[wk_af.args[0]]
        w_wk = wk_af.kwargs["packed_weight"]
        _fw._check_shape("wk out", wk_af.kwargs["out"], (None, _W_WK_SHAPE[0]))
        wk_out = next(
            (
                u
                for u in wk_af.users
                if _fw._is_getitem(u) and u.args[1] == _IDX_LINEAR_OUT
            ),
            None,
        )
        if wk_out is None:
            raise _SKIP("indexer wk result missing")

    # All-or-nothing, mirroring the two operators' interface split: a shared
    # layer has neither node; exactly one is a model/shape mismatch.
    if (wq_b_af is None) != (wk_af is None):
        raise _SKIP("only one of the two indexer projections present")
    full = wq_b_af is not None
    # full 与 wk_af 同真同假（上面的 XOR 检查），直接收窄 bf16_backend。
    if full and bf16_backend is not None:
        if not _wkwq_backend_supported(fp8_backend, bf16_backend):
            raise _SKIP(
                f"backend pair {fp8_backend} x {bf16_backend} unsupported by "
                "fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache"
            )
    elif not _shared_backend_supported(fp8_backend):
        raise _SKIP(
            f"fp8 backend {fp8_backend} unsupported by "
            "fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache"
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
    redirects: dict[Node, Node | None] = {
        hidden_gi: None,
        m.q_norm_out: None,
        cache_out: None,
        m.cat: None,
    }
    if residual_gi is not None:
        redirects[residual_gi] = None
    if wq_b_out is not None:
        redirects[wq_b_out] = None
    if wk_out is not None:
        redirects[wk_out] = None
    region = _fw.close_region(
        graph,
        [
            *([gather_af] if gather_af is not None else []),
            add_norm_af,
            qkva_af,
            m.q_norm_af,
            m.kv_norm_af,
            m.q_b_af,
            *([wq_b_af] if wq_b_af is not None else []),
            *([wk_af] if wk_af is not None else []),
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
        # qkv workspace: the linear kernels write whole 16-row groups, so the
        # buffer must hold aligned_linear_rows(tokens) rows. The qkv_a linear's
        # own out buffer is exactly that ([16*ceil(T/16), 2624]); on the full
        # layers the two absorbed indexer GEMMs' out buffers are the same deal
        # for the indexer_q ([16*ceil(T/16), 4096]) / indexer_kw ([16*ceil(T/16),
        # 160]) workspaces.
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
        fused_target = _FUSED_WKWQ if full else _FUSED
        fused_kwargs = dict(
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
        )
        if full and wq_b_af is not None and wk_af is not None:
            # The wk_wq operator's five indexer arguments are required -- it
            # has no indexer-free form to leave them out of. w_wk is the BF16
            # linear backend's packed weight (the graph's bf16_linear node
            # already reads it), and bf16_backend tells the operator which
            # packed layout to expect for it.
            fused_kwargs.update(
                w_wq=wq_b_af.kwargs["packed_weight"],
                w_wq_scale=wq_b_af.kwargs["weight_scale"],
                w_wk=w_wk,
                bf16_backend=bf16_backend,
                indexer_q=wq_b_af.kwargs["out"],
                indexer_kw=wk_af.kwargs["out"],
            )
        fused = graph.create_node(
            "call_function", _AF, (fused_target,), fused_kwargs
        )
        idx = _IDX_WKWQ if full else _IDX
        hidden_out = graph.create_node(
            "call_function", _GETITEM, (fused, idx["hidden"])
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
            "call_function", _GETITEM, (fused, idx["kv_cache"])
        )
        redirects[m.q_norm_out] = graph.create_node(
            "call_function", _GETITEM, (fused, idx["q_c_norm"])
        )
        if residual_gi is not None:
            redirects[residual_gi] = graph.create_node(
                "call_function", _GETITEM, (fused, idx["residual"])
            )
        redirects[m.cat] = graph.create_node(
            "call_function", _GETITEM, (fused, idx["out_q"])
        )
        if full:
            assert wq_b_out is not None and wk_out is not None
            redirects[wq_b_out] = graph.create_node(
                "call_function", _GETITEM, (fused, idx["indexer_q"])
            )
            redirects[wk_out] = graph.create_node(
                "call_function", _GETITEM, (fused, idx["indexer_kw"])
            )

    _fw.rebase_and_erase(graph, region, redirects)
    return True


def replace_with_fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache(graph) -> int:
    if not _backend_supported():
        logger.info(
            "deepseek_v32 pass: fused_mla_add_norm_qkva[_wk_wq]_qnorm_qb_uk"
            "_rope_cache backend unsupported on this build, graph left unfused"
        )
        return 0
    cache_afs = [n for n in list(graph.nodes) if _fw._is_af_of(n, _CACHE_WRITE)]
    if not cache_afs:
        return 0
    replaced = 0
    for cache_af in cache_afs:
        try:
            replaced += bool(_replace_one(graph, cache_af))
        except _SKIP as reason:
            logger.info(
                "deepseek_v32 fused_mla_add_norm[_wk_wq]_qnorm_qb_uk_rope"
                "_cache pass: layer skipped (%s)",
                reason,
            )
    return replaced
