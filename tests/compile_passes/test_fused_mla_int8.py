# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-V3.2 INT8 [16, 16] UE8M0 + int8_ds_mla compile passes."""

from __future__ import annotations

import uuid

import pytest
import torch
import torch.nn as nn
import torch_xcpu  # noqa: F401
from torch_xcpu import ops as xcpu_ops
from torch_xcpu.ops_defs.bf16_linear import Bf16LinearBackend
from torch_xcpu.ops_defs.fp8_linear import Fp8LinearBackend
from torch_xcpu.ops_defs.int8_linear import Int8LinearBackend
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.plugins import load_general_plugins

from vllm_xcpu_plugin import envs as xcpu_envs
from vllm_xcpu_plugin import ir_kernels as _ir_kernels  # noqa: F401
from vllm_xcpu_plugin.compile_passes import XcpuPostGradPass, pass_uuid

load_general_plugins()

pytestmark = pytest.mark.skipif(
    not xcpu_ops.int8_scaled_mm_supported(),
    reason="INT8 linear backend unavailable on this build",
)

_AF = torch.ops.higher_order.auto_functionalized
_RMS_NORM_OP = torch.ops.torch_xcpu.rms_norm_bf16.default
_FUSED_P1_INT8 = torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache_int8.default
_FUSED_P2_INT8 = (
    torch.ops.torch_xcpu.fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache_int8.default
)
_FUSED_P2_WKWQ_INT8 = (
    torch.ops.torch_xcpu
    .fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache_int8
    .default
)
_FUSED_P1_FP8 = torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache.default

_HEADS = 16
_HIDDEN = 6144
_Q_LORA = 2048
_KV_LORA = 512
_NOPE = 192
_ROPE = 64
_HEAD_DIM = _NOPE + _ROPE
_QKV_WIDTH = _Q_LORA + _KV_LORA + _ROPE
_N_QB = _HEADS * _HEAD_DIM
_N_WK = 160
_EPS = 1e-6
_PAGE_SIZE = 16
_PAGES = 4


@pytest.fixture
def compile_vllm_env(monkeypatch):
    import vllm.ir as ir
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setenv("VLLM_USE_XCPU_LINEAR", "1")
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    cfg = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    with (
        set_current_vllm_config(cfg),
        ir.enable_torch_wrap(False),
        ir.ops.rms_norm.set_priority(["torch_xcpu", "native"]),
        ir.ops.fused_add_rms_norm.set_priority(["torch_xcpu", "native"]),
    ):
        yield cfg


class _CapturingPostPass(XcpuPostGradPass):
    def __init__(self):
        super().__init__()
        self._instance_id = uuid.uuid4().hex
        self.af_ops: list[list[object]] = []

    def uuid(self):
        return (super().uuid(), self._instance_id)

    def __call__(self, graph) -> None:
        super().__call__(graph)
        self.af_ops.append([
            n.args[0]
            for n in graph.nodes
            if n.op == "call_function" and n.target == _AF and n.args
        ])


def _pick_int8_backend() -> Int8LinearBackend:
    for backend in (
        Int8LinearBackend.INTEL_AMX,
        Int8LinearBackend.PORTABLE_LUT,
        Int8LinearBackend.ACC,
    ):
        if xcpu_ops.int8_scaled_mm_supported(backend) and bool(
            torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache_int8_backend_supported(
                int(backend)
            )
        ):
            return backend
    pytest.skip("No supported INT8 MLA mega-kernel backend")


def _pick_bf16_pair(int8_backend: Int8LinearBackend) -> Bf16LinearBackend:
    gate = (
        torch.ops.torch_xcpu
        .fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache_int8_backend_supported
    )
    for bf16 in (
        Bf16LinearBackend.INTEL_AMX,
        Bf16LinearBackend.ACC,
        Bf16LinearBackend.PORTABLE,
    ):
        if xcpu_ops.bf16_linear_supported(bf16) and bool(
            gate(int(int8_backend), int(bf16))
        ):
            return bf16
    pytest.skip(f"No compatible BF16 backend for INT8 backend {int8_backend}")


def _make_int8_linear(
    n: int,
    k: int,
    gen: torch.Generator,
    backend: Int8LinearBackend,
    block_size: tuple[int, int] = (16, 16),
    scale_dtype: torch.dtype = torch.uint8,
):
    bn, bk = block_size
    w = torch.randint(-64, 65, (n, k), generator=gen, dtype=torch.int8, device="cpu")
    s_u8 = torch.randint(
        117, 123, (n // bn, k // bk), generator=gen, dtype=torch.uint8, device="cpu"
    )
    scale = s_u8 if scale_dtype == torch.uint8 else s_u8.view(torch.float8_e8m0fnu)
    return xcpu_ops.initialize_int8_linear(
        w.to("mcpu"), scale.to("mcpu"), block_size, backend=backend
    )


def _alloc_cache(kv_cache_dtype: str):
    assert kv_cache_dtype in ("fp8_ds_mla", "int8_ds_mla")
    storage = torch.full((_PAGES, _PAGE_SIZE, 656), 3, dtype=torch.uint8, device="mcpu")
    return storage.clone(), storage.clone()


def _make_positions_and_slots(tokens: int):
    positions = (torch.arange(tokens, dtype=torch.int64, device="cpu") * 3 % 29).to(
        "mcpu"
    )
    slots = torch.arange(tokens, dtype=torch.int64, device="cpu")
    slots = (_PAGES - 1 - slots // _PAGE_SIZE) * _PAGE_SIZE + slots % _PAGE_SIZE
    if tokens > 1:
        slots[1] = -1
    return positions, slots.to("mcpu")


def _close(
    actual: torch.Tensor, expected: torch.Tensor, name: str, tol_mul: float = 1 / 16
):
    a = actual.cpu().float()
    e = expected.cpu().float()
    rms = e.pow(2).mean().sqrt()
    tol = e.abs().max() / 1024 + e.abs() / 64 + tol_mul * rms
    bad = ((a - e).abs() > tol).sum().item()
    assert bad == 0, f"{name}: {bad} elements exceeded tolerance"


def _check_q_out(
    actual: torch.Tensor,
    expected: torch.Tensor,
    slots: torch.Tensor,
    tokens: int,
    name: str,
):
    _close(
        actual[..., :_KV_LORA], expected[..., :_KV_LORA], f"{name} ql_nope T={tokens}"
    )
    # Fused MLA ops intentionally leave q_pe un-rotated for negative slots
    # (padding tokens).
    active = slots[:tokens].cpu() >= 0
    _close(
        actual.cpu()[active][..., _KV_LORA:],
        expected.cpu()[active][..., _KV_LORA:],
        f"{name} q_pe T={tokens}",
        tol_mul=1 / 4,
    )


def _check_cache(
    got_cache: torch.Tensor,
    want_cache: torch.Tensor,
    slots: torch.Tensor,
    tokens: int,
    kv_cache_dtype: str,
) -> None:
    written = [int(s) for s in slots[:tokens].cpu().tolist() if s >= 0]
    g_pages = got_cache.cpu().reshape(got_cache.shape[0], -1)
    w_pages = want_cache.cpu().reshape(want_cache.shape[0], -1)
    written_mask = torch.zeros_like(g_pages, dtype=torch.bool)
    for slot in written:
        page, off = slot // _PAGE_SIZE, slot % _PAGE_SIZE
        written_mask[page, off * 640 : (off + 1) * 640] = True
        written_mask[
            page, _PAGE_SIZE * 640 + off * 16 : _PAGE_SIZE * 640 + (off + 1) * 16
        ] = True
        g_row = g_pages[page]
        w_row = w_pages[page]
        if kv_cache_dtype == "fp8_ds_mla":
            g_lat = g_row[off * 640 : off * 640 + 512].view(torch.float8_e4m3fn).float()
            w_lat = w_row[off * 640 : off * 640 + 512].view(torch.float8_e4m3fn).float()
            g_sc = (
                g_row[_PAGE_SIZE * 640 + off * 16 :][:16]
                .view(torch.float32)
                .repeat_interleave(128)
            )
            w_sc = (
                w_row[_PAGE_SIZE * 640 + off * 16 :][:16]
                .view(torch.float32)
                .repeat_interleave(128)
            )
        else:
            g_lat = g_row[off * 640 : off * 640 + 512].view(torch.int8).float()
            w_lat = w_row[off * 640 : off * 640 + 512].view(torch.int8).float()
            g_sc = (
                g_row[_PAGE_SIZE * 640 + off * 16 :][:16]
                .view(torch.float8_e8m0fnu)
                .float()
                .repeat_interleave(32)
            )
            w_sc = (
                w_row[_PAGE_SIZE * 640 + off * 16 :][:16]
                .view(torch.float8_e8m0fnu)
                .float()
                .repeat_interleave(32)
            )
        _close(g_lat * g_sc, w_lat * w_sc, "cache latent", tol_mul=1 / 16)
        _close(
            g_row[off * 640 + 512 : off * 640 + 640].view(torch.bfloat16),
            w_row[off * 640 + 512 : off * 640 + 640].view(torch.bfloat16),
            "cache k_pe",
            tol_mul=1 / 8,
        )
    assert torch.equal(g_pages[~written_mask], w_pages[~written_mask]), (
        "cache unwritten bytes modified"
    )


def _mla_front_window_unfused(
    qkv_lora: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    q_a_w: torch.Tensor,
    kv_a_w: torch.Tensor,
    qb_linear,
    w_uk_t: torch.Tensor,
    kv_cache_dtype: str,
    is_neox: bool,
    *,
    hidden_for_indexer: torch.Tensor | None = None,
    wq_linear=None,
    wk_linear=None,
):
    q_c, kv_lora = qkv_lora.split([_Q_LORA, _KV_LORA + _ROPE], dim=-1)
    q_c_norm = torch.empty_like(q_c)
    xcpu_ops.rms_norm(q_c_norm, q_c, q_a_w, _EPS)
    kv_c, k_pe = kv_lora.split([_KV_LORA, _ROPE], dim=-1)
    kv_c_norm = torch.empty_like(kv_c)
    xcpu_ops.rms_norm(kv_c_norm, kv_c, kv_a_w, _EPS)
    k_pe = k_pe.unsqueeze(1)
    q = qb_linear(q_c_norm).view(-1, _HEADS, _HEAD_DIM)
    xcpu_ops.rotary_embedding(positions, q[..., _NOPE:], k_pe, _ROPE, cos_sin, is_neox)
    q_nope = q[..., :_NOPE]
    q_pe = q[..., _NOPE:]
    extra = ()
    if wq_linear is not None and wk_linear is not None:
        assert hidden_for_indexer is not None
        extra = (wq_linear(q_c_norm), wk_linear(hidden_for_indexer))
    xcpu_ops.reshape_and_cache(
        kv_c_norm,
        k_pe.squeeze(1),
        kv_cache,
        slot_mapping,
        kv_cache_dtype=kv_cache_dtype,
    )
    q_nope_uk = torch.empty(
        (q.shape[0], _HEADS, _KV_LORA), dtype=q.dtype, device=q.device
    )
    xcpu_ops.einsum_mhk_hkn_to_mhn(q_nope, w_uk_t, q_nope_uk)
    out_q = torch.cat((q_nope_uk, q_pe), dim=-1)
    return q_c_norm, out_q, *extra


class _Phase1Module(nn.Module):
    def __init__(
        self, seed: int, qb_linear, kv_cache_dtype: str, is_neox: bool = False
    ):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.q_a_w = nn.Parameter(
            (torch.randn(_Q_LORA, generator=g) * 0.2 + 1).bfloat16().to("mcpu"),
            requires_grad=False,
        )
        self.kv_a_w = nn.Parameter(
            (torch.randn(_KV_LORA, generator=g) * 0.2 + 1).bfloat16().to("mcpu"),
            requires_grad=False,
        )
        self.w_uk_t = nn.Parameter(
            (torch.randn(_HEADS, _NOPE, _KV_LORA, generator=g) * 0.1)
            .bfloat16()
            .to("mcpu"),
            requires_grad=False,
        )
        self.cos_sin = nn.Parameter(
            torch.randn(29, _ROPE, generator=g).bfloat16().to("mcpu"),
            requires_grad=False,
        )
        self.qb_linear = qb_linear
        self.kv_cache_dtype = kv_cache_dtype
        self.is_neox = is_neox

    def forward(self, hidden, qkv_w_placeholder, positions, slot_mapping, kv_cache):
        # Match a real layer where qkv comes from a preceding linear output:
        # qkva_linear output clone -> rms_norm -> ...
        qkv_lora = self.pre_linear(hidden)
        return _mla_front_window_unfused(
            qkv_lora,
            positions,
            self.cos_sin,
            slot_mapping,
            kv_cache,
            self.q_a_w,
            self.kv_a_w,
            self.qb_linear,
            self.w_uk_t,
            self.kv_cache_dtype,
            self.is_neox,
        )


def test_int8_pass_uuid_changes_with_switches(monkeypatch):
    for name in (
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8",
    ):
        assert name in xcpu_envs.env_variables

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "0",
    )
    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "0",
    )
    u0 = pass_uuid()

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    u1 = pass_uuid()

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    u2 = pass_uuid()
    assert len({u0, u1, u2}) == 3


@pytest.mark.parametrize("kv_cache_dtype", ["fp8_ds_mla", "int8_ds_mla"])
def test_phase1_int8_pass_fuses_and_matches(
    compile_vllm_env, monkeypatch, kv_cache_dtype: str
):
    backend = _pick_int8_backend()
    scale_dtype = (
        torch.float8_e8m0fnu if kv_cache_dtype == "int8_ds_mla" else torch.uint8
    )
    g = torch.Generator(device="cpu").manual_seed(11)
    qkva_lin = _make_int8_linear(
        _QKV_WIDTH, _HIDDEN, g, backend, scale_dtype=scale_dtype
    )
    qb_lin = _make_int8_linear(_N_QB, _Q_LORA, g, backend, scale_dtype=scale_dtype)

    mod = _Phase1Module(12, qb_lin, kv_cache_dtype, is_neox=False)
    mod.pre_linear = qkva_lin

    def run_unfused(hidden, positions, slot_mapping, kv_cache):
        return mod(hidden, None, positions, slot_mapping, kv_cache)

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    torch._dynamo.reset()
    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        run_unfused,
        backend="inductor",
        dynamic=True,
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": False,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    for tokens in (7, 1, 16, 33):
        h_cpu = (torch.randn((tokens, _HIDDEN), generator=g) * 0.5).bfloat16()
        positions, slots = _make_positions_and_slots(tokens)
        got_c, want_c = _alloc_cache(kv_cache_dtype)
        with torch.inference_mode():
            want = run_unfused(h_cpu.to("mcpu"), positions, slots, want_c)
            got = compiled(h_cpu.to("mcpu"), positions, slots, got_c)
            torch.mcpu.synchronize()
        _close(got[0], want[0], f"q_c_norm T={tokens}")
        _check_q_out(got[1], want[1], slots, tokens, "phase1")
        _check_cache(got_c, want_c, slots, tokens, kv_cache_dtype)

    assert post_pass.af_ops
    assert all(ops.count(_FUSED_P1_INT8) == 1 for ops in post_pass.af_ops)
    assert all(_RMS_NORM_OP not in ops for ops in post_pass.af_ops)
    # Only the preceding pre_linear int8_scaled_mm remains; q_b_proj was absorbed.
    assert all(
        sum(1 for op in ops if "int8_scaled_mm_" in str(op)) == 1
        for ops in post_pass.af_ops
    )


class _Phase2Module(nn.Module):
    def __init__(
        self,
        seed: int,
        backend: Int8LinearBackend,
        kv_cache_dtype: str,
        *,
        with_indexer: bool,
        with_sp_gather: bool = False,
    ):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        scale_dtype = (
            torch.float8_e8m0fnu if kv_cache_dtype == "int8_ds_mla" else torch.uint8
        )
        self.add_norm_w = nn.Parameter(
            (torch.randn(_HIDDEN, generator=g) * 0.2 + 1).bfloat16().to("mcpu"),
            requires_grad=False,
        )
        self.q_a_w = nn.Parameter(
            (torch.randn(_Q_LORA, generator=g) * 0.2 + 1).bfloat16().to("mcpu"),
            requires_grad=False,
        )
        self.kv_a_w = nn.Parameter(
            (torch.randn(_KV_LORA, generator=g) * 0.2 + 1).bfloat16().to("mcpu"),
            requires_grad=False,
        )
        self.w_uk_t = nn.Parameter(
            (torch.randn(_HEADS, _NOPE, _KV_LORA, generator=g) * 0.1)
            .bfloat16()
            .to("mcpu"),
            requires_grad=False,
        )
        self.cos_sin = nn.Parameter(
            torch.randn(29, _ROPE, generator=g).bfloat16().to("mcpu"),
            requires_grad=False,
        )
        self.qkva_linear = _make_int8_linear(
            _QKV_WIDTH, _HIDDEN, g, backend, scale_dtype=scale_dtype
        )
        self.qb_linear = _make_int8_linear(
            _N_QB, _Q_LORA, g, backend, scale_dtype=scale_dtype
        )
        self.with_indexer = with_indexer
        self.with_sp_gather = with_sp_gather
        self.kv_cache_dtype = kv_cache_dtype
        if with_indexer:
            bf16_backend = _pick_bf16_pair(backend)
            self.wq_linear = _make_int8_linear(
                _N_QB, _Q_LORA, g, backend, scale_dtype=scale_dtype
            )
            wk_w = (
                (torch.randn(_N_WK, _HIDDEN, generator=g) * 0.1).bfloat16().to("mcpu")
            )
            self.wk_linear = xcpu_ops.initialize_bf16_linear(wk_w, backend=bf16_backend)
        else:
            self.wq_linear = None
            self.wk_linear = None

    def forward(
        self, inp, residual, positions, slot_mapping, kv_cache, comm_wrapper=None
    ):
        if self.with_sp_gather:
            landing = torch.empty_like(inp)
            xcpu_ops.all_gather_into_tensor_out_v2(landing, inp, comm_wrapper, dim=0)
            hidden_in = landing[: residual.shape[0]]
        else:
            hidden_in = inp
        hidden = torch.empty_like(hidden_in)
        res_out = torch.empty_like(residual)
        xcpu_ops.fused_add_rms_norm_out(
            hidden, res_out, hidden_in, residual, self.add_norm_w, _EPS
        )
        qkv_lora = self.qkva_linear(hidden)
        q_c_norm, out_q, *extra = _mla_front_window_unfused(
            qkv_lora,
            positions,
            self.cos_sin,
            slot_mapping,
            kv_cache,
            self.q_a_w,
            self.kv_a_w,
            self.qb_linear,
            self.w_uk_t,
            self.kv_cache_dtype,
            False,
            hidden_for_indexer=hidden,
            wq_linear=self.wq_linear,
            wk_linear=self.wk_linear,
        )
        return hidden, res_out, q_c_norm, out_q, *extra


@pytest.mark.parametrize("with_indexer", [False, True], ids=["shared", "wkwq"])
@pytest.mark.parametrize("kv_cache_dtype", ["fp8_ds_mla", "int8_ds_mla"])
def test_phase2_int8_pass_fuses_and_matches(
    compile_vllm_env, monkeypatch, with_indexer: bool, kv_cache_dtype: str
):
    backend = _pick_int8_backend()
    mod = _Phase2Module(
        40 + int(with_indexer), backend, kv_cache_dtype, with_indexer=with_indexer
    )
    expected_op = _FUSED_P2_WKWQ_INT8 if with_indexer else _FUSED_P2_INT8

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    torch._dynamo.reset()
    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        mod,
        backend="inductor",
        dynamic=True,
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": False,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    g = torch.Generator(device="cpu").manual_seed(55)
    for tokens in (7, 1, 16, 33):
        h_cpu = (torch.randn((tokens, _HIDDEN), generator=g) * 0.5).bfloat16()
        r_cpu = (torch.randn((tokens, _HIDDEN), generator=g) * 0.5).bfloat16()
        positions, slots = _make_positions_and_slots(tokens)
        got_c, want_c = _alloc_cache(kv_cache_dtype)
        with torch.inference_mode():
            want = mod(h_cpu.to("mcpu"), r_cpu.to("mcpu"), positions, slots, want_c)
            got = compiled(h_cpu.to("mcpu"), r_cpu.to("mcpu"), positions, slots, got_c)
            torch.mcpu.synchronize()
        _close(got[0], want[0], f"hidden T={tokens}")
        torch.testing.assert_close(got[1].cpu(), want[1].cpu(), rtol=0, atol=0)
        _close(got[2], want[2], f"q_c_norm T={tokens}")
        _check_q_out(got[3], want[3], slots, tokens, "phase2")
        if with_indexer:
            _close(got[4], want[4], f"indexer_q T={tokens}")
            _close(got[5], want[5], f"indexer_kw T={tokens}")
        _check_cache(got_c, want_c, slots, tokens, kv_cache_dtype)

    assert post_pass.af_ops
    assert all(ops.count(expected_op) == 1 for ops in post_pass.af_ops)
    assert all(
        sum(1 for op in ops if "int8_scaled_mm_" in str(op)) == 0
        for ops in post_pass.af_ops
    )


def _self_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    if not MPI.Is_initialized() or MPI.Query_thread() < MPI.THREAD_SERIALIZED:
        return None
    return MPI.COMM_SELF.py2f()


@pytest.mark.parametrize("with_indexer", [False, True], ids=["shared", "wkwq"])
def test_phase2_int8_absorbs_sp_all_gather(
    compile_vllm_env, monkeypatch, with_indexer: bool
):
    comm = _self_comm()
    if comm is None:
        pytest.skip("MPI is unavailable or below MPI_THREAD_SERIALIZED")

    backend = _pick_int8_backend()
    mod = _Phase2Module(
        80 + int(with_indexer),
        backend,
        "int8_ds_mla",
        with_indexer=with_indexer,
        with_sp_gather=True,
    )
    expected_op = _FUSED_P2_WKWQ_INT8 if with_indexer else _FUSED_P2_INT8

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    torch._dynamo.reset()
    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        mod,
        backend="inductor",
        dynamic=True,
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": False,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    g = torch.Generator(device="cpu").manual_seed(85)
    comm_wrapper = torch.tensor([comm], dtype=torch.int64, device="cpu")
    for tokens in (7, 16, 20):
        shard_cpu = (torch.randn((tokens, _HIDDEN), generator=g) * 0.5).bfloat16()
        r_cpu = (torch.randn((tokens, _HIDDEN), generator=g) * 0.5).bfloat16()
        positions, slots = _make_positions_and_slots(tokens)
        got_c, want_c = _alloc_cache("int8_ds_mla")
        with torch.inference_mode():
            want = mod(
                shard_cpu.to("mcpu"),
                r_cpu.to("mcpu"),
                positions,
                slots,
                want_c,
                comm_wrapper,
            )
            got = compiled(
                shard_cpu.to("mcpu"),
                r_cpu.to("mcpu"),
                positions,
                slots,
                got_c,
                comm_wrapper,
            )
            torch.mcpu.synchronize()
        for idx in range(len(want)):
            if idx == 3:
                _check_q_out(got[idx], want[idx], slots, tokens, "sp")
            else:
                _close(
                    got[idx], want[idx], f"sp output {idx} T={tokens}", tol_mul=1 / 4
                )
        _check_cache(got_c, want_c, slots, tokens, "int8_ds_mla")

    assert post_pass.af_ops
    assert all(ops.count(expected_op) == 1 for ops in post_pass.af_ops)
    assert all(
        torch.ops.torch_xcpu.all_gather_into_tensor_out_v2.default not in ops
        for ops in post_pass.af_ops
    )


def test_two_layer_phase2_then_phase1_int8(compile_vllm_env, monkeypatch):
    backend = _pick_int8_backend()
    g = torch.Generator(device="cpu").manual_seed(91)
    qkva0 = _make_int8_linear(_QKV_WIDTH, _HIDDEN, g, backend)
    qb0 = _make_int8_linear(_N_QB, _Q_LORA, g, backend)
    l0 = _Phase1Module(92, qb0, "int8_ds_mla", is_neox=False)
    l0.pre_linear = qkva0
    l1 = _Phase2Module(93, backend, "int8_ds_mla", with_indexer=False)

    class TwoLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = l0
            self.l1 = l1
            self.bridge_w = nn.Parameter(
                (torch.randn(_HIDDEN, generator=g, device="cpu") * 0.2 + 1)
                .bfloat16()
                .to("mcpu"),
                requires_grad=False,
            )

        def forward(self, x, res, positions, slots, c0, c1):
            _, out0 = self.l0(x, None, positions, slots, c0)
            # In real models, custom attention / o_proj ops sit between layers so
            # Layer 1's backward region closure stops before Layer 0's cat.
            bridged_in = (
                out0[..., :_KV_LORA].reshape(x.shape[0], -1)[:, :_HIDDEN].contiguous()
            )
            bridged = torch.empty_like(bridged_in)
            xcpu_ops.rms_norm(bridged, bridged_in, self.bridge_w, _EPS)
            out0_proj = x * 0.5 + bridged * 0.01
            h1, res1, q1, out1 = self.l1(out0_proj, res, positions, slots, c1)
            return out0, h1, res1, q1, out1

    mod = TwoLayer()
    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    torch._dynamo.reset()
    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        mod,
        backend="inductor",
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": False,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    tokens = 8
    x = (torch.randn((tokens, _HIDDEN), generator=g) * 0.5).bfloat16().to("mcpu")
    res = (torch.randn((tokens, _HIDDEN), generator=g) * 0.5).bfloat16().to("mcpu")
    positions, slots = _make_positions_and_slots(tokens)
    g0, w0 = _alloc_cache("int8_ds_mla")
    g1, w1 = _alloc_cache("int8_ds_mla")
    with torch.inference_mode():
        want = mod(x, res, positions, slots, w0, w1)
        got = compiled(x, res, positions, slots, g0, g1)
        torch.mcpu.synchronize()
    for i in range(len(want)):
        if i in (0, 4):
            _check_q_out(got[i], want[i], slots, tokens, f"two_layer out[{i}]")
        else:
            _close(got[i], want[i], f"two_layer out[{i}]", tol_mul=1 / 4)
    _check_cache(g0, w0, slots, tokens, "int8_ds_mla")
    _check_cache(g1, w1, slots, tokens, "int8_ds_mla")

    ops = post_pass.af_ops[-1]
    assert ops.count(_FUSED_P1_INT8) == 1
    assert ops.count(_FUSED_P2_INT8) == 1


def test_int8_block32_stays_unfused(compile_vllm_env, monkeypatch):
    backend = _pick_int8_backend()
    g = torch.Generator(device="cpu").manual_seed(101)
    qkva32 = _make_int8_linear(_QKV_WIDTH, _HIDDEN, g, backend, block_size=(32, 32))
    qb32 = _make_int8_linear(_N_QB, _Q_LORA, g, backend, block_size=(32, 32))

    # Construct directly via Phase2Module structure using 32x32 weights:
    mod = _Phase1Module(102, qb32, "int8_ds_mla", is_neox=False)
    mod.pre_linear = qkva32

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8",
        "1",
    )
    torch._dynamo.reset()
    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        lambda h, pos, sl, c: mod(h, None, pos, sl, c),
        backend="inductor",
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": False,
            "post_grad_custom_post_pass": post_pass,
        },
    )
    h = (torch.randn((8, _HIDDEN), generator=g) * 0.5).bfloat16().to("mcpu")
    pos, slots = _make_positions_and_slots(8)
    _, c = _alloc_cache("int8_ds_mla")
    with torch.inference_mode():
        compiled(h, pos, slots, c)
    assert post_pass.af_ops
    assert all(_FUSED_P1_INT8 not in ops for ops in post_pass.af_ops)


def test_fp8_pass_accepts_int8_ds_mla_cache(compile_vllm_env, monkeypatch):
    fp8_backend = None
    for b in (
        Fp8LinearBackend.INTEL_AMX,
        Fp8LinearBackend.PORTABLE_LUT,
        Fp8LinearBackend.ACC,
    ):
        if xcpu_ops.fp8_scaled_mm_supported(b) and bool(
            torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache_backend_supported(
                int(b)
            )
        ):
            fp8_backend = b
            break
    if fp8_backend is None:
        pytest.skip("No supported FP8 MLA mega-kernel backend")

    g = torch.Generator(device="cpu").manual_seed(111)
    w_qkv = (torch.randn(_QKV_WIDTH, _HIDDEN, generator=g, device="cpu") * 0.1).to(
        torch.float8_e4m3fn
    )
    s_qkv = torch.full((21, _HIDDEN // 128), 0.125, dtype=torch.float32, device="cpu")
    qkva_fp8 = xcpu_ops.initialize_fp8_linear(
        w_qkv.to("mcpu"), s_qkv.to("mcpu"), (128, 128), backend=fp8_backend
    )
    w_qb = (torch.randn(_N_QB, _Q_LORA, generator=g, device="cpu") * 0.1).to(
        torch.float8_e4m3fn
    )
    s_qb = torch.full(
        (_N_QB // 128, _Q_LORA // 128), 0.125, dtype=torch.float32, device="cpu"
    )
    qb_fp8 = xcpu_ops.initialize_fp8_linear(
        w_qb.to("mcpu"), s_qb.to("mcpu"), (128, 128), backend=fp8_backend
    )

    mod = _Phase1Module(112, qb_fp8, "int8_ds_mla", is_neox=False)
    mod.pre_linear = qkva_fp8

    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE",
        "1",
    )
    torch._dynamo.reset()
    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        lambda h, pos, sl, c: mod(h, None, pos, sl, c),
        backend="inductor",
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": False,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    h = (torch.randn((8, _HIDDEN), generator=g) * 0.5).bfloat16().to("mcpu")
    pos, slots = _make_positions_and_slots(8)
    got_c, want_c = _alloc_cache("int8_ds_mla")
    with torch.inference_mode():
        want = mod(h, None, pos, slots, want_c)
        got = compiled(h, pos, slots, got_c)
        torch.mcpu.synchronize()

    _close(got[0], want[0], "fp8+int8_ds_mla out[0]", tol_mul=1 / 4)
    _check_q_out(got[1], want[1], slots, 8, "fp8+int8_ds_mla")
    _check_cache(got_c, want_c, slots, 8, "int8_ds_mla")
    assert post_pass.af_ops
    assert all(ops.count(_FUSED_P1_FP8) == 1 for ops in post_pass.af_ops)
