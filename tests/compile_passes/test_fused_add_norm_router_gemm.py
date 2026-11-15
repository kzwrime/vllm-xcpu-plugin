# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the post-grad fused_add_norm_router_gemm compile pass."""

from __future__ import annotations

import uuid

import pytest
import torch
import torch.nn as nn
import torch_xcpu  # noqa: F401
import vllm.envs as vllm_envs
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.plugins import load_general_plugins

from vllm_xcpu_plugin import ir_kernels as _ir_kernels  # noqa: F401
from vllm_xcpu_plugin.compile_passes import XcpuPostGradPass, pass_uuid
from vllm_xcpu_plugin.compile_passes.fused_add_norm_router_gemm import (
    replace_with_fused_add_norm_router_gemm,
)
from vllm_xcpu_plugin.layers.fused_moe.gate_linear import XcpuGateLinear

load_general_plugins()

_AF = torch.ops.higher_order.auto_functionalized
_AF_V2 = torch.ops.higher_order.auto_functionalized_v2
_AF_HOPS = (_AF, _AF_V2)
_ADD_NORM_OP = torch.ops.torch_xcpu.fused_add_rms_norm_out_bf16.default
_FUSED_OP = torch.ops.torch_xcpu.fused_add_norm_router_gemm.default


@pytest.fixture
def compile_vllm_env(monkeypatch):
    import vllm.ir as ir
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setenv("VLLM_USE_XCPU_LINEAR", "1")
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )
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


def _make_norm_and_gate(
    hidden_size: int = 128,
    num_experts: int = 64,
    *,
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = torch.float32,
    bias: bool = False,
    gemma_norm: bool = False,
    seed: int = 0,
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.device("cpu"):
        norm = (
            GemmaRMSNorm(hidden_size, eps=eps)
            if gemma_norm
            else RMSNorm(hidden_size, eps=eps)
        )
        gate = GateLinear(
            hidden_size,
            num_experts,
            bias=bias,
            out_dtype=out_dtype,
            params_dtype=torch.bfloat16,
        )
    assert isinstance(gate, XcpuGateLinear)
    with torch.no_grad():
        norm.weight.copy_(
            torch.randn(hidden_size, generator=gen, dtype=torch.bfloat16)
        )
        gate.weight.copy_(
            torch.randn(
                (num_experts, hidden_size), generator=gen, dtype=torch.bfloat16
            )
            * 0.05
        )
        if bias:
            gate.bias.copy_(
                torch.randn(num_experts, generator=gen, dtype=torch.bfloat16)
                * 0.05
            )
    norm = norm.to(device="mcpu", dtype=torch.bfloat16)
    gate = gate.to("mcpu")
    gate.quant_method.process_weights_after_loading(gate)
    if (
        not bias
        and out_dtype in (torch.float32, torch.bfloat16, None)
        and num_experts % 32 == 0
        and hidden_size % 32 == 0
    ):
        assert getattr(gate, "_xcpu_bf16_linear", None) is not None
    return norm, gate


class _CapturingPostPass(XcpuPostGradPass):
    def __init__(self):
        super().__init__()
        self._instance_id = uuid.uuid4().hex
        self.graphs: list[torch.fx.Graph] = []
        self.op_counts: list[dict[str, int]] = []

    def uuid(self):
        # Bypass Inductor FxGraphCache hits across test cases so __call__
        # always observes the post-grad graph in every test invocation.
        return (super().uuid(), self._instance_id)

    def __call__(self, graph) -> None:
        super().__call__(graph)
        self.graphs.append(graph)
        af_targets = [
            n.args[0]
            for n in graph.nodes
            if n.op == "call_function" and n.target in _AF_HOPS and n.args
        ]
        self.op_counts.append(
            {
                "fused": sum(t == _FUSED_OP for t in af_targets),
                "add_norm": sum(t == _ADD_NORM_OP for t in af_targets),
            }
        )


@pytest.mark.parametrize("auto_func_v2", [False, True])
@pytest.mark.parametrize(
    "hidden_size,num_experts",
    [
        (128, 64),
        (6144, 256),  # GLM-5.2 mega-kernel target shape
    ],
)
def test_compiled_moe_front_fuses_and_matches_unfused_across_batch_sizes(
    compile_vllm_env, monkeypatch, hidden_size, num_experts, auto_func_v2
):
    norm, gate = _make_norm_and_gate(hidden_size, num_experts, seed=42)
    shared_w = (
        torch.randn(
            (64, hidden_size),
            generator=torch.Generator().manual_seed(7),
            dtype=torch.bfloat16,
        )
        * 0.05
    ).to("mcpu")

    # Emit shared_experts projection BEFORE gate(hidden) to exercise hoisting
    # of the router_logits scratch buffer & SymInt alignment math above add_norm.
    class MoeFrontBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = norm
            self.gate = gate

        def forward(self, hidden: torch.Tensor, residual: torch.Tensor):
            hidden, residual = self.norm(hidden, residual)
            shared_out = torch.matmul(hidden, shared_w.t())
            logits, _ = self.gate(hidden)
            return hidden, residual, logits, shared_out

    block = MoeFrontBlock()
    monkeypatch.setenv("VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM", "1")
    torch._dynamo.reset()

    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        block,
        backend="inductor",
        dynamic=True,
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": auto_func_v2,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    # Exercise both T <= 32 (mega-kernel eligible on 6144x256) and T > 32
    # (including non-16-aligned T=33 prefill fallback).
    for num_tokens in (7, 32, 33, 64, 1):
        gen = torch.Generator(device="cpu").manual_seed(100 + num_tokens)
        h_cpu = torch.randn(
            (num_tokens, hidden_size), generator=gen, dtype=torch.bfloat16
        )
        r_cpu = torch.randn(
            (num_tokens, hidden_size), generator=gen, dtype=torch.bfloat16
        )

        with torch.inference_mode():
            ref_h, ref_r, ref_logits, ref_shared = block(
                h_cpu.to("mcpu"), r_cpu.to("mcpu")
            )
            got_h, got_r, got_logits, got_shared = compiled(
                h_cpu.to("mcpu"), r_cpu.to("mcpu")
            )
            ref_logits_from_got_h, _ = gate(got_h)
            torch.mcpu.synchronize()

        assert got_logits.dtype == torch.float32
        assert got_logits.shape == (num_tokens, num_experts)
        torch.testing.assert_close(got_r.cpu(), ref_r.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(
            got_h.cpu(), ref_h.cpu(), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            got_shared.cpu(), ref_shared.cpu(), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            got_logits.cpu(),
            ref_logits_from_got_h.cpu(),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            got_logits.cpu(), ref_logits.cpu(), rtol=5e-3, atol=5e-3
        )

    assert 1 <= len(post_pass.op_counts) <= 2
    assert all(
        c == {"fused": 1, "add_norm": 0} for c in post_pass.op_counts
    )


def test_live_input_sum_residual_is_not_corrupted(compile_vllm_env, monkeypatch):
    """Simulates EAGLE3 capture_input_sum: residual passed into post-attn norm
    also escapes as aux_hidden_states and must not be clobbered in place."""
    norm1, _ = _make_norm_and_gate(128, 64, seed=1)
    norm2, gate2 = _make_norm_and_gate(128, 64, seed=2)

    class LayerWithAuxCapture(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = norm1
            self.post_attention_layernorm = norm2
            self.gate = gate2

        def forward(self, hidden: torch.Tensor, residual: torch.Tensor):
            hidden, residual = self.input_layernorm(hidden, residual)
            input_sum = residual
            attn_out = hidden * 0.5
            hidden, residual = self.post_attention_layernorm(
                attn_out, input_sum
            )
            logits, _ = self.gate(hidden)
            return hidden, residual, logits, input_sum

    layer = LayerWithAuxCapture()
    monkeypatch.setenv("VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM", "1")
    torch._dynamo.reset()

    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        layer,
        backend="inductor",
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    h_cpu = torch.randn((8, 128), dtype=torch.bfloat16)
    r_cpu = torch.randn((8, 128), dtype=torch.bfloat16)
    with torch.inference_mode():
        ref_h, ref_r, ref_logits, ref_sum = layer(
            h_cpu.to("mcpu"), r_cpu.to("mcpu")
        )
        got_h, got_r, got_logits, got_sum = compiled(
            h_cpu.to("mcpu"), r_cpu.to("mcpu")
        )
        torch.mcpu.synchronize()

    # input_layernorm stays unfused; post_attention_layernorm + gate fuses.
    assert post_pass.op_counts[-1] == {"fused": 1, "add_norm": 1}
    torch.testing.assert_close(got_sum.cpu(), ref_sum.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(got_r.cpu(), ref_r.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(got_h.cpu(), ref_h.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        got_logits.cpu(), ref_logits.cpu(), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    "case",
    [
        "pass_disabled",
        "gemma_norm",
        "no_residual",
        "aliased_hidden_and_residual",
        "sp_slice_between_norm_and_gate",
        "gate_has_bias",
        "bf16_gate_output",
        "ambiguous_two_fp32_gates",
        "unaligned_num_experts",
    ],
)
def test_non_matching_graphs_stay_unfused(compile_vllm_env, monkeypatch, case):
    enabled = case != "pass_disabled"
    monkeypatch.setenv(
        "VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM",
        "1" if enabled else "0",
    )

    norm, gate = _make_norm_and_gate(
        128,
        60 if case == "unaligned_num_experts" else 64,
        gemma_norm=(case == "gemma_norm"),
        bias=(case == "gate_has_bias"),
        out_dtype=(
            torch.bfloat16 if case == "bf16_gate_output" else torch.float32
        ),
        seed=11,
    )
    _, gate_extra = _make_norm_and_gate(128, 64, seed=12)

    def fn(hidden: torch.Tensor, residual: torch.Tensor):
        if case == "no_residual":
            normed = norm(hidden)
            logits, _ = gate(normed)
            return normed, logits
        if case == "aliased_hidden_and_residual":
            normed, res_out = norm(hidden, hidden)
            logits, _ = gate(normed)
            return normed, res_out, logits
        normed, res_out = norm(hidden, residual)
        if case == "sp_slice_between_norm_and_gate":
            # Simulate pre-chunk norm followed by post-chunk gate on rank != 0.
            chunk = normed[normed.shape[0] // 2 :]
            logits, _ = gate(chunk)
            return chunk, res_out, logits
        if case == "ambiguous_two_fp32_gates":
            logits1, _ = gate(normed)
            logits2, _ = gate_extra(normed)
            return normed, res_out, logits1 + logits2
        logits, _ = gate(normed)
        return normed, res_out, logits

    torch._dynamo.reset()
    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        fn,
        backend="inductor",
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    h_cpu = torch.randn((8, 128), dtype=torch.bfloat16)
    r_cpu = torch.randn((8, 128), dtype=torch.bfloat16)
    with torch.inference_mode():
        ref = fn(h_cpu.to("mcpu"), r_cpu.to("mcpu"))
        got = compiled(h_cpu.to("mcpu"), r_cpu.to("mcpu"))
        torch.mcpu.synchronize()

    assert post_pass.op_counts[-1]["fused"] == 0
    expected_add_norm = 0 if case in ("gemma_norm", "no_residual") else 1
    assert post_pass.op_counts[-1]["add_norm"] == expected_add_norm
    for g, r in zip(got, ref):
        torch.testing.assert_close(g.cpu(), r.cpu(), rtol=1e-2, atol=1e-2)


def test_pass_uuid_changes_with_switch_and_env_registered(monkeypatch):
    assert (
        "VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM"
        in vllm_envs.environment_variables
    )
    monkeypatch.setenv("VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM", "0")
    u0 = pass_uuid()
    monkeypatch.setenv("VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM", "1")
    u1 = pass_uuid()
    assert u0 != u1


def test_direct_graph_pass_lint(compile_vllm_env):
    from torch._functorch.aot_autograd import aot_export_module
    from torch._subclasses.fake_tensor import FakeTensorMode

    norm, gate = _make_norm_and_gate(128, 64, seed=9)

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = norm
            self.gate = gate

        def forward(self, h, r):
            h, r = self.norm(h, r)
            logits, _ = self.gate(h)
            return h, r, logits

    mod = Mod()
    h = torch.randn((5, 128), dtype=torch.bfloat16, device="mcpu")
    r = torch.randn((5, 128), dtype=torch.bfloat16, device="mcpu")
    with torch.inference_mode(), FakeTensorMode(allow_non_fake_inputs=True):
        exported_gm, _ = aot_export_module(mod, (h, r), trace_joint=False)
    replaced = replace_with_fused_add_norm_router_gemm(exported_gm.graph)
    assert replaced == 1
    exported_gm.graph.lint()
    exported_gm.recompile()


@pytest.mark.parametrize("auto_func_v2", [False, True])
def test_intermediate_tensors_reinplace_without_clones(
    compile_vllm_env, monkeypatch, auto_func_v2
):
    """When hidden/residual entering post-attn norm are dead-after-use
    intermediates (not graph inputs), Inductor reinplaces fused_add_norm_router_gemm
    directly onto those buffers without cloning."""
    norm, gate = _make_norm_and_gate(128, 64, seed=21)

    class TwoStepBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = norm
            self.gate = gate

        def forward(self, h_in: torch.Tensor, r_in: torch.Tensor):
            h = h_in * 0.5
            r = r_in + 0.25
            h, r = self.norm(h, r)
            logits, _ = self.gate(h)
            return h, r, logits

    block = TwoStepBlock()
    monkeypatch.setenv("VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM", "1")
    torch._dynamo.reset()

    post_pass = _CapturingPostPass()
    compiled = torch.compile(
        block,
        backend="inductor",
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "enable_auto_functionalized_v2": auto_func_v2,
            "post_grad_custom_post_pass": post_pass,
        },
    )

    h_cpu = torch.randn((8, 128), dtype=torch.bfloat16)
    r_cpu = torch.randn((8, 128), dtype=torch.bfloat16)
    with torch.inference_mode():
        ref_h, ref_r, ref_logits = block(h_cpu.to("mcpu"), r_cpu.to("mcpu"))
        got_h, got_r, got_logits = compiled(h_cpu.to("mcpu"), r_cpu.to("mcpu"))
        torch.mcpu.synchronize()

    assert post_pass.op_counts[-1] == {"fused": 1, "add_norm": 0}
    torch.testing.assert_close(got_r.cpu(), ref_r.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(got_h.cpu(), ref_h.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        got_logits.cpu(), ref_logits.cpu(), rtol=1e-4, atol=1e-4
    )
    # Inductor mutates gm.graph in place during reinplace_inplaceable_ops +
    # decompose_auto_functionalized right after post_grad_custom_post_pass.
    final_graph = post_pass.graphs[-1]
    fused_direct = [
        n
        for n in final_graph.nodes
        if n.op == "call_function" and n.target == _FUSED_OP
    ]
    clone_ops = [
        n
        for n in final_graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.clone.default
    ]
    assert len(fused_direct) == 1
    assert clone_ops == []

