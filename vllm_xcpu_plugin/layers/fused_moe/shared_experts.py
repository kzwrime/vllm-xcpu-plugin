"""Fuse replicated shared MLPs into XCPU EP v5 / AF-EP v8 ops after loading."""

import os

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_xcpu_plugin import envs

from .ep_experts import XcpuEPExperts, XcpuSharedExpertWeights

logger = init_logger(__name__)

# gate_up -> SiluAndMul[WithClamp] -> down [-> sigmoid(expert_gate) *], nothing
# else.
_SHARED_MLP_TYPES = ("DeepseekV2MLP", "DeepseekV4MLP", "Qwen2MoeMLP")
# Fixed by torch_xcpu's MXFP4 linear (ops_defs/mxfp4_linear.py).
_MXFP4_SCALE_BLOCK_SIZE = (1, 32)


def _packed_linear(linear: nn.Module):
    """(weight, scale, backend, scale blocks) the unfused layer runs with."""
    if getattr(linear, "bias", None) is not None:
        return None
    fp8 = getattr(linear, "_xcpu_fp8_linear", None)
    if fp8 is not None:
        params = fp8.params
        return (
            params.packed_weight,
            params.weight_scale,
            int(params.backend),
            tuple(params.scale_block_size),
        )
    int8_lin = getattr(linear, "_xcpu_int8_linear", None)
    if int8_lin is not None:
        params = int8_lin.params
        return (
            params.packed_weight,
            params.weight_scale,
            int(params.backend),
            tuple(params.scale_block_size),
        )
    for name in ("_xcpu_mxfp4_linear", "_xcpu_quark_mxfp4_linear"):
        mxfp4 = getattr(linear, name, None)
        if mxfp4 is not None:
            params = mxfp4.params
            return (
                params.packed_weight,
                params.packed_weight_scale,
                int(params.backend),
                _MXFP4_SCALE_BLOCK_SIZE,
            )
    if isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod):
        return linear.weight, None, -1, (0, 0)
    return None


def _extract_shared_expert_weights(
    runner: MoERunner,
    *,
    hidden_size: int,
    activation_dtype: torch.dtype,
    routed_swiglu_limit: float | None,
) -> tuple[XcpuSharedExpertWeights | None, float, str]:
    """Extract the runner's shared MLP weights, swiglu_limit, or rejection reason."""
    if runner.routed_input_transform is not None or (
        runner.routed_output_transform is not None
    ):
        return None, 0.0, "routed input/output transforms"
    mlp = runner.shared_experts._layer
    if type(mlp).__name__ not in _SHARED_MLP_TYPES or not isinstance(
        mlp.act_fn, (SiluAndMul, SiluAndMulWithClamp)
    ):
        return None, 0.0, f"unsupported shared MLP {type(mlp).__name__}"
    limit = float(getattr(mlp.act_fn, "swiglu_limit", 0.0))
    if routed_swiglu_limit is not None and limit != routed_swiglu_limit:
        return (
            None,
            0.0,
            "shared MLP swiglu_limit differs from the routed experts",
        )
    # The op runs on this rank's rows with full weights and no all-reduce.
    config = runner.moe_config
    if not (config.is_sequence_parallel or config.tp_size == 1) or any(
        linear.tp_size != 1 for linear in (mlp.gate_up_proj, mlp.down_proj)
    ):
        return None, 0.0, "tensor-parallel shared MLP"
    gate_up = _packed_linear(mlp.gate_up_proj)
    down = _packed_linear(mlp.down_proj)
    if gate_up is None or down is None:
        return None, 0.0, "shared MLP linear format"
    if gate_up[0].dtype != down[0].dtype or gate_up[2:] != down[2:]:
        return None, 0.0, "shared MLP linears differ in format or backend"
    if gate_up[2] < 0 and gate_up[0].dtype != activation_dtype:
        return None, 0.0, "unquantized shared MLP dtype"
    gate = getattr(mlp, "expert_gate", None)
    gate_weight = None
    if gate is not None:
        gate_weight = gate.weight
        if (
            getattr(gate, "bias", None) is not None
            or not isinstance(gate.quant_method, UnquantizedLinearMethod)
            or gate_weight.dtype != activation_dtype
            or gate_weight.numel() != hidden_size
        ):
            return None, 0.0, "shared expert gate format"
    return (
        XcpuSharedExpertWeights(
            w13=gate_up[0],
            w2=down[0],
            w13_scale=gate_up[1],
            w2_scale=down[1],
            linear_backend=gate_up[2],
            scale_block_size=gate_up[3],
            gate=gate_weight,
        ),
        limit,
        "",
    )


def _shared_expert_weights(
    runner: MoERunner, ep_experts: XcpuEPExperts
) -> tuple[XcpuSharedExpertWeights | None, str]:
    """The runner's shared MLP as the v5 op takes it, or why it cannot."""
    shared, _, reason = _extract_shared_expert_weights(
        runner,
        hidden_size=ep_experts.compute.params.hidden,
        activation_dtype=ep_experts.activation_dtype,
        routed_swiglu_limit=ep_experts.swiglu_limit,
    )
    return shared, reason


def install_fused_shared_experts(model: nn.Module) -> int:
    """Move every eligible MoE layer's shared MLP into its v5 EP or v8 AF-EP op.

    On by default (VLLM_XCPU_FUSE_SHARED_EXPERTS); layers on other MoE backends
    keep the runner's shared path.

    Runs once the whole model is post-processed, so the shared MLP's linear
    layers already hold their packed weights; the op reuses them. A fused
    runner then runs as a routed-only MoE: the op returns the shared output
    plus the scaled routed output. Returns the number of fused layers.
    """
    if not envs.VLLM_XCPU_FUSE_SHARED_EXPERTS:
        return 0
    from vllm_xcpu_plugin.af_ep.attn.client_v8 import ExpertsClientV8
    from vllm_xcpu_plugin.af_ep.attn.remote_fused_moe import (
        RemoteExpertsFusedMoEMethod,
    )

    fused_v5 = 0
    fused_af_v8 = 0
    for name, runner in model.named_modules():
        if not isinstance(runner, MoERunner) or runner.shared_experts is None:
            continue
        ep_experts = getattr(runner.routed_experts, "_xcpu_ep_experts", None)
        if ep_experts is not None and ep_experts.version == 5:
            shared, reason = _shared_expert_weights(runner, ep_experts)
            if shared is None:
                logger.warning_once(
                    "Keeping the unfused shared expert path (%s), e.g. %s",
                    reason,
                    name,
                    scope="process",
                )
                continue
            ep_experts.bind_shared_experts(shared, runner.routed_scaling_factor)
            runner._shared_experts = None
            runner.routed_scaling_factor = 1.0
            runner._forward_entry = runner._select_forward()
            fused_v5 += 1
            continue

        quant_method = getattr(runner.routed_experts, "quant_method", None)
        if isinstance(quant_method, RemoteExpertsFusedMoEMethod) and isinstance(
            quant_method.client, ExpertsClientV8
        ):
            moe_cfg = runner.moe_config
            shared, shared_limit, reason = _extract_shared_expert_weights(
                runner,
                hidden_size=moe_cfg.hidden_dim,
                activation_dtype=moe_cfg.in_dtype,
                routed_swiglu_limit=None,
            )
            if shared is None:
                logger.warning_once(
                    "Keeping the unfused AF-EP v8 shared expert path (%s), e.g. %s",
                    reason,
                    name,
                    scope="process",
                )
                continue
            quant_method.client.bind_shared_experts(
                quant_method.layer_idx,
                shared,
                runner.routed_scaling_factor,
                swiglu_limit=shared_limit,
            )
            runner._shared_experts = None
            runner.routed_scaling_factor = 1.0
            runner._forward_entry = runner._select_forward()
            fused_af_v8 += 1

    if fused_v5:
        # Parsed like torch_xcpu, which reads it once per process (default off).
        async_dispatch = os.getenv(
            "TORCH_XCPU_FUSED_EP_MOE_V5_ASYNC_DISPATCH", ""
        ) not in ("", "0")
        logger.warning_once(
            "Fused shared experts into XCPU EP v5 for %d MoE layers "
            "(TORCH_XCPU_FUSED_EP_MOE_V5_ASYNC_DISPATCH: %s)",
            fused_v5,
            "on" if async_dispatch else "off",
            scope="process",
        )
    if fused_af_v8:
        async_dispatch = os.getenv(
            "TORCH_XCPU_FUSED_AF_V8_ASYNC_DISPATCH", ""
        ) not in ("", "0")
        logger.warning_once(
            "Fused shared experts into XCPU AF-EP v8 for %d MoE layers "
            "(TORCH_XCPU_FUSED_AF_V8_ASYNC_DISPATCH: %s)",
            fused_af_v8,
            "on" if async_dispatch else "off",
            scope="process",
        )
    return fused_v5 + fused_af_v8
