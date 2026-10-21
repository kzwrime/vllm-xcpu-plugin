"""Fuse replicated shared MLPs into the XCPU EP v5 op after weight loading."""

import os

from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_xcpu_plugin import envs

from .ep_experts import XcpuEPExperts, XcpuSharedExpertWeights

logger = init_logger(__name__)

# gate_up -> SiluAndMul -> down [-> sigmoid(expert_gate) *], nothing else.
_SHARED_MLP_TYPES = ("DeepseekV2MLP", "Qwen2MoeMLP")
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


def _shared_expert_weights(
    runner: MoERunner, ep_experts: XcpuEPExperts
) -> tuple[XcpuSharedExpertWeights | None, str]:
    """The runner's shared MLP as the v5 op takes it, or why it cannot."""
    if runner.routed_input_transform is not None or (
        runner.routed_output_transform is not None
    ):
        return None, "routed input/output transforms"
    mlp = runner.shared_experts._layer
    if type(mlp).__name__ not in _SHARED_MLP_TYPES or not isinstance(
        mlp.act_fn, SiluAndMul
    ):
        return None, f"unsupported shared MLP {type(mlp).__name__}"
    # The op runs on this rank's rows with full weights and no all-reduce.
    config = runner.moe_config
    if not (config.is_sequence_parallel or config.tp_size == 1) or any(
        linear.tp_size != 1 for linear in (mlp.gate_up_proj, mlp.down_proj)
    ):
        return None, "tensor-parallel shared MLP"
    gate_up = _packed_linear(mlp.gate_up_proj)
    down = _packed_linear(mlp.down_proj)
    if gate_up is None or down is None:
        return None, "shared MLP linear format"
    if gate_up[0].dtype != down[0].dtype or gate_up[2:] != down[2:]:
        return None, "shared MLP linears differ in format or backend"
    if gate_up[2] < 0 and gate_up[0].dtype != ep_experts.activation_dtype:
        return None, "unquantized shared MLP dtype"
    gate = getattr(mlp, "expert_gate", None)
    gate_weight = None
    if gate is not None:
        gate_weight = gate.weight
        if (
            getattr(gate, "bias", None) is not None
            or not isinstance(gate.quant_method, UnquantizedLinearMethod)
            or gate_weight.dtype != ep_experts.activation_dtype
            or gate_weight.numel() != ep_experts.compute.params.hidden
        ):
            return None, "shared expert gate format"
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
        "",
    )


def install_fused_shared_experts(model: nn.Module) -> int:
    """Move every eligible MoE layer's shared MLP into its v5 EP op.

    On by default (VLLM_XCPU_FUSE_SHARED_EXPERTS); layers without the v5 EP
    op (other MoE backends) keep the runner's shared path.

    Runs once the whole model is post-processed, so the shared MLP's linear
    layers already hold their packed weights; the op reuses them. A fused
    runner then runs as a routed-only MoE: the op returns the shared output
    plus the scaled routed output. Returns the number of fused layers.
    """
    if not envs.VLLM_XCPU_FUSE_SHARED_EXPERTS:
        return 0
    fused = 0
    for name, runner in model.named_modules():
        if not isinstance(runner, MoERunner) or runner.shared_experts is None:
            continue
        ep_experts = getattr(runner.routed_experts, "_xcpu_ep_experts", None)
        if ep_experts is None or ep_experts.version != 5:
            continue
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
        fused += 1
    if fused:
        # Parsed like torch_xcpu, which reads it once per process (default off).
        async_dispatch = os.getenv(
            "TORCH_XCPU_FUSED_EP_MOE_V5_ASYNC_DISPATCH", ""
        ) not in ("", "0")
        logger.warning_once(
            "Fused shared experts into XCPU EP v5 for %d MoE layers "
            "(TORCH_XCPU_FUSED_EP_MOE_V5_ASYNC_DISPATCH: %s)",
            fused,
            "on" if async_dispatch else "off",
            scope="process",
        )
    return fused

