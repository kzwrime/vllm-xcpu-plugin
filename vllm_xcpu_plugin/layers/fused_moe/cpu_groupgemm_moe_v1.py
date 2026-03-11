# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

# Modular kernel interface for CPU MoE
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.utils.torch_utils import direct_register_custom_op


class CPUGroupGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    CPU implementation of FusedMoEPermuteExpertsUnpermute.
    This wraps the existing CPUFusedMOE implementation to conform
    to the standard modular kernel interface.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        quant_config: FusedMoEQuantConfig,
        topk_reduce: bool = True,
    ):
        super().__init__(quant_config)
        self.layer = layer
        self.topk_reduce = topk_reduce

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # CPUFusedMOE already handles weight application and reduction
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # CPU implementation doesn't need intermediate workspaces
        # It produces the final output directly
        workspace13 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace13, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Execute CPU MoE computation using native torch operations.
        The computation follows:
          Permute -> Grouped GEMM -> Activation -> Grouped GEMM -> Unpermute and reduce.
        Note: topk_weights and topk_ids should already be computed by router.
        """
        # No quantization support
        assert a1q_scale is None, "CPU MoE does not support input quantization"
        assert a2_scale is None, "CPU MoE does not support intermediate quantization"
        assert self.quant_dtype is None, "CPU MoE does not support weight quantization"

        torch.ops.vllm.fused_moe_compute(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            expert_num_tokens=expert_tokens_meta.expert_num_tokens
            if expert_tokens_meta
            else None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            topk_reduce=self.topk_reduce,
        )


def fused_moe_compute(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    expert_num_tokens: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    topk_reduce: bool,
) -> None:
    """
    Execute CPU MoE computation using native torch operations.
    The computation follows:
      Permute -> Grouped GEMM -> Activation -> Grouped GEMM -> Unpermute and reduce.
    Note: topk_weights and topk_ids should already be computed by router.
    """
    assert not apply_router_weight_on_input, (
        "CPU MoE does not support apply_router_weight_on_input"
    )
    # No quantization support
    assert a1q_scale is None, "CPU MoE does not support input quantization"
    assert a2_scale is None, "CPU MoE does not support intermediate quantization"

    assert activation == "silu"

    # Slice inputs to actual valid tokens if expert_num_tokens is provided
    # (handles static buffer allocation where buffer may be larger than actual data)
    if expert_num_tokens is not None:
        num_valid_tokens = int(expert_num_tokens.sum().item())
        hidden_states = hidden_states[:num_valid_tokens]
        topk_ids = topk_ids[:num_valid_tokens]
        topk_weights = topk_weights[:num_valid_tokens]

    from torch_xcpu import ops as xcpu_ops

    M, topk = topk_weights.shape
    K = hidden_states.shape[-1]
    num_experts = w1.shape[0]
    device = hidden_states.device
    fdtype = hidden_states.dtype

    if expert_map is None:
        expert_map = torch.arange(num_experts, device=device)

    permuted_hidden_states = torch.empty((M * topk, K), device=device, dtype=fdtype)
    sorted_by_expert = torch.empty(M * topk, device=device, dtype=torch.int32)
    expert_offsets = torch.empty(num_experts + 1, device=device, dtype=torch.int32)

    num_valid_tokens = xcpu_ops.moe_permute(
        permuted_hidden_states,
        sorted_by_expert,
        expert_offsets,
        hidden_states,
        topk_ids.to(torch.int32),
        expert_map.to(torch.int32),
        num_experts,
        global_num_experts,
        M,
    )

    if num_valid_tokens == 0:
        output.zero_()
        return

    # Step 2: Grouped GEMM (first layer) - compute gate_up projections
    # Use moe_grouped_gemm to compute all experts in one pass
    intermediate_output = torch.empty(
        (permuted_hidden_states.shape[0], w1.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    xcpu_ops.moe_grouped_gemm(
        intermediate_output,     # [:num_valid_tokens, intermediate_size]
        permuted_hidden_states,  # [:num_valid_tokens, K]
        w1,  # [num_experts, 2 * intermediate_size, K]
        expert_offsets,  # [num_experts + 1]
        True,
        num_valid_tokens,
        # trans_b: w1 is [num_experts, 2 * intermediate_size, K],
        # need to transpose last 2 dims
    )  # intermediate_output is [num_valid_tokens, 2 * intermediate_size]

    # Step 3: Activation function (SiluAndMul for SwiGLU)
    activated = torch.empty(
        (intermediate_output.shape[0], intermediate_output.shape[1] // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    xcpu_ops.silu_and_mul(
        activated[:num_valid_tokens], intermediate_output[:num_valid_tokens]
    )

    # Step 4: Grouped GEMM (second layer) - compute down projections
    # Reuse permuted_hidden_states as the output buffer for the second grouped_gemm.
    # The original permuted_hidden_states is no longer needed after this point.
    xcpu_ops.moe_grouped_gemm(
        permuted_hidden_states,  # output buffer: [:num_valid_tokens, K]
        activated,  # input: [:num_valid_tokens, intermediate_size // 2]
        w2,
        expert_offsets,
        True,
        num_valid_tokens,
    )

    if topk_reduce:
        workspace_unpermute_and_reduce = torch.empty(
            M, K, dtype=topk_weights.dtype, device=hidden_states.device
        )
        xcpu_ops.moe_unpermute(
            output,
            permuted_hidden_states,
            sorted_by_expert.to(torch.int32),
            topk_weights=topk_weights,
            workspace_unpermute_and_reduce=workspace_unpermute_and_reduce,
            M=M,
            num_valid_tokens=num_valid_tokens,  # Only process num_valid_tokens
        )
    else:
        xcpu_ops.moe_unpermute(
            output,
            permuted_hidden_states,
            sorted_by_expert.to(torch.int32),
            topk=topk,
            M=M,
            num_valid_tokens=num_valid_tokens,  # Only process num_valid_tokens
        )


direct_register_custom_op(
    op_name="fused_moe_compute",
    op_func=fused_moe_compute,
    mutates_args=["output"],
)
