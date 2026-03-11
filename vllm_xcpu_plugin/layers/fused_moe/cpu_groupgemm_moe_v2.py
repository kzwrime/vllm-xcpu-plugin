# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

# Modular kernel interface for CPU MoE
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


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

        # Allocate all intermediate buffers here instead of in fused_moe_compute
        M_full_padding = hidden_states.shape[0]
        topk = topk_weights.shape[1]
        K = hidden_states.shape[-1]
        num_experts = w1.shape[0]
        device = hidden_states.device
        fdtype = hidden_states.dtype

        # Allocate buffers using M_full_padding
        permuted_hidden_states = torch.empty(
            (M_full_padding * topk, K), device=device, dtype=fdtype
        )
        sorted_by_expert = torch.empty(
            M_full_padding * topk, device=device, dtype=torch.int32
        )
        sorted_by_expert_back = torch.empty(
            M_full_padding * topk, device=device, dtype=torch.int32
        )
        expert_offsets = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
        intermediate_output = torch.empty(
            (M_full_padding * topk, w1.shape[1]), device=device, dtype=fdtype
        )
        activated = torch.empty(
            (M_full_padding * topk, w1.shape[1] // 2), device=device, dtype=fdtype)

        # Handle expert_num_tokens: allocate empty tensor if None
        if expert_tokens_meta is None:
            _expert_num_tokens = torch.empty(0, device=device, dtype=torch.int32)
        else:
            _expert_num_tokens = expert_tokens_meta.expert_num_tokens

        # Handle expert_map default value
        if expert_map is None:
            expert_map = torch.arange(w1.shape[0], device=device, dtype=torch.int32)
        else:
            expert_map = expert_map.to(torch.int32)

        # Only allocate workspace_unpermute_and_reduce if topk_reduce=True
        if self.topk_reduce:
            workspace_unpermute_and_reduce = torch.empty(
                M_full_padding, K, dtype=topk_weights.dtype, device=device)
        else:
            workspace_unpermute_and_reduce = torch.empty(0, device=device)

        # Call the fused C++ operator from torch_xcpu
        from torch_xcpu import ops as xcpu_ops
        xcpu_ops.fused_moe_compute(
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
            expert_num_tokens=_expert_num_tokens,
            apply_router_weight_on_input=apply_router_weight_on_input,
            topk_reduce=self.topk_reduce,
            permuted_hidden_states=permuted_hidden_states,
            sorted_by_expert=sorted_by_expert,
            sorted_by_expert_back=sorted_by_expert_back,
            expert_offsets=expert_offsets,
            intermediate_output=intermediate_output,
            activated=activated,
            workspace_unpermute_and_reduce=workspace_unpermute_and_reduce,
        )
