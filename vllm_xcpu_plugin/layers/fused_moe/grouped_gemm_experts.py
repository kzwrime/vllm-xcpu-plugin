"""Single Modular Experts implementation for all XCPU MoE weight formats."""

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform

from .workspace import FusedMoeWorkspacePlan

logger = init_logger(__name__)


class XcpuGroupedGemmExperts(mk.FusedMoEExpertsModular):
    """Map Modular Kernel routing state to one ``fused_moe_compute`` call."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        fused_moe,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.fused_moe = fused_moe
        parallel_config = moe_config.moe_parallel_config
        # V5/V6 preserve the complete top-k on each token-destination record.
        # Their Experts stage performs the destination-local weighted reduce;
        # Finalize then sums only one partial per destination rank.
        self.topk_reduce = not parallel_config.use_ep or (
            parallel_config.all2all_backend in {"mpi_alltoallv_v5", "mpi_alltoallv_v6"}
        )
        gemm1 = fused_moe.params.gemm1
        logger.warning_once(
            "Using XcpuGroupedGemmExperts: format=%s backend=%s "
            "implementation=%s use_ep=%s topk_reduce=%s",
            gemm1.weight_format.name,
            gemm1.backend.name.lower(),
            fused_moe.resolved_backend,
            parallel_config.use_ep,
            self.topk_reduce,
            scope="process",
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.device_name == "mcpu"

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kMxfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.enable_eplb

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        if self.topk_reduce:
            return TopKWeightAndReduceNoOP()
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, global_num_experts, local_num_experts, expert_tokens_meta, activation
        output = (M, K) if self.topk_reduce else (M * topk, K)
        return (0,), (0,), output

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del w1, w2, workspace13, workspace2
        if activation != MoEActivation.SILU:
            raise ValueError("XCPU grouped GEMM experts only support SiLU")
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError("XCPU W8A16/W4A16 experts require BF16 activations")
        if apply_router_weight_on_input:
            raise ValueError("XCPU grouped GEMM experts weight routes on output")

        capacity, hidden = hidden_states.shape
        topk = topk_ids.shape[1]
        params = self.fused_moe.params
        intermediate = params.intermediate
        workspace = FusedMoeWorkspacePlan(
            input_capacity=capacity,
            topk=topk,
            local_experts=params.experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            topk_reduce=self.topk_reduce,
        ).allocate(hidden_states.device, hidden_states.dtype)

        if expert_tokens_meta is None:
            expert_num_tokens = torch.empty(
                0, device=hidden_states.device, dtype=torch.int32
            )
            num_input_rows_valid = torch.empty(
                0, device=hidden_states.device, dtype=torch.int32
            )
        else:
            expert_num_tokens = expert_tokens_meta.expert_num_tokens
            if expert_num_tokens is None:
                expert_num_tokens = torch.empty(
                    0, device=hidden_states.device, dtype=torch.int32
                )
            raw_num_input_rows_valid = getattr(
                expert_tokens_meta, "num_input_rows_valid", None
            )
            if raw_num_input_rows_valid is None:
                num_input_rows_valid = torch.empty(
                    0, device=hidden_states.device, dtype=torch.int32
                )
            elif not isinstance(raw_num_input_rows_valid, torch.Tensor):
                raise TypeError("num_input_rows_valid must be a tensor or None")
            else:
                num_input_rows_valid = raw_num_input_rows_valid

        expert_num_tokens = expert_num_tokens.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        num_input_rows_valid = num_input_rows_valid.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        expert_map = (
            None
            if expert_map is None
            else expert_map.to(
                device=hidden_states.device, dtype=torch.int32
            ).contiguous()
        )

        from torch_xcpu import ops as xcpu_ops

        xcpu_ops.fused_moe_compute(
            output=output,
            hidden_states=hidden_states,
            backend=self.fused_moe,
            topk_weights=topk_weights.float().contiguous(),
            topk_ids=topk_ids.to(torch.int32).contiguous(),
            activation="silu",
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            expert_num_tokens=expert_num_tokens,
            num_input_rows_valid=num_input_rows_valid,
            topk_reduce=self.topk_reduce,
            permuted_hidden_states=workspace.permuted_hidden_states,
            sorted_by_expert=workspace.sorted_by_expert,
            sorted_by_expert_back=workspace.sorted_by_expert_back,
            expert_offsets=workspace.expert_offsets,
            intermediate_output=workspace.intermediate_output,
            activated=workspace.activated,
            workspace_unpermute_and_reduce=workspace.unpermute_and_reduce,
        )
