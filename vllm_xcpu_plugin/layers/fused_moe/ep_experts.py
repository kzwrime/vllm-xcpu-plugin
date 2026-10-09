"""vLLM binding and invocation storage for direct routed EP operators."""

from typing import TYPE_CHECKING

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from torch import Tensor
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

if TYPE_CHECKING:
    from torch_xcpu.ops import FusedMoe


class XcpuEPExperts(torch.nn.Module):
    """Bind one routed EP op; allocate independent storage for each invocation.

    vLLM owns routing, shared experts, scaling and final output transforms.
    Calls sharing an MPI window must use one serialized stream.
    Communication metadata must outlive submitted work.
    """

    def __init__(
        self,
        compute: "FusedMoe",
        comm_metadata: Tensor,
        comm_ptr_wrapper: Tensor,
        expert_map: Tensor,
        *,
        version: int,
        ep_size: int,
        ep_rank: int,
        max_num_tokens: int,
        topk: int,
    ):
        super().__init__()
        if version not in (5, 6):
            raise ValueError("XCPU EP requires MPI V5/V6")
        if not 0 < ep_size <= 128 or not 0 <= ep_rank < ep_size:
            raise ValueError("invalid XCPU EP topology")
        if max_num_tokens <= 0 or topk not in (6, 8):
            raise ValueError("XCPU EP requires positive capacity and topk=6/8")
        self.version = version
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.max_num_tokens = max_num_tokens
        self.topk = topk
        if (
            comm_metadata.device.type != "cpu"
            or comm_metadata.dtype != torch.int64
            or comm_metadata.numel() < 2
            or not comm_metadata.is_contiguous()
            or comm_ptr_wrapper.device.type != "cpu"
            or comm_ptr_wrapper.dtype != torch.int64
            or comm_ptr_wrapper.numel() != 1
            or not comm_ptr_wrapper.is_contiguous()
        ):
            raise ValueError("EP MoE requires persistent CPU int64 MPI metadata")
        if comm_metadata[:2].tolist() != [ep_size, ep_rank]:
            raise ValueError("EP topology disagrees with communicator metadata")
        self.compute = compute
        self.activation_dtype = (
            torch.float32
            if compute.params.gemm1.params.packed_weight.dtype == torch.float32
            else torch.bfloat16
        )
        self.comm_metadata = comm_metadata
        self.comm_ptr_wrapper = comm_ptr_wrapper
        self.global_num_experts = compute.params.experts * ep_size
        expected = torch.arange(self.global_num_experts, device="cpu")
        expected -= ep_rank * compute.params.experts
        expected = torch.where(
            (expected >= 0) & (expected < compute.params.experts), expected, -1
        ).to(torch.int32)
        # Initialization only: V5/V6 wire routing requires linear placement.
        if not torch.equal(expert_map.cpu(), expected):
            raise ValueError("EP MoE requires the uniform linear expert map")
        self.expert_map = expert_map.to(
            device=compute.params.gemm1.params.packed_weight.device,
            dtype=torch.int32,
        ).contiguous()
        self._op = getattr(
            torch.ops.torch_xcpu,
            f"fused_ep_moe_v{version}_{compute.backend_type.__name__}",
        ).default

        gemm1, gemm2 = compute.params.gemm1.params, compute.params.gemm2.params
        block_n, block_k = gemm1.scale_block_size or (0, 0)
        self.workspace_bytes = getattr(
            torch.ops.torch_xcpu,
            f"fused_ep_moe_v{version}_workspace_size_{compute.backend_type.__name__}",
        )(
            ep_size,
            max_num_tokens,
            topk,
            compute.params.hidden,
            compute.params.intermediate,
            compute.params.experts,
            4 if self.activation_dtype == torch.float32 else 2,
            block_n,
            block_k,
            gemm1.bias is not None,
            gemm2.bias is not None,
        )

    def forward(
        self, hidden_states: Tensor, topk_weights: Tensor, topk_ids: Tensor
    ) -> Tensor:
        params = self.compute.params
        if hidden_states.dtype != self.activation_dtype:
            raise ValueError("EP MoE activation dtype disagrees with compute backend")
        torch._check(hidden_states.ndim == 2)
        torch._check(hidden_states.size(1) == params.hidden)
        torch._check(hidden_states.size(0) <= self.max_num_tokens)
        torch._check(topk_ids.shape == (hidden_states.size(0), self.topk))
        torch._check(topk_weights.shape == topk_ids.shape)
        torch._check(hidden_states.device == self.expert_map.device)
        torch._check(topk_ids.device == hidden_states.device)
        torch._check(topk_weights.device == hidden_states.device)
        torch._check(params.hidden * hidden_states.element_size() % 4 == 0)
        hidden_states = hidden_states.contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_weights.float().contiguous()

        gemm1, gemm2 = params.gemm1.params, params.gemm2.params
        block_n, block_k = gemm1.scale_block_size or (0, 0)
        workspace = torch.empty(
            self.workspace_bytes, device=hidden_states.device, dtype=torch.uint8
        )
        output = torch.empty_like(hidden_states)
        self._op(
            output,
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1.packed_weight,
            gemm2.packed_weight,
            gemm1.packed_weight_scale,
            gemm2.packed_weight_scale,
            gemm1.bias,
            gemm2.bias,
            block_n,
            block_k,
            self.global_num_experts,
            self.expert_map,
            self.max_num_tokens,
            self.comm_metadata,
            self.comm_ptr_wrapper,
            workspace,
        )
        return output


class XcpuEPKernelAdapter(mk.FusedMoEKernel):
    """Keep 0.25 capability queries compatible; execute through EPExperts.

    The inherited Prepare/Finalize and Experts objects supply capability and
    initialization checks only. The hot path never calls their apply methods.
    Future vLLM integration can call EPExperts.forward directly.
    """

    def __init__(self, prepare_finalize, fused_experts, ep_experts):
        super().__init__(prepare_finalize, fused_experts)
        self.ep_experts = ep_experts

    @property
    def can_overlap_shared_experts(self) -> bool:
        # Shared experts stay on the upstream MoERunner path.
        return False

    def validate_routed_contract(
        self, activation, global_num_experts, apply_router_weight_on_input
    ):
        if activation != MoEActivation.SILU or apply_router_weight_on_input:
            raise ValueError("XCPU EP requires SiLU and output router weighting")
        if global_num_experts != self.ep_experts.global_num_experts:
            raise ValueError("XCPU EP expert count changed after initialization")

    def apply(
        self,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        apply_router_weight_on_input,
        shared_experts=None,
        shared_experts_input=None,
    ):
        # The packed weights/map are bound after loading and cannot be replaced
        # during serving. Shared execution and merging stay on MoERunner.
        del w1, w2, expert_map, shared_experts, shared_experts_input
        self.validate_routed_contract(
            activation, global_num_experts, apply_router_weight_on_input
        )
        return self.ep_experts(hidden_states, topk_weights, topk_ids)
