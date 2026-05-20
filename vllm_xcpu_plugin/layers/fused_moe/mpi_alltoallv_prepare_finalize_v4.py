# SPDX-License-Identifier: Apache-2.0
"""
ep_all2all.py

Optimized MoE Expert Parallel (EP) communication module for CPU platforms.
Implements vector-based permutation and reduction to replace slow Python loops.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator


class MpiAlltoallvPrepareAndFinalizeV4(mk.FusedMoEPrepareAndFinalizeModular):
    """
    High-performance CPU implementation of Expert Parallel communication.

    Improvements over original:
    1. Removes all Python loops for index generation (vectorized).
    2. Uses C++ kernel for finalize (alltoallv + unpermute+reduce fused).
    3. Removes quantization overhead.
    """

    def __init__(
        self,
        max_num_tokens: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank_expert_offset: int,
        dp_rank: int,
        dp_size: int,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers
        self.rank_expert_offset = rank_expert_offset

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_rank = dist.get_rank(self.ep_group)
        self.ep_size = dist.get_world_size(self.ep_group)

        self.num_local_experts_ranks = self._compute_expert_distribution(
            num_experts, self.ep_size
        )
        self.experts_local_rank = self.num_local_experts_ranks[self.ep_rank]
        assert self.experts_local_rank == num_local_experts, (
            f"{self.num_local_experts_ranks}"
        )

        # Context storage for finalize phase
        # We need to know where to put the received data back
        self._sort_indices_back: torch.Tensor | None = None
        self.topk: int = -1

        # Communication metadata
        self._full_send_split_sizes: torch.Tensor | None = None
        self._recv_split_sizes: torch.Tensor | None = None

        self._topk_weights: torch.Tensor | None = None

        communicator = get_ep_group().device_communicator
        assert isinstance(communicator, CpuMPICommunicator)
        self.comm_ptr_wrapper = communicator.comm_ptr_wrapper

        # Communication metadata tensor:
        # [ep_size, ep_rank, tp_rank, tp_size, dp_rank, dp_size]
        self._comm_metadata = torch.tensor(
            [
                self.ep_size,
                self.ep_rank,
                self.tp_rank,
                self.tp_size,
                self.dp_rank,
                self.dp_size,
            ],
            dtype=torch.int64,
            device="cpu",  # Keep on CPU for C++ access
        )

        if not envs.VLLM_ENABLE_MOE_DP_CHUNK:
            vllm_config = get_current_vllm_config()
            assert (
                vllm_config.scheduler_config.max_num_batched_tokens
                <= self.max_num_tokens
            )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def _compute_expert_distribution(
        self, num_experts: int, ep_size: int
    ) -> torch.Tensor:
        """Compute expert distribution across ranks (deterministic)."""
        base_experts = num_experts // ep_size
        extra_experts = num_experts % ep_size

        num_local_experts_ranks = torch.empty(ep_size, dtype=torch.int32)
        for i in range(ep_size):
            num_local_experts_ranks[i] = base_experts + (1 if i < extra_experts else 0)

        return num_local_experts_ranks

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig | None = None,  # Not used
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """
        Synchronous wrapper for prepare_async.
        """
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        return receiver()

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig | None = None,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )

        assert not apply_router_weight_on_input
        assert a1.shape[0] <= self.max_num_tokens, (
            "Check --max-num-batched-tokens and VLLM_MOE_DP_CHUNK_SIZE"
        )

        from torch_xcpu import ops as xcpu_ops

        num_tokens, hidden_dim = a1.shape
        _, topk = topk_ids.shape
        device = a1.device
        self.topk = topk
        self._topk_weights = topk_weights

        total_tokens = num_tokens * topk

        # =============================================================================
        # Allocate tensors (shape-independent first, then shape-dependent)
        # =============================================================================

        # --- Communication metadata (persistent, cached) ---
        self._full_send_split_sizes = torch.empty(
            self.ep_size, dtype=torch.int32, device=device
        )
        self._recv_split_sizes = torch.empty(
            self.ep_size, dtype=torch.int32, device=device
        )

        # --- For fused operator ---
        sort_indices_back = torch.empty(total_tokens, dtype=torch.int32, device=device)
        expert_num_tokens = torch.empty(num_experts, dtype=torch.int32, device=device)

        static_buffer_padding_per_dp = (
            self.num_local_experts + hidden_dim - 1
        ) // hidden_dim

        # static_buffer_size = (
        #     self.dp_size * self.max_num_tokens * min(topk, self.num_local_experts)
        #     + self.ep_size * static_buffer_padding_per_dp
        # )
        static_buffer_size = (
            self.dp_size * self.max_num_tokens * min(topk, self.num_local_experts)
            + self.ep_size * static_buffer_padding_per_dp
        )
        assert static_buffer_size % self.ep_size == 0

        recv_topk_ids = torch.empty(
            static_buffer_size, dtype=torch.int32, device=device
        )
        recv_hidden_states = torch.empty(
            (static_buffer_size, hidden_dim), dtype=a1.dtype, device=device
        )

        # --- Pre-allocate send_hidden_states (shape-dependent) ---
        # Size depends on local_tokens and hidden_dim
        send_hidden_states = torch.empty(
            total_tokens + self.ep_size, hidden_dim, dtype=a1.dtype, device=device
        )
        # send_hidden_states = torch.empty(
        #     local_tokens, hidden_dim, dtype=a1.dtype, device=device
        # )

        # --- Pre-allocate sort_indices (needed for phase2) ---
        sort_indices = torch.empty(
            total_tokens + self.ep_size, dtype=torch.int32, device=device
        )

        # Workspace is no longer used internally, pass empty tensor for compatibility
        workspace = torch.empty(0, dtype=torch.int32, device=device)

        ret_topk_ids_shape = (static_buffer_size, 1)

        # =============================================================================
        # Call fused operator (phase1 + alltoall + phase2 + alltoallv)
        # =============================================================================

        xcpu_ops.moe_prepare_fused_v4(
            sort_indices_back,  # output
            recv_hidden_states,  # output
            recv_topk_ids,  # output
            expert_num_tokens,  # output
            self._recv_split_sizes,  # output
            self._full_send_split_sizes,  # output
            send_hidden_states,
            sort_indices,
            workspace,
            a1,
            topk_ids.to(torch.int32),
            static_buffer_size,
            num_experts,
            self.num_local_experts,
            hidden_dim,
            self._comm_metadata,
            self.comm_ptr_wrapper,
        )

        self._sort_indices_back = sort_indices_back

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.cpu(),
        )

        def _receiver() -> mk.PrepareResultType:
            ret_topk_ids = recv_topk_ids.unsqueeze(1)
            return (
                recv_hidden_states,
                None,  # no quant scale
                expert_tokens_meta,
                ret_topk_ids,
                torch.empty(
                    ret_topk_ids_shape, device=device, dtype=topk_weights.dtype
                ),
            )

        return _receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        receiver, _ = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        receiver()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        from torch_xcpu import ops as xcpu_ops

        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP)

        finalize_send_sizes_tensor = self._recv_split_sizes
        finalize_recv_sizes_tensor = self._full_send_split_sizes

        device = output.device
        hidden_dim = output.size(1)
        M = output.size(0)
        total_output_tokens = M * self.topk

        # =============================================================================
        # Allocate tensors
        # =============================================================================

        _finalize_recv_hidden_states = torch.empty(
            (total_output_tokens, hidden_dim),
            dtype=output.dtype,
            device=device,
        )
        _finalize_workspace = torch.empty(
            (M, hidden_dim),
            dtype=torch.float32,
            device=device,
        )

        # =============================================================================
        # Call moe_finalize
        # =============================================================================

        assert self._sort_indices_back is not None

        xcpu_ops.moe_finalize_v3(
            output,
            fused_expert_output,
            self._sort_indices_back,
            finalize_send_sizes_tensor,
            finalize_recv_sizes_tensor,
            self._comm_metadata,
            self.comm_ptr_wrapper,
            _finalize_recv_hidden_states,
            _finalize_workspace,
            self.topk,
            self._topk_weights,
        )
        self._topk_weights = None
        self._sort_indices_back = None
        self._full_send_split_sizes = None
        self._recv_split_sizes = None

        def _receiver():
            pass

        return _receiver, lambda: None
