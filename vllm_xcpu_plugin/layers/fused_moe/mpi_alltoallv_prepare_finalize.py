# SPDX-License-Identifier: Apache-2.0
"""
ep_all2all.py

Optimized MoE Expert Parallel (EP) communication module for CPU platforms.
Implements vector-based permutation and reduction to replace slow Python loops.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
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


class MpiAlltoallvPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
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
        self.max_num_tokens_across_ep = max_num_tokens * self.ep_size

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
        self.comm_ptr = communicator.comm_ptr

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

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
    ) -> mk.PrepareResultType:
        """
        Synchronous wrapper for prepare_async.
        """
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
    ) -> Callable:

        assert not apply_router_weight_on_input
        import torch_mpi_ext

        from torch_xcpu import ops as xcpu_ops

        num_tokens, hidden_dim = a1.shape
        _, topk = topk_ids.shape
        device = a1.device
        self.topk = topk
        self._topk_weights = topk_weights

        experts_per_rank = num_experts // self.ep_size

        total_tokens = num_tokens * topk
        avg_size = total_tokens // self.tp_size
        extras = total_tokens % self.tp_size
        # tp_start = self.tp_rank * avg_size + min(self.tp_rank, extras)
        # tp_end = tp_start + avg_size + (1 if self.tp_rank < extras else 0)
        local_tokens = avg_size + (1 if self.tp_rank < extras else 0)

        # =============================================================================
        # Allocate tensors (shape-independent first, then shape-dependent)
        # =============================================================================

        # --- Communication metadata (persistent, cached) ---
        if self._full_send_split_sizes is None:
            self._full_send_split_sizes = torch.empty(
                self.ep_size, dtype=torch.int32, device=device
            )
        if self._recv_split_sizes is None:
            self._recv_split_sizes = torch.empty(
                self.ep_size, dtype=torch.int32, device=device
            )

        # --- For moe_prepare_phase1 ---
        sort_indices_back = torch.empty(total_tokens, dtype=torch.int32, device=device)
        sort_indices = torch.empty(local_tokens, dtype=torch.int32, device=device)
        expert_count = torch.empty(num_experts, dtype=torch.int32, device=device)

        # --- For alltoall ---
        recv_expert_count_flat = torch.empty_like(expert_count)

        # --- For moe_prepare_phase2 (shape-independent) ---
        send_hidden_states = torch.empty(
            local_tokens, hidden_dim, dtype=a1.dtype, device=device
        )
        expert_num_tokens = torch.empty(
            experts_per_rank, dtype=torch.int32, device=device
        )
        send_split_sizes = torch.empty(self.ep_size, dtype=torch.int32, device=device)
        sdispls_hs = torch.empty(self.ep_size, dtype=torch.int32, device=device)
        rdispls_hs = torch.empty(self.ep_size, dtype=torch.int32, device=device)

        # =============================================================================
        # Call operators
        # =============================================================================

        xcpu_ops.moe_prepare_phase1(
            sort_indices,
            sort_indices_back,
            expert_count,
            self._full_send_split_sizes,
            topk_ids,
            experts_per_rank,
            self.ep_size,
            hidden_dim,
            self.tp_rank,
            self.tp_size,
        )

        self._sort_indices_back = sort_indices_back

        assert num_experts % self.ep_size == 0, (
            "num_experts must be divisible by ep_size"
        )
        torch_mpi_ext.ops.alltoall_out(
            recv_expert_count_flat,
            expert_count,
            self.comm_ptr,
        )

        static_buffer_size = (
            self.max_num_tokens * self.dp_size * min(topk, self.num_local_experts)
        )

        recv_topk_ids = torch.empty(
            static_buffer_size, dtype=torch.int32, device=device
        )
        recv_hidden_states = torch.empty(
            (static_buffer_size, hidden_dim), dtype=a1.dtype, device=device
        )

        ret_topk_ids_shape = (static_buffer_size, 1)

        xcpu_ops.moe_prepare_phase2(
            send_hidden_states,
            recv_topk_ids,
            send_split_sizes,
            self._recv_split_sizes,
            expert_num_tokens,
            sdispls_hs,
            rdispls_hs,
            a1,
            sort_indices,
            expert_count,
            recv_expert_count_flat,
            self.ep_rank,
            experts_per_rank,
            self.ep_size,
            topk,
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.cpu(),
        )

        torch_mpi_ext.ops.alltoallv_out(
            recvbuf=recv_hidden_states,
            sendbuf=send_hidden_states,
            sendcounts=send_split_sizes,
            sdispls=sdispls_hs,
            recvcounts=self._recv_split_sizes,
            rdispls=rdispls_hs,
            comm_ptr=self.comm_ptr,
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
        receiver = self.finalize_async(
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
    ) -> Callable:
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

        xcpu_ops.moe_finalize(
            output,
            fused_expert_output,
            self._sort_indices_back,
            finalize_send_sizes_tensor,
            finalize_recv_sizes_tensor,
            self.ep_size,
            self.tp_size,
            self.comm_ptr,
            _finalize_recv_hidden_states,
            _finalize_workspace,
            self.topk,
            self._topk_weights,
        )
        self._topk_weights = None
        self._sort_indices_back = None

        def _receiver():
            pass

        return _receiver
