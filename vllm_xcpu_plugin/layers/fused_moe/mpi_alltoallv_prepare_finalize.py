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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator

logger = init_logger(__name__)


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
        ep_group: dist.ProcessGroup,
        num_local_experts: int,
        num_dispatchers: int,
        rank_expert_offset: int,
    ):
        super().__init__()
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers
        self.rank_expert_offset = rank_expert_offset

        self.ep_rank = dist.get_rank(self.ep_group)
        self.ep_size = dist.get_world_size(self.ep_group)

        # Context storage for finalize phase
        # We need to know where to put the received data back
        self._sort_indices_back: torch.Tensor | None = None
        self.topk: int = -1

        # Communication metadata
        self._send_split_sizes: torch.Tensor | None = None
        self._recv_split_sizes: torch.Tensor | None = None

        self._topk_weights: torch.Tensor | None = None

        communicator = get_ep_group().device_communicator
        assert isinstance(communicator, CpuMPICommunicator)
        self.comm_ptr = communicator.comm_ptr

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None  # Dynamic sizing

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

        # Input shapes
        # a1: [num_tokens, hidden_dim]
        # topk_ids: [num_tokens, topk]
        num_tokens, hidden_dim = a1.shape
        _, topk = topk_ids.shape
        device = a1.device
        self.topk = topk
        self._topk_weights = topk_weights

        # Move pre-allocated arrays to device
        if self._send_split_sizes is None:
            self._send_split_sizes = torch.empty(
                self.ep_size, dtype=torch.int32, device=device
            )
        if self._recv_split_sizes is None:
            self._recv_split_sizes = torch.empty(
                self.ep_size, dtype=torch.int32, device=device
            )

        # Repeat input for each topk choice: [num_tokens * topk, hidden_dim]
        hidden_states_source = a1.repeat_interleave(topk, dim=0)

        experts_per_rank = num_experts // self.ep_size

        # 2. PERMUTE LOGIC (Vectorized) - Now using C++ operator
        # Allocate output tensors for the C++ operator
        total_tokens = num_tokens * topk
        sort_indices = torch.empty(total_tokens, dtype=torch.int32, device=device)
        sort_indices_back = torch.empty(total_tokens, dtype=torch.int32, device=device)
        expert_count = torch.empty(num_experts, dtype=torch.int32, device=device)

        xcpu_ops.moe_prepare_phase1(
            sort_indices,
            sort_indices_back,
            expert_count,
            topk_ids,
            experts_per_rank,
            self.ep_size,
        )

        # 3. Save state for Finalize
        # sort_indices_back maps original position -> sorted position
        # We'll use this in finalize to do sequential writes
        self._sort_indices_back = sort_indices_back

        # 4. Exchange Expert Count via alltoall (BEFORE phase2 operator)
        assert num_experts % self.ep_size == 0, (
            "num_experts must be divisible by ep_size"
        )
        recv_expert_count_flat = torch.empty_like(expert_count)
        torch_mpi_ext.ops.alltoall_out(
            recv_expert_count_flat,
            expert_count,
            self.comm_ptr,
        )

        # 5. Compute total_recv to correctly size recv_topk_ids and recv_hidden_states
        total_recv = int(recv_expert_count_flat.sum().item())

        # 6. Call moe_prepare_phase2: compute
        #      split_sizes + permute + reconstruct + recv_topk_ids
        send_hidden_states = torch.empty_like(hidden_states_source)
        recv_topk_ids = torch.empty(total_recv, dtype=torch.int32, device=device)
        expert_num_tokens = torch.empty(
            experts_per_rank, dtype=torch.int32, device=device
        )

        xcpu_ops.moe_prepare_phase2(
            send_hidden_states,
            recv_topk_ids,
            self._send_split_sizes,
            self._recv_split_sizes,
            expert_num_tokens,
            hidden_states_source,
            sort_indices,
            expert_count,
            recv_expert_count_flat,
            self.ep_rank,
            experts_per_rank,
            self.ep_size,
        )

        # 7. Store split_sizes multiplied by hidden_dim for finalize
        # In-place multiplication: self._send_split_sizes *= hidden_dim
        self._send_split_sizes.mul_(hidden_dim)
        self._recv_split_sizes.mul_(hidden_dim)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.cpu(),
        )

        # 8. Allocate Recv Buffers and Perform All-to-All for hidden_states
        recv_hidden_states = torch.empty(
            (total_recv, hidden_dim), dtype=a1.dtype, device=device
        )

        # Use pre-computed split_sizes (already multiplied by hidden_dim)
        sdispls_hs = torch.nn.functional.pad(
            torch.cumsum(self._send_split_sizes[:-1], dim=0), (1, 0)
        )
        rdispls_hs = torch.nn.functional.pad(
            torch.cumsum(self._recv_split_sizes[:-1], dim=0), (1, 0)
        )
        torch_mpi_ext.ops.alltoallv_out(
            recvbuf=recv_hidden_states,
            sendbuf=send_hidden_states,
            sendcounts=self._send_split_sizes,
            sdispls=sdispls_hs,
            recvcounts=self._recv_split_sizes,
            rdispls=rdispls_hs,
            comm_ptr=self.comm_ptr,
        )

        def _receiver() -> mk.PrepareResultType:
            # vLLM expects 2D topk_ids/weights [tokens, topk] usually, but since we
            # broke down the batch into individual tokens for EP,
            # we return [total_recv, 1].
            ret_topk_ids = recv_topk_ids.unsqueeze(1)
            return (
                recv_hidden_states,
                None,  # no quant scale
                expert_tokens_meta,
                ret_topk_ids,
                torch.empty(
                    ret_topk_ids.shape, device=device, dtype=topk_weights.dtype
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

        # 1. Reverse Communication (Experts -> Original Ranks) + Unpermute and Reduce
        # Send back what we received.
        # send_split_sizes (for finalize) == recv_split_sizes (from prepare)
        # Note: these are already multiplied by hidden_dim
        finalize_send_sizes_tensor = self._recv_split_sizes
        finalize_recv_sizes_tensor = self._send_split_sizes

        device = output.device
        hidden_dim = output.size(1)
        M = output.size(0)
        total_output_tokens = M * self.topk

        # Allocate workspace buffers
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

        # The magic happens here:
        # We use sort_indices_back from the prepare phase for sequential writes
        assert self._sort_indices_back is not None

        # Call the C++ operator that combines:
        # 1. Alltoallv reverse communication
        # 2. Zero output
        # 3. Unpermute and reduce (sequential write using sort_indices_back)
        xcpu_ops.moe_finalize(
            output,
            fused_expert_output,
            self._sort_indices_back,
            finalize_send_sizes_tensor,
            finalize_recv_sizes_tensor,
            self.ep_size,
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
