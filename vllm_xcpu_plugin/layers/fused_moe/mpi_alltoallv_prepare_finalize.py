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
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens

from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator

logger = init_logger(__name__)


def permute_before_alltoallv(
    topk_ids: torch.Tensor,
    experts_per_rank: int,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare indices for all_to_all_single communication using vectorized operations.

    This function flattens the token-expert assignments, sorts them by target rank,
    and calculates split sizes.

    NOTE: This implementation effectively
          expands [batch_size, topk] to [batch_size * topk].

    Args:
        topk_ids: [num_tokens, topk] Tensor containing expert IDs.
        experts_per_rank: Number of experts hosted on each rank.
        ep_size: Expert Parallel world size.

    Returns:
        sort_indices: [num_tokens * topk]
                        Indices to reorder data for sending.
        split_sizes:  [ep_size]
                        CPU Tensor containing number of tokens to send to each rank.
        target_ranks: [num_tokens * topk]
                        Rank index for each flattened token (auxiliary).
    """
    # Flatten structure: [token0_k0, token0_k1, token1_k0, token1_k1, ...]
    flat_topk_ids = topk_ids.flatten()

    # Calculate which rank owns each selected expert
    # shape: [num_tokens * topk]
    target_ranks = torch.div(flat_topk_ids, experts_per_rank, rounding_mode="floor")

    # Sort by target rank to cluster data for contiguous memory sending.
    # argsort is fast enough on CPU for this purpose compared to python loops.
    sort_indices = torch.argsort(target_ranks)

    # Calculate how many tokens go to each rank (input_split_sizes)
    # bincount is highly optimized.
    split_sizes = torch.bincount(target_ranks[sort_indices], minlength=ep_size)

    return sort_indices, split_sizes, target_ranks


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
        self._row_indices_restore: torch.Tensor | None = None
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

        # Input shapes
        # a1: [num_tokens, hidden_dim]
        # topk_ids: [num_tokens, topk]
        num_tokens, hidden_dim = a1.shape
        _, topk = topk_ids.shape
        device = a1.device
        self.topk = topk
        self._topk_weights = topk_weights

        # Repeat input for each topk choice: [num_tokens * topk, hidden_dim]
        hidden_states_source = a1.repeat_interleave(topk, dim=0)

        experts_per_rank = num_experts // self.ep_size

        # 2. PERMUTE LOGIC (Vectorized)
        # We calculate sort indices based on target rank
        sort_indices, send_split_sizes_tensor, _ = permute_before_alltoallv(
            topk_ids, experts_per_rank, self.ep_size
        )

        # 3. Save state for Finalize
        # We need to map the ORDER of data we sent
        # (which will be the order we receive results back)
        # back to the ORIGINAL row index (0..num_tokens-1) for reduction.
        # Create a mapping: [0, 0, 1, 1, 2, 2 ...] for topk=2
        row_indices_restore = torch.arange(num_tokens * topk, device=device)
        # Reorder this mapping to match the data we are sending
        self._row_indices_restore = row_indices_restore[sort_indices].to(torch.int32)

        # 4. Prepare Send Tensors
        # Use index_select or advanced indexing.
        # For [N*k, D], simple indexing `tensor[indices]` is efficient.
        send_hidden_states = hidden_states_source[sort_indices]

        # Flatten and sort metadata
        send_topk_ids = topk_ids.flatten()[sort_indices]

        # 5. Exchange Split Sizes
        self._send_split_sizes = send_split_sizes_tensor.to(torch.int32)
        recv_split_sizes_tensor = torch.empty_like(send_split_sizes_tensor)

        torch_mpi_ext.ops.alltoall_out(
            recv_split_sizes_tensor,
            send_split_sizes_tensor,
            self.comm_ptr,
        )
        self._recv_split_sizes = recv_split_sizes_tensor.to(torch.int32)

        total_recv = int(recv_split_sizes_tensor.sum().item())

        # 6. Allocate Recv Buffers
        recv_hidden_states = torch.empty(
            (total_recv, hidden_dim), dtype=a1.dtype, device=device
        )
        recv_topk_ids = torch.empty((total_recv,), dtype=topk_ids.dtype, device=device)

        # 7. Perform All-to-All
        # Note: We issue multiple all_to_all calls.
        # On CPU/TCP this introduces some latency overhead
        # vs packing, but keeps logic significantly simpler
        # and avoids memory copy for packing.
        send_split_sizes_hs = send_split_sizes_tensor * hidden_dim
        recv_split_sizes_hs = recv_split_sizes_tensor * hidden_dim
        sdispls_hs = torch.nn.functional.pad(
            torch.cumsum(send_split_sizes_hs[:-1], dim=0), (1, 0)
        )
        rdispls_hs = torch.nn.functional.pad(
            torch.cumsum(recv_split_sizes_hs[:-1], dim=0), (1, 0)
        )
        torch_mpi_ext.ops.alltoallv_out(
            recvbuf=recv_hidden_states,
            sendbuf=send_hidden_states,
            sendcounts=send_split_sizes_hs,
            sdispls=sdispls_hs,
            recvcounts=recv_split_sizes_hs,
            rdispls=rdispls_hs,
            comm_ptr=self.comm_ptr,
        )

        sdispls = torch.nn.functional.pad(
            torch.cumsum(send_split_sizes_tensor[:-1], dim=0), (1, 0)
        )
        rdispls = torch.nn.functional.pad(
            torch.cumsum(recv_split_sizes_tensor[:-1], dim=0), (1, 0)
        )
        torch_mpi_ext.ops.alltoallv_out(
            recvbuf=recv_topk_ids,
            sendbuf=send_topk_ids,
            sendcounts=send_split_sizes_tensor,
            sdispls=sdispls,
            recvcounts=recv_split_sizes_tensor,
            rdispls=rdispls,
            comm_ptr=self.comm_ptr,
        )

        # 8. Calculate Metadata for Expert Computation
        # We need to reshape 1D arrays back to the format
        # expected by kernels if necessary,
        # but standard MoE kernels usually handle flattened lists or need count.
        # Here we mimic the standard vLLM metadata generation.

        # Since recv_topk_ids is now 1D [total_tokens],
        # we treat it as topk=1 for the expert counter
        # The local expert execution will treat these as individual items.
        expert_num_tokens = count_expert_num_tokens(
            recv_topk_ids.unsqueeze(1), self.num_local_experts, expert_map
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.cpu(),
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
                torch.ones(ret_topk_ids.shape, device=device, dtype=topk_weights.dtype),
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
        finalize_send_sizes_tensor = self._recv_split_sizes
        finalize_recv_sizes_tensor = self._send_split_sizes

        assert finalize_recv_sizes_tensor is not None
        assert finalize_send_sizes_tensor is not None

        device = output.device
        hidden_dim = output.size(1)

        # Calculate workspace sizes
        total_output_tokens = int(finalize_recv_sizes_tensor.sum().item())
        M = output.size(0)

        # Allocate or resize workspace buffers
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
        # We use the restored indices from the prepare phase.
        assert self._row_indices_restore is not None

        # Call the C++ operator that combines:
        # 1. Alltoallv reverse communication
        # 2. Zero output
        # 3. Unpermute and reduce (index_add)
        xcpu_ops.moe_finalize(
            output,
            fused_expert_output,
            self._row_indices_restore,
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
        self._row_indices_restore = None
        self._recv_split_sizes = None
        self._send_split_sizes = None

        def _receiver():
            pass

        return _receiver
