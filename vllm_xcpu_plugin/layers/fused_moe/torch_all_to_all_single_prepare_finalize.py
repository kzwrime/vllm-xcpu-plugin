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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens

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


def unpermute_and_reduce_after_alltoallv(
    output: torch.Tensor,
    fused_expert_output: torch.Tensor,
    row_indices: torch.Tensor,
) -> None:
    """
    Scatter results back to original positions and reduce (sum) them.

    Args:
        output: [num_tokens, hidden_dim]
                 The destination tensor (will be modified in-place).
        fused_expert_output: [total_recv_tokens, hidden_dim]
                              Results received from experts.
        row_indices: [total_recv_tokens] Indices mapping each result row back to
                     the original token index (0..num_tokens-1).
    """
    # fused_expert_output contains results for
    #   [token_i_exp_a, token_i_exp_b, token_j_exp_a...]
    # row_indices contains [i, i, j...] matching the order of fused_expert_output.
    # index_add_ performs the reduction: output[idx] += src[i]
    output.index_add_(0, row_indices, fused_expert_output)


class TorchAlltoallSinglePrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    High-performance CPU implementation of Expert Parallel communication.

    Improvements over original:
    1. Removes all Python loops for index generation (vectorized).
    2. Uses index_add_ for fast unpermute-and-reduce.
    3. Removes quantization overhead.
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        num_local_experts: int,
        num_dispatchers: int,
        rank_expert_offset: int,
        tp_rank: int,
        tp_size: int,
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

        # Communication metadata
        self._send_split_sizes: list[int] | None = None
        self._recv_split_sizes: list[int] | None = None

        # communicator = get_ep_group().device_communicator
        # assert isinstance(communicator, CpuMPICommunicator)
        # self.comm_ptr = communicator.comm_ptr

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

        # Input shapes
        # a1: [num_tokens, hidden_dim]
        # topk_ids: [num_tokens, topk]
        num_tokens, hidden_dim = a1.shape
        _, topk = topk_ids.shape
        device = a1.device

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
        original_row_indices = torch.arange(
            num_tokens, device=device
        ).repeat_interleave(topk)
        # Reorder this mapping to match the data we are sending
        self._row_indices_restore = original_row_indices[sort_indices]

        # 4. Prepare Send Tensors
        # Use index_select or advanced indexing.
        # For [N*k, D], simple indexing `tensor[indices]` is efficient.
        send_hidden_states = hidden_states_source[sort_indices]

        # Flatten and sort metadata
        send_topk_ids = topk_ids.flatten()[sort_indices]
        send_topk_weights = topk_weights.flatten()[sort_indices]

        # 5. Exchange Split Sizes
        self._send_split_sizes = send_split_sizes_tensor.tolist()
        recv_split_sizes_tensor = torch.empty_like(send_split_sizes_tensor)

        dist.all_to_all_single(
            recv_split_sizes_tensor,
            send_split_sizes_tensor,
            group=self.ep_group,
        )
        self._recv_split_sizes = recv_split_sizes_tensor.tolist()

        total_recv = int(recv_split_sizes_tensor.sum().item())

        # 6. Allocate Recv Buffers
        recv_hidden_states = torch.empty(
            (total_recv, hidden_dim), dtype=a1.dtype, device=device
        )
        recv_topk_ids = torch.empty((total_recv,), dtype=topk_ids.dtype, device=device)
        recv_topk_weights = torch.empty(
            (total_recv,), dtype=topk_weights.dtype, device=device
        )

        # 7. Perform All-to-All
        # Note: We issue multiple all_to_all calls.
        # On CPU/TCP this introduces some latency overhead
        # vs packing, but keeps logic significantly simpler
        # and avoids memory copy for packing.
        dist.all_to_all_single(
            recv_hidden_states,
            send_hidden_states,
            output_split_sizes=self._recv_split_sizes,
            input_split_sizes=self._send_split_sizes,
            group=self.ep_group,
        )

        dist.all_to_all_single(
            recv_topk_ids,
            send_topk_ids,
            output_split_sizes=self._recv_split_sizes,
            input_split_sizes=self._send_split_sizes,
            group=self.ep_group,
        )

        dist.all_to_all_single(
            recv_topk_weights,
            send_topk_weights,
            output_split_sizes=self._recv_split_sizes,
            input_split_sizes=self._send_split_sizes,
            group=self.ep_group,
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

        print(f"expert_num_tokens is {expert_num_tokens.sum().item()}")

        def _receiver() -> mk.PrepareResultType:
            # vLLM expects 2D topk_ids/weights [tokens, topk] usually, but since we
            # broke down the batch into individual tokens for EP,
            # we return [total_recv, 1].
            return (
                recv_hidden_states,
                None,  # no quant scale
                expert_tokens_meta,
                recv_topk_ids.unsqueeze(1),
                recv_topk_weights.unsqueeze(1),
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

        # # Validation
        # if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
        #      # Ensure we use a contiguous reducer logic if delegated
        #      weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP)

        # 1. Apply Weights (if not done on input)
        # Note: topk_weights here comes from the 'prepare' return,
        # so it is [total_recv, 1] fused_expert_output is [total_recv, hidden_dim]
        # if not apply_router_weight_on_input and fused_expert_output.numel() > 0:
        #     fused_expert_output = fused_expert_output * topk_weights

        # 2. Reverse Communication (Experts -> Original Ranks)
        # Send back what we received.
        # send_split_sizes (for finalize) == recv_split_sizes (from prepare)
        finalize_send_sizes = self._recv_split_sizes
        finalize_recv_sizes = self._send_split_sizes

        assert finalize_recv_sizes is not None

        total_output_tokens = sum(finalize_recv_sizes)
        hidden_dim = output.size(1)
        dtype = output.dtype
        device = output.device

        recv_hidden_states = torch.empty(
            (total_output_tokens, hidden_dim), dtype=dtype, device=device
        )

        # 3. All-to-All Single
        dist.all_to_all_single(
            recv_hidden_states,
            fused_expert_output,
            output_split_sizes=finalize_recv_sizes,
            input_split_sizes=finalize_send_sizes,
            group=self.ep_group,
        )

        # 4. UNPERMUTE AND REDUCE
        # We need to reset output first because we are accumulating
        output.zero_()

        # The magic happens here:
        # We use the restored indices from the prepare phase.
        # recv_hidden_states is ordered exactly as we sent them in prepare.
        # self._row_indices_restore maps that order to the original row index.
        assert self._row_indices_restore is not None
        unpermute_and_reduce_after_alltoallv(
            output, recv_hidden_states, self._row_indices_restore
        )

        def _receiver():
            pass

        return _receiver
