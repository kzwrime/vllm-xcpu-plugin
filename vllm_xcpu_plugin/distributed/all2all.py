# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.forward_context import get_forward_context


class All2allvSingleAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all_to_all_single (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

        self.get_cpu_view_from_mcpu_tensor = torch.mcpu.get_cpu_view_from_mcpu_tensor  # type: ignore
        self.get_mcpu_view_from_cpu_tensor = torch.mcpu.get_mcpu_view_from_cpu_tensor  # type: ignore

    def _requires_cpu_staging(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type != "cpu"

    def _to_cpu_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._requires_cpu_staging(tensor):
            if self.get_cpu_view_from_mcpu_tensor is not None:
                return self.get_cpu_view_from_mcpu_tensor(tensor)
            return tensor.to("cpu")
        return tensor

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        assert hidden_states.dim() == 2
        output_hidden_states = torch.empty(
            (sum(sizes), hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        assert router_logits.dim() == 2
        output_router_logits = torch.empty(
            (sum(sizes), router_logits.shape[1]),
            dtype=router_logits.dtype,
            device=router_logits.device,
        )
        input_split_sizes = [
            sizes[dist_group.rank_in_group] for _ in range(dist_group.world_size)
        ]

        # TODO use expand instead of repeat ?
        cpu_output_hidden_states = self._to_cpu_from_device(output_hidden_states)
        cpu_input_hidden_states = self._to_cpu_from_device(hidden_states).repeat(
            dist_group.world_size, 1
        )
        cpu_output_router_logits = self._to_cpu_from_device(output_router_logits)
        cpu_input_router_logits = self._to_cpu_from_device(router_logits).repeat(
            dist_group.world_size, 1
        )
        dist.all_to_all_single(
            cpu_output_hidden_states,
            cpu_input_hidden_states,
            output_split_sizes=sizes,
            input_split_sizes=input_split_sizes,
            group=dist_group.cpu_group,
        )
        dist.all_to_all_single(
            cpu_output_router_logits,
            cpu_input_router_logits,
            output_split_sizes=sizes,
            input_split_sizes=input_split_sizes,
            group=dist_group.cpu_group,
        )
        return output_hidden_states, output_router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        assert hidden_states.dim() == 2
        output_hidden_states = torch.empty(
            (sum(sizes), hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        assert topk_weights.dim() == 2
        output_topk_weights = torch.empty(
            (sum(sizes), topk_weights.shape[1]),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        assert topk_ids.dim() == 2
        output_topk_ids = torch.empty(
            (sum(sizes), topk_ids.shape[1]),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        input_split_sizes = [
            sizes[dist_group.rank_in_group] for _ in range(dist_group.world_size)
        ]
        dist.all_to_all_single(
            output_hidden_states,
            hidden_states.repeat(dist_group.world_size, 1),  # TODO use expand
            output_split_sizes=sizes,
            input_split_sizes=input_split_sizes,
        )
        dist.all_to_all_single(
            output_topk_weights,
            topk_weights.repeat(dist_group.world_size, 1),  # TODO use expand
            output_split_sizes=sizes,
            input_split_sizes=input_split_sizes,
        )
        dist.all_to_all_single(
            output_topk_ids,
            topk_ids.repeat(dist_group.world_size, 1),  # TODO use expand
            output_split_sizes=sizes,
            input_split_sizes=input_split_sizes,
        )
        return output_hidden_states, output_topk_weights, output_topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        recv_hidden_states = dist_group.all_reduce(hidden_states)

        size_sum = 0
        offsets = [0 for i in range(dist_group.world_size)]
        for i in range(len(sizes)):
            size_sum += sizes[i]
            offsets[i] = size_sum

        start = (
            0
            if dist_group.rank_in_group == 0
            else int(offsets[dist_group.rank_in_group - 1])
        )
        end = int(offsets[dist_group.rank_in_group])
        hidden_states = recv_hidden_states[start:end]

        return hidden_states

    def destroy(self):
        pass
