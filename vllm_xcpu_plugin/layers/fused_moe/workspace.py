"""Workspace contract shared by every XCPU MoE weight format."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FusedMoeWorkspace:
    permuted_hidden_states: torch.Tensor
    sorted_by_expert: torch.Tensor
    sorted_by_expert_back: torch.Tensor
    expert_offsets: torch.Tensor
    intermediate_output: torch.Tensor
    activated: torch.Tensor
    unpermute_and_reduce: torch.Tensor


@dataclass(frozen=True, slots=True)
class FusedMoeWorkspacePlan:
    input_capacity: int
    topk: int
    local_experts: int
    hidden_size: int
    intermediate_size: int
    topk_reduce: bool

    def allocate(self, device: torch.device, dtype: torch.dtype) -> FusedMoeWorkspace:
        route_capacity = self.input_capacity * self.topk
        return FusedMoeWorkspace(
            permuted_hidden_states=torch.empty(
                (route_capacity, self.hidden_size), device=device, dtype=dtype
            ),
            sorted_by_expert=torch.empty(
                route_capacity, device=device, dtype=torch.int32
            ),
            sorted_by_expert_back=torch.empty(
                route_capacity, device=device, dtype=torch.int32
            ),
            expert_offsets=torch.empty(
                self.local_experts + 1, device=device, dtype=torch.int32
            ),
            intermediate_output=torch.empty(
                (route_capacity, 2 * self.intermediate_size),
                device=device,
                dtype=dtype,
            ),
            activated=torch.empty(
                (route_capacity, self.intermediate_size),
                device=device,
                dtype=dtype,
            ),
            unpermute_and_reduce=(
                torch.empty(
                    (self.input_capacity, self.hidden_size),
                    device=device,
                    dtype=torch.float32,
                )
                if self.topk_reduce
                else torch.empty(0, device=device, dtype=torch.float32)
            ),
        )
