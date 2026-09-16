"""Synchronous F-rank execution endpoint for AF-EP V7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

from ..common.session_v7 import AfV7Session

if TYPE_CHECKING:
    from vllm_xcpu_plugin.layers.fused_moe.workspace import FusedMoeWorkspace

    from .model import RoutedExpertsModel


@dataclass(frozen=True)
class _ExpertBuffers:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    num_input_rows_valid: torch.Tensor
    hidden_elements_per_a_rank: torch.Tensor
    hidden_element_offsets_per_a_rank: torch.Tensor
    expert_num_tokens: torch.Tensor
    workspace: FusedMoeWorkspace


@dataclass(frozen=True)
class _ExpertLayer:
    layer_idx: int
    backend: Any
    expert_map: torch.Tensor | None


class ExpertServiceV7:
    """Run one dispatch -> local experts -> combine transaction at a time."""

    def __init__(
        self,
        model: RoutedExpertsModel,
        session: AfV7Session,
    ) -> None:
        assert session.cluster_type == ClusterType.MOE
        assert session.ep_size == model.ep_size
        assert session.role_rank == model.ep_rank
        self.model = model
        self.session = session
        self._capacity = session.expert_capacity
        self._buffers = self._allocate_buffers()
        self._layers = tuple(
            self._prepare_layer(layer_idx) for layer_idx in model.layer_indices
        )

    def initialize(self) -> None:
        self.session.initialize(
            self.model.hidden_size,
            self.model.top_k,
            self._buffers.hidden_states.dtype,
        )

    def execute_model_pass(self) -> None:
        self.session.validate_initialized(
            self.model.hidden_size,
            self.model.top_k,
            self._buffers.hidden_states.dtype,
        )
        self.session.sync_forward_entry()
        for layer in self._layers:
            self._execute_layer(layer)
        # Same-stream ordering protects per-layer reuse. Bound host submissions
        # to one model pass so the infinite service loop cannot grow the queue.
        torch.accelerator.synchronize()

    def _execute_layer(self, layer: _ExpertLayer) -> None:
        from torch_xcpu import ops as xcpu_ops

        self._receive(layer.layer_idx, xcpu_ops)
        self._compute(layer, xcpu_ops)
        self._send(layer.layer_idx, xcpu_ops)

    def _receive(self, layer_idx: int, xcpu_ops: Any) -> None:
        buffers = self._buffers
        xcpu_ops.moe_af_dispatch_recv_v7(
            buffers.hidden_states,
            buffers.topk_ids,
            buffers.topk_weights,
            buffers.num_input_rows_valid,
            buffers.hidden_elements_per_a_rank,
            buffers.hidden_element_offsets_per_a_rank,
            layer_idx,
            self.session.metadata,
            self.session.communicator_handle,
        )

    def _compute(self, layer: _ExpertLayer, xcpu_ops: Any) -> None:
        buffers = self._buffers
        workspace = buffers.workspace
        xcpu_ops.fused_moe_compute(
            output=buffers.output,
            hidden_states=buffers.hidden_states,
            backend=layer.backend,
            topk_weights=buffers.topk_weights,
            topk_ids=buffers.topk_ids,
            activation=self.model.hidden_act,
            global_num_experts=self.model.num_experts,
            expert_map=layer.expert_map,
            expert_num_tokens=buffers.expert_num_tokens,
            num_input_rows_valid=buffers.num_input_rows_valid,
            topk_reduce=True,
            permuted_hidden_states=workspace.permuted_hidden_states,
            sorted_by_expert=workspace.sorted_by_expert,
            sorted_by_expert_back=workspace.sorted_by_expert_back,
            expert_offsets=workspace.expert_offsets,
            intermediate_output=workspace.intermediate_output,
            activated=workspace.activated,
            workspace_unpermute_and_reduce=workspace.unpermute_and_reduce,
        )

    def _send(self, layer_idx: int, xcpu_ops: Any) -> None:
        buffers = self._buffers
        xcpu_ops.moe_af_combine_send_v7(
            buffers.output,
            buffers.hidden_elements_per_a_rank,
            buffers.hidden_element_offsets_per_a_rank,
            self.session.metadata,
            self.session.communicator_handle,
            self.session.max_rows_per_attention_rank,
            self.model.top_k,
            layer_idx,
        )
        # 调试错误时可开启:
        # torch.accelerator.synchronize()

    def _prepare_layer(self, layer_idx: int) -> _ExpertLayer:
        expert_map = self.model.routed_experts[str(layer_idx)].expert_map
        if expert_map is not None:
            expert_map = expert_map.to(
                device=self._buffers.hidden_states.device,
                dtype=torch.int32,
            ).contiguous()
        return _ExpertLayer(
            layer_idx=layer_idx,
            backend=self.model.fused_moe_for_layer(layer_idx),
            expert_map=expert_map,
        )

    def _allocate_buffers(self) -> _ExpertBuffers:
        from vllm_xcpu_plugin.layers.fused_moe.workspace import FusedMoeWorkspacePlan

        device = self.model.device
        workspace = FusedMoeWorkspacePlan(
            input_capacity=self._capacity,
            topk=self.model.top_k,
            local_experts=self.model.num_experts // self.model.ep_size,
            hidden_size=self.model.hidden_size,
            intermediate_size=self.model.intermediate_size,
            topk_reduce=True,
        ).allocate(device, self.model.dtype)
        return _ExpertBuffers(
            hidden_states=torch.empty(
                self._capacity,
                self.model.hidden_size,
                dtype=self.model.dtype,
                device=device,
            ),
            topk_ids=torch.empty(
                self._capacity,
                self.model.top_k,
                dtype=torch.int32,
                device=device,
            ),
            topk_weights=torch.empty(
                self._capacity,
                self.model.top_k,
                dtype=torch.float32,
                device=device,
            ),
            output=torch.empty(
                self._capacity,
                self.model.hidden_size,
                dtype=self.model.dtype,
                device=device,
            ),
            num_input_rows_valid=torch.empty(1, dtype=torch.int32, device=device),
            hidden_elements_per_a_rank=torch.empty(
                self.session.num_attention_ranks,
                dtype=torch.int32,
                device=device,
            ),
            hidden_element_offsets_per_a_rank=torch.empty(
                self.session.num_attention_ranks,
                dtype=torch.int32,
                device=device,
            ),
            expert_num_tokens=torch.empty(0, dtype=torch.int32, device=device),
            workspace=workspace,
        )
