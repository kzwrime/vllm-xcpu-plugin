"""Attention-rank client for synchronous AF-EP V7 routed experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

from ..common.session_v7 import AfV7Session
from .runtime import ExpertsClient

# All A/F ranks enter dispatch/combine in the same model layer order.


def _dispatch_record_bytes(hidden_size: int, topk: int, dtype: torch.dtype) -> int:
    """Match dispatch_record_bytes in torch_xcpu/csrc/moe_ep:

    two int32 header fields, top-k IDs/weights, then one hidden-state row.
    """
    return (
        2 * torch.int32.itemsize
        + topk * (torch.int32.itemsize + torch.float32.itemsize)
        + hidden_size * dtype.itemsize
    )


@dataclass(frozen=True)
class AttentionWorkspace:
    """Scratch buffers shared by all routed layers of one A worker."""

    return_row_indices: torch.Tensor
    send_rows_per_expert_rank: torch.Tensor
    dispatch_send_buffer: torch.Tensor
    combine_workspace: torch.Tensor


class ExpertsClientV7(ExpertsClient):
    """Execute remote routed experts with one workspace shared across layers.

    Calls must be serialized on one device and one execution stream; this client
    is not reentrant. Session initialization fixes hidden size, top-k and dtype.
    Ops may enqueue device work: returning a tensor does not synchronize it.
    Same-stream ordering keeps workspace reuse safe between consecutive layers.
    """

    def __init__(self, session: AfV7Session) -> None:
        assert session.cluster_type == ClusterType.ATTN
        self._session = session
        self._workspace: AttentionWorkspace | None = None

    def sync_forward_entry(self) -> None:
        self._session.sync_forward_entry()

    def initialize(self, hidden_size: int, topk: int, dtype: torch.dtype) -> None:
        self._session.initialize(hidden_size, topk, dtype)

    def execute_layer(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
    ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 2
        assert hidden_states.size(0) <= self._session.max_rows_per_attention_rank
        assert topk_ids.dim() == topk_weights.dim() == 2
        assert topk_ids.shape == topk_weights.shape
        assert topk_ids.size(0) == hidden_states.size(0)
        assert topk_ids.size(1) in (6, 8)
        assert num_experts > 0 and num_local_experts > 0
        if not hidden_states.is_contiguous():
            raise ValueError("AF-EP hidden states must be contiguous")
        if not (hidden_states.device == topk_ids.device == topk_weights.device):
            raise ValueError(
                "AF-EP hidden states and top-k tensors must share a device"
            )
        if (
            self._workspace is not None
            and self._workspace.return_row_indices.device != hidden_states.device
        ):
            raise ValueError("AF-EP client cannot change workspace device")

        from torch_xcpu import ops as xcpu_ops

        num_rows, hidden_size = hidden_states.shape
        topk = topk_ids.size(1)
        # The A-side layer is partitioned over A ranks; remote experts are
        # independently partitioned over F ranks.
        remote_num_local_experts = num_experts // self._session.ep_size
        if num_experts % self._session.ep_size:
            raise ValueError("AF-EP requires a uniform expert partition across F ranks")

        self._session.validate_initialized(hidden_size, topk, hidden_states.dtype)
        workspace = self._ensure_workspace(hidden_states, topk)
        return_row_indices = workspace.return_row_indices[:num_rows]
        xcpu_ops.moe_af_dispatch_send_v7(
            return_row_indices,
            workspace.send_rows_per_expert_rank,
            workspace.dispatch_send_buffer,
            hidden_states,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.float().contiguous(),
            num_experts,
            remote_num_local_experts,
            self._session.max_rows_per_attention_rank,
            layer_idx,
            self._session.metadata,
            self._session.communicator_handle,
        )

        output = torch.empty_like(hidden_states)
        xcpu_ops.moe_af_combine_recv_v7(
            output,
            return_row_indices,
            self._session.metadata,
            self._session.communicator_handle,
            workspace.combine_workspace[:num_rows],
            self._session.max_rows_per_attention_rank,
            layer_idx,
        )
        return output

    def _ensure_workspace(
        self,
        hidden_states: torch.Tensor,
        topk: int,
    ) -> AttentionWorkspace:
        if self._workspace is not None:
            return self._workspace

        capacity = self._session.max_rows_per_attention_rank
        hidden_size = hidden_states.size(1)
        record_bytes = _dispatch_record_bytes(hidden_size, topk, hidden_states.dtype)
        self._workspace = AttentionWorkspace(
            return_row_indices=torch.empty(
                capacity, topk, dtype=torch.int32, device=hidden_states.device
            ),
            send_rows_per_expert_rank=torch.empty(
                self._session.ep_size,
                dtype=torch.int32,
                device=hidden_states.device,
            ),
            dispatch_send_buffer=torch.empty(
                self._session.ep_size * capacity * record_bytes,
                dtype=torch.uint8,
                device=hidden_states.device,
            ),
            combine_workspace=torch.empty(
                capacity,
                hidden_size,
                dtype=torch.float32,
                device=hidden_states.device,
            ),
        )
        return self._workspace
