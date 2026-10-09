"""Attention-rank client for synchronous AF-EP V8 routed experts."""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

from ..common.session_v8 import AfV8Session
from .runtime import ExpertsClient

# All A/F ranks enter dispatch/combine in the same model layer order.


class ExpertsClientV8(ExpertsClient):
    """Execute remote routed experts with one workspace shared across layers.

    Calls must be serialized on one device and one execution stream; this client
    is not reentrant. Session initialization fixes hidden size, top-k and dtype.
    Ops may enqueue device work: returning a tensor does not synchronize it.
    Same-stream ordering keeps workspace reuse safe between consecutive layers.
    """

    def __init__(self, session: AfV8Session) -> None:
        assert session.cluster_type == ClusterType.ATTN
        self._session = session
        self._workspace: torch.Tensor | None = None
        self._workspace_bytes = 0
        self._op: Callable[..., None] | None = None

    def register_layer_capacity(self, sp_size: int) -> None:
        self._session.register_layer_capacity(sp_size)

    def sync_forward_entry(self) -> None:
        self._session.sync_forward_entry()

    def initialize(self, hidden_size: int, topk: int, dtype: torch.dtype) -> None:
        self._session.initialize(hidden_size, topk, dtype)
        self._workspace_bytes = (
            torch.ops.torch_xcpu.fused_af_a_dispatch_combine_v8_workspace_size(
                self._session.ep_size,
                self._session.max_rows_per_attention_rank,
                topk,
                hidden_size,
                dtype.itemsize,
            )
        )
        self._op = getattr(
            torch.ops.torch_xcpu,
            "fused_af_a_dispatch_combine_v8_"
            + ("bf16" if dtype == torch.bfloat16 else "fp32"),
        ).default

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
            and self._workspace.device != hidden_states.device
        ):
            raise ValueError("AF-EP client cannot change workspace device")

        num_rows, hidden_size = hidden_states.shape
        topk = topk_ids.size(1)
        if num_experts % self._session.ep_size:
            raise ValueError("AF-EP requires a uniform expert partition across F ranks")
        self._session.validate_initialized(hidden_size, topk, hidden_states.dtype)
        assert self._op is not None
        workspace = self._ensure_workspace(hidden_states, topk)
        output = torch.empty_like(hidden_states)
        self._op(
            output,
            hidden_states,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.float().contiguous(),
            num_experts,
            self._session.max_rows_per_attention_rank,
            layer_idx,
            self._session.metadata,
            self._session.communicator_handle,
            workspace,
        )
        return output

    def _ensure_workspace(self, hidden_states: torch.Tensor, topk: int) -> torch.Tensor:
        if self._workspace is None:
            self._workspace = torch.empty(
                self._workspace_bytes, dtype=torch.uint8, device=hidden_states.device
            )
        return self._workspace
