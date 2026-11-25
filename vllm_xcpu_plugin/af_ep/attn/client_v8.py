"""Attention-rank client for synchronous AF-EP V8 routed experts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

from ..common.session_v8 import AfV8Session
from .runtime import ExpertsClient

if TYPE_CHECKING:
    from vllm_xcpu_plugin.layers.fused_moe.ep_experts import XcpuSharedExpertWeights

# All A/F ranks enter dispatch/combine in the same model layer order.


@dataclass(frozen=True)
class _BoundSharedExpert:
    weights: XcpuSharedExpertWeights
    routed_scaling_factor: float
    swiglu_limit: float


class ExpertsClientV8(ExpertsClient):
    """Execute remote routed experts with one workspace shared across layers.

    Mirrors fused_ep_moe_v5 on A ranks: v5 fixed per-rank region wire protocol
    reusing A's combine window as the dispatch send buffer, optional chunked
    multi-round execution, sequence-parallel padding skip
    (topk_ids.size(0) <= hidden_states.size(0)), topk=3/6/8, non-divisible
    expert padding, and optional fused shared expert overlapped with remote F
    compute.
    """

    supports_num_valid_tokens = True

    def __init__(self, session: AfV8Session) -> None:
        assert session.cluster_type == ClusterType.ATTN
        self._session = session
        self._workspace: torch.Tensor | None = None
        self._workspace_bytes = 0
        self._op: Callable[..., None] | None = None
        self._shared_by_layer: dict[int, _BoundSharedExpert] = {}
        self._max_shared_intermediate = 0
        self._init_args: tuple[int, int, torch.dtype] | None = None

    def register_layer_capacity(self, sp_size: int) -> None:
        self._session.register_layer_capacity(sp_size)

    def register_num_experts(self, num_experts: int) -> None:
        self._session.register_num_experts(num_experts)

    def bind_shared_experts(
        self,
        layer_idx: int,
        shared: XcpuSharedExpertWeights,
        routed_scaling_factor: float,
        *,
        swiglu_limit: float = 0.0,
    ) -> None:
        """Move a layer's replicated shared MLP into the A-side AF-EP V8 op."""
        self._shared_by_layer[layer_idx] = _BoundSharedExpert(
            weights=shared,
            routed_scaling_factor=float(routed_scaling_factor),
            swiglu_limit=float(swiglu_limit),
        )
        if shared.intermediate > self._max_shared_intermediate:
            self._max_shared_intermediate = shared.intermediate
            if self._init_args is not None:
                hidden_size, topk, dtype = self._init_args
                self._recompute_workspace_bytes(hidden_size, topk, dtype)

    def sync_forward_entry(self) -> None:
        self._session.sync_forward_entry()

    def _recompute_workspace_bytes(
        self, hidden_size: int, topk: int, dtype: torch.dtype
    ) -> None:
        self._workspace_bytes = (
            torch.ops.torch_xcpu.fused_af_a_dispatch_combine_v8_workspace_size(
                self._session.ep_size,
                self._session.max_rows_per_attention_rank,
                topk,
                hidden_size,
                dtype.itemsize,
                self._session.max_recv_tokens_per_rank,
                self._max_shared_intermediate,
            )
        )
        if (
            self._workspace is not None
            and self._workspace.numel() < self._workspace_bytes
        ):
            self._workspace = None

    def initialize(
        self,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        *,
        num_experts: int | None = None,
    ) -> None:
        if num_experts is not None:
            self._session.register_num_experts(num_experts)
        self._session.initialize(hidden_size, topk, dtype)
        self._init_args = (hidden_size, topk, dtype)
        self._recompute_workspace_bytes(hidden_size, topk, dtype)
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
        assert hidden_states.dtype in (torch.bfloat16, torch.float32)
        assert hidden_states.dim() == 2
        assert hidden_states.size(0) <= self._session.max_rows_per_attention_rank
        assert topk_ids.dim() == topk_weights.dim() == 2
        assert topk_ids.shape == topk_weights.shape
        assert topk_ids.size(0) <= hidden_states.size(0)
        assert topk_ids.size(1) in (3, 6, 8)
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

        _, hidden_size = hidden_states.shape
        topk = topk_ids.size(1)
        ep_size = self._session.ep_size
        # Non-divisible global expert counts (e.g. 256 experts on EP24) are
        # uniformly padded to ceil(G / F) * F, matching F's padded expert_map.
        local_experts = (num_experts + ep_size - 1) // ep_size
        padded_num_experts = local_experts * ep_size
        self._session.validate_initialized(hidden_size, topk, hidden_states.dtype)
        assert self._op is not None
        workspace = self._ensure_workspace(hidden_states, topk)
        output = torch.empty_like(hidden_states)
        bound = self._shared_by_layer.get(layer_idx)
        shared_args: tuple[object, ...]
        if bound is None:
            shared_args = (None, None, None, None, -1, 0, 0, None, 1.0, 0.0)
        else:
            s = bound.weights
            shared_args = (
                s.w13,
                s.w2,
                s.w13_scale,
                s.w2_scale,
                s.linear_backend,
                s.scale_block_size[0],
                s.scale_block_size[1],
                s.gate,
                bound.routed_scaling_factor,
                bound.swiglu_limit,
            )
        self._op(
            output,
            hidden_states,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.float().contiguous(),
            padded_num_experts,
            self._session.max_rows_per_attention_rank,
            layer_idx,
            self._session.metadata,
            self._session.communicator_handle,
            workspace,
            self._session.max_recv_tokens_per_rank,
            *shared_args,
        )
        return output

    def _ensure_workspace(self, hidden_states: torch.Tensor, topk: int) -> torch.Tensor:
        del topk
        if self._workspace is None:
            self._workspace = torch.empty(
                self._workspace_bytes, dtype=torch.uint8, device=hidden_states.device
            )
        return self._workspace
