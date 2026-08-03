# SPDX-License-Identifier: Apache-2.0

"""MPI v5 EP with 2D token MoE prepare/finalize implementation."""

from collections.abc import Callable

import torch
import torch.distributed as dist
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator

from .expert_tokens_metadata import XCPUExpertTokensMetadata


class MpiAlltoallvPrepareAndFinalizeV5(mk.FusedMoEPrepareAndFinalizeModular):
    """V3-derived dispatch that sends a token once per destination rank."""

    def __init__(
        self,
        max_num_tokens: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank_expert_offset: int,
        dp_rank: int,
        dp_size: int,
        sp_size: int = 1,
        is_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.ep_group = ep_group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers
        self.rank_expert_offset = rank_expert_offset
        self.is_sequence_parallel = is_sequence_parallel

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.max_moe_tokens_per_rank = (
            (max_num_tokens + sp_size - 1) // sp_size
            if is_sequence_parallel
            else max_num_tokens
        )
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_size = dist.get_world_size(ep_group)

        if num_experts != self.ep_size * num_local_experts:
            raise ValueError("MPI v5 requires a uniform expert partition")

        communicator = get_ep_group().device_communicator
        assert isinstance(communicator, CpuMPICommunicator)
        self.comm_ptr_wrapper = communicator.comm_ptr_wrapper
        # [ep_size, ep_rank, moe_tp_rank, moe_tp_size, dp_rank, dp_size]
        # v5 不使用模型级 TP 分片；TP 槽位仅为兼容底层公共元数据布局。
        self._comm_metadata = torch.tensor(
            [
                self.ep_size,
                self.ep_rank,
                0,
                1,
                self.dp_rank,
                self.dp_size,
            ],
            dtype=torch.int64,
            device="cpu",  # Keep on CPU for C++ access
        )

        self._send_record_input_rows: torch.Tensor | None = None
        self._recv_input_rows_per_source: torch.Tensor | None = None
        self._send_input_rows_per_destination: torch.Tensor | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_moe_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

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
        quant_config: FusedMoEQuantConfig | None = None,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support "
                "defer_input_quant=True"
            )
        return self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )()

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig | None = None,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support "
                "defer_input_quant=True"
            )
        del quant_config
        assert not apply_router_weight_on_input
        # Linear EP still supplies the normal global-to-local expert map to
        # Experts. Prepare routes by the validated uniform global layout.
        del expert_map
        if a1.size(0) > self.max_moe_tokens_per_rank:
            raise ValueError(
                f"MoE input has {a1.size(0)} rows, capacity is "
                f"{self.max_moe_tokens_per_rank}"
            )
        if num_experts != self.num_experts:
            raise ValueError(
                f"num_experts changed from {self.num_experts} to {num_experts}"
            )

        from torch_xcpu import ops as xcpu_ops

        num_input_rows, hidden_dim = a1.shape
        topk = topk_ids.size(1)
        if topk < 1:
            raise ValueError("MPI v5 requires topk >= 1")
        device = a1.device

        # A source rank can send each local input row to this destination at
        # most once. Reserve one such segment for every source rank.
        recv_input_rows_capacity = self.ep_size * self.max_moe_tokens_per_rank
        send_input_rows_capacity = num_input_rows * min(topk, self.ep_size)

        send_record_input_rows = torch.empty(
            send_input_rows_capacity,
            dtype=torch.int32,
            device=device,
        )
        recv_hidden_states = torch.empty(
            recv_input_rows_capacity,
            hidden_dim,
            dtype=a1.dtype,
            device=device,
        )
        recv_topk_ids = torch.full(
            (
                recv_input_rows_capacity,
                topk,
            ),
            -1,
            dtype=torch.int32,
            device=device,
        )
        recv_topk_weights = torch.ones(
            (
                recv_input_rows_capacity,
                topk,
            ),
            dtype=topk_weights.dtype,
            device=device,
        )
        expert_num_tokens = torch.empty(num_experts, dtype=torch.int32, device=device)
        num_input_rows_valid = torch.empty(1, dtype=torch.int32, device=device)
        recv_input_rows_per_source = torch.empty(
            self.ep_size, dtype=torch.int32, device=device
        )
        send_input_rows_per_destination = torch.empty(
            self.ep_size, dtype=torch.int32, device=device
        )
        send_hidden_states = torch.empty(
            send_input_rows_capacity,
            hidden_dim,
            dtype=a1.dtype,
            device=device,
        )
        send_topk_ids = torch.empty(
            send_input_rows_capacity,
            topk,
            dtype=torch.int32,
            device=device,
        )
        send_topk_weights = torch.empty(
            send_input_rows_capacity,
            topk,
            dtype=topk_weights.dtype,
            device=device,
        )

        xcpu_ops.moe_prepare_fused_v5(
            send_record_input_rows,
            recv_hidden_states,
            recv_topk_ids,
            recv_topk_weights,
            expert_num_tokens,
            num_input_rows_valid,
            recv_input_rows_per_source,
            send_input_rows_per_destination,
            send_hidden_states,
            send_topk_ids,
            send_topk_weights,
            a1,
            topk_ids,
            topk_weights,
            num_experts,
            self.num_local_experts,
            self._comm_metadata,
            self.comm_ptr_wrapper,
        )

        self._send_record_input_rows = send_record_input_rows
        self._recv_input_rows_per_source = recv_input_rows_per_source
        self._send_input_rows_per_destination = send_input_rows_per_destination

        local_expert_num_tokens = expert_num_tokens.narrow(
            0, self.rank_expert_offset, self.num_local_experts
        ).contiguous()
        expert_tokens_meta = XCPUExpertTokensMetadata(
            expert_num_tokens=local_expert_num_tokens,
            expert_num_tokens_cpu=local_expert_num_tokens.cpu(),
            num_input_rows_valid=num_input_rows_valid,
        )

        def _receiver() -> mk.PrepareResultType:
            return (
                recv_hidden_states,
                None,
                expert_tokens_meta,
                recv_topk_ids,
                recv_topk_weights,
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
        receiver, _ = self.finalize_async(
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
    ) -> tuple[Callable, Callable]:
        del topk_weights, topk_ids
        if apply_router_weight_on_input:
            raise ValueError("MPI v5 requires expert-side output weighting")
        if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            raise TypeError("MPI v5 requires expert-side local route reduction")
        assert self._send_record_input_rows is not None
        assert self._recv_input_rows_per_source is not None
        assert self._send_input_rows_per_destination is not None

        from torch_xcpu import ops as xcpu_ops

        _, hidden_dim = output.shape
        recv_hidden_states = torch.empty(
            self._send_record_input_rows.numel(),
            hidden_dim,
            dtype=output.dtype,
            device=output.device,
        )
        workspace = torch.empty_like(output, dtype=torch.float32)
        xcpu_ops.moe_finalize_v5(
            output,
            fused_expert_output,
            self._send_record_input_rows,
            self._recv_input_rows_per_source,
            self._send_input_rows_per_destination,
            self._comm_metadata,
            self.comm_ptr_wrapper,
            recv_hidden_states,
            workspace,
        )

        self._send_record_input_rows = None
        self._recv_input_rows_per_source = None
        self._send_input_rows_per_destination = None

        return (lambda: None), (lambda: None)
