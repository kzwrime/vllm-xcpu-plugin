# SPDX-License-Identifier: Apache-2.0

"""MPI v6 destination-parallel fixed-record RMA prepare/finalize."""

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


class MpiAlltoallvPrepareAndFinalizeV6(mk.FusedMoEPrepareAndFinalizeModular):
    """Dispatch one fixed record per input row and destination rank."""

    version = "v6"

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
            raise ValueError(f"MPI {self.version} requires a uniform expert partition")

        communicator = get_ep_group().device_communicator
        assert isinstance(communicator, CpuMPICommunicator)
        self.comm_ptr_wrapper = communicator.comm_ptr_wrapper
        # [ep_size, ep_rank, moe_tp_rank, moe_tp_size, dp_rank, dp_size].
        # V6 uses no model-level TP sharding; the TP fields preserve the common
        # communication metadata layout.
        self._comm_metadata = torch.tensor(
            [self.ep_size, self.ep_rank, 0, 1, self.dp_rank, self.dp_size],
            dtype=torch.int64,
            device="cpu",
        )

        self._return_row_indices: torch.Tensor | None = None
        self._recv_input_rows_per_source: torch.Tensor | None = None
        self._send_input_rows_per_destination: torch.Tensor | None = None
        self._dispatch_send_buffer: torch.Tensor | None = None

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
        if (
            not defer_input_quant
            and quant_config is not None
            and quant_config.quant_dtype is not None
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support activation "
                "quantization in Prepare. It only dispatches unquantized "
                "activations."
            )

        assert not apply_router_weight_on_input
        # Linear EP supplies its normal global-to-local expert map to Experts;
        # V6 routes using the validated uniform global expert layout.
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

        num_input_rows, hidden_dim = a1.shape
        topk = topk_ids.size(1)
        if topk not in (6, 8):
            raise ValueError(f"MPI v6 supports only topk=6 or topk=8, got {topk}")

        from torch_xcpu import ops as xcpu_ops

        device = a1.device
        recv_input_rows_capacity = self.ep_size * self.max_moe_tokens_per_rank
        return_row_indices = torch.empty(
            num_input_rows,
            topk,
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
            (recv_input_rows_capacity, topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        recv_topk_weights = torch.ones(
            (recv_input_rows_capacity, topk),
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

        # Fixed record:
        # [num_records, source_input_row, topk_ids, topk_weights, hidden_state].
        dispatch_record_bytes = (
            2 * torch.int32.itemsize
            + topk * (torch.int32.itemsize + topk_weights.element_size())
            + hidden_dim * a1.element_size()
        )
        dispatch_send_buffer = torch.empty(
            self.ep_size * self.max_moe_tokens_per_rank * dispatch_record_bytes,
            dtype=torch.uint8,
            device=device,
        )

        xcpu_ops.moe_prepare_fused_v6(
            return_row_indices,
            recv_hidden_states,
            recv_topk_ids,
            recv_topk_weights,
            expert_num_tokens,
            num_input_rows_valid,
            recv_input_rows_per_source,
            send_input_rows_per_destination,
            dispatch_send_buffer,
            a1,
            topk_ids,
            topk_weights,
            num_experts,
            self.num_local_experts,
            self._comm_metadata,
            self.comm_ptr_wrapper,
        )

        # Keep all asynchronous kernel inputs/outputs alive until finalize.
        self._return_row_indices = return_row_indices
        self._recv_input_rows_per_source = recv_input_rows_per_source
        self._send_input_rows_per_destination = send_input_rows_per_destination
        self._dispatch_send_buffer = dispatch_send_buffer

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
            raise ValueError(
                f"MPI {self.version} requires expert-side output weighting"
            )
        if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            raise TypeError(
                f"MPI {self.version} requires expert-side local route reduction"
            )

        assert self._return_row_indices is not None
        assert self._recv_input_rows_per_source is not None
        assert self._send_input_rows_per_destination is not None
        assert self._dispatch_send_buffer is not None

        from torch_xcpu import ops as xcpu_ops

        workspace = torch.empty_like(output, dtype=torch.float32)
        xcpu_ops.moe_finalize_v6(
            output,
            fused_expert_output,
            self._return_row_indices,
            self._recv_input_rows_per_source,
            self._comm_metadata,
            self.comm_ptr_wrapper,
            workspace,
            self.max_moe_tokens_per_rank,
        )

        self._return_row_indices = None
        self._recv_input_rows_per_source = None
        self._send_input_rows_per_destination = None
        self._dispatch_send_buffer = None
        return (lambda: None), (lambda: None)
