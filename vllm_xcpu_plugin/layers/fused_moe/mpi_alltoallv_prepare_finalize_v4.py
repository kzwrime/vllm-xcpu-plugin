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
    TopKWeightAndReduceDelegate,
)

import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator
from vllm_xcpu_plugin.layers.fused_moe.expert_tokens_metadata import (
    XCPUExpertTokensMetadata,
)
from vllm_xcpu_plugin.layers.fused_moe.mpi_alltoallv_v4_plan import (
    compute_send_rounds,
    validate_single_sender_capacity,
)

logger = init_logger(__name__)


class MpiAlltoallvPrepareAndFinalizeV4(mk.FusedMoEPrepareAndFinalizeModular):
    """
    High-performance CPU implementation of Expert Parallel communication.

    Improvements over original:
    1. Removes all Python loops for index generation (vectorized).
    2. Uses C++ kernel for finalize (alltoallv + unpermute+reduce fused).
    3. Removes quantization overhead.
    """

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
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.max_recv_tokens = envs_xcpu.VLLM_XCPU_MOE_MAX_RECV_TOKENS
        self.num_dispatchers_ = num_dispatchers
        self.rank_expert_offset = rank_expert_offset
        self.is_sequence_parallel = is_sequence_parallel

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        if self.is_sequence_parallel:
            self.max_moe_tokens_per_rank = (
                self.max_num_tokens + sp_size - 1
            ) // sp_size
        else:
            self.max_moe_tokens_per_rank = self.max_num_tokens
        self.ep_rank = dist.get_rank(self.ep_group)
        self.ep_size = dist.get_world_size(self.ep_group)

        self.num_local_experts_ranks = self._compute_expert_distribution(
            num_experts, self.ep_size
        )
        self.experts_local_rank = self.num_local_experts_ranks[self.ep_rank]
        assert self.experts_local_rank == num_local_experts, (
            f"{self.num_local_experts_ranks}"
        )

        # Context storage for finalize phase
        # We need to know where to put the received data back
        self._sort_indices_back: torch.Tensor | None = None
        self.topk: int = -1

        # Communication metadata
        self._full_send_split_sizes: torch.Tensor | None = None
        self._recv_split_sizes: torch.Tensor | None = None
        self._send_count_overall: torch.Tensor | None = None
        self._send_rounds: list[tuple[int, int]] | None = None

        self._topk_weights: torch.Tensor | None = None

        self.ep_group_coordinator = get_ep_group()
        communicator = self.ep_group_coordinator.device_communicator
        assert isinstance(communicator, CpuMPICommunicator)
        self.comm_ptr_wrapper = communicator.comm_ptr_wrapper

        # Communication metadata tensor:
        # [ep_size, ep_rank, moe_tp_rank, moe_tp_size, dp_rank, dp_size]
        # v4 不使用模型级 TP 分片；TP 槽位仅为兼容底层公共元数据布局。
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

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_moe_tokens_per_rank

    def _compute_expert_distribution(
        self, num_experts: int, ep_size: int
    ) -> torch.Tensor:
        """Compute expert distribution across ranks (deterministic)."""
        base_experts = num_experts // ep_size
        extra_experts = num_experts % ep_size

        num_local_experts_ranks = torch.empty(ep_size, dtype=torch.int32)
        for i in range(ep_size):
            num_local_experts_ranks[i] = base_experts + (1 if i < extra_experts else 0)

        return num_local_experts_ranks

    def _get_static_buffer_size(self, topk: int, hidden_dim: int) -> int:
        del hidden_dim
        if self.max_recv_tokens > 0:
            return self.max_recv_tokens

        # v4 performs one MPI_Alltoallv over every EP sender. The receive
        # tensor therefore needs aggregate capacity, not merely DP capacity.
        return (
            self.ep_size
            * self.max_moe_tokens_per_rank
            * min(topk, self.num_local_experts)
        )

    def _check_single_sender_capacity(
        self,
        topk: int,
        static_buffer_size: int,
    ) -> None:
        max_num_local_experts = int(self.num_local_experts_ranks.max().item())

        validate_single_sender_capacity(
            topk=topk,
            max_num_local_experts=max_num_local_experts,
            max_tokens=self.max_moe_tokens_per_rank,
            max_recv_tokens=static_buffer_size,
        )

    def _collect_send_count_overall(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        topk_ids_cpu = topk_ids.to(device="cpu", dtype=torch.int64).reshape(-1)
        invalid_ids = (topk_ids_cpu < -1) | (topk_ids_cpu >= num_experts)
        if invalid_ids.any():
            invalid_id = int(topk_ids_cpu[invalid_ids][0].item())
            raise ValueError(
                f"Invalid expert ID {invalid_id}; expected -1 or an ID in "
                f"[0, {num_experts})"
            )
        expert_count = torch.bincount(
            topk_ids_cpu[topk_ids_cpu >= 0],
            minlength=num_experts,
        )

        expert_prefix = torch.empty(self.ep_size + 1, dtype=torch.int64)
        expert_prefix[0] = 0
        expert_prefix[1:] = torch.cumsum(
            self.num_local_experts_ranks.to(dtype=torch.int64), dim=0
        )

        local_send_counts = torch.empty(self.ep_size, dtype=torch.int64)
        for rank in range(self.ep_size):
            start = int(expert_prefix[rank].item())
            end = int(expert_prefix[rank + 1].item())
            local_send_counts[rank] = expert_count[start:end].sum()

        gathered = self.ep_group_coordinator.all_gather(local_send_counts, dim=0)
        send_count_overall = gathered.reshape(self.ep_size, self.ep_size)
        return send_count_overall

    def compute_send_rounds_for_input(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> list[tuple[int, int]]:
        _, hidden_dim = hidden_states.shape
        _, topk = topk_ids.shape
        static_buffer_size = self._get_static_buffer_size(topk, hidden_dim)
        self._check_single_sender_capacity(topk, static_buffer_size)
        send_count_overall = self._collect_send_count_overall(topk_ids, num_experts)
        send_rounds = compute_send_rounds(send_count_overall, static_buffer_size)
        return send_rounds

    def _check_recv_buffer_capacity(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
        static_buffer_size: int,
    ) -> torch.Tensor:
        send_count_overall = self._collect_send_count_overall(topk_ids, num_experts)
        send_rounds = compute_send_rounds(send_count_overall, static_buffer_size)
        self._send_count_overall = send_count_overall
        self._send_rounds = send_rounds

        recv_counts = send_count_overall.sum(dim=0)
        max_recv_tokens = int(recv_counts.max().item()) if self.ep_size > 0 else 0

        if max_recv_tokens > static_buffer_size:
            max_recv_rank = int(recv_counts.argmax().item())
            raise RuntimeError(
                "moe_prepare_fused_v4 receive token buffer overflow: "
                f"limit_tokens={static_buffer_size}, "
                f"max_recv_tokens={max_recv_tokens}, "
                f"max_recv_rank={max_recv_rank}, "
                f"planned_send_rounds={send_rounds}, ep_rank={self.ep_rank}, "
                f"local_send_tokens={send_count_overall[self.ep_rank].tolist()}"
            )

        return send_count_overall

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
        defer_input_quant: bool = False,
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
            defer_input_quant,
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
        assert a1.shape[0] <= self.max_moe_tokens_per_rank, (
            f"MoE input has {a1.shape[0]} tokens, exceeding the per-rank "
            f"capacity {self.max_moe_tokens_per_rank}; check "
            "--max-num-batched-tokens and use_sequence_parallel_moe"
        )

        from torch_xcpu import ops as xcpu_ops

        num_tokens, hidden_dim = a1.shape
        _, topk = topk_ids.shape
        device = a1.device
        self.topk = topk
        self._topk_weights = topk_weights

        total_tokens = num_tokens * topk

        # =============================================================================
        # Allocate tensors (shape-independent first, then shape-dependent)
        # =============================================================================

        # --- Communication metadata (persistent, cached) ---
        self._full_send_split_sizes = torch.empty(
            self.ep_size, dtype=torch.int32, device=device
        )
        self._recv_split_sizes = torch.empty(
            self.ep_size, dtype=torch.int32, device=device
        )

        # --- For fused operator ---
        sort_indices_back = torch.empty(total_tokens, dtype=torch.int32, device=device)
        expert_num_tokens = torch.empty(num_experts, dtype=torch.int32, device=device)
        num_input_rows_valid = torch.empty(1, dtype=torch.int32, device=device)

        static_buffer_size = self._get_static_buffer_size(topk, hidden_dim)
        self._check_single_sender_capacity(topk, static_buffer_size)

        self._check_recv_buffer_capacity(
            topk_ids.to(torch.int32), num_experts, static_buffer_size
        )

        recv_topk_ids = torch.full(
            (static_buffer_size,), -1, dtype=torch.int32, device=device
        )
        recv_hidden_states = torch.empty(
            (static_buffer_size, hidden_dim), dtype=a1.dtype, device=device
        )

        # --- Pre-allocate send_hidden_states (shape-dependent) ---
        # Size depends on local_tokens and hidden_dim
        send_hidden_states = torch.empty(
            total_tokens + self.ep_size, hidden_dim, dtype=a1.dtype, device=device
        )
        # send_hidden_states = torch.empty(
        #     local_tokens, hidden_dim, dtype=a1.dtype, device=device
        # )

        # --- Pre-allocate sort_indices (needed for phase2) ---
        sort_indices = torch.empty(
            total_tokens + self.ep_size, dtype=torch.int32, device=device
        )

        # Workspace is no longer used internally, pass empty tensor for compatibility
        workspace = torch.empty(0, dtype=torch.int32, device=device)

        # =============================================================================
        # Call fused operator (phase1 + alltoall + phase2 + alltoallv)
        # =============================================================================

        xcpu_ops.moe_prepare_fused_v4(
            sort_indices_back,  # output
            recv_hidden_states,  # output
            recv_topk_ids,  # output
            expert_num_tokens,  # output
            num_input_rows_valid,  # output
            self._recv_split_sizes,  # output
            self._full_send_split_sizes,  # output
            send_hidden_states,
            sort_indices,
            workspace,
            a1,
            topk_ids.to(torch.int32),
            static_buffer_size,
            num_experts,
            self.num_local_experts,
            hidden_dim,
            self._comm_metadata,
            self.comm_ptr_wrapper,
        )

        self._sort_indices_back = sort_indices_back

        local_expert_num_tokens = expert_num_tokens.narrow(
            0, self.rank_expert_offset, self.num_local_experts
        ).contiguous()
        expert_tokens_meta = XCPUExpertTokensMetadata(
            expert_num_tokens=local_expert_num_tokens,
            expert_num_tokens_cpu=local_expert_num_tokens.cpu(),
            num_input_rows_valid=num_input_rows_valid,
        )

        def _receiver() -> mk.PrepareResultType:
            ret_topk_ids = recv_topk_ids.unsqueeze(1)
            return (
                recv_hidden_states,
                None,  # no quant scale
                expert_tokens_meta,
                ret_topk_ids,
                # Experts 只使用 [C,T] shape; Finalize 使用原始 router weight。
                torch.empty_like(ret_topk_ids, dtype=topk_weights.dtype),
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
        from torch_xcpu import ops as xcpu_ops

        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate)

        finalize_send_sizes_tensor = self._recv_split_sizes
        finalize_recv_sizes_tensor = self._full_send_split_sizes

        device = output.device
        hidden_dim = output.size(1)
        M = output.size(0)
        total_output_tokens = M * self.topk

        # =============================================================================
        # Allocate tensors
        # =============================================================================

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

        # =============================================================================
        # Call moe_finalize
        # =============================================================================

        assert self._sort_indices_back is not None

        xcpu_ops.moe_finalize_v3(
            output,
            fused_expert_output,
            self._sort_indices_back,
            finalize_send_sizes_tensor,
            finalize_recv_sizes_tensor,
            self._comm_metadata,
            self.comm_ptr_wrapper,
            _finalize_recv_hidden_states,
            _finalize_workspace,
            self.topk,
            self._topk_weights,
        )
        self._topk_weights = None
        self._sort_indices_back = None
        self._full_send_split_sizes = None
        self._recv_split_sizes = None
        self._send_count_overall = None
        self._send_rounds = None

        def _receiver():
            pass

        return _receiver, lambda: None
