"""V8 transport state shared by the Attention and MoE roles.

The common package shares code across roles, not across transport versions.
MpiWorld owns generic process discovery and the attention/MoE pair
communicator; PairSession the rank layout and tensor metadata ABI; this
module adds V8's v5-aligned fixed-region window and collective initialization
contract (including optional EP receive-quota chunking and multi-layout window
reuse).
"""

from __future__ import annotations

import torch

import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin.disagg import PairSession
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld

# Metadata ABI revision understood by the V8 C++ endpoints, not the backend's
# generation number. Keep this value aligned with their kProtocolVersion.
AF_PROTOCOL_VERSION = 2


class AfV8Session(PairSession):
    """Hold one endpoint's V8 inputs and cache its initialized model shape.

    This consumes an already connected MPI world; it does not connect jobs or
    negotiate a transport version. initialize() collectively validates the V8
    configuration and allocates its role's v5-protocol MPI window through
    torch_xcpu on the first call (and supports multi-layout reuse).
    """

    protocol_version = AF_PROTOCOL_VERSION
    peer_role = ClusterType.MOE
    name = "AF-EP"

    def __init__(
        self,
        mpi_world: MpiWorld,
        *,
        max_rows_per_attention_rank: int,
        max_recv_tokens_per_rank: int | None = None,
    ) -> None:
        assert max_rows_per_attention_rank > 0
        super().__init__(mpi_world)
        self.num_expert_ranks = self.num_peer_ranks
        self.ep_size = self.num_expert_ranks
        self._forward_allreduce = envs_xcpu.VLLM_XCPU_AF_FORWARD_ALLREDUCE
        self.max_rows_per_attention_rank = max_rows_per_attention_rank
        self._scheduler_capacity = max_rows_per_attention_rank
        self._layer_capacity = 0
        quota = (
            envs_xcpu.VLLM_XCPU_MOE_EP_MAX_RECV_TOKENS_PER_RANK
            if max_recv_tokens_per_rank is None
            else int(max_recv_tokens_per_rank)
        )
        if quota < 0:
            raise ValueError("AF-EP receive quota must be nonnegative")
        self.max_recv_tokens_per_rank = quota
        self._num_experts: int | None = None
        self.local_experts: int | None = None
        self._model_shape: tuple[int, int, torch.dtype] | None = None
        self._initialized_layouts: set[tuple[int, int, int, int, int, torch.dtype]] = (
            set()
        )

    def register_layer_capacity(self, sp_size: int) -> None:
        """A registers actual per-layer SP, before collective initialization."""
        if self.cluster_type != ClusterType.ATTN or self._model_shape is not None:
            raise RuntimeError(
                "AF layer capacity must be registered on A before initialize"
            )
        if sp_size < 1:
            raise ValueError("AF SP size must be positive")
        rows = (self._scheduler_capacity + sp_size - 1) // sp_size
        self._layer_capacity = max(self._layer_capacity, rows)

    def register_num_experts(self, num_experts: int) -> None:
        """Record logical global expert count so A derives padded E per F rank."""
        if num_experts <= 0:
            raise ValueError("AF num_experts must be positive")
        if self._num_experts is not None and self._num_experts != num_experts:
            raise ValueError(
                f"AF num_experts changed from {self._num_experts} to {num_experts}"
            )
        self._num_experts = num_experts

    @property
    def expert_capacity(self) -> int:
        return self.num_attention_ranks * self.max_rows_per_attention_rank

    def sync_forward_entry(self) -> None:
        if self._forward_allreduce:
            assert self.allreduce(1) == (
                self.num_attention_ranks + self.num_expert_ranks
            )

    def initialize(
        self,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        *,
        local_experts: int | None = None,
        max_recv_tokens_per_rank: int | None = None,
    ) -> None:
        """Allocate role MPI window after model load, before any forward."""
        model_shape = (hidden_size, topk, dtype)
        if max_recv_tokens_per_rank is not None:
            if max_recv_tokens_per_rank < 0:
                raise ValueError("AF-EP receive quota must be nonnegative")
            self.max_recv_tokens_per_rank = int(max_recv_tokens_per_rank)

        if local_experts is None:
            if self._num_experts is not None:
                local_experts = (
                    self._num_experts + self.ep_size - 1
                ) // self.ep_size
            elif self.local_experts is not None:
                local_experts = self.local_experts
            else:
                local_experts = 1
        if local_experts <= 0:
            raise ValueError("AF local_experts must be positive")

        if self._model_shape is not None:
            if self._model_shape != model_shape:
                raise RuntimeError(
                    f"AF V8 model shape changed: {model_shape} != {self._model_shape}"
                )
            layout_key = (
                self.max_rows_per_attention_rank,
                hidden_size,
                topk,
                local_experts,
                self.max_recv_tokens_per_rank,
                dtype,
            )
            if layout_key in self._initialized_layouts:
                return
            torch.accelerator.synchronize()
            torch.ops.torch_xcpu.fused_af_v8_initialize_windows(
                self.max_rows_per_attention_rank,
                topk,
                hidden_size,
                local_experts,
                dtype.itemsize,
                self.max_recv_tokens_per_rank,
                self.metadata,
                self.communicator_handle,
            )
            self._initialized_layouts.add(layout_key)
            return

        # Loading/weight conversion may have queued device work. Complete it
        # before entering MPI on this host thread.
        torch.accelerator.synchronize()
        # F has no SP execution: A reports its post-SP capacity. Use the
        # largest A/layer requirement for fixed-slot transport, never divide on F.
        proposal = (
            self._layer_capacity or self._scheduler_capacity
            if self.cluster_type == ClusterType.ATTN
            else 0
        )
        capacities = self.allgather(proposal)
        if (
            len(capacities) != self.num_attention_ranks + self.num_expert_ranks
            or any(n <= 0 for n in capacities[: self.num_attention_ranks])
            or any(n != 0 for n in capacities[self.num_attention_ranks :])
        ):
            raise ValueError("AF capacity negotiation has invalid role proposals")
        self.max_rows_per_attention_rank = max(capacities)
        torch.ops.torch_xcpu.fused_af_v8_initialize_windows(
            self.max_rows_per_attention_rank,
            topk,
            hidden_size,
            local_experts,
            dtype.itemsize,
            self.max_recv_tokens_per_rank,
            self.metadata,
            self.communicator_handle,
        )
        self.local_experts = local_experts
        self._model_shape = model_shape
        self._initialized_layouts.add((
            self.max_rows_per_attention_rank,
            hidden_size,
            topk,
            local_experts,
            self.max_recv_tokens_per_rank,
            dtype,
        ))
        print(
            f"AF-EP V8 transport ready: role={self.cluster_type.name} "
            f"rank={self.role_rank} hidden_size={hidden_size} topk={topk} "
            f"local_experts={local_experts} "
            f"max_rows_per_attention_rank={self.max_rows_per_attention_rank} "
            f"recv_quota={self.max_recv_tokens_per_rank}",
            flush=True,
        )

    def validate_initialized(
        self, hidden_size: int, topk: int, dtype: torch.dtype
    ) -> None:
        """Check a forward locally before submitting transport operations."""
        if self._model_shape is None:
            raise RuntimeError(
                "AF V8 transport must be initialized before model execution"
            )
        model_shape = (hidden_size, topk, dtype)
        if self._model_shape != model_shape:
            raise RuntimeError(
                f"AF V8 model shape changed: {model_shape} != {self._model_shape}"
            )
