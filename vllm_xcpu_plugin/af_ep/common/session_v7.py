"""V7 transport state shared by the Attention and MoE roles.

The common package shares code across roles, not across transport versions.
MpiWorld owns generic process discovery and the attention/MoE pair
communicator; PairSession the rank layout and tensor metadata ABI; this
module adds V7's fixed capacity and collective initialization contract.
"""

from __future__ import annotations

import torch

import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin.disagg import PairSession
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld

# Metadata ABI revision understood by the V7 C++ endpoints, not the backend's
# generation number. Keep this value aligned with their kProtocolVersion.
AF_PROTOCOL_VERSION = 1


class AfV7Session(PairSession):
    """Hold one endpoint's V7 inputs and cache its initialized model shape.

    This consumes an already connected MPI world; it does not connect jobs or
    negotiate a transport version. initialize() collectively validates the V7
    configuration and creates its windows through torch_xcpu on the
    first call.
    """

    protocol_version = AF_PROTOCOL_VERSION
    peer_role = ClusterType.MOE
    name = "AF-EP"

    def __init__(
        self,
        mpi_world: MpiWorld,
        *,
        max_rows_per_attention_rank: int,
    ) -> None:
        assert max_rows_per_attention_rank > 0
        super().__init__(mpi_world)
        self.num_expert_ranks = self.num_peer_ranks
        self.ep_size = self.num_expert_ranks
        self._forward_allreduce = envs_xcpu.VLLM_XCPU_AF_FORWARD_ALLREDUCE
        self.max_rows_per_attention_rank = max_rows_per_attention_rank
        self._scheduler_capacity = max_rows_per_attention_rank
        self._layer_capacity = 0
        self._model_shape: tuple[int, int, torch.dtype] | None = None

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

    @property
    def expert_capacity(self) -> int:
        return self.num_attention_ranks * self.max_rows_per_attention_rank

    def sync_forward_entry(self) -> None:
        if self._forward_allreduce:
            assert self.allreduce(1) == (
                self.num_attention_ranks + self.num_expert_ranks
            )

    def initialize(self, hidden_size: int, topk: int, dtype: torch.dtype) -> None:
        """Create windows after model load, before any forward is submitted."""
        model_shape = (hidden_size, topk, dtype)
        if self._model_shape is not None:
            if self._model_shape != model_shape:
                raise RuntimeError(
                    f"AF V7 model shape changed: {model_shape} != {self._model_shape}"
                )
            return

        from torch_xcpu import ops as xcpu_ops

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
        xcpu_ops.moe_af_v7_initialize(
            self.max_rows_per_attention_rank,
            hidden_size,
            topk,
            dtype,
            self.metadata,
            self.communicator_handle,
        )
        self._model_shape = model_shape
        print(
            f"AF-EP V7 transport ready: role={self.cluster_type.name} "
            f"rank={self.role_rank} hidden_size={hidden_size} topk={topk} "
            f"max_rows_per_attention_rank={self.max_rows_per_attention_rank}",
            flush=True,
        )

    def validate_initialized(
        self, hidden_size: int, topk: int, dtype: torch.dtype
    ) -> None:
        """Check a forward locally; never create windows from model execution."""
        if self._model_shape is None:
            raise RuntimeError(
                "AF V7 transport must be initialized before model execution"
            )
        model_shape = (hidden_size, topk, dtype)
        if self._model_shape != model_shape:
            raise RuntimeError(
                f"AF V7 model shape changed: {model_shape} != {self._model_shape}"
            )
