"""V7 transport state shared by the Attention and MoE roles.

The common package shares code across roles, not across transport versions.
MpiWorld owns generic process discovery; this module encodes V7's rank layout,
tensor metadata ABI, fixed capacity and collective initialization contract.
"""

from __future__ import annotations

import torch

import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld

# Metadata ABI revision understood by the V7 C++ endpoints, not the backend's
# generation number. Keep this value aligned with their kProtocolVersion.
AF_PROTOCOL_VERSION = 1


class AfV7Session:
    """Hold one endpoint's V7 inputs and cache its initialized model shape.

    This consumes an already connected MPI world; it does not connect jobs or
    negotiate a transport version. initialize() collectively validates the V7
    configuration and creates its windows through torch_xcpu on the first call.
    """

    def __init__(
        self,
        mpi_world: MpiWorld,
        *,
        max_rows_per_attention_rank: int,
    ) -> None:
        assert max_rows_per_attention_rank > 0
        num_attention_ranks, num_expert_ranks, role_rank = self._validate_world(
            mpi_world
        )

        self.cluster_type = mpi_world.cluster_type
        self.cluster_instance_id = mpi_world.cluster_instance_id
        self.cluster_rank = mpi_world.cluster_rank
        self.role_rank = role_rank
        self.num_attention_ranks = num_attention_ranks
        self.num_expert_ranks = num_expert_ranks
        self.ep_size = num_expert_ranks
        self.metadata = torch.tensor(
            [
                AF_PROTOCOL_VERSION,
                mpi_world.global_size,
                mpi_world.global_rank,
                self.num_attention_ranks,
                self.num_expert_ranks,
                self.num_attention_ranks,
                int(self.cluster_type),
                self.role_rank,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self.communicator_handle = torch.tensor(
            [mpi_world.global_world_comm.py2f()],
            dtype=torch.int64,
            device="cpu",
        )
        self._global_world_comm = mpi_world.global_world_comm
        self._forward_allreduce = envs_xcpu.VLLM_XCPU_AF_FORWARD_ALLREDUCE
        self.max_rows_per_attention_rank = max_rows_per_attention_rank
        self._model_shape: tuple[int, int, torch.dtype] | None = None

    @property
    def expert_capacity(self) -> int:
        return self.num_attention_ranks * self.max_rows_per_attention_rank

    def sync_forward_entry(self) -> None:
        if self._forward_allreduce:
            assert self._global_world_comm.allreduce(1) == (
                self.num_attention_ranks + self.num_expert_ranks
            )

    @staticmethod
    def _validate_world(mpi_world: MpiWorld) -> tuple[int, int, int]:
        clusters = tuple(
            mpi_world.clusters[instance_id]
            for instance_id in sorted(mpi_world.clusters)
        )
        attention_ranks = tuple(
            rank
            for cluster in clusters
            if cluster.cluster_type == ClusterType.ATTN
            for rank in cluster.global_ranks
        )
        expert_ranks = tuple(
            rank
            for cluster in clusters
            if cluster.cluster_type == ClusterType.MOE
            for rank in cluster.global_ranks
        )
        assert attention_ranks and expert_ranks
        expected_attention_ranks = tuple(range(len(attention_ranks)))
        expected_expert_ranks = tuple(
            range(len(attention_ranks), len(attention_ranks) + len(expert_ranks))
        )
        if (
            attention_ranks != expected_attention_ranks
            or expert_ranks != expected_expert_ranks
        ):
            raise ValueError(
                "AF-EP requires all Attention cluster instances before all MoE "
                f"cluster instances: Attention={attention_ranks}, MoE={expert_ranks}"
            )
        role_ranks = (
            attention_ranks
            if mpi_world.cluster_type == ClusterType.ATTN
            else expert_ranks
        )
        assert mpi_world.global_rank in role_ranks
        role_rank = role_ranks.index(mpi_world.global_rank)
        return len(attention_ranks), len(expert_ranks), role_rank

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
            f"rank={self.role_rank} hidden_size={hidden_size} topk={topk}",
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
