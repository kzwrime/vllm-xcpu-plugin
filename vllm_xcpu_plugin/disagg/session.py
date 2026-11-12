"""One rank's view of an attention <-> peer-role pair."""

from __future__ import annotations

from typing import Any

import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiWorld


class PairSession:
    """Transport metadata of one role pair, as the native endpoints parse it.

    ``metadata`` (int64 CPU, 8 elements) is ``[protocol, pair_size,
    pair_rank, num_attention_ranks, num_peer_ranks, peer_rank_offset (==
    num_attention_ranks), role (ClusterType value), role_rank]``, and
    ``communicator_handle`` holds the pair communicator's Fortran handle.
    Subclasses set ``protocol_version`` and ``peer_role`` and add the
    collective initialization of their transport.
    """

    protocol_version: int
    peer_role: ClusterType
    name: str = "disaggregation"

    def __init__(self, mpi_world: MpiWorld) -> None:
        cluster_type = mpi_world.cluster_type
        if cluster_type not in (ClusterType.ATTN, self.peer_role):
            raise ValueError(
                f"{self.name} session on a {cluster_type.name} rank; expected "
                f"ATTN or {self.peer_role.name}"
            )
        attention_ranks = mpi_world.ranks_of(ClusterType.ATTN)
        peer_ranks = mpi_world.ranks_of(self.peer_role)
        if not attention_ranks or not peer_ranks:
            raise ValueError(
                f"{self.name} needs attention and {self.peer_role.name} ranks in "
                f"the MPI world: attention={attention_ranks}, "
                f"{self.peer_role.name.lower()}={peer_ranks}"
            )
        # The pair communicator orders ranks by global rank; the endpoints
        # expect the attention ranks first, so they must precede the peers.
        if max(attention_ranks) >= min(peer_ranks):
            raise ValueError(
                f"{self.name} requires all attention ranks before all "
                f"{self.peer_role.name} ranks: attention={attention_ranks}, "
                f"{self.peer_role.name.lower()}={peer_ranks}"
            )
        own_ranks = attention_ranks if cluster_type == ClusterType.ATTN else peer_ranks
        if mpi_world.global_rank not in own_ranks:
            raise ValueError(f"rank {mpi_world.global_rank} is not in its own role")
        pair_comm = mpi_world.pair_comm(self.peer_role)
        if pair_comm is None:
            raise RuntimeError(f"{self.name}: this rank is outside the role pair")
        pair_ranks = sorted(attention_ranks + peer_ranks)
        pair_size, pair_rank = pair_comm.Get_size(), pair_comm.Get_rank()
        if pair_size != len(pair_ranks) or pair_rank != pair_ranks.index(
            mpi_world.global_rank
        ):
            raise RuntimeError(
                f"{self.name}: pair communicator {pair_rank}/{pair_size} does not "
                f"match the role directory {pair_ranks}"
            )

        self.cluster_type = cluster_type
        self.cluster_instance_id = mpi_world.cluster_instance_id
        self.cluster_rank = mpi_world.cluster_rank
        self.global_rank = mpi_world.global_rank
        self.role_rank = own_ranks.index(mpi_world.global_rank)
        self.num_attention_ranks = len(attention_ranks)
        self.num_peer_ranks = len(peer_ranks)
        self.pair_comm: Any = pair_comm
        self.pair_size = pair_size
        self.pair_rank = pair_rank
        self.metadata = torch.tensor(
            [
                self.protocol_version,
                pair_size,
                pair_rank,
                self.num_attention_ranks,
                self.num_peer_ranks,
                self.num_attention_ranks,
                int(cluster_type),
                self.role_rank,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self.communicator_handle = torch.tensor(
            [pair_comm.py2f()], dtype=torch.int64, device="cpu"
        )

    @property
    def is_attention(self) -> bool:
        return self.cluster_type == ClusterType.ATTN

    def allgather(self, value):
        """Collective over the pair (attention ranks first)."""
        return self.pair_comm.allgather(value)

    def allreduce(self, value):
        return self.pair_comm.allreduce(value)
