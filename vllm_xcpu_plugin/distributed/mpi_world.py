"""MPI process-world discovery for the attention, MoE and Engram role clusters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any


class ClusterType(IntEnum):
    ATTN = 0
    MOE = 1
    # Engram disaggregation: ranks that only serve DeepSeek-V4.1 Engram tables.
    ENGRAM = 2


@dataclass(frozen=True)
class MpiCluster:
    """One role group in the current MPI world."""

    instance_id: int
    cluster_type: ClusterType
    global_ranks: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.global_ranks)


@dataclass
class MpiWorld:
    """Expose the current cluster and the MPI world containing all clusters."""

    cluster_instance_id: int
    cluster_type: ClusterType
    cluster_comm: Any
    global_world_comm: Any
    clusters: Mapping[int, MpiCluster]
    # Role-pair communicators (attention ranks + one peer role), see pair_comm.
    _subcomms: dict[ClusterType, Any] = field(default_factory=dict, init=False)

    def ranks_of(self, cluster_type: ClusterType) -> tuple[int, ...]:
        """Global ranks of every cluster of this role, in instance order."""
        return tuple(
            rank
            for instance_id in sorted(self.clusters)
            if self.clusters[instance_id].cluster_type == cluster_type
            for rank in self.clusters[instance_id].global_ranks
        )

    def pair_comm(self, peer_role: ClusterType):
        """The communicator of the attention ranks plus the ``peer_role`` ranks.

        Ranks are ordered by global rank, so with the launcher placing the
        attention ranks first they occupy ``[0, A)`` of the pair and the peer
        ranks follow. Each disaggregation (AF-EP, Engram) exchanges on its own
        pair, which lets several coexist in one world.

        Creating it is a collective over the whole world (``MPI_Comm_split``):
        ``initialize_mpi_world`` creates every pair eagerly, in role order, so
        all ranks run the same sequence. A rank outside the pair holds None.
        """
        if peer_role == ClusterType.ATTN:
            raise ValueError("a role pair needs a non-attention peer role")
        if peer_role in self._subcomms:
            return self._subcomms[peer_role]
        member = self.cluster_type in (ClusterType.ATTN, peer_role)
        from mpi4py import MPI

        comm = self.global_world_comm.Split(
            color=0 if member else MPI.UNDEFINED, key=self.global_rank
        )
        if not member or comm == MPI.COMM_NULL:
            comm = None
        self._subcomms[peer_role] = comm
        return comm

    @property
    def cluster_rank(self) -> int:
        return self.cluster_comm.Get_rank()

    @property
    def cluster_size(self) -> int:
        return self.cluster_comm.Get_size()

    @property
    def global_rank(self) -> int:
        return self.global_world_comm.Get_rank()

    @property
    def global_size(self) -> int:
        return self.global_world_comm.Get_size()

    @property
    def cluster(self) -> MpiCluster:
        return self.clusters[self.cluster_instance_id]


_MPI_WORLD: MpiWorld | None = None


def initialize_mpi():
    """Import MPI using mpi4py's default initialization policy."""
    from mpi4py import MPI

    return MPI


def initialize_mpi_world(cluster_type: ClusterType) -> MpiWorld:
    """Split the global MPI world into Attention and MoE role domains."""
    global _MPI_WORLD
    cluster_type = ClusterType(cluster_type)
    if _MPI_WORLD is not None:
        assert _MPI_WORLD.cluster_type == cluster_type
        return _MPI_WORLD

    mpi = initialize_mpi()
    global_comm = mpi.COMM_WORLD
    print(
        f"XCPU MPI role={cluster_type.name} rank={global_comm.Get_rank()} "
        f"thread_provided={mpi.Query_thread()}",
        flush=True,
    )
    cluster_instance_id = int(cluster_type)

    local_comm = global_comm.Split(
        color=int(cluster_type),
        key=global_comm.Get_rank(),
    )

    descriptors = global_comm.allgather((
        cluster_instance_id,
        int(cluster_type),
        local_comm.Get_rank(),
        local_comm.Get_size(),
    ))
    clusters = _build_cluster_directory(descriptors)

    _MPI_WORLD = MpiWorld(
        cluster_instance_id=cluster_instance_id,
        cluster_type=cluster_type,
        cluster_comm=local_comm,
        global_world_comm=global_comm,
        clusters=MappingProxyType(clusters),
    )
    # Every rank builds every role pair, in the same order (collective).
    for peer_role in sorted({c.cluster_type for c in clusters.values()}):
        if peer_role != ClusterType.ATTN:
            _MPI_WORLD.pair_comm(peer_role)
    return _MPI_WORLD


def get_mpi_world() -> MpiWorld | None:
    return _MPI_WORLD


def require_mpi_world() -> MpiWorld:
    world = get_mpi_world()
    if world is None:
        raise RuntimeError("MPI world has not been initialized")
    return world


def _build_cluster_directory(
    descriptors: list[tuple[int, int, int, int]],
) -> dict[int, MpiCluster]:
    members: dict[int, dict[int, int]] = {}
    sizes: dict[int, int] = {}
    types: dict[int, ClusterType] = {}
    for global_rank, descriptor in enumerate(descriptors):
        instance_id, cluster_value, cluster_rank, cluster_size = descriptor
        assert instance_id >= 0
        cluster_type = ClusterType(cluster_value)
        assert cluster_size > 0 and 0 <= cluster_rank < cluster_size
        assert types.setdefault(instance_id, cluster_type) == cluster_type
        assert sizes.setdefault(instance_id, cluster_size) == cluster_size
        by_rank = members.setdefault(instance_id, {})
        assert cluster_rank not in by_rank
        by_rank[cluster_rank] = global_rank

    clusters = {}
    for instance_id, by_rank in members.items():
        assert set(by_rank) == set(range(sizes[instance_id]))
        clusters[instance_id] = MpiCluster(
            instance_id=instance_id,
            cluster_type=types[instance_id],
            global_ranks=tuple(by_rank[rank] for rank in range(sizes[instance_id])),
        )
    return clusters
