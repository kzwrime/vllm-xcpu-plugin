"""MPI process-world discovery for Attention and MoE role clusters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any


class ClusterType(IntEnum):
    ATTN = 0
    MOE = 1


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
    # Reserved for special-purpose communicators once a concrete consumer
    # defines membership, creation order, tag ownership, and lifecycle.
    _subcomms: dict[str, Any] = field(default_factory=dict, init=False)

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
