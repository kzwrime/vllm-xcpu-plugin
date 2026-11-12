import pytest

import vllm_xcpu_plugin.distributed.mpi_world as mpi_world_module
from vllm_xcpu_plugin.distributed.mpi_world import (
    ClusterType,
    _build_cluster_directory,
)


def test_initialize_splits_single_application_world_by_role(monkeypatch) -> None:
    local_comm = type(
        "LocalComm", (), {"Get_rank": lambda self: 0, "Get_size": lambda self: 2}
    )()
    pair_comm = type(
        "PairComm", (), {"Get_rank": lambda self: 4, "Get_size": lambda self: 6}
    )()

    class GlobalComm:
        def Get_rank(self):
            return 4

        def Get_size(self):
            return 6

        def Split(self, color, key):
            # Role split, then the attention/MoE pair (all ranks, in order).
            assert key == 4
            if color == int(ClusterType.MOE):
                return local_comm
            assert color == 0
            return pair_comm

        def allgather(self, descriptor):
            assert descriptor == (1, int(ClusterType.MOE), 0, 2)
            return [
                (0, int(ClusterType.ATTN), 0, 4),
                (0, int(ClusterType.ATTN), 1, 4),
                (0, int(ClusterType.ATTN), 2, 4),
                (0, int(ClusterType.ATTN), 3, 4),
                (1, int(ClusterType.MOE), 0, 2),
                (1, int(ClusterType.MOE), 1, 2),
            ]

    fake_mpi = type(
        "MPI",
        (),
        {"COMM_WORLD": GlobalComm(), "Query_thread": staticmethod(lambda: 3)},
    )()
    monkeypatch.setattr(mpi_world_module, "_MPI_WORLD", None)
    monkeypatch.setattr(mpi_world_module, "initialize_mpi", lambda: fake_mpi)

    world = mpi_world_module.initialize_mpi_world(ClusterType.MOE)

    assert world.cluster_rank == 0
    assert world.cluster_size == 2
    assert world.clusters[0].global_ranks == (0, 1, 2, 3)
    assert world.clusters[1].global_ranks == (4, 5)
    assert world.pair_comm(ClusterType.MOE) is pair_comm
    assert world.ranks_of(ClusterType.ATTN) == (0, 1, 2, 3)


def test_pair_comm_excludes_other_roles(monkeypatch) -> None:
    """A rank outside a pair splits with MPI.UNDEFINED and holds None."""
    from types import SimpleNamespace

    from vllm_xcpu_plugin.distributed.mpi_world import MpiCluster, MpiWorld

    calls = []

    class Comm:
        def Get_rank(self):
            return 2

        def Split(self, color, key):
            calls.append((color, key))
            return "null" if color == "undefined" else "pair"

    fake_mpi = SimpleNamespace(UNDEFINED="undefined", COMM_NULL="null")
    monkeypatch.setitem(
        __import__("sys").modules, "mpi4py", SimpleNamespace(MPI=fake_mpi)
    )
    world = MpiWorld(
        cluster_instance_id=1,
        cluster_type=ClusterType.MOE,
        cluster_comm=None,
        global_world_comm=Comm(),
        clusters={
            0: MpiCluster(0, ClusterType.ATTN, (0, 1)),
            1: MpiCluster(1, ClusterType.MOE, (2,)),
            2: MpiCluster(2, ClusterType.ENGRAM, (3,)),
        },
    )
    assert world.pair_comm(ClusterType.ENGRAM) is None
    assert world.pair_comm(ClusterType.MOE) == "pair"
    assert world.pair_comm(ClusterType.MOE) == "pair"  # cached
    assert calls == [("undefined", 2), (0, 2)]


def test_cluster_directory_preserves_local_rank_order() -> None:
    clusters = _build_cluster_directory([
        (0, 0, 0, 2),
        (0, 0, 1, 2),
        (1, 0, 1, 2),
        (1, 0, 0, 2),
        (2, 1, 0, 1),
    ])

    assert clusters[0].cluster_type == ClusterType.ATTN
    assert clusters[0].global_ranks == (0, 1)
    assert clusters[1].cluster_type == ClusterType.ATTN
    assert clusters[1].global_ranks == (3, 2)
    assert clusters[2].cluster_type == ClusterType.MOE
    assert clusters[2].global_ranks == (4,)


@pytest.mark.parametrize(
    "descriptors",
    [
        [(0, 0, 0, 2)],
        [(0, 0, 0, 2), (0, 0, 0, 2)],
        [(0, 0, 0, 2), (0, 0, 1, 3)],
        [(0, 0, 0, 2), (0, 1, 1, 2)],
        [(0, 7, 0, 1)],
    ],
)
def test_cluster_directory_rejects_inconsistent_membership(descriptors) -> None:
    with pytest.raises((AssertionError, ValueError)):
        _build_cluster_directory(descriptors)
